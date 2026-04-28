from __future__ import annotations

from copy import deepcopy
import math
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
)
from torchrl.data import Composite as CompositeSpec, LazyTensorStorage, TensorDictReplayBuffer, TensorSpec
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.modules.distributions import TanhNormalWithEntropy
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    CatTensors,
)
from active_adaptation.learning.ppo.ppo_base import PPOBase

from .common import NullVecNorm
from .action_bounds import (
    coerce_action_bounds_config,
    default_action_bounds,
    resolve_action_bounds,
)

BOOTSTRAP_KEY = "bootstrap"
ACTOR_INPUT_KEY = "_actor_input"
CRITIC_INPUT_KEY = "_critic_input"
Q_LOGITS_KEY = "_q_logits"


def _build_mlp(
    input_dim: int | None,
    hidden_dims: Sequence[int],
    *,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for idx, dim in enumerate(hidden_dims):
        if idx == 0 and last_dim is None:
            layers.append(nn.LazyLinear(dim))
        else:
            layers.append(nn.Linear(last_dim, dim))
        layers.append(nn.LayerNorm(dim) if use_layer_norm else nn.Identity())
        layers.append(nn.SiLU())
        last_dim = dim
    return nn.Sequential(*layers)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()

    expanded_mask = mask
    while expanded_mask.ndim < value.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(value)
    denom = expanded_mask.sum().clamp_min(1)
    return (value * expanded_mask).sum() / denom


def _safe_shape(spec: CompositeSpec, key: str) -> int:
    if key not in spec.keys(True, True):
        raise KeyError(f"Missing required observation key: {key!r}")
    return int(spec[key].shape[-1])


class FastSACActorCore(nn.Module):
    def __init__(
        self,
        action_dim: int,
        *,
        hidden_dim: int = 512,
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.net = _build_mlp(None, hidden_dims, use_layer_norm=use_layer_norm)
        last_dim = hidden_dims[-1]
        self.fc_mu = nn.Linear(last_dim, action_dim)
        self.fc_logstd = nn.Linear(last_dim, action_dim)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, actor_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(actor_input)
        loc = self.fc_mu(hidden)
        log_std = self.fc_logstd(hidden)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        loc = torch.clamp(loc, -10.0, 10.0)
        loc = torch.nan_to_num(loc, nan=0.0)
        log_std = torch.nan_to_num(log_std, nan=self.log_std_min)
        scale = torch.nan_to_num(
            log_std.exp(),
            nan=math.exp(self.log_std_min),
        )
        return loc, scale


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        *,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.net = _build_mlp(None, hidden_dims, use_layer_norm=use_layer_norm)
        self.out = nn.Linear(hidden_dims[-1], num_atoms)

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        hidden = self.net(critic_input)
        return self.out(hidden)

    def projection(
        self,
        critic_input: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(critic_input), dim=-1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        lower_indices = torch.clamp((lower + offset).reshape(-1), 0, proj_dist.numel() - 1)
        upper_indices = torch.clamp((upper + offset).reshape(-1), 0, proj_dist.numel() - 1)
        proj_dist.reshape(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).reshape(-1))
        proj_dist.reshape(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).reshape(-1))
        return proj_dist


class DistributionalCritic(nn.Module):
    q_support: torch.Tensor

    def __init__(
        self,
        *,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    num_atoms=num_atoms,
                    v_min=v_min,
                    v_max=v_max,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_q_networks)
            ]
        )
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        outputs = [qnet(critic_input) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
        self,
        critic_input: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        projections = [
            qnet.projection(
                critic_input,
                rewards,
                bootstrap,
                discount,
                self.q_support,
                self.q_support.device,
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.q_support, dim=-1)


class DistributionalCriticTD(TensorDictModuleBase):
    def __init__(
        self,
        *,
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.in_keys = [OBS_KEY, CMD_KEY, OBS_PRIV_KEY, ACTION_KEY]
        self.out_keys = [Q_LOGITS_KEY]
        self.cat_tensors = CatTensors(
            self.in_keys,
            CRITIC_INPUT_KEY,
            del_keys=False,
            sort=False,
        )
        self.model = DistributionalCritic(
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
            device=device,
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.cat_tensors(tensordict)
        tensordict.set(
            self.out_keys[0],
            self.model(tensordict[CRITIC_INPUT_KEY]).permute(1, 0, 2),
        )
        return tensordict

    def projection(
        self,
        tensordict: TensorDictBase,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        projection_td = tensordict.copy()
        self.cat_tensors(projection_td)
        return self.model.projection(
            projection_td[CRITIC_INPUT_KEY],
            rewards,
            bootstrap,
            discount,
        ).permute(1, 0, 2)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.model.q_support, dim=-1)


@dataclass
class FastSACConfig:
    _target_: str = f"{__package__}.fast_sac.FastSAC"

    name: str = "fast_sac"
    collect_steps: int = 1
    # Effective replay capacity = buffer_size * collect_steps * num_envs.
    buffer_size: int = 1024
    replay_batch_size: int = 4096
    # Effective transition warmup = warm_up_steps * collect_steps * num_envs.
    warm_up_steps: int = 128
    updates_per_step: int = 4
    policy_frequency: int = 2

    gamma: float = 0.99
    tau: float = 0.05
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 1e-3
    target_entropy_ratio: float = 1.0
    weight_decay: float = 1e-3

    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    action_bounds: dict[str, list[float]] = field(
        default_factory=default_action_bounds
    )
    action_min: float | None = None
    action_max: float | None = None
    num_atoms: int = 501
    v_min: float = -50.0
    v_max: float = 200.0
    actor_q_reduce: str = "min"
    log_std_max: float = 0.0
    log_std_min: float = -4.0
    use_layer_norm: bool = True
    max_grad_norm: float = 1.0

    vecnorm: bool = True
    freeze_vecnorm: bool = False
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)
    grad_sync_mode: str | None = "manual"

    def __post_init__(self) -> None:
        self.actor_q_reduce = str(self.actor_q_reduce).lower()
        if self.actor_q_reduce not in {"min", "mean", "q0", "q1"}:
            raise ValueError(
                "actor_q_reduce must be one of {'min', 'mean', 'q0', 'q1'}, "
                f"got {self.actor_q_reduce!r}"
            )

        if isinstance(self.grad_sync_mode, str):
            self.grad_sync_mode = self.grad_sync_mode.lower()
            if self.grad_sync_mode in {"none", "null"}:
                self.grad_sync_mode = None

        if self.grad_sync_mode not in {"manual", None, "ddp"}:
            raise ValueError(
                "grad_sync_mode must be one of {'manual', None, 'ddp'}, "
                f"got {self.grad_sync_mode!r}"
            )
        self.action_bounds = coerce_action_bounds_config(
            self.action_bounds,
            action_min=self.action_min,
            action_max=self.action_max,
        )


cs = ConfigStore.instance()
cs.store("fast_sac", node=FastSACConfig(), group="algo")


class FastSAC(PPOBase):
    def __init__(
        self,
        cfg: FastSACConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ) -> None:
        super().__init__()
        self.cfg = FastSACConfig(**cfg)
        if aa.is_distributed() and self.cfg.grad_sync_mode == "ddp":
            raise NotImplementedError("FastSAC only supports manual gradient sync.")

        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        object.__setattr__(self, "env", env)

        observation_keys = set(observation_spec.keys(True, True))
        missing_keys = sorted({OBS_KEY, CMD_KEY, OBS_PRIV_KEY}.difference(observation_keys))
        if missing_keys:
            raise KeyError(f"Missing required observation keys: {missing_keys}")

        self.num_envs = int(getattr(env, "num_envs", observation_spec.shape[0]))
        self.action_dim = int(env.action_manager.action_dim)
        self.joint_names = env.action_manager.joint_names
        self.gradient_step = 0

        self._build_vecnorm_modules(observation_spec)

        action_min, action_max = resolve_action_bounds(
            self.cfg.action_bounds,
            self.joint_names,
            self.device,
        )

        self.actor = ProbabilisticActor(
            module=Seq(
                CatTensors(
                    [OBS_KEY, CMD_KEY],
                    ACTOR_INPUT_KEY,
                    del_keys=False,
                    sort=False,
                ),
                Mod(
                    FastSACActorCore(
                        self.action_dim,
                        hidden_dim=self.cfg.actor_hidden_dim,
                        log_std_max=self.cfg.log_std_max,
                        log_std_min=self.cfg.log_std_min,
                        use_layer_norm=self.cfg.use_layer_norm,
                    ),
                    [ACTOR_INPUT_KEY],
                    ["loc", "scale"],
                ),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=TanhNormalWithEntropy,
            distribution_kwargs={
                "low": action_min,
                "high": action_max,
                "event_dims": 1,
            },
            return_log_prob=True,
        ).to(self.device)

        self.qnet = DistributionalCriticTD(
            num_atoms=self.cfg.num_atoms,
            v_min=self.cfg.v_min,
            v_max=self.cfg.v_max,
            hidden_dim=self.cfg.critic_hidden_dim,
            use_layer_norm=self.cfg.use_layer_norm,
            device=self.device,
        ).to(self.device)

        fake_input = observation_spec.zero()
        fake_critic_input = fake_input.copy()
        fake_critic_input.set(
            ACTION_KEY,
            torch.zeros(
                (*fake_input.batch_size, self.action_dim),
                device=self.device,
            ),
        )
        with VecNorm.freeze():
            self.vecnorm(fake_input)
            self.actor.get_dist(fake_input)
            self.qnet(fake_critic_input)

        self.qnet_target = deepcopy(self.qnet).to(self.device)
        self.qnet_target.requires_grad_(False)

        self.temperature = nn.ParameterDict(
            {"log_alpha": nn.Parameter(torch.tensor(math.log(self.cfg.alpha_init), device=self.device))}
        ).to(self.device)
        self.target_entropy = -float(self.action_dim) * self.cfg.target_entropy_ratio

        fused = str(self.device).startswith("cuda")
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.cfg.actor_lr,
            weight_decay=self.cfg.weight_decay,
            fused=fused,
            betas=(0.9, 0.95),
        )
        self.q_optimizer = torch.optim.AdamW(
            self.qnet.parameters(),
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
            fused=fused,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = torch.optim.AdamW(
            self.temperature.parameters(),
            lr=self.cfg.alpha_lr,
            weight_decay=0.0,
            fused=fused,
            betas=(0.9, 0.95),
        )

        self.replay_buffer_capacity = self.cfg.buffer_size * self.cfg.collect_steps * self.num_envs
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.replay_buffer_capacity),
            batch_size=self.cfg.replay_batch_size,
            prefetch=2,
        )

        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            self._broadcast_parameters()
        else:
            self.world_size = 1

    @property
    def log_alpha(self) -> torch.Tensor:
        return self.temperature["log_alpha"]

    def _build_vecnorm_modules(self, observation_spec: CompositeSpec) -> None:
        modules = []
        self.vecnorms: Mapping[str, VecNorm] = nn.ModuleDict()
        vecnorm_cls = VecNorm if self.cfg.vecnorm else NullVecNorm
        for key in (OBS_KEY, CMD_KEY, OBS_PRIV_KEY):
            if key not in observation_spec.keys(True, True):
                continue
            shape = observation_spec[key].shape[-1:]
            vecnorm = vecnorm_cls(input_shape=shape, stats_shape=shape, decay=0.9999)
            self.vecnorms[key] = vecnorm
            modules.append(Mod(vecnorm, [key], [key]))
        self.vecnorm = Seq(*modules).to(self.device)

    def _broadcast_parameters(self) -> None:
        with torch.no_grad():
            for module in (self.vecnorm, self.actor, self.qnet, self.qnet_target, self.temperature):
                for param in module.parameters():
                    dist.broadcast(param, src=0)
                for buf in module.buffers():
                    dist.broadcast(buf, src=0)

    @torch.no_grad()
    def _all_reduce_grads(self, *modules: nn.Module | nn.ParameterDict) -> None:
        for module in modules:
            for param in module.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    def _sync_vecnorms(self) -> None:
        if not aa.is_distributed() or not self.cfg.vecnorm:
            return
        for vecnorm in self.vecnorms.values():
            vecnorm.synchronize(mode="broadcast")

    def _run_frozen_vecnorm(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.cfg.vecnorm:
            return tensordict
        with VecNorm.freeze():
            self.vecnorm(tensordict)
            bootstrap_td = tensordict.get(BOOTSTRAP_KEY, None)
            if bootstrap_td is not None:
                self.vecnorm(bootstrap_td)
        return tensordict

    @torch.no_grad()
    def _update_vecnorm_from_batch(self, tensordict: TensorDictBase) -> None:
        if not self.cfg.vecnorm:
            return
        bootstrap_td = tensordict.get(BOOTSTRAP_KEY, None)
        for key, vecnorm in self.vecnorms.items():
            values = [tensordict[key].reshape(-1, tensordict[key].shape[-1])]
            if bootstrap_td is not None:
                values.append(
                    bootstrap_td[key].reshape(-1, bootstrap_td[key].shape[-1])
                )
            vecnorm._update(torch.cat(values, dim=0))

    def _sample_actor(
        self,
        tensordict: TensorDictBase,
    ) -> tuple[TensorDictBase, TanhNormalWithEntropy]:
        dist = self.actor.get_dist(tensordict)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        tensordict.set(ACTION_KEY, action)
        tensordict.set(f"{ACTION_KEY}_log_prob", log_prob)
        return tensordict, dist

    def _reduce_actor_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.actor_q_reduce == "min":
            return q_values.min(dim=1).values
        if self.cfg.actor_q_reduce == "mean":
            return q_values.mean(dim=1)
        if self.cfg.actor_q_reduce == "q0":
            return q_values[:, 0]
        if q_values.shape[1] < 2:
            raise ValueError(
                "actor_q_reduce='q1' requires at least two Q heads."
            )
        return q_values[:, 1]

    def _get_current_iter(self) -> int:
        return int(getattr(self.env, "current_iter", 0))

    def get_next_saved_keys(self) -> tuple[str, ...]:
        return (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)

    def make_tensordict_primer(self):
        return TensorDictPrimer({}, reset_key="done", expand_specs=False)

    def on_stage_start(self, stage: str) -> None:
        del stage
        return None

    def _reward_total(self, tensordict: TensorDictBase) -> torch.Tensor:
        reward = tensordict[REWARD_KEY]
        if reward.shape[-1] != 1:
            reward = reward.sum(-1, keepdim=True)
        return reward.squeeze(-1)

    def _discount(self, tensordict: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = tensordict["next"].get("terminated", None)
        if terminated is None:
            terminated = tensordict[DONE_KEY]
        terminated = terminated.float().squeeze(-1)

        discount = tensordict["next"].get("discount", None)
        if discount is None:
            bootstrap = 1.0 - terminated
            discount = torch.full_like(bootstrap, self.cfg.gamma)
            return bootstrap, discount
        bootstrap = (1.0 - terminated) * discount.float().squeeze(-1)
        return bootstrap, torch.full_like(bootstrap, self.cfg.gamma)

    def _collect_replay_data(self, tensordict: TensorDictBase) -> TensorDictBase:
        keys: list[Union[str, tuple[str, str]]] = [
            OBS_KEY,
            CMD_KEY,
            OBS_PRIV_KEY,
            ACTION_KEY,
            DONE_KEY,
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
            ("next", "discount"),
            REWARD_KEY,
        ]
        if "is_init" in tensordict.keys(True, True):
            keys.append("is_init")
        replay_td = tensordict.select(*keys, strict=False)
        next_td = tensordict["next"]
        for key in (OBS_KEY, CMD_KEY, OBS_PRIV_KEY):
            replay_td.set((BOOTSTRAP_KEY, key), next_td[key])
        return replay_td

    def observe(self, tensordict: TensorDictBase) -> None:
        self.replay_buffer.extend(self._collect_replay_data(tensordict).reshape(-1).cpu())

    def _update_critic(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        rewards = self._reward_total(tensordict)
        bootstrap, discount = self._discount(tensordict)

        with torch.no_grad():
            bootstrap_td = tensordict[BOOTSTRAP_KEY].copy()
            bootstrap_td, _ = self._sample_actor(bootstrap_td)
            next_log_probs = bootstrap_td[f"{ACTION_KEY}_log_prob"]
            adjusted_rewards = (
                rewards
                - discount
                * bootstrap
                * self.log_alpha.exp().detach()
                * next_log_probs
            )
            target_distributions = self.qnet_target.projection(
                bootstrap_td,
                adjusted_rewards,
                bootstrap,
                discount,
            )
            target_values = self.qnet_target.get_value(target_distributions)

        critic_td = tensordict.copy()
        self.qnet(critic_td)
        q_outputs = critic_td[Q_LOGITS_KEY]
        critic_log_probs = F.log_softmax(q_outputs, dim=-1).clamp(min=-30.0)
        critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        q_loss = sum(
            _masked_mean(critic_losses[:, i], mask)
            for i in range(critic_losses.shape[1])
        )

        reward_mean = _masked_mean(rewards.detach(), mask)
        reward_max = rewards.detach().max()
        reward_min = rewards.detach().min()
        target_q_mean = _masked_mean(target_values.mean(dim=1).detach(), mask)
        target_q_max = target_values.detach().max()
        target_q_min = target_values.detach().min()
        target_clamp_hi = (target_values.detach() >= (self.cfg.v_max - 1e-4)).float().mean()
        target_clamp_lo = (target_values.detach() <= (self.cfg.v_min + 1e-4)).float().mean()
        diagnostics = {
            "reward/mean": reward_mean.detach(),
            "reward/max": reward_max.detach(),
            "reward/min": reward_min.detach(),
            "critic/target_q_mean": target_q_mean.detach(),
            "critic/target_q_max": target_q_max.detach(),
            "critic/target_q_min": target_q_min.detach(),
            "critic/target_vmax_frac": target_clamp_hi.detach(),
            "critic/target_vmin_frac": target_clamp_lo.detach(),
            "policy/next_log_prob_mean": next_log_probs.detach().mean(),
            "policy/next_log_prob_min": next_log_probs.detach().min(),
            "policy/next_log_prob_max": next_log_probs.detach().max(),
        }

        return q_loss, target_values.mean(dim=1).detach(), next_log_probs.detach(), diagnostics

    def _update_actor(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        actor_td = tensordict.copy()
        actor_td, _ = self._sample_actor(actor_td)
        self.qnet(actor_td)
        q_outputs = actor_td[Q_LOGITS_KEY]
        q_probs = F.softmax(q_outputs, dim=-1)
        q_values = self.qnet.get_value(q_probs)
        q_value = self._reduce_actor_q_values(q_values)
        log_probs = actor_td[f"{ACTION_KEY}_log_prob"]
        actor_loss = _masked_mean(
            self.log_alpha.exp().detach() * log_probs - q_value,
            mask,
        )
        action_std = actor_td["scale"].mean(dim=-1)
        diagnostics = {
            "actor/q_value": _masked_mean(q_value.detach(), mask),
            "policy/log_prob_mean": _masked_mean(log_probs.detach(), mask),
            "policy/log_prob_min": log_probs.detach().min(),
            "policy/log_prob_max": log_probs.detach().max(),
        }
        return actor_loss, (-log_probs).detach(), action_std.detach(), diagnostics

    def _update_alpha(self, next_log_probs: torch.Tensor) -> torch.Tensor:
        alpha_loss = -(
            self.log_alpha.exp() * (next_log_probs.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.temperature)
        self.alpha_optimizer.step()
        return alpha_loss.detach()

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)

    def _update_step(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        tensordict = tensordict.copy()
        self._update_vecnorm_from_batch(tensordict)
        self._run_frozen_vecnorm(tensordict)
        mask = None
        if "is_init" in tensordict.keys(True, True):
            valid = ~tensordict["is_init"].squeeze(-1)
            mask = valid if valid.any() else None

        q_loss, target_values, next_log_probs, critic_diag = self._update_critic(tensordict, mask)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.qnet)
        if self.cfg.max_grad_norm > 0:
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.qnet.parameters(),
                self.cfg.max_grad_norm,
            )
        else:
            q_grad_norm = torch.zeros((), device=self.device)
        self.q_optimizer.step()

        actor_updated = self.gradient_step % self.cfg.policy_frequency == 0
        if actor_updated:
            actor_loss, entropy, action_std, actor_diag = self._update_actor(tensordict, mask)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
                self._all_reduce_grads(self.actor)
            if self.cfg.max_grad_norm > 0:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.cfg.max_grad_norm,
                )
            else:
                actor_grad_norm = torch.zeros((), device=self.device)
            self.actor_optimizer.step()
        else:
            actor_loss = torch.zeros((), device=self.device)
            entropy = torch.zeros_like(actor_loss)
            action_std = torch.zeros_like(actor_loss)
            actor_grad_norm = torch.zeros_like(actor_loss)
            actor_diag = {
                "actor/q_value": torch.zeros((), device=self.device),
                "policy/log_prob_mean": torch.zeros((), device=self.device),
                "policy/log_prob_min": torch.zeros((), device=self.device),
                "policy/log_prob_max": torch.zeros((), device=self.device),
            }

        alpha_loss = self._update_alpha(next_log_probs)
        self._soft_update_target()
        self.gradient_step += 1

        metrics = {
            "critic/loss": q_loss.detach(),
            "critic/grad_norm": q_grad_norm.detach(),
            "critic/target_q_mean": target_values.mean().detach(),
            "actor/loss": actor_loss.detach(),
            "actor/entropy": entropy.mean().detach(),
            "actor/action_std": action_std.mean().detach(),
            "actor/grad_norm": actor_grad_norm.detach(),
            "actor/updated": torch.tensor(float(actor_updated), device=self.device),
            "alpha/loss": alpha_loss.detach(),
            "alpha/value": self.log_alpha.exp().detach(),
        }
        metrics.update(critic_diag)
        metrics.update(actor_diag)
        return metrics

    def update(self) -> dict[str, float]:
        info: dict[str, float] = {
            "rb_size": float(len(self.replay_buffer)),
            "alpha/value": self.log_alpha.exp().item(),
        }
        warmup_transitions = self.cfg.warm_up_steps * self.cfg.collect_steps * self.num_envs
        if len(self.replay_buffer) < min(
            max(warmup_transitions, self.cfg.replay_batch_size),
            self.replay_buffer.storage.max_size,
        ):
            self._sync_vecnorms()
            self.num_updates += 1
            return info

        metric_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(self.cfg.updates_per_step):
            batch = self.replay_buffer.sample().to(self.device)
            step_metrics = self._update_step(batch)
            for key, value in step_metrics.items():
                metric_lists[key].append(value.detach())

        self._sync_vecnorms()
        for key, values in metric_lists.items():
            info[key] = torch.stack(values).float().mean().item()
        info["rb_size"] = float(len(self.replay_buffer))
        info["gradient_step"] = float(self.gradient_step)
        self.num_updates += 1
        return info

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        del mode, critic
        rollout_policy = Seq(
            self.vecnorm,
            self.actor,
            selected_out_keys=[f"{ACTION_KEY}_log_prob", ACTION_KEY, "loc", "scale"],
        )
        rollout_policy.forward = VecNorm.freeze()(rollout_policy.forward)
        return rollout_policy

    def train_op(self, tensordict: TensorDictBase) -> dict[str, float]:
        self.observe(tensordict.exclude("stats"))
        return self.update()

    def compute_value(self, tensordict: TensorDictBase) -> TensorDictBase:
        work_td = tensordict.copy()
        with torch.no_grad():
            self._run_frozen_vecnorm(work_td)
            work_td, _ = self._sample_actor(work_td)
            self.qnet(work_td)
            q_probs = F.softmax(work_td[Q_LOGITS_KEY], dim=-1)
            q_values = self.qnet.get_value(q_probs)
            q_value = self._reduce_actor_q_values(q_values).unsqueeze(-1)
        tensordict.set("state_value", q_value)
        return tensordict

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["gradient_step"] = self.gradient_step
        state_dict["num_updates"] = self.num_updates
        state_dict["last_iter"] = self._get_current_iter()
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            module_state = state_dict.get(name, {})
            try:
                module.load_state_dict(module_state, strict=strict)
                succeed_keys.append(name)
            except Exception as exc:
                warnings.warn(f"Failed to load state dict for {name}: {str(exc)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        self.gradient_step = int(state_dict.get("gradient_step", 0))
        self.num_updates = int(state_dict.get("num_updates", 0))
        start_iter = int(state_dict.get("last_iter", 0))
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)

        return failed_keys
