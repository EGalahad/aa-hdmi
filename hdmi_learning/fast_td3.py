from __future__ import annotations

import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Mapping, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.data import Composite as CompositeSpec, LazyTensorStorage, TensorDictReplayBuffer, TensorSpec
from torchrl.envs.transforms import TensorDictPrimer

import active_adaptation as aa
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
from .fast_sac import (
    DistributionalCritic,
    _build_mlp,
    _masked_mean,
    _safe_shape,
)


class FastTD3ActorCore(nn.Module):
    action_min: torch.Tensor
    action_max: torch.Tensor
    action_center: torch.Tensor
    action_scale: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        init_scale: float = 0.01,
        use_layer_norm: bool = True,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ) -> None:
        super().__init__()
        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.net = _build_mlp(input_dim, hidden_dims, use_layer_norm=use_layer_norm)
        self.fc_action = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.normal_(self.fc_action.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_action.bias, 0.0)

        action_min = action_min.to(dtype=torch.float32)
        action_max = action_max.to(dtype=torch.float32)
        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)
        self.register_buffer("action_center", (action_max + action_min) * 0.5)
        self.register_buffer("action_scale", (action_max - action_min) * 0.5)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        hidden = self.net(actor_input)
        action = torch.tanh(self.fc_action(hidden))
        return self.action_center + self.action_scale * action


class TD3ExplorationNoise(nn.Module):
    action_min: torch.Tensor
    action_max: torch.Tensor
    log_std_min: torch.Tensor
    log_std_max: torch.Tensor
    noise_scales: torch.Tensor

    def __init__(
        self,
        *,
        action_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("action_min", action_min.to(dtype=torch.float32))
        self.register_buffer("action_max", action_max.to(dtype=torch.float32))
        self.register_buffer("log_std_min", torch.tensor(log_std_min, dtype=torch.float32))
        self.register_buffer("log_std_max", torch.tensor(log_std_max, dtype=torch.float32))
        self.register_buffer("noise_scales", torch.empty(0, 1))

    def _sample_scales(self, batch_size: int, device: torch.device) -> torch.Tensor:
        std_min = self.log_std_min.exp()
        std_max = self.log_std_max.exp()
        return torch.rand(batch_size, 1, device=device) * (std_max - std_min) + std_min

    def _ensure_noise_scales(self, batch_size: int, device: torch.device) -> None:
        if self.noise_scales.shape != (batch_size, 1) or self.noise_scales.device != device:
            self.noise_scales = self._sample_scales(batch_size, device)

    def forward(
        self,
        action: torch.Tensor,
        is_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        batch_size = action.shape[0]
        self._ensure_noise_scales(batch_size, action.device)

        if is_init is not None:
            reset_mask = is_init.reshape(batch_size, -1).any(dim=-1, keepdim=True)
            if reset_mask.any():
                new_scales = self._sample_scales(batch_size, action.device)
                self.noise_scales = torch.where(reset_mask, new_scales, self.noise_scales)

        noise = torch.randn_like(action) * self.noise_scales
        noisy_action = torch.max(
            torch.min(action + noise, self.action_max),
            self.action_min,
        )
        return (noisy_action,)


@dataclass
class FastTD3Config:
    _target_: str = f"{__package__}.fast_td3.FastTD3"

    name: str = "fast_td3"
    collect_steps: int = 1
    # Effective replay capacity = buffer_size * collect_steps * num_envs.
    buffer_size: int = 1024
    replay_batch_size: int = 4096
    # Effective transition warmup = warm_up_steps * collect_steps * num_envs.
    warm_up_steps: int = 128
    updates_per_step: int = 4
    policy_frequency: int = 2

    gamma: float = 0.97
    tau: float = 0.01
    actor_lr: float = 3e-4
    # critic_lr: float = 3e-4
    critic_lr: float = 1e-3
    weight_decay: float = 1e-3

    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 512
    init_scale: float = 0.01
    action_min: float = -5.0
    action_max: float = 5.0
    num_atoms: int = 101
    v_min: float = -100.0
    v_max: float = 400.0
    log_std_max: float = 0.0
    log_std_min: float = -1.0
    policy_noise: float = 0.1
    noise_clip: float = 0.2
    use_cdq: bool = True
    use_layer_norm: bool = True
    max_grad_norm: float = 1.0

    vecnorm: bool = True
    freeze_vecnorm: bool = False
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)
    grad_sync_mode: str | None = "manual"

    def __post_init__(self) -> None:
        if isinstance(self.grad_sync_mode, str):
            self.grad_sync_mode = self.grad_sync_mode.lower()
            if self.grad_sync_mode in {"none", "null"}:
                self.grad_sync_mode = None

        if self.grad_sync_mode not in {"manual", None, "ddp"}:
            raise ValueError(
                "grad_sync_mode must be one of {'manual', None, 'ddp'}, "
                f"got {self.grad_sync_mode!r}"
            )


cs = ConfigStore.instance()
cs.store("fast_td3", node=FastTD3Config(), group="algo")


class FastTD3(PPOBase):
    def __init__(
        self,
        cfg: FastTD3Config,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ) -> None:
        super().__init__()
        self.cfg = FastTD3Config(**cfg)
        if aa.is_distributed() and self.cfg.grad_sync_mode == "ddp":
            raise NotImplementedError("FastTD3 only supports manual gradient sync.")

        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        object.__setattr__(self, "env", env)

        self.obs_dim = _safe_shape(observation_spec, OBS_KEY)
        self.cmd_dim = _safe_shape(observation_spec, CMD_KEY)
        self.priv_dim = _safe_shape(observation_spec, OBS_PRIV_KEY)
        self.num_envs = int(getattr(env, "num_envs", observation_spec.shape[0]))
        self.action_dim = int(env.action_manager.action_dim)
        self.joint_names = env.action_manager.joint_names
        self.gradient_step = 0

        self._build_vecnorm_modules(observation_spec)

        actor_input_dim = self.obs_dim + self.cmd_dim
        critic_input_dim = self.obs_dim + self.cmd_dim + self.priv_dim + self.action_dim
        action_min = torch.full(
            (self.action_dim,),
            float(self.cfg.action_min),
            device=self.device,
            dtype=torch.float32,
        )
        action_max = torch.full(
            (self.action_dim,),
            float(self.cfg.action_max),
            device=self.device,
            dtype=torch.float32,
        )
        if not torch.all(action_max > action_min):
            raise ValueError(
                f"action_max must be greater than action_min, got {self.cfg.action_min} and {self.cfg.action_max}"
            )

        actor_core = FastTD3ActorCore(
            actor_input_dim,
            self.action_dim,
            hidden_dim=self.cfg.actor_hidden_dim,
            init_scale=self.cfg.init_scale,
            use_layer_norm=self.cfg.use_layer_norm,
            action_min=action_min,
            action_max=action_max,
        ).to(self.device)
        object.__setattr__(self, "actor_core", actor_core)
        self.actor = Seq(
            CatTensors([OBS_KEY, CMD_KEY], "_actor_input", del_keys=False, sort=False),
            Mod(actor_core, ["_actor_input"], [ACTION_KEY]),
        ).to(self.device)

        self.actor_target = FastTD3ActorCore(
            actor_input_dim,
            self.action_dim,
            hidden_dim=self.cfg.actor_hidden_dim,
            init_scale=self.cfg.init_scale,
            use_layer_norm=self.cfg.use_layer_norm,
            action_min=action_min,
            action_max=action_max,
        ).to(self.device)
        self.actor_target.load_state_dict(actor_core.state_dict())
        self.actor_target.requires_grad_(False)

        self.qnet = DistributionalCritic(
            critic_input_dim,
            num_atoms=self.cfg.num_atoms,
            v_min=self.cfg.v_min,
            v_max=self.cfg.v_max,
            hidden_dim=self.cfg.critic_hidden_dim,
            use_layer_norm=self.cfg.use_layer_norm,
            device=self.device,
        ).to(self.device)
        self.qnet_target = DistributionalCritic(
            critic_input_dim,
            num_atoms=self.cfg.num_atoms,
            v_min=self.cfg.v_min,
            v_max=self.cfg.v_max,
            hidden_dim=self.cfg.critic_hidden_dim,
            use_layer_norm=self.cfg.use_layer_norm,
            device=self.device,
        ).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_target.requires_grad_(False)

        self.exploration = Mod(
            TD3ExplorationNoise(
                action_dim=self.action_dim,
                log_std_min=self.cfg.log_std_min,
                log_std_max=self.cfg.log_std_max,
                action_min=action_min,
                action_max=action_max,
            ),
            [ACTION_KEY, "is_init"],
            [ACTION_KEY],
        ).to(self.device)

        fused = str(self.device).startswith("cuda")
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_core.parameters(),
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
            for module in (
                self.vecnorm,
                self.actor,
                self.actor_target,
                self.qnet,
                self.qnet_target,
                self.exploration,
            ):
                for param in module.parameters():
                    dist.broadcast(param, src=0)
                for buf in module.buffers():
                    dist.broadcast(buf, src=0)

    @torch.no_grad()
    def _all_reduce_grads(self, *modules: nn.Module) -> None:
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
        discount = tensordict["next"].get("discount", None)
        if discount is None:
            bootstrap = (1.0 - tensordict[DONE_KEY].float()).squeeze(-1)
            discount = torch.full_like(bootstrap, self.cfg.gamma)
            return bootstrap, discount
        bootstrap = discount.float().squeeze(-1)
        return bootstrap, torch.full_like(bootstrap, self.cfg.gamma)

    def _encode_inputs(self, tensordict: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
        with VecNorm.freeze():
            features = []
            for key in (OBS_KEY, CMD_KEY, OBS_PRIV_KEY):
                value = tensordict[key]
                if key in self.vecnorms:
                    value = self.vecnorms[key](value)
                features.append(value)

        obs, cmd, priv = features
        actor_input = torch.cat([obs, cmd], dim=-1)
        critic_prefix = torch.cat([obs, cmd, priv], dim=-1)
        return actor_input, critic_prefix

    def _reduce_q_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_cdq:
            return torch.minimum(q_values[0], q_values[1])
        return q_values.mean(dim=0)

    def _collect_replay_data(self, tensordict: TensorDictBase) -> TensorDictBase:
        keys: list[Union[str, tuple[str, str]]] = [
            OBS_KEY,
            CMD_KEY,
            OBS_PRIV_KEY,
            ACTION_KEY,
            DONE_KEY,
            ("next", "discount"),
            REWARD_KEY,
            ("next", OBS_KEY),
            ("next", CMD_KEY),
            ("next", OBS_PRIV_KEY),
        ]
        if "is_init" in tensordict.keys(True, True):
            keys.append("is_init")
        return tensordict.select(*keys, strict=False)

    def observe(self, tensordict: TensorDictBase) -> None:
        self.replay_buffer.extend(self._collect_replay_data(tensordict).reshape(-1).cpu())

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor_core.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)
            for target_param, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)

    def _update_critic(
        self,
        tensordict: TensorDictBase,
        critic_prefix: torch.Tensor,
        next_actor_input: torch.Tensor,
        next_critic_prefix: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = self._reward_total(tensordict)
        bootstrap, discount = self._discount(tensordict)

        with torch.no_grad():
            next_actions = self.actor_target(next_actor_input)
            target_noise = torch.randn_like(next_actions) * self.cfg.policy_noise
            target_noise = target_noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_actions = torch.maximum(
                torch.minimum(next_actions + target_noise, self.actor_target.action_max),
                self.actor_target.action_min,
            )
            target_distributions = self.qnet_target.projection(
                torch.cat([next_critic_prefix, next_actions], dim=-1),
                rewards,
                bootstrap,
                discount,
            )
            target_values = self.qnet_target.get_value(target_distributions)
            if self.cfg.use_cdq:
                min_distribution = torch.where(
                    (target_values[0] < target_values[1]).unsqueeze(-1),
                    target_distributions[0],
                    target_distributions[1],
                )
                target_distributions = torch.stack([min_distribution, min_distribution], dim=0)
            target_summary = self._reduce_q_values(target_values).detach()

        critic_logits = self.qnet(torch.cat([critic_prefix, tensordict[ACTION_KEY]], dim=-1))
        critic_log_probs = F.log_softmax(critic_logits, dim=-1).clamp(min=-30.0)
        critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        critic_loss = sum(_masked_mean(loss, mask) for loss in critic_losses.unbind(0))

        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.qnet)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.cfg.max_grad_norm)
        self.q_optimizer.step()
        return critic_loss, critic_grad_norm, target_summary.mean().detach(), target_summary.detach()

    def _update_actor(
        self,
        actor_input: torch.Tensor,
        critic_prefix: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = self.actor_core(actor_input)
        actor_logits = self.qnet(torch.cat([critic_prefix, actions], dim=-1))
        actor_values = self._reduce_q_values(self.qnet.get_value(F.softmax(actor_logits, dim=-1)))
        actor_loss = _masked_mean(-actor_values, mask)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.actor_core)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_core.parameters(),
            self.cfg.max_grad_norm,
        )
        self.actor_optimizer.step()
        self._soft_update_target()
        return (
            actor_loss.detach(),
            actor_grad_norm.detach(),
            actor_values.mean().detach(),
            actions.abs().mean().detach(),
        )

    def _update_step(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        mask = None
        if "is_init" in tensordict.keys(True, True):
            valid = ~tensordict["is_init"].squeeze(-1)
            mask = valid if valid.any() else None

        actor_input, critic_prefix = self._encode_inputs(tensordict)
        next_actor_input, next_critic_prefix = self._encode_inputs(tensordict["next"])
        q_loss, q_grad_norm, target_q_mean, target_summary = self._update_critic(
            tensordict,
            critic_prefix,
            next_actor_input,
            next_critic_prefix,
            mask,
        )

        zero = torch.zeros((), device=self.device)
        actor_updated = self.gradient_step % self.cfg.policy_frequency == 0
        if actor_updated:
            actor_loss, actor_grad_norm, actor_q_mean, action_abs = self._update_actor(
                actor_input,
                critic_prefix,
                mask,
            )
        else:
            actor_loss = zero
            actor_q_mean = zero
            action_abs = zero
            actor_grad_norm = zero

        self.gradient_step += 1

        return {
            "critic/loss": q_loss.detach(),
            "critic/grad_norm": q_grad_norm.detach(),
            "critic/target_q_mean": target_q_mean,
            "critic/q_min": target_summary.min().detach(),
            "critic/q_max": target_summary.max().detach(),
            "actor/loss": actor_loss.detach(),
            "actor/q_mean": actor_q_mean,
            "actor/action_abs": action_abs,
            "actor/grad_norm": actor_grad_norm.detach(),
            "actor/updated": torch.tensor(float(actor_updated), device=self.device),
        }

    def update(self) -> dict[str, float]:
        info: dict[str, float] = {
            "rb_size": float(len(self.replay_buffer)),
        }
        warmup_transitions = self.cfg.warm_up_steps * self.cfg.collect_steps * self.num_envs
        if len(self.replay_buffer) < min(warmup_transitions, self.replay_buffer.storage.max_size):
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
        del critic
        modules = [self.vecnorm, self.actor]
        if mode == "train":
            modules.append(self.exploration)
        rollout_policy = Seq(*modules, selected_out_keys=[ACTION_KEY])
        if self.cfg.freeze_vecnorm:
            rollout_policy.forward = VecNorm.freeze()(rollout_policy.forward)
        return rollout_policy

    def train_op(self, tensordict: TensorDictBase) -> dict[str, float]:
        self.observe(tensordict.exclude("stats"))
        return self.update()

    def compute_value(self, tensordict: TensorDictBase) -> TensorDictBase:
        actor_input, critic_prefix = self._encode_inputs(tensordict)
        with torch.no_grad():
            action = self.actor_core(actor_input)
            q_probs = F.softmax(self.qnet(torch.cat([critic_prefix, action], dim=-1)), dim=-1)
            q_value = self._reduce_q_values(self.qnet.get_value(q_probs)).unsqueeze(-1)
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
