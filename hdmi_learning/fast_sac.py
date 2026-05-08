from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import math
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
)
from torchrl.data import Composite as CompositeSpec, LazyTensorStorage, TensorDictReplayBuffer, TensorSpec
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import ProbabilisticActor

import active_adaptation as aa
from active_adaptation.learning.modules.distributions import TanhNormalWithEntropy
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    GAE,
    TERM_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
)
from active_adaptation.learning.ppo.ppo_base import PPOBase

from .common import EmpiricalNormalizer
from .action_bounds import (
    coerce_action_bounds_config,
    default_action_bounds,
    resolve_action_bounds,
)

BOOTSTRAP_KEY = "bootstrap"
ACTOR_INPUT_KEY = "_actor_input"
CRITIC_OBS_KEY = "_critic_obs"
CRITIC_INPUT_KEY = "_critic_input"
Q_LOGITS_KEY = "_q_logits"
ENV_ID_KEY = "_env_id"
N_STEP_REWARD_KEY = "_n_step_reward"
N_STEP_BOOTSTRAP_KEY = "_n_step_bootstrap"
N_STEP_DISCOUNT_KEY = "_n_step_discount"


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


def _masked_flat_values(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.reshape(-1)

    expanded_mask = mask
    while expanded_mask.ndim < value.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(value)
    return value[expanded_mask]


def _prefix_stats(prefix: str, value: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
    flat = _masked_flat_values(value.detach(), mask)
    if flat.numel() == 0:
        zero = torch.zeros((), device=value.device, dtype=value.dtype)
        return {
            f"{prefix}_mean": zero,
            f"{prefix}_q01": zero,
            f"{prefix}_q05": zero,
            f"{prefix}_q95": zero,
            f"{prefix}_q99": zero,
        }

    quantiles = torch.quantile(
        flat.float(),
        torch.tensor([0.01, 0.05, 0.95, 0.99], device=flat.device),
    )
    return {
        f"{prefix}_mean": flat.float().mean(),
        f"{prefix}_q01": quantiles[0],
        f"{prefix}_q05": quantiles[1],
        f"{prefix}_q95": quantiles[2],
        f"{prefix}_q99": quantiles[3],
    }


class CudaPrefetchNStepReplayBuffer:
    def __init__(
        self,
        *,
        capacity_per_env: int,
        num_envs: int,
        n_step: int,
        batch_size: int,
        gamma: float,
        device: torch.device,
        prefetch: int = 2,
        compact: bool = True,
    ) -> None:
        self.capacity_per_env = int(capacity_per_env)
        self.num_envs = int(num_envs)
        self.n_step = int(n_step)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.prefetch = max(0, int(prefetch))
        self.compact = bool(compact)
        self.storage: TensorDictBase | None = None
        self.cursor = 0
        self.length = 0
        self.lock = threading.RLock()
        self.prefetch_queue: deque[Future[TensorDictBase]] = deque()
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="fast-sac-replay")
            if self.prefetch > 0
            else None
        )
        self.copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    def __len__(self) -> int:
        return self.length * self.num_envs

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def __del__(self) -> None:
        self.close()

    def extend(self, data: TensorDictBase) -> None:
        data = data.detach()
        if data.device is not None and data.device.type != "cpu":
            data = data.cpu()
        if data.batch_dims == 1:
            data = data.unsqueeze(0)
        if data.batch_dims != 2 or data.batch_size[1] != self.num_envs:
            raise ValueError(
                "Expected replay data with batch size [T, num_envs] or [num_envs], "
                f"got {tuple(data.batch_size)} for num_envs={self.num_envs}."
            )

        with self.lock:
            if self.storage is None:
                sample = data[0]
                self.storage = sample.apply(
                    lambda value: torch.empty(
                        (self.capacity_per_env, *value.shape),
                        dtype=value.dtype,
                        device="cpu",
                    ),
                    batch_size=(self.capacity_per_env, *sample.batch_size),
                )

            num_steps = int(data.batch_size[0])
            start = 0
            while start < num_steps:
                write_count = min(num_steps - start, self.capacity_per_env - self.cursor)
                self.storage[self.cursor : self.cursor + write_count] = data[
                    start : start + write_count
                ]
                self.cursor = (self.cursor + write_count) % self.capacity_per_env
                self.length = min(self.capacity_per_env, self.length + write_count)
                start += write_count

        self._fill_prefetch()

    def sample(self) -> TensorDictBase:
        if self.executor is None:
            return self._sample_to_device()

        self._fill_prefetch()
        if not self.prefetch_queue:
            return self._sample_to_device()
        future = self.prefetch_queue.popleft()
        batch = future.result()
        self._fill_prefetch()
        return batch

    def _fill_prefetch(self) -> None:
        if self.executor is None or self.length < self.n_step:
            return
        while len(self.prefetch_queue) < self.prefetch:
            self.prefetch_queue.append(self.executor.submit(self._sample_to_device))

    def _sample_to_device(self) -> TensorDictBase:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        with self.lock:
            if self.storage is None or self.length < self.n_step:
                raise RuntimeError(
                    f"Not enough replay data to sample n_step={self.n_step}: "
                    f"length={self.length}."
                )
            length = self.length
            cursor = self.cursor
            oldest = cursor if length == self.capacity_per_env else 0
            starts = torch.randint(
                0,
                length - self.n_step + 1,
                (self.batch_size,),
                dtype=torch.long,
            )
            envs = torch.randint(
                0,
                self.num_envs,
                (self.batch_size, 1),
                dtype=torch.long,
            )
            offsets = torch.arange(self.n_step, dtype=torch.long).unsqueeze(0)
            time_idx = (oldest + starts.unsqueeze(1) + offsets) % self.capacity_per_env
            env_idx = envs.expand(-1, self.n_step)
            if self.compact:
                batch = self._sample_compact_locked(time_idx, env_idx)
            else:
                batch = self.storage[time_idx, env_idx]

        if self.device.type != "cuda":
            return batch.to(self.device)

        batch = batch.pin_memory()
        assert self.copy_stream is not None
        with torch.cuda.stream(self.copy_stream):
            batch = batch.to(self.device, non_blocking=True)
        self.copy_stream.synchronize()
        return batch

    def _sample_compact_locked(
        self,
        time_idx: torch.Tensor,
        env_idx: torch.Tensor,
    ) -> TensorDictBase:
        assert self.storage is not None
        batch_size, n_step = time_idx.shape
        env_flat = env_idx[:, 0]
        rewards = self.storage[REWARD_KEY][time_idx, env_idx]
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)
        rewards = rewards.squeeze(-1)
        dones = self.storage[DONE_KEY][time_idx, env_idx].bool().squeeze(-1)
        terminated = self.storage[TERM_KEY][time_idx, env_idx].bool().squeeze(-1)

        adjusted_rewards = torch.zeros(batch_size, dtype=rewards.dtype)
        bootstrap = torch.zeros(batch_size, dtype=rewards.dtype)
        bootstrap_discount = torch.zeros(batch_size, dtype=rewards.dtype)
        bootstrap_idx = torch.zeros(batch_size, dtype=torch.long)
        discount = torch.ones(batch_size, dtype=rewards.dtype)
        active = torch.ones(batch_size, dtype=torch.bool)

        for step_idx in range(n_step):
            step_active = active
            if not step_active.any():
                break

            adjusted_rewards = adjusted_rewards + torch.where(
                step_active,
                discount * rewards[:, step_idx],
                torch.zeros_like(adjusted_rewards),
            )
            can_bootstrap = step_active & ~terminated[:, step_idx]
            next_discount = discount * self.gamma
            boundary = can_bootstrap & dones[:, step_idx]
            if boundary.any():
                bootstrap_idx = torch.where(
                    boundary,
                    torch.full_like(bootstrap_idx, step_idx),
                    bootstrap_idx,
                )
                bootstrap_discount = torch.where(boundary, next_discount, bootstrap_discount)
                bootstrap = torch.where(boundary, torch.ones_like(bootstrap), bootstrap)

            active = can_bootstrap & ~dones[:, step_idx]
            discount = torch.where(active, next_discount, discount)

        if active.any():
            bootstrap_idx = torch.where(
                active,
                torch.full_like(bootstrap_idx, n_step - 1),
                bootstrap_idx,
            )
            bootstrap_discount = torch.where(active, discount, bootstrap_discount)
            bootstrap = torch.where(active, torch.ones_like(bootstrap), bootstrap)

        batch_idx = torch.arange(batch_size)
        compact = self.storage[time_idx[:, 0], env_flat].copy()
        compact.set(
            BOOTSTRAP_KEY,
            self.storage[BOOTSTRAP_KEY][time_idx[batch_idx, bootstrap_idx], env_flat],
        )
        compact.set(N_STEP_REWARD_KEY, adjusted_rewards.unsqueeze(-1))
        compact.set(N_STEP_BOOTSTRAP_KEY, bootstrap.unsqueeze(-1))
        compact.set(N_STEP_DISCOUNT_KEY, bootstrap_discount.unsqueeze(-1))
        return compact.unsqueeze(1)


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

        # loc = torch.clamp(loc, -10.0, 10.0)
        # loc = torch.nan_to_num(loc, nan=0.0)
        # log_std = torch.nan_to_num(log_std, nan=self.log_std_min)
        # scale = torch.nan_to_num(
        #     log_std.exp(),
        #     nan=math.exp(self.log_std_min),
        # )
        scale = log_std.exp()
        return loc, scale


def _column_like(
    value: torch.Tensor | float,
    reference: torch.Tensor,
) -> torch.Tensor:
    if torch.is_tensor(value):
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand_as(reference)
    else:
        value = torch.full_like(reference, float(value))
    return value.reshape(-1, 1)


def project_distributional_q(
    q_logits: torch.Tensor,
    rewards: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: torch.Tensor | float,
    q_support: torch.Tensor,
) -> torch.Tensor:
    q_support = q_support.to(device=q_logits.device, dtype=q_logits.dtype)
    num_atoms = q_support.shape[0]
    v_min = q_support[0]
    v_max = q_support[-1]
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = rewards.shape[0]

    rewards = rewards.to(device=q_logits.device, dtype=q_logits.dtype).reshape(-1)
    bootstrap = _column_like(bootstrap, rewards)
    discount = _column_like(discount, rewards)
    target_z = rewards.unsqueeze(1) + bootstrap * discount * q_support
    target_z = target_z.clamp(v_min.item(), v_max.item())
    b = (target_z - v_min) / delta_z
    lower = torch.floor(b).long()
    upper = torch.ceil(b).long()

    is_integer = upper == lower
    lower_mask = torch.logical_and((lower > 0), is_integer)
    upper_mask = torch.logical_and((lower == 0), is_integer)
    lower = torch.where(lower_mask, lower - 1, lower)
    upper = torch.where(upper_mask, upper + 1, upper)

    offset = (
        torch.arange(batch_size, device=q_logits.device)
        .mul(num_atoms)
        .unsqueeze(1)
        .expand(batch_size, num_atoms)
        .long()
    )
    max_index = batch_size * num_atoms - 1
    lower_indices = torch.clamp((lower + offset).reshape(-1), 0, max_index)
    upper_indices = torch.clamp((upper + offset).reshape(-1), 0, max_index)
    lower_weight = upper.to(dtype=q_logits.dtype) - b
    upper_weight = b - lower.to(dtype=q_logits.dtype)

    projections = []
    for next_logits in q_logits.unbind(dim=1):
        next_dist = F.softmax(next_logits, dim=-1)
        proj_dist = torch.zeros_like(next_dist)
        flat_proj = proj_dist.reshape(-1)
        flat_proj.index_add_(
            0,
            lower_indices,
            (next_dist * lower_weight).reshape(-1),
        )
        flat_proj.index_add_(
            0,
            upper_indices,
            (next_dist * upper_weight).reshape(-1),
        )
        projections.append(proj_dist)
    return torch.stack(projections, dim=1)


def distributional_q_value(
    probs: torch.Tensor,
    q_support: torch.Tensor,
) -> torch.Tensor:
    q_support = q_support.to(device=probs.device, dtype=probs.dtype)
    return torch.sum(probs * q_support, dim=-1)


class DistributionalCritic(TensorDictModuleBase):
    def __init__(
        self,
        *,
        num_atoms: int = 101,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
    ) -> None:
        super().__init__()
        self.in_keys = [CRITIC_INPUT_KEY]
        self.out_keys = [Q_LOGITS_KEY]
        self.num_atoms = num_atoms

        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.qnets = nn.ModuleList(
            [
                nn.Sequential(
                    _build_mlp(None, hidden_dims, use_layer_norm=use_layer_norm),
                    nn.Linear(hidden_dims[-1], num_atoms),
                )
                for _ in range(num_q_networks)
            ]
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        critic_input = tensordict[CRITIC_INPUT_KEY]
        outputs = [qnet(critic_input) for qnet in self.qnets]
        tensordict.set(Q_LOGITS_KEY, torch.stack(outputs, dim=1))
        return tensordict


class ValueProbe(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 768,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.net = nn.Sequential(
            _build_mlp(None, hidden_dims, use_layer_norm=use_layer_norm),
            nn.Linear(hidden_dims[-1], 1),
        )

    def forward(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.net(critic_obs)


class WarmupUniformRolloutPolicy:
    def __init__(self, policy: "FastSAC", actor_rollout_policy: TensorDictModuleBase) -> None:
        object.__setattr__(self, "_policy", policy)
        self.actor_rollout_policy = actor_rollout_policy

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.forward(tensordict)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        policy = self._policy
        if len(policy.replay_buffer) >= policy.warmup_transition_threshold:
            return self.actor_rollout_policy(tensordict)

        action = torch.rand(
            (*tensordict.batch_size, policy.action_dim),
            device=policy.action_min.device,
            dtype=policy.action_min.dtype,
        )
        action = policy.action_min + action * (policy.action_max - policy.action_min)
        tensordict.set(ACTION_KEY, action)
        return tensordict


class FastSACRolloutPolicy(TensorDictModuleBase):
    def __init__(self, policy: "FastSAC") -> None:
        super().__init__()
        object.__setattr__(self, "_policy", policy)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        policy = self._policy
        policy._prepare_actor_input(tensordict, update_normalizer=False)
        return policy.actor(tensordict)


@dataclass
class FastSACConfig:
    _target_: str = f"{__package__}.fast_sac.FastSAC"

    name: str = "fast_sac"
    collect_steps: int = 1
    # Effective replay capacity = buffer_size * collect_steps * num_envs.
    buffer_size: int = 1024
    replay_batch_size: int = 4096
    # Effective transition warmup = warm_up_steps * collect_steps * num_envs.
    warm_up_steps: int = 10
    updates_per_step: int = 4
    policy_frequency: int = 2
    n_step: int = 1
    custom_replay_buffer: bool = True
    custom_replay_prefetch: int = 2

    gamma: float = 0.995
    tau: float = 0.125
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 4e-3
    target_entropy_ratio: float | None = None # 1.0
    weight_decay: float = 1e-3

    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    action_space_mode: str = "holosoma"
    holosoma_action_scale: float = 0.25
    holosoma_use_actor_boundary: bool = True
    action_bounds: dict[str, list[float]] = field(
        default_factory=default_action_bounds
    )
    action_min: float | None = None
    action_max: float | None = None
    v_step: float = 0.05
    v_min: float = -5.0
    v_max: float = 15.0
    actor_q_reduce: str = "min"
    critic_q_reduce: str = "min"
    actor_update_scope: str = "first"
    n_step_entropy_mode: str = "bootstrap"
    debug_timing: bool = False
    debug_timing_interval: int = 1
    log_std_max: float = 0.0
    log_std_min: float = -4.0
    use_layer_norm: bool = True
    max_grad_norm: float = 1.0

    vecnorm: bool = True
    freeze_vecnorm: bool = False
    enable_value_probe: bool = True
    value_probe_hidden_dim: int | None = None
    value_probe_lr: float | None = None
    value_probe_update_every: int = 32
    value_probe_trace_steps: int = 32
    value_probe_inner: int = 2
    gae_lambda: float = 0.95
    action_bound_epsilon_ratio: float = 0.02
    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)
    grad_sync_mode: str | None = "manual"

    def __post_init__(self) -> None:
        self.n_step = int(self.n_step)
        if self.n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {self.n_step}.")
        self.custom_replay_prefetch = int(self.custom_replay_prefetch)
        if self.custom_replay_prefetch < 0:
            raise ValueError(
                "custom_replay_prefetch must be >= 0, "
                f"got {self.custom_replay_prefetch}."
            )

        self.actor_q_reduce = str(self.actor_q_reduce).lower()
        if self.actor_q_reduce not in {"min", "mean", "q0", "q1"}:
            raise ValueError(
                "actor_q_reduce must be one of {'min', 'mean', 'q0', 'q1'}, "
                f"got {self.actor_q_reduce!r}"
            )
        self.critic_q_reduce = str(self.critic_q_reduce).lower()
        if self.critic_q_reduce not in {"min", "mean", "each"}:
            raise ValueError(
                "critic_q_reduce must be one of {'min', 'mean', 'each'}, "
                f"got {self.critic_q_reduce!r}"
            )
        self.actor_update_scope = str(self.actor_update_scope).lower()
        if self.actor_update_scope not in {"first", "all"}:
            raise ValueError(
                "actor_update_scope must be one of {'first', 'all'}, "
                f"got {self.actor_update_scope!r}"
            )
        self.n_step_entropy_mode = str(self.n_step_entropy_mode).lower()
        if self.n_step_entropy_mode not in {"bootstrap", "all"}:
            raise ValueError(
                "n_step_entropy_mode must be one of {'bootstrap', 'all'}, "
                f"got {self.n_step_entropy_mode!r}"
            )
        self.debug_timing_interval = int(self.debug_timing_interval)
        if self.debug_timing_interval < 1:
            raise ValueError(
                f"debug_timing_interval must be >= 1, got {self.debug_timing_interval}."
            )
        self.value_probe_update_every = int(self.value_probe_update_every)
        if self.value_probe_update_every < 1:
            raise ValueError(
                "value_probe_update_every must be >= 1, "
                f"got {self.value_probe_update_every}."
            )
        self.value_probe_trace_steps = int(self.value_probe_trace_steps)
        if self.value_probe_trace_steps < 2:
            raise ValueError(
                "value_probe_trace_steps must be >= 2, "
                f"got {self.value_probe_trace_steps}."
            )
        self.value_probe_inner = int(self.value_probe_inner)
        if self.value_probe_inner < 1:
            raise ValueError(
                f"value_probe_inner must be >= 1, got {self.value_probe_inner}."
            )
        if self.value_probe_hidden_dim is None:
            self.value_probe_hidden_dim = self.critic_hidden_dim
        if self.value_probe_lr is None:
            self.value_probe_lr = self.critic_lr

        if isinstance(self.grad_sync_mode, str):
            self.grad_sync_mode = self.grad_sync_mode.lower()
            if self.grad_sync_mode in {"none", "null"}:
                self.grad_sync_mode = None

        if self.grad_sync_mode not in {"manual", None, "ddp"}:
            raise ValueError(
                "grad_sync_mode must be one of {'manual', None, 'ddp'}, "
                f"got {self.grad_sync_mode!r}"
            )
        self.action_space_mode = str(self.action_space_mode).lower()
        if self.action_space_mode not in {"manual", "holosoma"}:
            raise ValueError(
                "action_space_mode must be one of {'manual', 'holosoma'}, "
                f"got {self.action_space_mode!r}"
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
        object.__setattr__(self, "env", env)

        observation_keys = set(observation_spec.keys(True, True))
        missing_keys = sorted({OBS_KEY, CMD_KEY, OBS_PRIV_KEY}.difference(observation_keys))
        if missing_keys:
            raise KeyError(f"Missing required observation keys: {missing_keys}")

        self.num_envs = int(getattr(env, "num_envs", observation_spec.shape[0]))
        self.action_dim = int(env.action_manager.action_dim)
        self.joint_names = env.action_manager.joint_names
        self.gradient_step = 0

        self.actor_obs_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY)
        self.critic_obs_keys: Tuple[str, ...] = (OBS_KEY, CMD_KEY, OBS_PRIV_KEY)
        self.actor_obs_dim = sum(int(observation_spec[key].shape[-1]) for key in self.actor_obs_keys)
        self.critic_obs_dim = sum(
            int(observation_spec[key].shape[-1]) for key in self.critic_obs_keys
        )
        self._build_obs_normalizers()

        action_min, action_max = self._resolve_action_space_bounds()
        self.register_buffer("action_min", action_min.clone())
        self.register_buffer("action_max", action_max.clone())
        self.action_min: torch.Tensor
        self.action_max: torch.Tensor

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=Mod(
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
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=TanhNormalWithEntropy,
            distribution_kwargs={
                "low": self.action_min,
                "high": self.action_max,
                "event_dims": 1,
            },
            return_log_prob=True,
        ).to(self.device)

        num_atoms = int((self.cfg.v_max - self.cfg.v_min) / self.cfg.v_step) + 1
        self.qnet = DistributionalCritic(
            num_atoms=num_atoms,
            hidden_dim=self.cfg.critic_hidden_dim,
            use_layer_norm=self.cfg.use_layer_norm,
        ).to(self.device)
        self.register_buffer(
            "q_support",
            torch.linspace(
                self.cfg.v_min,
                self.cfg.v_max,
                num_atoms,
                device=self.device,
            ),
        )
        self.q_support: torch.Tensor

        fake_input = observation_spec.zero()
        fake_critic_input = fake_input.copy()
        fake_critic_input.set(
            ACTION_KEY,
            torch.zeros(
                (*fake_input.batch_size, self.action_dim),
                device=self.device,
            ),
        )
        with torch.no_grad():
            self._prepare_actor_input(fake_input, update_normalizer=False)
            self.actor.get_dist(fake_input)
            self._prepare_critic_obs(fake_critic_input, update_normalizer=False)
            self._prepare_critic_input(fake_critic_input)
            self.qnet(fake_critic_input)

        self.qnet_target = deepcopy(self.qnet).to(self.device)
        self.qnet_target.requires_grad_(False)
        fused = str(self.device).startswith("cuda")
        self.gae = GAE(self.cfg.gamma, self.cfg.gae_lambda).to(self.device)
        self.enable_value_probe = bool(self.cfg.enable_value_probe)
        self.value_probe = None
        self.value_optimizer = None
        self.value_trace: deque[TensorDictBase] = deque(
            maxlen=self.cfg.value_probe_trace_steps
        )
        if self.enable_value_probe:
            self.value_probe = ValueProbe(
                hidden_dim=int(self.cfg.value_probe_hidden_dim),
                use_layer_norm=self.cfg.use_layer_norm,
            ).to(self.device)
            self.value_optimizer = torch.optim.AdamW(
                self.value_probe.parameters(),
                lr=float(self.cfg.value_probe_lr),
                weight_decay=self.cfg.weight_decay,
                fused=fused,
                betas=(0.9, 0.95),
            )

        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(self.cfg.alpha_init), device=self.device)
        )
        self.fixed_alpha = self.cfg.target_entropy_ratio is None
        self.target_entropy = (
            None
            if self.fixed_alpha
            else -float(self.action_dim) * float(self.cfg.target_entropy_ratio)
        )
        self.log_alpha.requires_grad_(not self.fixed_alpha)

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
        self.alpha_optimizer = None
        if not self.fixed_alpha:
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha],
                lr=self.cfg.alpha_lr,
                weight_decay=0.0,
                fused=fused,
                betas=(0.9, 0.95),
            )

        self.replay_buffer_capacity_per_env = self.cfg.buffer_size * self.cfg.collect_steps
        self.replay_buffer_capacity = self.replay_buffer_capacity_per_env * self.num_envs
        warmup_transitions = self.cfg.warm_up_steps * self.cfg.collect_steps * self.num_envs
        self.warmup_transition_threshold = min(
            max(warmup_transitions, self.cfg.replay_batch_size),
            self.replay_buffer_capacity,
        )
        self.min_replay_sample_transitions = max(
            self.warmup_transition_threshold,
            self.cfg.n_step * self.num_envs,
        )
        self.use_slice_replay = self.cfg.n_step > 1
        self.use_custom_replay_buffer = bool(self.cfg.custom_replay_buffer)
        pin_replay_memory = str(self.device).startswith("cuda")
        if self.use_custom_replay_buffer:
            self.replay_buffer = CudaPrefetchNStepReplayBuffer(
                capacity_per_env=self.replay_buffer_capacity_per_env,
                num_envs=self.num_envs,
                n_step=self.cfg.n_step,
                batch_size=self.cfg.replay_batch_size,
                gamma=self.cfg.gamma,
                device=self.device,
                prefetch=self.cfg.custom_replay_prefetch,
                compact=(
                    self.cfg.actor_update_scope == "first"
                    and self.cfg.n_step_entropy_mode == "bootstrap"
                ),
            )
            self.replay_samples_on_device = True
        elif self.use_slice_replay:
            self.replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=self.replay_buffer_capacity, ndim=2),
                sampler=SliceSampler(
                    slice_len=self.cfg.n_step,
                    traj_key=ENV_ID_KEY,
                    strict_length=True,
                ),
                batch_size=self.cfg.replay_batch_size * self.cfg.n_step,
                dim_extend=0,
                pin_memory=pin_replay_memory,
                prefetch=2,
            )
            self.replay_samples_on_device = False
        else:
            self.replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=self.replay_buffer_capacity),
                batch_size=self.cfg.replay_batch_size,
                pin_memory=pin_replay_memory,
                prefetch=2,
            )
            self.replay_samples_on_device = False

        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            self._broadcast_parameters()
        else:
            self.world_size = 1

    def _sync_if_timing(self) -> None:
        if self.cfg.debug_timing and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def _build_obs_normalizers(self) -> None:
        self.use_obs_normalization = bool(self.cfg.vecnorm)
        self.update_obs_normalization = bool(self.cfg.vecnorm) and not self.cfg.freeze_vecnorm
        if self.use_obs_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalizer(
                self.actor_obs_dim,
                self.device,
            )
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalizer(
                self.critic_obs_dim,
                self.device,
            )
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

    def _resolve_action_space_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.action_space_mode == "holosoma":
            return self._configure_holosoma_action_space()
        return resolve_action_bounds(
            self.cfg.action_bounds,
            self.joint_names,
            self.device,
        )

    def _configure_holosoma_action_space(self) -> tuple[torch.Tensor, torch.Tensor]:
        manager = getattr(self.env, "action_manager", None)
        if manager is None:
            raise RuntimeError("Holosoma action scaling requires env.action_manager.")

        env_action_scale = self._compute_holosoma_env_action_scale(manager)
        if env_action_scale.numel() != self.action_dim:
            raise ValueError(
                f"FastSAC env action scale has {env_action_scale.numel()} entries, "
                f"expected action_dim={self.action_dim}."
            )

        manager.action_scaling = env_action_scale.to(manager.device)
        if self.cfg.holosoma_use_actor_boundary:
            actor_low, actor_high = self._compute_action_bounds_from_limits(
                manager,
                env_action_scale,
            )
            actor_scale = 0.5 * (actor_high - actor_low)
            actor_bias = 0.5 * (actor_high + actor_low)
            print(
                "[Info] FastSAC Holosoma action scaling: "
                f"env_scale_min={env_action_scale.min().item():.4f}, "
                f"env_scale_max={env_action_scale.max().item():.4f}, "
                f"actor_low_min={actor_low.min().item():.4f}, "
                f"actor_high_max={actor_high.max().item():.4f}, "
                f"actor_scale_min={actor_scale.min().item():.4f}, "
                f"actor_scale_max={actor_scale.max().item():.4f}, "
                f"actor_bias_absmax={actor_bias.abs().max().item():.4f}",
                flush=True,
            )
            return actor_low, actor_high

        print(
            "[Info] FastSAC Holosoma action scaling: "
            f"env_scale_min={env_action_scale.min().item():.4f}, "
            f"env_scale_max={env_action_scale.max().item():.4f}, "
            "actor_boundary_mode=fixed_unit",
            flush=True,
        )
        return -torch.ones_like(env_action_scale), torch.ones_like(env_action_scale)

    def _compute_holosoma_env_action_scale(self, manager) -> torch.Tensor:
        asset = manager.asset
        actuator_names = list(asset.actuator_names)
        ctrl_ids = torch.as_tensor(
            asset.indexing.ctrl_ids,
            device=self.device,
            dtype=torch.long,
        )
        if len(actuator_names) != int(ctrl_ids.numel()):
            raise RuntimeError(
                f"Expected one actuator name per control id, got {len(actuator_names)} names "
                f"and {int(ctrl_ids.numel())} control ids."
            )
        name_to_ctrl_id = {name: ctrl_ids[i] for i, name in enumerate(actuator_names)}
        missing = [name for name in manager.joint_names if name not in name_to_ctrl_id]
        if missing:
            raise RuntimeError(
                f"Cannot compute Holosoma action scale; missing actuators for joints: {missing}"
            )

        selected_ctrl_ids = torch.stack([name_to_ctrl_id[name] for name in manager.joint_names])
        force_range = manager.env.sim.get_default_field("actuator_forcerange").to(self.device)
        gainprm = manager.env.sim.get_default_field("actuator_gainprm").to(self.device)
        effort_limit = force_range[selected_ctrl_ids].abs().max(dim=-1).values
        stiffness = gainprm[selected_ctrl_ids, 0].abs().clamp_min(1.0e-6)
        return (
            float(self.cfg.holosoma_action_scale) * effort_limit / stiffness
        ).clamp_min(1.0e-6)

    def _compute_action_bounds_from_limits(
        self,
        manager,
        env_action_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(manager.asset.data, "joint_pos_limits"):
            raise RuntimeError(
                "FastSAC Holosoma action boundary requires asset.data.joint_pos_limits."
            )

        limits = manager.asset.data.joint_pos_limits[0, manager.joint_ids].to(self.device)
        default_pos = manager.default_joint_pos[0, manager.joint_ids].to(self.device)
        lower = limits[..., 0]
        upper = limits[..., 1]
        scale = env_action_scale.abs().clamp_min(1.0e-6)
        actor_low = (lower - default_pos) / scale
        actor_high = (upper - default_pos) / scale
        return actor_low, actor_high

    def _broadcast_parameters(self) -> None:
        with torch.no_grad():
            dist.broadcast(self.log_alpha.data, src=0)
            dist.broadcast(self.q_support, src=0)
            for module in (
                self.actor_obs_normalizer,
                self.critic_obs_normalizer,
                self.actor,
                self.qnet,
                self.qnet_target,
                self.value_probe,
            ):
                if module is None:
                    continue
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

    @torch.no_grad()
    def _all_reduce_param_grad(self, param: nn.Parameter) -> None:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    def _sync_vecnorms(self) -> None:
        return None

    def _cat_obs(self, tensordict: TensorDictBase, keys: Sequence[str]) -> torch.Tensor:
        return torch.cat([tensordict[key].float() for key in keys], dim=-1)

    def normalize_actor_obs(self, obs: torch.Tensor, *, update: bool = False) -> torch.Tensor:
        if not self.use_obs_normalization:
            return obs.float()
        flat = obs.reshape(-1, obs.shape[-1]).float()
        normed = self.actor_obs_normalizer(
            flat,
            update=bool(update and self.update_obs_normalization),
        )
        return normed.reshape_as(obs)

    def normalize_critic_obs(self, obs: torch.Tensor, *, update: bool = False) -> torch.Tensor:
        if not self.use_obs_normalization:
            return obs.float()
        flat = obs.reshape(-1, obs.shape[-1]).float()
        normed = self.critic_obs_normalizer(
            flat,
            update=bool(update and self.update_obs_normalization),
        )
        return normed.reshape_as(obs)

    def _prepare_actor_input(
        self,
        tensordict: TensorDictBase,
        *,
        update_normalizer: bool,
    ) -> TensorDictBase:
        actor_obs = self._cat_obs(tensordict, self.actor_obs_keys)
        tensordict.set(
            ACTOR_INPUT_KEY,
            self.normalize_actor_obs(actor_obs, update=update_normalizer),
        )
        return tensordict

    def _prepare_critic_obs(
        self,
        tensordict: TensorDictBase,
        *,
        update_normalizer: bool,
    ) -> TensorDictBase:
        critic_obs = self._cat_obs(tensordict, self.critic_obs_keys)
        tensordict.set(
            CRITIC_OBS_KEY,
            self.normalize_critic_obs(critic_obs, update=update_normalizer),
        )
        return tensordict

    def _prepare_critic_input(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict.set(
            CRITIC_INPUT_KEY,
            torch.cat([tensordict[CRITIC_OBS_KEY], tensordict[ACTION_KEY]], dim=-1),
        )
        return tensordict

    def _prepare_batch_inputs(
        self,
        tensordict: TensorDictBase,
        *,
        update_normalizers: bool,
    ) -> TensorDictBase:
        self._prepare_actor_input(tensordict, update_normalizer=update_normalizers)
        self._prepare_critic_obs(tensordict, update_normalizer=update_normalizers)
        bootstrap_td = tensordict.get(BOOTSTRAP_KEY, None)
        if bootstrap_td is not None:
            self._prepare_actor_input(bootstrap_td, update_normalizer=update_normalizers)
            self._prepare_critic_obs(bootstrap_td, update_normalizer=update_normalizers)
        return tensordict

    def _sample_actor(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self._prepare_actor_input(tensordict, update_normalizer=False)
        dist = self.actor.get_dist(tensordict)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        tensordict.set(ACTION_KEY, action)
        tensordict.set(f"{ACTION_KEY}_log_prob", log_prob)
        return tensordict

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

    def _reduce_target_distributions(
        self,
        target_distributions: torch.Tensor,
        target_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.critic_q_reduce == "each":
            return target_distributions, target_values

        if self.cfg.critic_q_reduce == "min":
            selected_idx = target_values.argmin(dim=1)
            selected = target_distributions[
                torch.arange(target_distributions.shape[0], device=target_distributions.device),
                selected_idx,
            ]
        else:
            selected = target_distributions.mean(dim=1)

        shared_distributions = selected.unsqueeze(1).expand_as(target_distributions)
        shared_values = distributional_q_value(shared_distributions, self.q_support)
        return shared_distributions, shared_values

    def _critic_stats_values(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.critic_q_reduce == "each":
            return q_values
        if self.cfg.critic_q_reduce == "min":
            return q_values.min(dim=1).values
        return q_values.mean(dim=1)

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
        env_id = torch.arange(self.num_envs, device=replay_td.device)
        replay_td.set(ENV_ID_KEY, env_id)
        return replay_td

    def _collect_value_probe_data(self, tensordict: TensorDictBase) -> TensorDictBase:
        keys: list[Union[str, tuple[str, str]]] = [
            OBS_KEY,
            CMD_KEY,
            OBS_PRIV_KEY,
            REWARD_KEY,
            DONE_KEY,
            TERM_KEY,
            ("next", OBS_KEY),
            ("next", CMD_KEY),
            ("next", OBS_PRIV_KEY),
        ]
        if "is_init" in tensordict.keys(True, True):
            keys.append("is_init")
        return tensordict.select(*keys, strict=False).detach()

    def observe(self, tensordict: TensorDictBase) -> None:
        if self.enable_value_probe:
            self.value_trace.append(self._collect_value_probe_data(tensordict).cpu())
        replay_td = self._collect_replay_data(tensordict)
        if self.use_custom_replay_buffer:
            self.replay_buffer.extend(replay_td.cpu())
            return
        if self.use_slice_replay:
            replay_td = replay_td.unsqueeze(0)
        else:
            replay_td = replay_td.reshape(-1)
        self.replay_buffer.extend(replay_td.cpu())

    def _compute_n_step_target_inputs(
        self,
        tensordict: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        timing = self.cfg.debug_timing
        timing_data: dict[str, float] = {}
        batch_size, n_step = tensordict.batch_size
        rewards = self._reward_total(tensordict)
        dones = tensordict[DONE_KEY].bool().squeeze(-1)
        terminated = tensordict[TERM_KEY].bool().squeeze(-1)

        if N_STEP_REWARD_KEY in tensordict.keys(True, True):
            adjusted_rewards = tensordict[N_STEP_REWARD_KEY].reshape(-1)
            bootstrap = tensordict[N_STEP_BOOTSTRAP_KEY].reshape(-1)
            bootstrap_discount = tensordict[N_STEP_DISCOUNT_KEY].reshape(-1)
            bootstrap_td = tensordict[BOOTSTRAP_KEY].reshape(-1).copy()
            self._sync_if_timing()
            t0 = time.perf_counter()
            self._sample_actor(bootstrap_td)
            self._sync_if_timing()
            if timing:
                timing_data["critic_target_actor_ms"] = (time.perf_counter() - t0) * 1000.0
            next_log_probs = bootstrap_td[f"{ACTION_KEY}_log_prob"]
            adjusted_rewards = adjusted_rewards + torch.where(
                bootstrap.bool(),
                -bootstrap_discount * self.log_alpha.exp().detach() * next_log_probs,
                torch.zeros_like(adjusted_rewards),
            )
            self._prepare_critic_input(bootstrap_td)
            self._sync_if_timing()
            t0 = time.perf_counter()
            self.qnet_target(bootstrap_td)
            self._sync_if_timing()
            if timing:
                timing_data["critic_target_qnet_ms"] = (time.perf_counter() - t0) * 1000.0
            return (
                adjusted_rewards,
                bootstrap,
                bootstrap_discount,
                bootstrap_td[Q_LOGITS_KEY],
                next_log_probs.reshape(-1),
                timing_data,
            )

        bootstrap_td_flat = tensordict[BOOTSTRAP_KEY].reshape(-1)
        next_log_probs = None
        if self.cfg.n_step_entropy_mode == "all":
            bootstrap_td_all = bootstrap_td_flat.copy()
            self._sync_if_timing()
            t0 = time.perf_counter()
            self._sample_actor(bootstrap_td_all)
            self._sync_if_timing()
            if timing:
                timing_data["critic_target_actor_ms"] = (time.perf_counter() - t0) * 1000.0
            next_log_probs = bootstrap_td_all[f"{ACTION_KEY}_log_prob"].reshape(
                batch_size,
                n_step,
            )

        adjusted_rewards = torch.zeros(batch_size, device=self.device, dtype=rewards.dtype)
        bootstrap = torch.zeros(batch_size, device=self.device, dtype=rewards.dtype)
        bootstrap_discount = torch.zeros(batch_size, device=self.device, dtype=rewards.dtype)
        bootstrap_idx = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        discount = torch.ones(batch_size, device=self.device, dtype=rewards.dtype)
        active = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        alpha = self.log_alpha.exp().detach()

        self._sync_if_timing()
        t0 = time.perf_counter()
        for step_idx in range(n_step):
            step_active = active
            if not step_active.any():
                break

            adjusted_rewards = adjusted_rewards + torch.where(
                step_active,
                discount * rewards[:, step_idx],
                torch.zeros_like(adjusted_rewards),
            )

            step_terminated = step_active & terminated[:, step_idx]
            can_bootstrap = step_active & ~terminated[:, step_idx]
            next_discount = discount * self.cfg.gamma
            if next_log_probs is not None:
                adjusted_rewards = adjusted_rewards + torch.where(
                    can_bootstrap,
                    -next_discount * alpha * next_log_probs[:, step_idx],
                    torch.zeros_like(adjusted_rewards),
                )

            boundary = can_bootstrap & dones[:, step_idx]
            if boundary.any():
                bootstrap_idx = torch.where(
                    boundary,
                    torch.full_like(bootstrap_idx, step_idx),
                    bootstrap_idx,
                )
                bootstrap_discount = torch.where(boundary, next_discount, bootstrap_discount)
                bootstrap = torch.where(boundary, torch.ones_like(bootstrap), bootstrap)

            active = can_bootstrap & ~dones[:, step_idx]
            discount = torch.where(active, next_discount, discount)

            if step_terminated.any():
                active = active & ~step_terminated

        if active.any():
            bootstrap_idx = torch.where(
                active,
                torch.full_like(bootstrap_idx, n_step - 1),
                bootstrap_idx,
            )
            bootstrap_discount = torch.where(active, discount, bootstrap_discount)
            bootstrap = torch.where(active, torch.ones_like(bootstrap), bootstrap)

        batch_idx = torch.arange(batch_size, device=self.device)
        bootstrap_flat_idx = batch_idx * n_step + bootstrap_idx
        self._sync_if_timing()
        if timing:
            timing_data["critic_target_accumulate_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        bootstrap_td = bootstrap_td_flat[bootstrap_flat_idx].copy()
        self._sync_if_timing()
        if timing:
            timing_data["critic_target_select_ms"] = (time.perf_counter() - t0) * 1000.0
        if next_log_probs is None:
            self._sync_if_timing()
            t0 = time.perf_counter()
            self._sample_actor(bootstrap_td)
            self._sync_if_timing()
            if timing:
                timing_data["critic_target_actor_ms"] = (time.perf_counter() - t0) * 1000.0
            bootstrap_log_probs = bootstrap_td[f"{ACTION_KEY}_log_prob"]
            adjusted_rewards = adjusted_rewards + torch.where(
                bootstrap.bool(),
                -bootstrap_discount * alpha * bootstrap_log_probs,
                torch.zeros_like(adjusted_rewards),
            )
            self._prepare_critic_input(bootstrap_td)
            next_log_probs = bootstrap_log_probs
        self._sync_if_timing()
        t0 = time.perf_counter()
        self.qnet_target(bootstrap_td)
        self._sync_if_timing()
        if timing:
            timing_data["critic_target_qnet_ms"] = (time.perf_counter() - t0) * 1000.0
        bootstrap_q_logits = bootstrap_td[Q_LOGITS_KEY]
        return (
            adjusted_rewards,
            bootstrap,
            bootstrap_discount,
            bootstrap_q_logits,
            next_log_probs.reshape(-1),
            timing_data,
        )

    def _update_critic(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        critic_td = tensordict[:, 0].copy()
        rewards = self._reward_total(tensordict)
        timing = self.cfg.debug_timing
        timing_data: dict[str, float] = {}

        with torch.no_grad():
            adjusted_rewards, bootstrap, bootstrap_discount, bootstrap_q_logits, next_log_probs, timing_data = (
                self._compute_n_step_target_inputs(tensordict)
            )
            self._sync_if_timing()
            t0 = time.perf_counter()
            target_distributions = project_distributional_q(
                bootstrap_q_logits,
                adjusted_rewards,
                bootstrap,
                bootstrap_discount,
                self.q_support,
            )
            self._sync_if_timing()
            if timing:
                timing_data["critic_projection_ms"] = (time.perf_counter() - t0) * 1000.0
            target_values = distributional_q_value(target_distributions, self.q_support)
            target_distributions, target_values = self._reduce_target_distributions(
                target_distributions,
                target_values,
            )

        self._sync_if_timing()
        t0 = time.perf_counter()
        self._prepare_critic_input(critic_td)
        self.qnet(critic_td)
        self._sync_if_timing()
        if timing:
            timing_data["critic_online_qnet_ms"] = (time.perf_counter() - t0) * 1000.0
        q_outputs = critic_td[Q_LOGITS_KEY]
        critic_log_probs = F.log_softmax(q_outputs, dim=-1).clamp(min=-30.0)
        critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        q_loss = _masked_mean(critic_losses, mask)

        first_rewards = rewards[:, 0]
        q_probs = F.softmax(q_outputs.detach(), dim=-1)
        q_values = distributional_q_value(q_probs, self.q_support)
        q_stats_values = self._critic_stats_values(q_values)
        target_stats_values = self._critic_stats_values(target_values)
        reward_mean = _masked_mean(first_rewards.detach(), mask)
        reward_max = first_rewards.detach().max()
        reward_min = first_rewards.detach().min()
        target_q_max = target_values.detach().max()
        target_q_min = target_values.detach().min()
        target_clamp_hi = (target_values.detach() >= (self.cfg.v_max - 1e-4)).float().mean()
        target_clamp_lo = (target_values.detach() <= (self.cfg.v_min + 1e-4)).float().mean()
        info = {
            "reward/mean": reward_mean.detach(),
            "reward/max": reward_max.detach(),
            "reward/min": reward_min.detach(),
            "critic/target_q_max": target_q_max.detach(),
            "critic/target_q_min": target_q_min.detach(),
            "critic/target_vmax_frac": target_clamp_hi.detach(),
            "critic/target_vmin_frac": target_clamp_lo.detach(),
        }
        info.update(_prefix_stats("critic/q", q_stats_values, mask))
        info.update(_prefix_stats("critic/target_q", target_stats_values, mask))

        self.q_optimizer.zero_grad(set_to_none=True)
        self._sync_if_timing()
        t0 = time.perf_counter()
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
        self._sync_if_timing()
        if timing:
            timing_data["critic_backward_step_ms"] = (time.perf_counter() - t0) * 1000.0
            for key, value in timing_data.items():
                info[f"debug_timing/{key}"] = torch.tensor(value, device=self.device)

        return (
            q_loss.detach(),
            q_grad_norm.detach(),
            target_stats_values.detach(),
            next_log_probs.detach(),
            info,
        )

    def _update_actor(
        self,
        tensordict: TensorDictBase,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        actor_td = tensordict.copy()
        actor_td = self._sample_actor(actor_td)
        self._prepare_critic_input(actor_td)
        self.qnet(actor_td)
        q_probs = F.softmax(actor_td[Q_LOGITS_KEY], dim=-1)
        q_values = distributional_q_value(q_probs, self.q_support)
        q_value = self._reduce_actor_q_values(q_values)
        log_probs = actor_td[f"{ACTION_KEY}_log_prob"]
        action = actor_td[ACTION_KEY]
        clamped_action = action.clamp(
            self.action_min + 1.0e-6,
            self.action_max - 1.0e-6,
        )
        action_span = (self.action_max - self.action_min).clamp_min(1.0e-6)
        edge_margin = action_span * float(self.cfg.action_bound_epsilon_ratio)
        near_low = (action - self.action_min) <= edge_margin
        near_high = (self.action_max - action) <= edge_margin
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
            "policy/action_clamp_frac": (
                (clamped_action - action).abs() > 1.0e-7
            ).float().mean(),
            "policy/action_bound_frac": (near_low | near_high).float().mean(),
        }
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

        return (
            actor_loss.detach(),
            (-log_probs).detach(),
            action_std.detach(),
            actor_grad_norm.detach(),
            diagnostics,
        )

    def _update_alpha(self, next_log_probs: torch.Tensor) -> torch.Tensor:
        if self.fixed_alpha:
            return torch.zeros((), device=self.device)
        alpha_loss = -(
            self.log_alpha.exp() * (next_log_probs.detach() + self.target_entropy)
        ).mean()
        assert self.alpha_optimizer is not None
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_param_grad(self.log_alpha)
        self.alpha_optimizer.step()
        return alpha_loss.detach()

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)

    def _train_value_probe(self) -> dict[str, torch.Tensor]:
        if (
            not self.enable_value_probe
            or self.value_probe is None
            or self.value_optimizer is None
            or len(self.value_trace) < self.cfg.value_probe_trace_steps
        ):
            return {}

        batch = torch.stack(list(self.value_trace), dim=0).to(self.device)
        critic_obs_tn = self._cat_obs(batch, self.critic_obs_keys)
        next_critic_obs_tn = self._cat_obs(batch["next"], self.critic_obs_keys)
        critic_obs_tn = self.normalize_critic_obs(critic_obs_tn, update=False)
        next_critic_obs_tn = self.normalize_critic_obs(next_critic_obs_tn, update=False)
        T, N = critic_obs_tn.shape[:2]
        flat = T * N
        values_tn = self.value_probe(critic_obs_tn.reshape(flat, -1)).reshape(T, N, 1)
        next_values_tn = self.value_probe(next_critic_obs_tn.reshape(flat, -1)).reshape(T, N, 1)

        rewards_nt = self._reward_total(batch).transpose(0, 1).unsqueeze(-1)
        terms_nt = batch[TERM_KEY].transpose(0, 1).float()
        dones_nt = batch[DONE_KEY].transpose(0, 1).float()
        values_nt = values_tn.transpose(0, 1)
        next_values_nt = next_values_tn.transpose(0, 1)
        _, returns_nt = self.gae(
            rewards_nt,
            terms_nt,
            dones_nt,
            values_nt,
            next_values_nt,
        )
        if "is_init" in batch.keys(True, True):
            valid_mask = (~batch["is_init"].transpose(0, 1).bool()).squeeze(-1)
        else:
            valid_mask = torch.ones_like(rewards_nt[..., 0], dtype=torch.bool)
        value_errors = (values_nt - returns_nt).square().squeeze(-1)
        value_loss = _masked_mean(value_errors, valid_mask)

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        if aa.is_distributed() and self.cfg.grad_sync_mode == "manual":
            self._all_reduce_grads(self.value_probe)
        if self.cfg.max_grad_norm > 0:
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.value_probe.parameters(),
                self.cfg.max_grad_norm,
            )
        else:
            value_grad_norm = torch.zeros((), device=self.device)
        self.value_optimizer.step()

        with torch.no_grad():
            probe_td = batch.reshape(-1).copy()
            self._prepare_batch_inputs(probe_td, update_normalizers=False)
            probe_td = self._sample_actor(probe_td)
            self._prepare_critic_input(probe_td)
            self.qnet(probe_td)
            q_probs = F.softmax(probe_td[Q_LOGITS_KEY], dim=-1)
            q_values = distributional_q_value(q_probs, self.q_support)
            q_pi = self._reduce_actor_q_values(q_values)
            critic_obs_flat = probe_td[CRITIC_OBS_KEY]
            value_pred = self.value_probe(critic_obs_flat).squeeze(-1)
            if "is_init" in probe_td.keys(True, True):
                flat_mask = ~probe_td["is_init"].bool().squeeze(-1)
            else:
                flat_mask = None
            value_info = {
                "value/loss": value_loss.detach(),
                "value/grad_norm": value_grad_norm.detach(),
                "value/pred_mean": _masked_mean(value_pred.detach(), flat_mask),
                "value/return_mean": _masked_mean(returns_nt.detach().squeeze(-1), valid_mask),
                "value/q_pi_mean": _masked_mean(q_pi.detach(), flat_mask),
                "value/q_pi_gap": _masked_mean((q_pi - value_pred).detach(), flat_mask),
            }
        return value_info

    def _update_step(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        timing = self.cfg.debug_timing
        timing_data: dict[str, float] = {}
        self._sync_if_timing()
        t0 = time.perf_counter()
        tensordict = tensordict.copy()
        self._prepare_batch_inputs(
            tensordict,
            update_normalizers=self.update_obs_normalization,
        )
        self._sync_if_timing()
        if timing:
            timing_data["vecnorm_ms"] = (time.perf_counter() - t0) * 1000.0

        critic_mask = None
        actor_mask = None
        if "is_init" in tensordict.keys(True, True):
            valid = ~tensordict["is_init"].squeeze(-1)
            critic_valid = valid[:, 0]
            critic_mask = critic_valid if critic_valid.any() else None
            if self.cfg.actor_update_scope == "first":
                actor_valid = critic_valid
            else:
                actor_valid = valid.reshape(-1)
            actor_mask = actor_valid if actor_valid.any() else None

        self._sync_if_timing()
        t0 = time.perf_counter()
        q_loss, q_grad_norm, target_values, next_log_probs, critic_diag = self._update_critic(
            tensordict,
            critic_mask,
        )
        self._sync_if_timing()
        if timing:
            timing_data["critic_ms"] = (time.perf_counter() - t0) * 1000.0

        actor_updated = self.gradient_step % self.cfg.policy_frequency == 0
        self._sync_if_timing()
        t0 = time.perf_counter()
        if actor_updated:
            if self.cfg.actor_update_scope == "first":
                actor_td = tensordict[:, 0]
            else:
                actor_td = tensordict.reshape(-1)
            actor_loss, entropy, action_std, actor_grad_norm, actor_diag = self._update_actor(
                actor_td,
                actor_mask,
            )
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
        self._sync_if_timing()
        if timing:
            timing_data["actor_ms"] = (time.perf_counter() - t0) * 1000.0

        self._sync_if_timing()
        t0 = time.perf_counter()
        alpha_loss = self._update_alpha(next_log_probs)
        self._soft_update_target()
        self._sync_if_timing()
        if timing:
            timing_data["alpha_target_ms"] = (time.perf_counter() - t0) * 1000.0

        value_metrics: dict[str, torch.Tensor] = {}
        if self.enable_value_probe and self.gradient_step % self.cfg.value_probe_update_every == 0:
            for _ in range(self.cfg.value_probe_inner):
                value_metrics = self._train_value_probe()
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
        metrics.update(value_metrics)
        if timing:
            for key, value in timing_data.items():
                metrics[f"debug_timing/{key}"] = torch.tensor(value, device=self.device)
        return metrics

    def update(self) -> dict[str, float]:
        timing = self.cfg.debug_timing
        update_start = time.perf_counter()
        self.num_updates += 1
        info: dict[str, float] = {
            "rb_size": float(len(self.replay_buffer)),
            "alpha/value": self.log_alpha.exp().item(),
        }
        if len(self.replay_buffer) < self.min_replay_sample_transitions:
            return info

        metric_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(self.cfg.updates_per_step):
            t_sample = time.perf_counter()
            batch = self.replay_buffer.sample()
            if self.use_custom_replay_buffer:
                pass
            elif self.use_slice_replay:
                batch = batch.reshape(
                    self.cfg.replay_batch_size,
                    self.cfg.n_step,
                )
            else:
                batch = batch.unsqueeze(1)
            t_to_device = time.perf_counter()
            if not self.replay_samples_on_device:
                batch = batch.to(self.device)
            self._sync_if_timing()
            t_update = time.perf_counter()
            step_metrics = self._update_step(batch)
            self._sync_if_timing()
            t_done = time.perf_counter()
            if timing:
                step_metrics["debug_timing/sample_ms"] = torch.tensor(
                    (t_to_device - t_sample) * 1000.0,
                    device=self.device,
                )
                step_metrics["debug_timing/to_device_ms"] = torch.tensor(
                    (t_update - t_to_device) * 1000.0,
                    device=self.device,
                )
                step_metrics["debug_timing/update_step_ms"] = torch.tensor(
                    (t_done - t_update) * 1000.0,
                    device=self.device,
                )
            for key, value in step_metrics.items():
                metric_lists[key].append(value.detach())

        self._sync_vecnorms()
        for key, values in metric_lists.items():
            info[key] = torch.stack(values).float().mean().item()
        info["rb_size"] = float(len(self.replay_buffer))
        info["gradient_step"] = float(self.gradient_step)
        if timing and self.gradient_step % self.cfg.debug_timing_interval == 0:
            debug_items = {
                key.removeprefix("debug_timing/"): value
                for key, value in info.items()
                if key.startswith("debug_timing/")
            }
            debug_items["update_total_ms"] = (time.perf_counter() - update_start) * 1000.0
            ordered = " ".join(f"{key}={value:.2f}" for key, value in sorted(debug_items.items()))
            print(
                f"[FastSAC timing] step={self.gradient_step} "
                f"n_step={self.cfg.n_step} batch={self.cfg.replay_batch_size} {ordered}",
                flush=True,
            )
        return info

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        del critic
        rollout_policy = FastSACRolloutPolicy(self)
        if mode == "train":
            return WarmupUniformRolloutPolicy(self, rollout_policy)
        return rollout_policy

    def train_op(self, tensordict: TensorDictBase) -> dict[str, float]:
        self.observe(tensordict.exclude("stats"))
        return self.update()

    def compute_value(self, tensordict: TensorDictBase) -> TensorDictBase:
        work_td = tensordict.copy()
        with torch.no_grad():
            self._prepare_batch_inputs(work_td, update_normalizers=False)
            work_td = self._sample_actor(work_td)
            self._prepare_critic_input(work_td)
            self.qnet(work_td)
            q_probs = F.softmax(work_td[Q_LOGITS_KEY], dim=-1)
            q_values = distributional_q_value(q_probs, self.q_support)
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
        state_dict["log_alpha"] = self.log_alpha.detach().clone()
        state_dict["q_support"] = self.q_support.detach().clone()
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

        if "log_alpha" in state_dict:
            self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.log_alpha.device))
        if "q_support" in state_dict:
            self.q_support.copy_(state_dict["q_support"].to(self.q_support.device))
        self.gradient_step = int(state_dict.get("gradient_step", 0))
        self.num_updates = int(state_dict.get("num_updates", 0))
        start_iter = int(state_dict.get("last_iter", 0))
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(start_iter)

        return failed_keys
