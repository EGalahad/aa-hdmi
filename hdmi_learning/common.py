import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY


REF_JPOS_KEY = "ref_joint_pos_"
PRIV_TEACHER_KEY = "priv_teacher"
PRIV_STUDENT_KEY = "priv_student"
CMD_SHORT_KEY = "command_short"


class NullVecNorm(VecNorm):
    """Identity VecNorm that keeps the module/state interface intact."""

    def forward(self, input_vector: torch.Tensor):
        return input_vector

    def _update(self, input_vector: torch.Tensor):
        raise RuntimeError("NullVecNorm does not support updating statistics.")

    def _compute(self):
        raise RuntimeError("NullVecNorm does not compute normalization.")

    def synchronize(self, mode: str = "broadcast"):
        del mode
        return None


def check_vecnorm_divergence(vecnorm: VecNorm):
    world_size = aa.get_world_size()

    loc, scale = vecnorm._compute()
    gather_loc = [torch.empty_like(loc) for _ in range(world_size)]
    gather_scale = [torch.empty_like(scale) for _ in range(world_size)]
    dist.all_gather(gather_loc, loc)
    dist.all_gather(gather_scale, scale)

    loc_diffs = []
    scale_diffs = []
    for i in range(world_size):
        loc_diff = torch.abs(gather_loc[i] - loc).sum().item()
        scale_diff = torch.abs(gather_scale[i] - scale).sum().item()
        loc_diffs.append(loc_diff)
        scale_diffs.append(scale_diff)
    return loc_diffs, scale_diffs


class MeanAction(TensorDictModuleBase):
    in_keys = ["loc"]
    out_keys = [ACTION_KEY]

    def forward(self, td):
        td[ACTION_KEY] = td["loc"]
        return td


class ObsOODDetector(TensorDictModuleBase):
    def __init__(self, in_keys, sigma: float = 5.0):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [("next", f"{k}_ood_ratio") for k in in_keys] + [
            ("next", k) for k in in_keys
        ]
        self.sigma = sigma

    def forward(self, tensordict: TensorDict):
        for in_key in self.in_keys:
            obs = tensordict.get(in_key, None)
            if obs is not None:
                ood_ratio = (obs.abs() > self.sigma).float().mean(dim=-1, keepdim=True)
                tensordict.set(("next", f"{in_key}_ood_ratio"), ood_ratio)
                tensordict.set(("next", in_key), obs)
        return tensordict


class ActorROA(nn.Module):
    def __init__(
        self,
        action_dim: int,
        init_noise_scale: float = 1.0,
        load_noise_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.load_noise_scale is not None:
            self.actor_std.data.fill_(self.load_noise_scale)
