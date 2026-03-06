from dataclasses import dataclass

import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq

from active_adaptation.learning.modules.vecnorm import VecNorm

from .ppo_roa import (
    ACTION_KEY,
    CMD_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    MeanAction,
    ObsOODDetector,
    PPOConfig,
    PPOROA,
)


@dataclass
class PPOConfigTest(PPOConfig):
    _target_: str = f"{__package__}.ppo_roa_test.PPOROATest"


cs = ConfigStore.instance()
cs.store(
    "ppo_roa_train-test",
    node=PPOConfigTest(
        phase="train", vecnorm="train", entropy_coef_start=0.004, entropy_coef_end=0.001
    ),
    group="algo",
)


class PPOROATest(PPOROA):
    """PPOROA variant with non-inplace VecNorm outputs."""

    def _build_vecnorm_modules(self, observation_spec):
        modules = []
        self.norm_map = {}
        self.norm_inv_map = {}
        self.vecnorms = nn.ModuleDict()

        keys_to_norm = [self.cmd_key, OBS_KEY, OBS_PRIV_KEY]
        for key in keys_to_norm:
            if key not in observation_spec.keys(True, True):
                continue
            shape = observation_spec[key].shape[-1:]
            vecnorm = VecNorm(input_shape=shape, stats_shape=shape, decay=1.0)
            out_key = f"{key}_normed"
            self.vecnorms[key] = vecnorm
            modules.append(Mod(vecnorm, [key], [out_key]))
            self.norm_map[key] = out_key
            self.norm_inv_map[out_key] = key

        self.vecnorm = Seq(*modules).to(self.device)

    def get_rollout_policy(self, mode: str = "train"):
        if mode == "deploy":
            vecnorms = []
            in_keys = set(self.adapt_module.in_keys).union(set(self.actor_adapt.in_keys))
            for out_key in in_keys:
                src_key = self.norm_inv_map.get(out_key, None)
                if src_key is not None:
                    vecnorms.append(Mod(self.vecnorms[src_key], [src_key], [out_key]))
            vecnorm = Seq(*vecnorms).to(self.device)
            ood_detector = ObsOODDetector(list(in_keys), sigma=5.0)
            modules = [vecnorm, ood_detector]
        else:
            modules = [self.vecnorm]

        if self.cfg.phase == "train":
            modules += [self.encoder_priv, self.actor]
        elif self.cfg.phase in ("adapt", "finetune"):
            modules += [self.adapt_module, self.actor_adapt]

        if mode == "deploy":
            modules[-1] = modules[-1].module[0]
            modules.append(MeanAction())
            out_keys = [ACTION_KEY]
        else:
            out_keys = ["sample_log_prob", ACTION_KEY] + self.dist_keys

        out_keys += self.vecnorm.out_keys

        return Seq(*modules, selected_out_keys=out_keys)
