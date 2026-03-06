"""Common reward aliases for HDMI tasks."""

import active_adaptation as aa
from active_adaptation.envs.mdp.base import Reward as BaseReward
from typing import List, TYPE_CHECKING
import torch

try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names

if TYPE_CHECKING:
    from mjlab.sensor import ContactSensor


class joint_pos_limits(BaseReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        weight: float,
        joint_names: List[str] | str = ".*",
        soft_factor: float = 0.9,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(
            joint_names, self.asset.joint_names
        )
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self):
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(dim=1, keepdim=True)


class joint_torque_limits(BaseReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        weight: float,
        joint_names: List[str] | str = ".*",
        soft_factor: float = 0.9,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(
            joint_names, self.asset.joint_names
        )
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        self.soft_limits = (
            self.asset.data.joint_effort_limits[:, self.joint_ids] * soft_factor
        )

    def compute(self):
        if hasattr(self.asset.data, "actuator_force"):
            applied_torque = self.asset.data.actuator_force[:, self.joint_ids]
        else:
            applied_torque = self.asset.data.applied_torque[:, self.joint_ids]
        violation_high = (applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        violation_low = (-applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        return -(violation_high + violation_low).sum(dim=1, keepdim=True)


class self_collisions(BaseReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        weight: float,
        sensor_name: str = "self_collision",
        force_threshold: float = 10.0,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        if aa.get_backend() != "mjlab":
            raise NotImplementedError(
                f"hdmi.self_collisions only supports mjlab backend, got '{aa.get_backend()}'."
            )
        self.sensor_name = sensor_name
        self.force_threshold = force_threshold
        self.contact_sensor: "ContactSensor" = self.env.scene.sensors[sensor_name]

    def compute(self):
        data = self.contact_sensor.data
        if data.force_history is not None:
            # force_history: [B, N, H, 3]
            force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
            hit = (force_mag > self.force_threshold).any(dim=1)  # [B, H]
            return hit.sum(dim=-1, keepdim=True).float()  # [B, 1]

        if data.found is None:
            raise RuntimeError(
                f"Contact sensor '{self.sensor_name}' does not provide force_history or found."
            )
        return data.found.squeeze(-1).unsqueeze(1).float()
