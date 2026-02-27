"""Common reward aliases for HDMI tasks."""

import active_adaptation as aa
from active_adaptation.envs.mdp.base import Reward as BaseReward
from typing import Any, List, cast
import torch

try:
    from isaaclab.utils.string import (
        resolve_matching_names,
        resolve_matching_names_values,
    )
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import (
        resolve_matching_names,
        resolve_matching_names_values,
    )

if aa.get_backend() == "isaac":
    from isaaclab.sensors import ContactSensor as ContactSensorType
elif aa.get_backend() == "mjlab":
    from active_adaptation.sensors.mjlab import CfrcContactSensor as ContactSensorType
else:
    ContactSensorType = Any


def _substep_in_contact(contact_sensor: ContactSensorType, body_ids: torch.Tensor) -> torch.Tensor:
    if aa.get_backend() == "isaac":
        return contact_sensor.data.net_forces_w[:, body_ids].norm(dim=-1) > 0.0
    if aa.get_backend() == "mjlab":
        found = contact_sensor.data.found
        if found is None:
            num_envs = contact_sensor.data.force.shape[0]
            return torch.zeros(
                (num_envs, len(body_ids)),
                dtype=torch.bool,
                device=body_ids.device,
            )
        return found[:, body_ids] > 0
    current_contact_time = contact_sensor.data.current_contact_time
    if current_contact_time is None:
        return torch.zeros(
            (contact_sensor.data.force.shape[0], len(body_ids)),
            dtype=torch.bool,
            device=body_ids.device,
        )
    return current_contact_time[:, body_ids] > 0.0


class joint_pos_limits(BaseReward):
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


class joint_torque_limits(BaseReward):
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


class feet_slip(BaseReward):
    def __init__(
        self,
        env,
        body_names: str,
        weight: float,
        tolerance: float = 0.0,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.env.scene["robot"]
        self.articulation_body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.articulation_body_ids = torch.as_tensor(
            self.articulation_body_ids, device=self.device
        )
        sensors = self.env.scene.sensors
        self.contact_sensor = cast(
            ContactSensorType,
            sensors.get("feet_ground_contact", sensors.get("contact_forces")),
        )
        if self.contact_sensor is None:
            raise KeyError("Neither 'feet_ground_contact' nor 'contact_forces' sensor exists.")
        if hasattr(self.contact_sensor, "find_bodies"):
            sensor_ids, _ = self.contact_sensor.find_bodies(body_names)
            self.sensor_body_ids = torch.as_tensor(sensor_ids, device=self.device)
        else:
            self.sensor_body_ids = self.articulation_body_ids
        self.tolerance = tolerance
        self.in_contact_step = torch.zeros(
            self.num_envs, len(self.sensor_body_ids), dtype=torch.bool, device=self.device
        )

    def pre_step(self, substep: int):
        if substep == 0:
            self.in_contact_step[:] = False

    def post_step(self, substep: int):
        self.in_contact_step |= _substep_in_contact(self.contact_sensor, self.sensor_body_ids)

    def compute(self):
        feet_vel = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, :2]
        feet_vel = (feet_vel.norm(dim=-1) - self.tolerance).clamp(min=0.0, max=1.0)
        slip = (self.in_contact_step * feet_vel).sum(dim=1, keepdim=True)
        return -slip


class feet_air_time(BaseReward, namespace="hdmi"):
    supported_backends = ("isaac", "mjlab")

    def __init__(
        self,
        env,
        body_names: str,
        thres: float,
        weight: float,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.thres = thres
        self.asset = self.env.scene.articulations["robot"]
        self.articulation_body_ids, self.body_names = self.asset.find_bodies(body_names)
        sensors = self.env.scene.sensors
        self.contact_sensor = cast(ContactSensorType, sensors.get("contact_forces"))
        if hasattr(self.contact_sensor, "find_bodies"):
            sensor_ids, _ = self.contact_sensor.find_bodies(body_names)
            self.sensor_body_ids = torch.as_tensor(sensor_ids, device=self.device)
        else:
            self.sensor_body_ids = torch.as_tensor(self.articulation_body_ids, device=self.device)
        num_bodies = len(self.sensor_body_ids)
        self.in_contact_last = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.bool, device=self.device
        )
        self.in_contact_step = torch.zeros_like(self.in_contact_last)

    def reset(self, env_ids):
        self.in_contact_last.index_fill_(0, env_ids, False)
        self.in_contact_step.index_fill_(0, env_ids, False)

    def pre_step(self, substep: int):
        if substep == 0:
            self.in_contact_step[:] = False

    def post_step(self, substep: int):
        self.in_contact_step |= _substep_in_contact(self.contact_sensor, self.sensor_body_ids)

    def compute(self):
        in_contact_this = self.in_contact_step
        first_contact = (~in_contact_this) & self.in_contact_last
        self.in_contact_last[:] = in_contact_this
        last_air_time = self.contact_sensor.data.last_air_time
        if last_air_time is None:
            last_air_time = self.contact_sensor.data.current_air_time
        last_air_time = last_air_time[:, self.sensor_body_ids]
        reward = torch.sum(
            (last_air_time - self.thres).clamp_max(0.0) * first_contact,
            dim=1,
            keepdim=True,
        )
        reward *= ~self.command_manager.is_standing_env
        return reward


class feet_contact_count(BaseReward):
    supported_backends = ("isaac", "mjlab")
    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight=weight, enabled=enabled)
        self.asset = self.env.scene["robot"]
        self.contact_sensor = cast(ContactSensorType, self.env.scene.sensors["contact_forces"])

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.in_contact_step = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=torch.bool, device=self.env.device
        )

    def reset(self, env_ids):
        self.in_contact_step.index_fill_(0, env_ids, False)

    def pre_step(self, substep: int):
        if substep == 0:
            self.in_contact_step[:] = False

    def post_step(self, substep: int):
        self.in_contact_step |= _substep_in_contact(self.contact_sensor, self.body_ids)

    def compute(self):
        return self.in_contact_step.float().sum(1, keepdim=True)
