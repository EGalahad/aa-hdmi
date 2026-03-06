"""Feet-related reward aliases for HDMI tasks."""

import active_adaptation as aa
from active_adaptation.envs.mdp.base import Reward as BaseReward
from typing import cast
import torch
try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names

if aa.get_backend() == "isaac":
    from isaaclab.sensors import ContactSensor as IsaacContactSensor
elif aa.get_backend() == "mjlab":
    from mjlab.sensor import ContactSensor as MJLabContactSensor


def _sensor_body_names(contact_sensor) -> list[str]:
    if hasattr(contact_sensor, "body_names"):
        return list(contact_sensor.body_names)
    slots = getattr(contact_sensor, "_slots", None)
    if slots is not None:
        names: list[str] = []
        for slot in slots:
            name = getattr(slot, "primary_name", None)
            if name is not None and name not in names:
                names.append(name)
        if names:
            return names
    raise AttributeError("Cannot infer body names from contact sensor.")


def _find_sensor_bodies(contact_sensor, body_names: str):
    if hasattr(contact_sensor, "find_bodies"):
        return contact_sensor.find_bodies(body_names)
    sensor_body_names = _sensor_body_names(contact_sensor)
    return resolve_matching_names(body_names, sensor_body_names)


def _substep_in_contact(contact_sensor, body_ids: torch.Tensor) -> torch.Tensor:
    if aa.get_backend() == "isaac":
        contact_sensor = cast(IsaacContactSensor, contact_sensor)
        return contact_sensor.data.net_forces_w[:, body_ids].norm(dim=-1) > 0.0
    if aa.get_backend() == "mjlab":
        contact_sensor = cast(MJLabContactSensor, contact_sensor)
        found = contact_sensor.data.found
        if found is not None:
            found = found[:, body_ids]
            if found.ndim == 3:
                # [N, B, S] -> [N, B]
                found = found.any(dim=-1)
            return found > 0
        force = contact_sensor.data.force
        if force is not None:
            force = force[:, body_ids]
            if force.ndim == 4:
                # [N, B, S, 3] -> [N, B]
                return force.norm(dim=-1).any(dim=-1) > 0.0
            return force.norm(dim=-1) > 0.0

    current_contact_time = contact_sensor.data.current_contact_time
    if current_contact_time is None:
        return torch.zeros(
            (contact_sensor.data.force.shape[0], len(body_ids)),
            dtype=torch.bool,
            device=body_ids.device,
        )
    return current_contact_time[:, body_ids] > 0.0


class _ContactMajorityCache:
    """Shared per-env-step contact state using substep majority voting."""

    def __init__(self, env, contact_sensor):
        if aa.get_backend() == "isaac":
            contact_sensor = cast(IsaacContactSensor, contact_sensor)
        elif aa.get_backend() == "mjlab":
            contact_sensor = cast(MJLabContactSensor, contact_sensor)

        self.env = env
        self.contact_sensor = contact_sensor
        self.decimation = max(int(self.env.decimation), 1)

        self.body_names = _sensor_body_names(contact_sensor)
        self.num_bodies = len(self.body_names)
        self.all_body_ids = torch.arange(self.num_bodies, device=self.env.device)

        self.substep_found = torch.zeros(
            (self.env.num_envs, self.num_bodies, self.decimation),
            dtype=torch.bool,
            device=self.env.device,
        )
        self.current_contact = torch.zeros(
            (self.env.num_envs, self.num_bodies), dtype=torch.bool, device=self.env.device
        )
        self.is_first_contact = torch.zeros_like(self.current_contact)
        self.is_first_detached = torch.zeros_like(self.current_contact)

        self._last_post_stamp = (-1, -1)
        self._last_update_stamp = -1

    def reset(self, env_ids: torch.Tensor):
        self.substep_found[env_ids] = False
        self.current_contact[env_ids] = False
        self.is_first_contact[env_ids] = False
        self.is_first_detached[env_ids] = False

    def post_step(self, substep: int):
        stamp = (int(self.env.timestamp), int(substep))
        if stamp == self._last_post_stamp:
            return
        self.substep_found[:, :, substep] = _substep_in_contact(
            self.contact_sensor, self.all_body_ids
        )
        self._last_post_stamp = stamp

    def update(self):
        stamp = int(self.env.timestamp)
        if stamp == self._last_update_stamp:
            return
        votes = self.substep_found.sum(dim=-1)
        contact_majority = votes >= (self.decimation // 2)
        prev_contact = self.current_contact
        self.is_first_contact[:] = (~prev_contact) & contact_majority
        self.is_first_detached[:] = prev_contact & (~contact_majority)
        self.current_contact[:] = contact_majority
        self.substep_found.zero_()
        self._last_update_stamp = stamp

    def current_contact_for(self, body_ids: torch.Tensor):
        return self.current_contact[:, body_ids]

    def first_contact_for(self, body_ids: torch.Tensor):
        return self.is_first_contact[:, body_ids]


def _get_contact_majority_cache(env, contact_sensor):
    cache = getattr(env, "_hdmi_contact_majority_cache", None)
    if cache is None:
        cache = _ContactMajorityCache(env, contact_sensor)
        env._hdmi_contact_majority_cache = cache
    elif cache.contact_sensor is not contact_sensor:
        raise RuntimeError("Multiple contact sensors are not supported by shared contact cache.")
    return cache


class feet_slip(BaseReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        body_names: str,
        weight: float,
        tolerance: float = 0.0,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.env.scene.articulations["robot"]
        self.articulation_body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.articulation_body_ids = torch.as_tensor(
            self.articulation_body_ids, device=self.device
        )
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        sensor_ids, _ = _find_sensor_bodies(self.contact_sensor, body_names)
        self.sensor_body_ids = torch.as_tensor(sensor_ids, device=self.device)
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
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]
        sensor_ids, _ = _find_sensor_bodies(self.contact_sensor, body_names)

        self.sensor_body_ids = torch.as_tensor(sensor_ids, device=self.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

        self.thres = thres

    def reset(self, env_ids):
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        first_contact = self.contact_cache.first_contact_for(self.sensor_body_ids)
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


class feet_contact_count(BaseReward, namespace="hdmi"):
    supported_backends = ("isaac", "mjlab")

    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight=weight, enabled=enabled)
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]

        body_ids, self.body_names = _find_sensor_bodies(self.contact_sensor, body_names)
        self.body_ids = torch.as_tensor(body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

    def reset(self, env_ids):
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        first_contact = self.contact_cache.first_contact_for(self.body_ids)
        return first_contact.float().mean(1, keepdim=True)


class feet_contact_duration(BaseReward, namespace="hdmi"):
    supported_backends = ("isaac", "mjlab")

    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight=weight, enabled=enabled)
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]

        body_ids, self.body_names = _find_sensor_bodies(self.contact_sensor, body_names)
        self.body_ids = torch.as_tensor(body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

    def reset(self, env_ids):
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        current_contact = self.contact_cache.current_contact_for(self.body_ids)
        return current_contact.float().mean(1, keepdim=True)
