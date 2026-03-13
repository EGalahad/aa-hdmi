"""Feet-related reward aliases for HDMI tasks."""

import active_adaptation as aa
from hdmi.tasks.command import RobotTracking
from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward
from typing import TYPE_CHECKING, cast
import torch
try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names

if aa.get_backend() == "isaac":
    from isaaclab.sensors import ContactSensor as IsaacContactSensor
elif aa.get_backend() == "mjlab":
    from mjlab.sensor import ContactSensor as MJLabContactSensor

if TYPE_CHECKING:
    from mjlab.viewer.viser import ViserMujocoScene

TrackReward = BaseReward[RobotTracking]

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


def _find_sensor_bodies_by_names(contact_sensor, body_names: list[str]) -> tuple[list[int], list[str]]:
    sensor_body_names = _sensor_body_names(contact_sensor)
    sensor_name_to_id = {name: i for i, name in enumerate(sensor_body_names)}
    ids = []
    names = []
    for name in body_names:
        sensor_id = sensor_name_to_id.get(name)
        if sensor_id is None:
            continue
        ids.append(sensor_id)
        names.append(name)
    return ids, names


def _current_in_contact(contact_sensor, body_ids: torch.Tensor) -> torch.Tensor:
    """Return [N, B] in-contact mask using the configured history window."""
    if aa.get_backend() == "isaac":
        contact_sensor = cast(IsaacContactSensor, contact_sensor)
    elif aa.get_backend() == "mjlab":
        contact_sensor = cast(MJLabContactSensor, contact_sensor)

    data = contact_sensor.data

    # mjlab ContactSensor: [N, B, H, 3]
    force_history = getattr(data, "force_history", None)
    if force_history is not None:
        force_mag = force_history[:, body_ids].norm(dim=-1)  # [N, B, H]
        return force_mag.gt(0.0).any(dim=-1)

    # isaac ContactSensor: commonly [N, H, B, 3], but keep robust to [N, B, H, 3].
    current_contact_time = getattr(data, "current_contact_time", None)
    if current_contact_time is not None:
        return current_contact_time[:, body_ids] > 1e-6

    raise RuntimeError("Contact sensor does not expose usable contact fields.")


class feet_slip(TrackReward, namespace="hdmi"):
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
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        _, matched_body_names = self.asset.find_bodies(body_names)
        self.body_names = sorted(matched_body_names)
        self.articulation_body_ids = torch.as_tensor(
            [self.asset.body_names.index(name) for name in self.body_names],
            device=self.device,
        )
        sensor_ids, sensor_names = _find_sensor_bodies_by_names(
            self.contact_sensor, self.body_names
        )
        if set(sensor_names) != set(self.body_names):
            missing = sorted(set(self.body_names) - set(sensor_names))
            raise RuntimeError(
                f"feet_slip: missing feet in contact sensor: {missing}"
            )
        self.sensor_body_ids = torch.as_tensor(sensor_ids, device=self.device)
        self.tolerance = tolerance

    def _compute(self):
        in_contact_step = _current_in_contact(self.contact_sensor, self.sensor_body_ids)
        feet_vel = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, :2]
        feet_vel = (feet_vel.norm(dim=-1) - self.tolerance).clamp(min=0.0, max=1.0)
        slip = (in_contact_step * feet_vel).sum(dim=1, keepdim=True)
        return -slip


class feet_air_time(TrackReward, namespace="hdmi"):
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
        self.current_contact = torch.zeros(
            self.num_envs, len(self.sensor_body_ids), dtype=torch.bool, device=self.device
        )
        self.prev_contact = torch.zeros_like(self.current_contact)
        self.is_first_contact = torch.zeros_like(self.current_contact)

        self.thres = thres

    def reset(self, env_ids):
        self.current_contact[env_ids] = False
        self.prev_contact[env_ids] = False
        self.is_first_contact[env_ids] = False

    def update(self):
        self.prev_contact[:] = self.current_contact
        self.current_contact[:] = _current_in_contact(
            self.contact_sensor, self.sensor_body_ids
        )
        self.is_first_contact[:] = (~self.prev_contact) & self.current_contact

    def _compute(self):
        last_air_time = self.contact_sensor.data.last_air_time
        if last_air_time is None:
            last_air_time = self.contact_sensor.data.current_air_time
        last_air_time = last_air_time[:, self.sensor_body_ids]
        reward = torch.sum(
            (last_air_time - self.thres).clamp_max(0.0) * self.is_first_contact,
            dim=1,
            keepdim=True,
        )
        reward *= ~self.command_manager.is_standing_env
        return reward


class feet_contact_count(TrackReward, namespace="hdmi"):
    supported_backends = ("isaac", "mjlab")

    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight=weight, enabled=enabled)
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]

        body_ids, self.body_names = _find_sensor_bodies(self.contact_sensor, body_names)
        self.body_ids = torch.as_tensor(body_ids, device=self.env.device)
        self.current_contact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=torch.bool, device=self.device
        )
        self.prev_contact = torch.zeros_like(self.current_contact)
        self.is_first_contact = torch.zeros_like(self.current_contact)

    def reset(self, env_ids):
        self.current_contact[env_ids] = False
        self.prev_contact[env_ids] = False
        self.is_first_contact[env_ids] = False

    def update(self):
        self.prev_contact[:] = self.current_contact
        self.current_contact[:] = _current_in_contact(
            self.contact_sensor, self.body_ids
        )
        self.is_first_contact[:] = (~self.prev_contact) & self.current_contact

    def _compute(self):
        return self.is_first_contact.float().mean(1, keepdim=True)


class feet_contact_duration(TrackReward, namespace="hdmi"):
    supported_backends = ("isaac", "mjlab")

    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight=weight, enabled=enabled)
        self.contact_sensor: "IsaacContactSensor" | "MJLabContactSensor" = self.env.scene.sensors["contact_forces"]

        body_ids, self.body_names = _find_sensor_bodies(self.contact_sensor, body_names)
        self.body_ids = torch.as_tensor(body_ids, device=self.env.device)
        self.current_contact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids):
        self.current_contact[env_ids] = False

    def update(self):
        self.current_contact[:] = _current_in_contact(
            self.contact_sensor, self.body_ids
        )

    def _compute(self):
        return self.current_contact.float().mean(1, keepdim=True)


class ref_contact(TrackReward, namespace="hdmi"):
    supported_backends = ("isaac", "mjlab")

    def __init__(
        self,
        env,
        body_names: str | list[str],
        body2_names: str | list[str] | None = None,
        air_h_low: float = 0.035,
        air_h_high: float = 0.155,
        contact_h_low: float = 0.035,
        contact_h_high: float = 0.125,
        feet_standing_z_enter: float = 0.18,
        feet_standing_z_exit: float = 0.25,
        feet_standing_vxy_enter: float = 0.2,
        feet_standing_vxy_exit: float = 0.3,
        feet_standing_vz_enter: float = 0.15,
        feet_standing_vz_exit: float = 0.25,
        debug_draw_enabled: bool = True,
        debug_target_color: tuple[float, float, float, float] = (1.0, 0.9, 0.1, 1.0),
        debug_current_color: tuple[float, float, float, float] = (0.1, 0.9, 1.0, 1.0),
        debug_target_z_offset: float = 0.02,
        debug_current_z_offset: float = 0.0,
        debug_point_size: float = 50.0,
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, weight=weight, **kwargs)
        self.asset = self.command_manager.asset
        self.contact_sensor = self.env.scene.sensors["contact_forces"]

        _, motion_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        _, asset_names = resolve_matching_names(body_names, self.asset.body_names)
        matched_names = sorted(set(motion_names) & set(asset_names))
        if not matched_names:
            raise RuntimeError("feet_air_time_ref_dense: no matched feet in motion and asset.")

        self.body_indices_motion = torch.tensor(
            [self.command_manager.tracking_body_names.index(n) for n in matched_names],
            dtype=torch.long,
            device=self.device,
        )
        self.body_indices_asset = torch.tensor(
            [self.asset.body_names.index(n) for n in matched_names],
            dtype=torch.long,
            device=self.device,
        )

        sensor_ids, sensor_names = _find_sensor_bodies_by_names(self.contact_sensor, matched_names)
        if not sensor_ids:
            raise RuntimeError("feet_air_time_ref_dense: no matched feet in contact sensor.")
        if set(sensor_names) != set(matched_names):
            missing = sorted(set(matched_names) - set(sensor_names))
            raise RuntimeError(
                f"feet_air_time_ref_dense: missing feet in contact sensor: {missing}"
            )
        self.sensor_body_ids = torch.tensor(sensor_ids, dtype=torch.long, device=self.device)

        if body2_names is None:
            self.body2_indices_asset = self.body_indices_asset
        else:
            body2_indices_asset, _ = self.asset.find_bodies(body2_names)
            self.body2_indices_asset = torch.as_tensor(
                body2_indices_asset, dtype=torch.long, device=self.device
            )
            if len(self.body2_indices_asset) != len(self.body_indices_asset):
                raise ValueError(
                    "body2_names must match body_names length for feet_air_time_ref_dense."
                )

        self.air_h_low = float(air_h_low)
        self.air_h_high = float(air_h_high)
        self.air_h_span = max(self.air_h_high - self.air_h_low, 1e-6)
        self.contact_h_low = float(contact_h_low)
        self.contact_h_high = float(contact_h_high)
        self.contact_h_span = max(self.contact_h_high - self.contact_h_low, 1e-6)

        self.feet_standing_z_enter = float(feet_standing_z_enter)
        self.feet_standing_z_exit = float(feet_standing_z_exit)
        self.feet_standing_vxy_enter = float(feet_standing_vxy_enter)
        self.feet_standing_vxy_exit = float(feet_standing_vxy_exit)
        self.feet_standing_vz_enter = float(feet_standing_vz_enter)
        self.feet_standing_vz_exit = float(feet_standing_vz_exit)

        self.current_contact = torch.zeros(
            self.num_envs, len(self.sensor_body_ids), dtype=torch.bool, device=self.device
        )
        self.target_contact = torch.zeros_like(self.current_contact)

        self.debug_draw_enabled = bool(debug_draw_enabled)
        self.debug_target_color = debug_target_color
        self.debug_current_color = debug_current_color
        self.debug_target_z_offset = float(debug_target_z_offset)
        self.debug_current_z_offset = float(debug_current_z_offset)
        self.debug_point_size = float(debug_point_size)

    def reset(self, env_ids):
        self.current_contact[env_ids] = False
        self.target_contact[env_ids] = False

    def _compute_target_contact(self) -> torch.Tensor:
        ref_vel = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
        ref_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        root_vxy = self.command_manager.current_ref_motion.body_lin_vel_w[
            :, self.command_manager.root_body_idx_motion, :2
        ].norm(dim=-1, keepdim=True).clamp_min(1.0)

        feet_vxy = ref_vel[..., :2].norm(dim=-1)
        feet_vz_abs = ref_vel[..., 2].abs()
        feet_z = ref_pos[..., 2]

        enter_contact = (
            (feet_z < self.feet_standing_z_enter)
            & (feet_vxy < self.feet_standing_vxy_enter * root_vxy)
            & (feet_vz_abs < self.feet_standing_vz_enter * root_vxy)
        )
        exit_contact = (
            (feet_z > self.feet_standing_z_exit)
            | (feet_vxy > self.feet_standing_vxy_exit * root_vxy)
            | (feet_vz_abs > self.feet_standing_vz_exit * root_vxy)
        )
        self.target_contact = (self.target_contact & (~exit_contact)) | enter_contact
        return self.target_contact

    def _compute(self):
        self.current_contact[:] = _current_in_contact(
            self.contact_sensor, self.sensor_body_ids
        )
        target_contact = self._compute_target_contact()
        current_contact = self.current_contact

        mismatch = current_contact ^ target_contact
        both_air = (~current_contact) & (~target_contact)
        both_contact = current_contact & target_contact

        penalty = torch.zeros_like(current_contact, dtype=torch.float32)
        penalty[mismatch] = -1.0

        feet_height_air = torch.minimum(
            self.asset.data.body_link_pos_w[:, self.body_indices_asset, 2],
            self.asset.data.body_link_pos_w[:, self.body2_indices_asset, 2],
        )
        air_ratio = ((feet_height_air - self.air_h_low) / self.air_h_span).clamp(0.0, 1.0)
        air_penalty = -(1.0 - air_ratio)
        penalty = torch.where(both_air, air_penalty, penalty)

        feet_height_contact = torch.maximum(
            self.asset.data.body_link_pos_w[:, self.body_indices_asset, 2],
            self.asset.data.body_link_pos_w[:, self.body2_indices_asset, 2],
        )
        contact_ratio = ((feet_height_contact - self.contact_h_low) / self.contact_h_span).clamp(
            0.0, 1.0
        )
        contact_penalty = -contact_ratio
        penalty = torch.where(both_contact, contact_penalty, penalty)

        return penalty.mean(dim=1, keepdim=True)

    def debug_draw(self):
        if not self.debug_draw_enabled:
            return

        viewer = getattr(self.env.sim, "viewer", None)
        if viewer is None:
            return
        scene: "ViserMujocoScene" | None = getattr(viewer, "scene", None)
        if scene is None:
            return

        point_radius = scene.meansize * 0.001 * self.debug_point_size
        target_contact = self.target_contact
        current_contact = self.current_contact
        target_points = self.command_manager.ref_body_pos_w[
            :, self.body_indices_motion
        ].clone()
        target_points[..., 2] += self.debug_target_z_offset
        if target_contact.any():
            target_env_ids, target_point_ids = target_contact.nonzero(as_tuple=True)
            target_points_np = target_points[target_contact].detach().cpu().numpy()
            target_labels = [
                f"target_contact_env_{env_idx}_{point_idx}"
                for env_idx, point_idx in zip(
                    target_env_ids.tolist(), target_point_ids.tolist()
                )
            ]
            for label, point in zip(target_labels, target_points_np):
                scene.add_sphere(
                    point,
                    radius=point_radius,
                    color=self.debug_target_color,
                    label=label,
                )

        current_points = self.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ].clone()
        current_points[..., 2] += self.debug_current_z_offset
        if current_contact.any():
            current_env_ids, current_point_ids = current_contact.nonzero(as_tuple=True)
            current_points_np = current_points[current_contact].detach().cpu().numpy()
            current_labels = [
                f"current_contact_env_{env_idx}_{point_idx}"
                for env_idx, point_idx in zip(
                    current_env_ids.tolist(), current_point_ids.tolist()
                )
            ]
            for label, point in zip(current_labels, current_points_np):
                scene.add_sphere(
                    point,
                    radius=point_radius,
                    color=self.debug_current_color,
                    label=label,
                )
