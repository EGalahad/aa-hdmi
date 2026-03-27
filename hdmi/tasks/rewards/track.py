from hdmi.tasks.command import RobotTracking

from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward

from typing import List, Dict, TYPE_CHECKING
from omegaconf import DictConfig

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

import torch

if TYPE_CHECKING:
    from mjlab.sensor import ContactSensor

TrackReward = BaseReward[RobotTracking]


class _tracking_keypoint(TrackReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        sigma: float = 0.03,
        tolerance: float | Dict[str, float] = 0.0,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_body_names

        self.sigma = sigma
        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert (
            set(matched_names) == set(matched_names_motion) == set(matched_names_asset)
        ), "body names in motion dataset and robot not matched"
        assert set(matched_names) <= set(
            self.command_manager.tracking_body_names
        ), "Some body names in motion dataset not found in tracking body names"

        self.body_indices_motion = []
        self.body_indices_asset = []
        self.body_indices_tracking = []
        self.body_names = list(sorted(matched_names))
        self.num_bodies = len(self.body_names)
        for body_name in self.body_names:
            body_idx_tracking = self.command_manager.tracking_body_names.index(
                body_name
            )
            body_idx_motion = body_idx_tracking
            body_idx_asset = self.command_manager.asset.body_names.index(body_name)

            self.body_indices_motion.append(body_idx_motion)
            self.body_indices_asset.append(body_idx_asset)
            self.body_indices_tracking.append(body_idx_tracking)

        self.tolerance = torch.zeros(len(self.body_names), device=self.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = (
                resolve_matching_names_values(tolerance, self.body_names)
            )
            self.tolerance[tolerance_indices] = torch.tensor(
                tolerance_values, device=self.device
            )
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

    def _compute(self):
        raise NotImplementedError


class keypoint_pos_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_pos_error[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_pos_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_pos_error(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_pos_error[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)


class keypoint_pos_error_local(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_ori_error[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ori_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ori_error(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_ori_error[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_error_local(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)


class keypoint_lin_vel_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_lin_vel_error[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ang_vel_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.body_ang_vel_error[:, self.body_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class _tracking_joint(TrackReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        joint_names: List[str] | str | None = None,
        sigma: float = 0.03,
        tolerance: float | Dict[str, float] = 0.0,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names

        self.sigma = sigma
        joint_indices_asset, matched_names_asset = resolve_matching_names(
            joint_names, self.command_manager.asset.joint_names
        )
        joint_indices_motion, matched_names_motion = resolve_matching_names(
            joint_names, self.command_manager.tracking_joint_names
        )

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert (
            set(matched_names) == set(matched_names_motion) == set(matched_names_asset)
        ), "joint names in motion dataset and robot not matched"
        assert set(matched_names) <= set(
            self.command_manager.tracking_joint_names
        ), "Some joint names in motion dataset not found in tracking joint names"

        self.joint_indices_motion = []
        self.joint_indices_asset = []
        self.joint_indices_tracking = []
        self.joint_names = list(sorted(matched_names))
        for joint_name in self.joint_names:
            joint_idx_tracking = self.command_manager.tracking_joint_names.index(
                joint_name
            )
            joint_idx_motion = joint_idx_tracking
            joint_idx_asset = self.command_manager.asset.joint_names.index(joint_name)

            self.joint_indices_motion.append(joint_idx_motion)
            self.joint_indices_asset.append(joint_idx_asset)
            self.joint_indices_tracking.append(joint_idx_tracking)

        self.tolerance = torch.zeros(len(self.joint_names), device=self.env.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = (
                resolve_matching_names_values(tolerance, matched_names_motion)
            )
            self.tolerance[tolerance_indices] = torch.tensor(
                tolerance_values, device=self.env.device
            )
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")


class joint_pos_tracking_product(_tracking_joint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class joint_pos_error(_tracking_joint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)


class joint_vel_tracking_product(_tracking_joint, namespace="hdmi"):
    def _compute(self):
        error = (
            self.command_manager.joint_vel_error[:, self.joint_indices_tracking]
            - self.tolerance
        ).clamp_min(0.0)
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class feet_air_time_ref(TrackReward, namespace="hdmi"):
    def __init__(self, env, body_names: List[str] | str, thres: float, **kwargs):
        super().__init__(env, **kwargs)
        self.thres = thres
        self.asset = self.command_manager.asset
        self.contact_sensor: "ContactSensor" = self.env.scene["feet_ground_contact"]

        # map body indices in motion & asset space
        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        matched_names = sorted(set(matched_names_motion) & set(matched_names_asset))
        assert matched_names, "feet_air_time_ref: no feet matched"
        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(n) for n in matched_names
        ]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(n) for n in matched_names
        ]
        sensor_ids, _ = self.contact_sensor.find_bodies(matched_names)
        self.sensor_body_ids = torch.tensor(sensor_ids, device=self.device)

        num_bodies = len(matched_names)
        self.reward_time = torch.zeros(self.num_envs, num_bodies, device=self.device)
        self.last_contact = torch.zeros(
            self.num_envs, num_bodies, dtype=bool, device=self.device
        )

        # height-dependent scaling
        self.h_low, self.h_high = 0.035, 0.12
        self.c_low, self.c_high = 0.5, 2.0
        self.exp_log_c_ratio = torch.log(
            torch.tensor(self.c_high / self.c_low, device=self.device)
        )

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def _compute(self):
        # current contact from sensor
        current_contact = (
            self.contact_sensor.data.current_contact_time[:, self.sensor_body_ids] > 0.0
        )
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        # reference stance: slow & low feet in the reference motion
        ref_vel = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_motion]
        ref_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        ref_feet_standing = (ref_vel.norm(dim=-1) < 0.2) & (ref_pos[..., 2] < 0.15)

        # height-based scaling using current robot foot height
        feet_height = self.asset.data.body_link_pos_w[:, self.body_indices_asset, 2]
        t = (feet_height - self.h_low) / (self.h_high - self.h_low)
        t = torch.clamp(t, 0.0, 1.0)
        feet_height_coef = self.c_low * torch.exp(self.exp_log_c_ratio * t)

        contact_diff = ref_feet_standing ^ current_contact
        self.reward_time = self.reward_time + torch.where(
            contact_diff, -self.env.step_dt, self.env.step_dt * feet_height_coef
        )

        reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * first_contact,
            dim=1,
            keepdim=True,
        )

        # reset timer for feet that are on ground
        self.reward_time = self.reward_time * (~current_contact)
        return reward


# --------------------------------------------------------------------------- #
# Keypoint position tracking with root alignment & look-ahead buffer
# --------------------------------------------------------------------------- #
class keypoint_pos_tracking_aligned(TrackReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        body_names: List[str] | str,
        sigma: float = 0.3,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.sigma = sigma

        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        matched_names = sorted(set(matched_names_motion) & set(matched_names_asset))
        assert matched_names, "keypoint_pos_tracking_aligned: no body matched"

        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(n) for n in matched_names
        ]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(n) for n in matched_names
        ]
        self.body_names = matched_names
        self.num_bodies = len(self.body_indices_asset)

    # --- reward --- #
    def _compute(self):
        target_anchor_pos = self.command_manager.target_anchor_pos_buf[:, :, 0].mean(
            dim=1
        )
        ref_body_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        ref_anchor_pos = self.command_manager.ref_anchor_pos_w
        aligned_body_pos_w = (
            ref_body_pos - ref_anchor_pos.unsqueeze(1) + target_anchor_pos.unsqueeze(1)
        )
        robot_body_pos_w = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        error = (aligned_body_pos_w - robot_body_pos_w).norm(dim=-1)
        return torch.exp(-error.mean(dim=1, keepdim=True) / self.sigma)
