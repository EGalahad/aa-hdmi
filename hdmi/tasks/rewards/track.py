from hdmi.tasks.command import RobotTracking

from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward

from typing import List, TYPE_CHECKING

try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names

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
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_body_names

        self.sigma = sigma
        body_indices_tracking, matched_body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        assert matched_body_names, "No body names matched in tracking_body_names"
        self.body_indices_tracking = list(body_indices_tracking)
        self.body_names = list(matched_body_names)
        self.num_bodies = len(self.body_names)

    def _compute(self):
        raise NotImplementedError


class keypoint_pos_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_pos_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_pos_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_pos_error(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_pos_error[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_pos_error_local(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_ori_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ori_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ori_error(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_ori_error[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_error_local(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_lin_vel_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_lin_vel_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ang_vel_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.body_ang_vel_error[:, self.body_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class _tracking_joint(TrackReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        joint_names: List[str] | str | None = None,
        sigma: float = 0.03,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names

        self.sigma = sigma
        joint_indices_tracking, matched_joint_names = resolve_matching_names(
            joint_names, self.command_manager.tracking_joint_names
        )
        assert matched_joint_names, "No joint names matched in tracking_joint_names"
        self.joint_indices_tracking = list(joint_indices_tracking)
        self.joint_names = list(matched_joint_names)

    def _compute(self):
        raise NotImplementedError

class joint_pos_tracking_product(_tracking_joint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class joint_pos_error(_tracking_joint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
        return error.mean(dim=1).unsqueeze(1)


class joint_vel_tracking_product(_tracking_joint, namespace="hdmi"):
    def _compute(self):
        error = self.command_manager.joint_vel_error[:, self.joint_indices_tracking]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class feet_air_time_ref(TrackReward, namespace="hdmi"):
    def __init__(self, env, body_names: List[str] | str, thres: float, **kwargs):
        super().__init__(env, **kwargs)
        self.thres = thres
        self.contact_sensor: "ContactSensor" = self.env.scene["feet_ground_contact"]

        body_indices_tracking, matched_body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        assert matched_body_names, "feet_air_time_ref: no feet matched"
        self.body_indices_tracking = list(body_indices_tracking)
        sensor_ids, _ = self.contact_sensor.find_bodies(matched_body_names)
        self.sensor_body_ids = torch.tensor(sensor_ids, device=self.device)

        num_bodies = len(matched_body_names)
        self.reward_time = torch.zeros(self.num_envs, num_bodies, device=self.device)
        self.last_contact = torch.zeros(
            self.num_envs, num_bodies, dtype=bool, device=self.device
        )

        self.h_low, self.h_high = 0.035, 0.12
        self.c_low, self.c_high = 0.5, 2.0
        self.exp_log_c_ratio = torch.log(
            torch.tensor(self.c_high / self.c_low, device=self.device)
        )

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def _compute(self):
        current_contact = (
            self.contact_sensor.data.current_contact_time[:, self.sensor_body_ids] > 0.0
        )
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        ref_vel = self.command_manager.ref_body_lin_vel_w[:, self.body_indices_tracking]
        ref_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_tracking]
        ref_feet_standing = (ref_vel.norm(dim=-1) < 0.2) & (ref_pos[..., 2] < 0.15)

        feet_height = self.command_manager.robot_body_link_pos_w[
            :, self.body_indices_tracking, 2
        ]
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

        body_indices_tracking, matched_body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        assert matched_body_names, "keypoint_pos_tracking_aligned: no body matched"
        self.body_indices_tracking = list(body_indices_tracking)
        self.body_names = list(matched_body_names)
        self.num_bodies = len(self.body_indices_tracking)

    # --- reward --- #
    def _compute(self):
        target_anchor_pos = self.command_manager.target_anchor_pos_buf[:, :, 0].mean(
            dim=1
        )
        ref_body_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_tracking]
        ref_anchor_pos = self.command_manager.ref_anchor_pos_w
        aligned_body_pos_w = (
            ref_body_pos - ref_anchor_pos.unsqueeze(1) + target_anchor_pos.unsqueeze(1)
        )
        robot_body_pos_w = self.command_manager.robot_body_link_pos_w[
            :, self.body_indices_tracking
        ]
        error = (aligned_body_pos_w - robot_body_pos_w).norm(dim=-1)
        return torch.exp(-error.mean(dim=1, keepdim=True) / self.sigma)
