from hdmi.hdmi_tasks.command import RobotTracking
from active_adaptation.envs.mdp.base import Termination as BaseTermination

import torch
from typing import List
try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names
from active_adaptation.utils.math import (
    quat_rotate_inverse as quat_apply_inverse,
    yaw_quat,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    batchify,
)

quat_apply_inverse = batchify(quat_apply_inverse)


class _cum_error_mixin:
    def __init__(self, env, min_steps: int = 1, threshold: float = 0.25, **kwargs):
        super().__init__(env, **kwargs)
        self.min_steps = min_steps
        self.threshold = threshold

        with torch.device(self.device):
            self.error = torch.zeros(self.num_envs)
            self.__exceeded = torch.zeros(self.num_envs, dtype=torch.bool)
            self.__cum_steps = torch.zeros(self.num_envs, dtype=torch.int32)

    def update(self):
        self.__exceeded = self.error >= self.threshold
        self.__cum_steps[self.__exceeded] += 1
        self.__cum_steps[~self.__exceeded] = 0

    def reset(self, env_ids):
        self.__cum_steps[env_ids] = 0

    def compute(self, termination: torch.Tensor):
        return (self.__cum_steps >= self.min_steps).unsqueeze(-1)


RobotTrackTermination = BaseTermination[RobotTracking]


class motion_timeout(RobotTrackTermination):
    """
    Terminates when the motion clip is consumed (or always true in replay mode).
    """

    def __init__(self, env, is_timeout: bool = True, **kwargs):
        super().__init__(env, is_timeout=is_timeout, **kwargs)

    def compute(self, termination: torch.Tensor):
        if self.command_manager.replay_motion:
            return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
        return (self.command_manager.t >= self.command_manager.motion_len).unsqueeze(1)


class cum_body_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(name)
            for name in self.body_names
        ]
        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        ref_body_link_pos_w = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        robot_body_link_pos_w = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_link_pos_w - robot_body_link_pos_w).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_z_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(name)
            for name in self.body_names
        ]
        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        ref_body_link_pos_w = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        robot_body_link_pos_w = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_link_pos_w - robot_body_link_pos_w)[..., 2].abs()
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_ori_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(name)
            for name in self.body_names
        ]
        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        ref_body_link_quat_w = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]
        robot_body_link_quat_w = self.command_manager.asset.data.body_link_quat_w[
            :, self.body_indices_asset
        ]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_quat_diff = quat_mul(
            quat_conjugate(ref_body_link_quat_w), robot_body_link_quat_w
        )
        body_ori_error = axis_angle_from_quat(body_quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()


class cum_body_pos_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(name)
            for name in self.body_names
        ]
        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        ref_body_link_pos_w = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        ref_anchor_link_pos_w = self.command_manager.ref_anchor_link_pos_w[
            :, None, :
        ].clone()
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w[:, None, :]

        robot_body_link_pos_w = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        robot_anchor_link_pos_w = self.command_manager.robot_anchor_link_pos_w[
            :, None, :
        ].clone()
        robot_anchor_link_quat_w = self.command_manager.robot_anchor_link_quat_w[
            :, None, :
        ]

        ref_anchor_link_pos_w[..., 2] = 0.0
        robot_anchor_link_pos_w[..., 2] = 0.0
        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w)
        robot_anchor_link_quat_w = yaw_quat(robot_anchor_link_quat_w)

        ref_body_pos_local = quat_apply_inverse(
            ref_anchor_link_quat_w, ref_body_link_pos_w - ref_anchor_link_pos_w
        )
        robot_body_pos_local = quat_apply_inverse(
            robot_anchor_link_quat_w, robot_body_link_pos_w - robot_anchor_link_pos_w
        )

        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_pos_local - robot_body_pos_local).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_ori_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(name)
            for name in self.body_names
        ]
        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        ref_body_link_quat_w = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w[:, None, :]

        robot_body_link_quat_w = self.command_manager.asset.data.body_link_quat_w[
            :, self.body_indices_asset
        ]
        robot_anchor_link_quat_w = self.command_manager.robot_anchor_link_quat_w[
            :, None, :
        ]

        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w).expand_as(
            ref_body_link_quat_w
        )
        robot_anchor_link_quat_w = yaw_quat(robot_anchor_link_quat_w).expand_as(
            robot_body_link_quat_w
        )

        ref_body_quat_local = quat_mul(
            quat_conjugate(ref_anchor_link_quat_w), ref_body_link_quat_w
        )
        robot_body_quat_local = quat_mul(
            quat_conjugate(robot_anchor_link_quat_w), robot_body_link_quat_w
        )

        # shape: [num_envs, num_tracking_bodies, 3]
        body_quat_diff = quat_mul(
            quat_conjugate(ref_body_quat_local), robot_body_quat_local
        )
        body_ori_error = axis_angle_from_quat(body_quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()


class cum_joint_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, joint_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.joint_names = resolve_matching_names(
            joint_names, self.command_manager.tracking_joint_names
        )[1]
        self.joint_indices_asset = [
            self.command_manager.asset.joint_names.index(name)
            for name in self.joint_names
        ]
        self.joint_indices_motion = [
            self.command_manager.tracking_joint_names.index(name)
            for name in self.joint_names
        ]

    def update(self):
        ref_joint_pos = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        robot_joint_pos = self.command_manager.asset.data.joint_pos[
            :, self.joint_indices_asset
        ]

        joint_pos_error = (ref_joint_pos - robot_joint_pos).abs()
        self.error[:] = joint_pos_error.max(dim=1).values
        super().update()
