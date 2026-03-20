from hdmi.tasks.command import RobotTracking
from active_adaptation.envs.mdp.terminations.base import Termination as BaseTermination

import torch
from typing import List
try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names


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
        # if self.command_manager.replay_motion:
        #     return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
        return (self.command_manager.t >= self.command_manager.motion_len).unsqueeze(1)


class cum_body_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_pos_error = self.command_manager.body_pos_error[:, self.body_indices_tracking]
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_z_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_pos_error = (
            self.command_manager.ref_body_pos_w[:, self.body_indices_tracking, 2]
            - self.command_manager.robot_body_link_pos_w[:, self.body_indices_tracking, 2]
        ).abs()
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_ori_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_ori_error = self.command_manager.body_ori_error[:, self.body_indices_tracking]
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()


class cum_body_lin_vel_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_lin_vel_error = self.command_manager.body_lin_vel_error[
            :, self.body_indices_tracking
        ]
        self.error[:] = body_lin_vel_error.max(dim=1).values
        super().update()


class cum_body_ang_vel_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_ang_vel_error = self.command_manager.body_ang_vel_error[
            :, self.body_indices_tracking
        ]
        self.error[:] = body_ang_vel_error.max(dim=1).values
        super().update()


class cum_body_pos_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_pos_error = self.command_manager.body_pos_error_local[:, self.body_indices_tracking]
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()


class cum_body_ori_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )[1]
        self.body_indices_tracking = [
            self.command_manager.tracking_body_names.index(name)
            for name in self.body_names
        ]

    def update(self):
        body_ori_error = self.command_manager.body_ori_error_local[:, self.body_indices_tracking]
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()


class cum_body_pos_error_aligned(_cum_error_mixin, RobotTrackTermination):
    def __init__(
        self,
        env,
        body_names: str | List[str] = ".*",
        look_ahead: int = 50,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.look_ahead = int(look_ahead)

        _, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        _, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        matched_names = sorted(set(matched_names_motion) & set(matched_names_asset))
        assert matched_names, "cum_body_pos_error_aligned: no body matched"

        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(name)
            for name in matched_names
        ]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(name)
            for name in matched_names
        ]

        with torch.device(self.device):
            self.look_ahead_idx = torch.tensor(
                [self.look_ahead - 1], dtype=torch.long
            )
            self.look_ahead_indices = torch.arange(self.look_ahead)
            self.target_anchor_pos_buf = torch.zeros(self.num_envs, self.look_ahead, 3)

    def reset(self, env_ids):
        super().reset(env_ids)
        future_ref_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids[env_ids],
            self.command_manager.t[env_ids],
            steps=self.look_ahead_indices,
        )
        ref_pos = future_ref_motion.body_pos_w[
            :, :, self.command_manager.anchor_body_idx_motion
        ] + self.command_manager.env.scene.env_origins[env_ids].unsqueeze(1)
        self.target_anchor_pos_buf[env_ids] = ref_pos

    def update(self):
        target_anchor_pos = self.target_anchor_pos_buf[:, 0]
        ref_body_pos = self.command_manager.ref_body_pos_w[:, self.body_indices_motion]
        ref_anchor_pos = self.command_manager.ref_anchor_pos_w

        aligned_body_pos_w = (
            ref_body_pos - ref_anchor_pos.unsqueeze(1) + target_anchor_pos.unsqueeze(1)
        )
        robot_body_pos_w = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        body_pos_error = (aligned_body_pos_w - robot_body_pos_w).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values

        ref_pos_current = self.command_manager.ref_anchor_pos_w.clone()
        robot_pos_current = self.command_manager.robot_anchor_pos_w.clone()
        ref_pos_current[:, 2] = 0.0
        robot_pos_current[:, 2] = 0.0

        look_ahead_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids,
            self.command_manager.t,
            steps=self.look_ahead_idx,
        ).squeeze(1)
        ref_pos_look_ahead = (
            look_ahead_motion.body_pos_w[:, self.command_manager.anchor_body_idx_motion]
            + self.command_manager.env.scene.env_origins
        )
        aligned_anchor_pos_w = ref_pos_look_ahead - ref_pos_current + robot_pos_current

        self.target_anchor_pos_buf[:] = self.target_anchor_pos_buf.roll(-1, dims=1)
        self.target_anchor_pos_buf[:, -1] = aligned_anchor_pos_w
        super().update()


class cum_joint_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, env, joint_names: str | List[str] = ".*", **kwargs):
        super().__init__(env, **kwargs)
        self.joint_names = resolve_matching_names(
            joint_names, self.command_manager.tracking_joint_names
        )[1]
        self.joint_indices_tracking = [
            self.command_manager.tracking_joint_names.index(name)
            for name in self.joint_names
        ]

    def update(self):
        joint_pos_error = self.command_manager.joint_pos_error[:, self.joint_indices_tracking]
        self.error[:] = joint_pos_error.max(dim=1).values
        super().update()
