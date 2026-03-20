from hdmi.tasks.command import RobotTracking
from hdmi.tasks.actions import JointPosition

from active_adaptation.envs.mdp.observations.base import Observation as BaseObservation

import torch
from typing import cast

TrackObservation = BaseObservation[RobotTracking]


class ref_joint_pos_future(TrackObservation, namespace="hdmi"):
    def compute(self):
        return self.command_manager.ref_joint_pos_future_.view(self.num_envs, -1)


class ref_joint_vel_future(TrackObservation, namespace="hdmi"):
    def compute(self):
        return self.command_manager.ref_joint_vel_future_.view(self.num_envs, -1)


class ref_joint_action(TrackObservation, namespace="hdmi"):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        action_manager = cast(JointPosition, self.env.action_manager)
        self.action_joint_ids = action_manager.joint_ids
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_manager.joint_names
        ]

        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[
            :, self.action_joint_ids
        ]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[
            :, self.action_indices_motion
        ]
        ref_joint_action = (
            ref_joint_pos - self.default_joint_pos
        ) / self.action_scaling
        return ref_joint_action

# root_diff_obs

class ref_root_pos_future_b(TrackObservation, namespace="hdmi"):
    """
    Reference root position in robot root frame
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_pos_future_b = torch.zeros(
            self.num_envs, num_future_steps, 3, device=self.device
        )


    def compute(self):
        return self.command_manager.ref_root_pos_future_b.view(self.num_envs, -1)


class ref_root_ori_future_b(TrackObservation, namespace="hdmi"):
    """
    Reference root orientation in robot root frame
    """

    def __init__(self, env, noise_std=0.0, **kwargs):
        super().__init__(env, **kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_ori_future_b = torch.zeros(
            self.num_envs, num_future_steps, 2, 3, device=self.device
        )
        self.noise_std = noise_std


    def compute(self):
        ref_root_ori_future_b = self.command_manager.ref_root_ori_future_b_matrix
        if self.noise_std > 0.0:
            ref_root_ori_future_b = ref_root_ori_future_b.clone()
            ref_root_ori_future_b += (
                torch.randn_like(ref_root_ori_future_b).clamp(-3.0, 3.0) * self.noise_std
            )
        return ref_root_ori_future_b[:, :, :2, :].reshape(self.num_envs, -1)


# motion_local_obs

class ref_body_pos_future_local(TrackObservation, namespace="hdmi"):
    """
    Reference body position in motion anchor frame
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.ref_body_pos_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )

    def compute(self):
        return self.command_manager.ref_body_pos_future_local.view(self.num_envs, -1)


class ref_body_ori_future_local(TrackObservation, namespace="hdmi"):
    """
    Reference body orientation in motion anchor frame
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.ref_body_ori_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            3,
            device=self.device,
        )


    def compute(self):
        return self.command_manager.ref_body_ori_future_local_matrix[:, :, :, :2, :].reshape(
            self.num_envs, -1
        )

# body_local_diff_obs

class diff_body_pos_future_local(TrackObservation, namespace="hdmi"):
    """
    Reference body position in each motion anchor frame - Robot body position in robot anchor frame.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.diff_body_pos_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )


    def compute(self):
        return self.command_manager.diff_body_pos_future_local.view(self.num_envs, -1)


class diff_body_lin_vel_future_local(TrackObservation, namespace="hdmi"):
    """
    Reference body linear velocity in motion anchor frame - Robot body linear velocity in robot anchor frame.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.diff_body_lin_vel_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )


    def compute(self):
        return self.command_manager.diff_body_lin_vel_future_local.view(self.num_envs, -1)


class diff_body_ori_future_local(TrackObservation, namespace="hdmi"):
    """
    Reference body orientation in motion anchor frame - Robot body orientation in robot anchor frame.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.diff_body_ori_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            3,
            device=self.device,
        )


    def compute(self):
        return self.command_manager.diff_body_ori_future_local_matrix[:, :, :, :2, :].reshape(
            self.num_envs, -1
        )


class diff_body_ang_vel_future_local(TrackObservation, namespace="hdmi"):
    """
    Reference body linear velocity in motion anchor frame - Robot body linear velocity in robot anchor frame.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.diff_body_ang_vel_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )


    def compute(self):
        return self.command_manager.diff_body_ang_vel_future_local.view(self.num_envs, -1)


class ref_motion_phase(TrackObservation, namespace="hdmi"):
    def compute(self):
        return (self.command_manager.obs_motion_t / self.command_manager.motion_len).unsqueeze(1)
