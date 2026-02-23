from hdmi.hdmi_tasks.command import RobotTracking
from active_adaptation.envs.mdp.base import Observation as BaseObservation

import torch
from active_adaptation.utils.math import (
    quat_rotate_inverse as quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    matrix_from_quat,
    yaw_quat,
    batchify,
)

quat_apply_inverse = batchify(quat_apply_inverse)

RobotTrackObservation = BaseObservation[RobotTracking]


def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3.0, 3.0) * std


class root_ang_vel_history(RobotTrackObservation):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()

    def reset(self, env_ids):
        value = self.asset.data.root_com_ang_vel_b[env_ids]
        value = value.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.buffer[env_ids] = value

    def update(self):
        value = self.asset.data.root_com_ang_vel_b
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = value

    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)


class projected_gravity_history(RobotTrackObservation):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()

    def reset(self, env_ids):
        value = self.asset.data.projected_gravity_b[env_ids]
        value = value.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.buffer[env_ids] = value

    def update(self):
        value = self.asset.data.projected_gravity_b
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = value

    def compute(self):
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)


class joint_pos_history(RobotTrackObservation):
    def __init__(
        self,
        env,
        joint_names: str = ".*",
        history_steps: list[int] = [0],
        noise_std: float = 0.0,
    ):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(noise_std, 0.0)
        self.asset = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        self.num_joints = len(self.joint_ids)
        self.buffer = torch.zeros(
            (self.num_envs, self.buffer_size, self.num_joints), device=self.device
        )

    def reset(self, env_ids):
        value = self.asset.data.joint_pos[env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)]
        self.buffer[env_ids] = value.unsqueeze(1)

    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        value = self.asset.data.joint_pos[:, self.joint_ids]
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.buffer[:, 0] = value

    def compute(self):
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)


class ref_joint_pos_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_pos_future_.view(self.num_envs, -1)


class ref_joint_vel_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_vel_future_.view(self.num_envs, -1)


class ref_joint_pos_action(RobotTrackObservation):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_joint_names
        ]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[
            :, self.action_indices_motion
        ]
        return ref_joint_pos


class ref_joint_pos_action_policy(RobotTrackObservation):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_joint_names
        ]

        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[
            :, action_manager.joint_ids
        ]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[
            :, self.action_indices_motion
        ]
        ref_joint_action = (
            ref_joint_pos - self.default_joint_pos
        ) / self.action_scaling
        return ref_joint_action


class ref_root_pos_future_b(RobotTrackObservation):
    """
    Reference root position in robot root frame
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_pos_future_b = torch.zeros(
            self.num_envs, num_future_steps, 3, device=self.device
        )

    def update(self):
        ref_root_pos_future_w = (
            self.command_manager.ref_root_pos_future_w
        )  # shape: [num_envs, num_future_steps, 3]
        robot_root_link_pos_w = self.command_manager.robot_root_link_pos_w[
            :, None, :
        ]  # shape: [num_envs, 1, 3]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_pos_future_b = quat_apply_inverse(
            robot_root_link_quat_w, ref_root_pos_future_w - robot_root_link_pos_w
        )
        self.ref_root_pos_future_b = ref_root_pos_future_b

    def compute(self):
        return self.ref_root_pos_future_b.view(self.num_envs, -1)


class ref_root_ori_future_b(RobotTrackObservation):
    """
    Reference root orientation in robot root frame
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_ori_future_b = torch.zeros(
            self.num_envs, num_future_steps, 2, 3, device=self.device
        )

    def update(self):
        ref_root_quat_future_w = (
            self.command_manager.ref_root_quat_future_w
        )  # shape: [num_envs, num_future_steps, 4]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_link_quat_w).expand_as(ref_root_quat_future_w),
            ref_root_quat_future_w,
        )
        ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
        self.ref_root_ori_future_b = ref_root_ori_future_b[:, :, :2, :]

    def compute(self):
        return self.ref_root_ori_future_b.reshape(self.num_envs, -1)


class ref_body_pos_future_local(RobotTrackObservation):
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

    def update(self):
        ref_body_pos_future_w = (
            self.command_manager.ref_body_pos_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_anchor_link_pos_w = self.command_manager.ref_anchor_link_pos_w
        ref_anchor_link_pos_w = ref_anchor_link_pos_w.unsqueeze(1).unsqueeze(2)
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w
        ref_anchor_link_quat_w = ref_anchor_link_quat_w.unsqueeze(1).unsqueeze(2)

        ref_anchor_link_pos_w = ref_anchor_link_pos_w.clone()
        ref_anchor_link_pos_w[..., 2] = 0.0
        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(
            ref_anchor_link_quat_w, ref_body_pos_future_w - ref_anchor_link_pos_w
        )
        self.ref_body_pos_future_local = ref_body_pos_future_local

    def compute(self):
        return self.ref_body_pos_future_local.view(self.num_envs, -1)


class ref_body_ori_future_local(RobotTrackObservation):
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

    def update(self):
        ref_body_quat_future_w = self.command_manager.ref_body_quat_future_w
        # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w
        ref_anchor_link_quat_w = ref_anchor_link_quat_w.unsqueeze(1).unsqueeze(2)
        # shape: [num_envs, 1, 1, 4]

        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_anchor_link_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w,
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)

    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)


class diff_body_pos_future_local(RobotTrackObservation):
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

    def update(self):
        ref_body_pos_future_w = (
            self.command_manager.ref_body_pos_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_anchor_link_pos_w = self.command_manager.ref_anchor_link_pos_w[
            :, None, None, :
        ].clone()  # shape: [num_envs, 1, 1, 3]
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]

        robot_body_link_pos_w = (
            self.command_manager.robot_body_link_pos_w
        )  # shape: [num_envs, num_tracking_bodies, 3]
        robot_anchor_link_pos_w = self.command_manager.robot_anchor_link_pos_w[
            :, None, :
        ].clone()  # shape: [num_envs, 1, 3]
        robot_anchor_link_quat_w = self.command_manager.robot_anchor_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_anchor_link_pos_w[..., 2] = 0.0
        robot_anchor_link_pos_w[..., 2] = 0.0
        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w)
        robot_anchor_link_quat_w = yaw_quat(robot_anchor_link_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(
            ref_anchor_link_quat_w, ref_body_pos_future_w - ref_anchor_link_pos_w
        )
        robot_body_pos_local = quat_apply_inverse(
            robot_anchor_link_quat_w, robot_body_link_pos_w - robot_anchor_link_pos_w
        )

        self.diff_body_pos_future_local = (
            ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1)
        )

    def compute(self):
        return self.diff_body_pos_future_local.view(self.num_envs, -1)


class diff_body_lin_vel_future_local(RobotTrackObservation):
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

    def update(self):
        ref_body_lin_vel_future_w = (
            self.command_manager.ref_body_lin_vel_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]
        robot_body_link_lin_vel_w = (
            self.command_manager.robot_body_com_lin_vel_w
        )  # shape: [num_envs, num_tracking_bodies, 3]
        robot_anchor_link_quat_w = self.command_manager.robot_anchor_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w)
        robot_anchor_link_quat_w = yaw_quat(robot_anchor_link_quat_w)

        ref_body_lin_vel_future_local = quat_apply_inverse(
            ref_anchor_link_quat_w, ref_body_lin_vel_future_w
        )
        robot_body_lin_vel_local = quat_apply_inverse(
            robot_anchor_link_quat_w, robot_body_link_lin_vel_w
        )

        self.diff_body_lin_vel_future_local = (
            ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1)
        )

    def compute(self):
        return self.diff_body_lin_vel_future_local.view(self.num_envs, -1)


class diff_body_ori_future_local(RobotTrackObservation):
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

    def update(self):
        ref_body_quat_future_w = (
            self.command_manager.ref_body_quat_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]
        robot_body_link_quat_w = (
            self.command_manager.robot_body_link_quat_w
        )  # shape: [num_envs, num_tracking_bodies, 4]
        robot_anchor_link_quat_w = self.command_manager.robot_anchor_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w)
        robot_anchor_link_quat_w = yaw_quat(robot_anchor_link_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_anchor_link_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w,
        )
        robot_body_quat_local = quat_mul(
            quat_conjugate(robot_anchor_link_quat_w).expand_as(robot_body_link_quat_w),
            robot_body_link_quat_w,
        ).unsqueeze(1)
        diff_body_quat_future = quat_mul(
            quat_conjugate(robot_body_quat_local).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_local,
        )
        self.diff_body_ori_future_local = matrix_from_quat(diff_body_quat_future)

    def compute(self):
        return self.diff_body_ori_future_local[:, :, :, :2, :].reshape(
            self.num_envs, -1
        )


class diff_body_ang_vel_future_local(RobotTrackObservation):
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

    def update(self):
        ref_body_ang_vel_future_w = (
            self.command_manager.ref_body_ang_vel_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_anchor_link_quat_w = self.command_manager.ref_anchor_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]
        robot_body_link_ang_vel_w = (
            self.command_manager.robot_body_com_ang_vel_w
        )  # shape: [num_envs, num_tracking_bodies, 3]
        robot_anchor_link_quat_w = self.command_manager.robot_anchor_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_anchor_link_quat_w = yaw_quat(ref_anchor_link_quat_w)
        robot_anchor_link_quat_w = yaw_quat(robot_anchor_link_quat_w)

        ref_body_ang_vel_future_local = quat_apply_inverse(
            ref_anchor_link_quat_w, ref_body_ang_vel_future_w
        )
        robot_body_ang_vel_local = quat_apply_inverse(
            robot_anchor_link_quat_w, robot_body_link_ang_vel_w
        )

        self.diff_body_ang_vel_future_local = (
            ref_body_ang_vel_future_local - robot_body_ang_vel_local.unsqueeze(1)
        )

    def compute(self):
        return self.diff_body_ang_vel_future_local.view(self.num_envs, -1)


class ref_motion_phase(RobotTrackObservation):
    def compute(self):
        return (self.command_manager.t / self.command_manager.motion_len).unsqueeze(1)
