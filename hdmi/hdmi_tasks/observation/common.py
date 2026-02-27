"""Common observation aliases for HDMI tasks."""

import torch
import active_adaptation as aa
from active_adaptation.envs.mdp.base import Observation as BaseObservation
from typing import cast
from active_adaptation.assets import (
    get_output_joint_indexing,
    get_output_body_indexing,
)
from hdmi.hdmi_tasks.actions import HDMIJointPosition
from active_adaptation.utils.math import quat_rotate_inverse, yaw_quat

if aa.get_backend() == "isaac":
    from isaaclab.assets import ArticulationData
elif aa.get_backend() == "mjlab":
    from mjlab.entity import EntityData


def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3.0, 3.0) * std


class root_ang_vel_history(BaseObservation):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
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


class projected_gravity_history(BaseObservation):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
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


class joint_pos_history(BaseObservation):
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
        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        self.output_indexing, _ = get_output_joint_indexing(
            "simulation",
            self.asset.cfg,
            self.joint_names,
            self.device,
        )
        self.num_joints = len(self.joint_ids)
        self.buffer = torch.zeros(
            (self.num_envs, self.buffer_size, self.num_joints), device=self.device
        )
        self.action_manager = cast(HDMIJointPosition, self.env.input_managers["action"])

    def reset(self, env_ids):
        value = self.asset.data.joint_pos[
            env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)
        ]
        self.buffer[env_ids] = value.unsqueeze(1)

    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        value = self.asset.data.joint_pos[:, self.joint_ids]
        if self.noise_std > 0:
            value = random_noise(value, self.noise_std)
        self.buffer[:, 0] = value

    def compute(self):
        # joint_pos = self.buffer - self.asset.data.encoder_bias[
        #     :, self.joint_ids
        # ].unsqueeze(1)
        joint_pos = self.buffer - self.action_manager.offset[
            :, self.joint_ids
        ].unsqueeze(1)
        joint_pos_selected = joint_pos[:, self.history_steps][:, :, self.output_indexing]
        return joint_pos_selected.reshape(self.num_envs, -1)


# class applied_action(BaseObservation):
#     def __init__(self, env, **kwargs):
#         super().__init__(env, **kwargs)
#         self.action_manager = cast(HDMIJointPosition, self.env.input_managers["action"])

#     def compute(self):
#         applied_action = self.action_manager.applied_action
#         return applied_action[:, self.action_manager._target_to_input_indexing]


class prev_actions(BaseObservation):
    def __init__(self, env, key: str = "action", steps: int = 1):
        super().__init__(env)
        self.steps = steps
        self.action_manager = self.env.input_managers[key]

    def compute(self):
        action_buf = self.action_manager.action_buf[:, : self.steps]
        return action_buf.reshape(self.num_envs, -1)


class body_pos_b(BaseObservation):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.output_indexing, _ = get_output_body_indexing(
            "simulation",
            self.asset.cfg,
            self.body_names,
            self.device,
        )
        self.update()

    def update(self):
        self.root_link_pos_w = self.asset.data.root_link_pos_w.unsqueeze(1).clone()
        self.root_link_quat_w = yaw_quat(self.asset.data.root_link_quat_w).unsqueeze(1)

        self.root_link_pos_w[..., 2] = 0.0

        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_indices]

    def compute(self):
        body_pos_b = quat_rotate_inverse(
            self.root_link_quat_w, self.body_link_pos_w - self.root_link_pos_w
        )
        body_pos_b = body_pos_b[:, self.output_indexing]
        return body_pos_b.reshape(self.num_envs, -1)


class body_vel_b(BaseObservation):
    def __init__(self, env, body_names: str, yaw_only: bool = False):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.output_indexing, _ = get_output_body_indexing(
            "simulation",
            self.asset.cfg,
            self.body_names,
            self.device,
        )
        self.update()

    def update(self):
        self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
        self.body_com_vel_w = self.asset.data.body_com_vel_w[:, self.body_indices]

    def compute(self):
        body_lin_vel_b = quat_rotate_inverse(
            self.root_link_quat_w, self.body_com_vel_w[:, :, :3]
        )
        body_lin_vel_b = body_lin_vel_b[:, self.output_indexing]
        return body_lin_vel_b.reshape(self.num_envs, -1)


class applied_torque(BaseObservation):
    def __init__(self, env, joint_names: str = ".*"):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.output_indexing, _ = get_output_joint_indexing(
            "simulation",
            self.asset.cfg,
            self.joint_names,
            self.device,
        )

    def compute(self):
        if aa.get_backend() == "isaac":
            asset_data = cast(ArticulationData, self.asset.data)
            applied_efforts = asset_data.applied_torque
        else:
            asset_data = cast(EntityData, self.asset.data)
            applied_efforts = asset_data.actuator_force
        # print("applied_efforts:", applied_efforts)
        efforts = applied_efforts[:, self.joint_ids]
        return efforts[:, self.output_indexing]
