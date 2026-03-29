from hdmi.tasks.command import RobotTracking
from hdmi.tasks.actions import JointPosition

from active_adaptation.envs.mdp.observations.base import Observation as BaseObservation

import torch
from typing import cast, List

try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names

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

    def compute(self):
        return self.command_manager.ref_root_pos_future_b.view(self.num_envs, -1)


class ref_root_ori_future_b(TrackObservation, namespace="hdmi"):
    """
    Reference root orientation in robot root frame
    """

    def __init__(self, env, noise_std=0.0, **kwargs):
        super().__init__(env, **kwargs)
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

class _tracking_body_future_observation(TrackObservation):
    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        future_steps: List[int] | int | None = None,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_body_names
        if future_steps is None:
            future_steps = self.command_manager.future_steps.tolist()
        elif isinstance(future_steps, int):
            future_steps = [future_steps]

        body_indices_tracking, matched_body_names = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        if not matched_body_names:
            raise ValueError("No tracking body matched for observation.")

        available_future_steps = [int(step) for step in self.command_manager.future_steps.tolist()]
        future_step_indices = []
        for step in future_steps:
            step = int(step)
            if step not in available_future_steps:
                raise ValueError(
                    f"future step {step} not in command.future_steps={available_future_steps}"
                )
            future_step_indices.append(available_future_steps.index(step))

        self.body_indices_tracking = torch.as_tensor(
            body_indices_tracking, dtype=torch.long, device=self.device
        )
        self.future_step_indices = torch.as_tensor(
            future_step_indices, dtype=torch.long, device=self.device
        )

    def _select_body_future(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.index_select(x, 1, self.future_step_indices)
        return torch.index_select(x, 2, self.body_indices_tracking)


class ref_body_pos_future_local(
    _tracking_body_future_observation, namespace="hdmi"
):
    """
    Reference body position in motion anchor frame
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.ref_body_pos_future_local
        ).reshape(self.num_envs, -1)


class ref_body_ori_future_local(
    _tracking_body_future_observation, namespace="hdmi"
):
    """
    Reference body orientation in motion anchor frame
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.ref_body_ori_future_local_matrix
        )[:, :, :, :2, :].reshape(self.num_envs, -1)

# body_local_diff_obs

class diff_body_pos_future_local(
    _tracking_body_future_observation, namespace="hdmi"
):
    """
    Reference body position in each motion anchor frame - Robot body position in robot anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_pos_future_local
        ).reshape(self.num_envs, -1)


class diff_body_lin_vel_future(
    _tracking_body_future_observation, namespace="hdmi"
):
    """
    Reference body linear velocity in motion anchor frame - Robot body linear velocity in robot anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_lin_vel_future
        ).reshape(self.num_envs, -1)


class diff_body_ori_future_local(
    _tracking_body_future_observation, namespace="hdmi"
):
    """
    Reference body orientation in motion anchor frame - Robot body orientation in robot anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_ori_future_local_matrix
        )[:, :, :, :2, :].reshape(self.num_envs, -1)


class diff_body_ang_vel_future(
    _tracking_body_future_observation, namespace="hdmi"
):
    """
    Reference body linear velocity in motion anchor frame - Robot body linear velocity in robot anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_ang_vel_future
        ).reshape(self.num_envs, -1)


class ref_motion_phase(TrackObservation, namespace="hdmi"):
    def compute(self):
        return (self.command_manager.obs_motion_t / self.command_manager.motion_len).unsqueeze(1)
