from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.assets.asset_cfg import (
    to_simulation_body_order,
    to_simulation_joint_order,
)
from hdmi.tasks.motion import MotionDataset, MotionData

from dataclasses import dataclass
from typing import List, Dict, Tuple, TYPE_CHECKING, Literal
import copy

if TYPE_CHECKING:
    from mjlab.viewer.viser import ViserMujocoScene

import torch
import numpy as np

from active_adaptation.utils.math import (
    sample_uniform as _sample_uniform,
    quat_from_euler_xyz as _quat_from_euler_xyz,
    quat_rotate_inverse as quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    quat_angle_magnitude,
    matrix_from_quat,
    quat_rotate,
    yaw_quat,
    batchify,
)
from tensordict import TensorDict


quat_apply_inverse = batchify(quat_apply_inverse)


_DESIRED_FRAME_COLORS = (
    (0.9, 0.3, 0.3, 0.9),
    (0.3, 0.9, 0.3, 0.9),
    (0.3, 0.3, 0.9, 0.9),
)


def sample_uniform(low, high, size, device):
    return _sample_uniform(size=size, low=low, high=high, device=device)


def quat_from_euler_xyz(roll, pitch, yaw):
    return _quat_from_euler_xyz(torch.stack([roll, pitch, yaw], dim=-1))


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_root_diff_obs(
    robot_root_pos_w: torch.Tensor,
    robot_root_quat_w: torch.Tensor,
    ref_root_pos_future_w: torch.Tensor,
    ref_root_quat_future_w: torch.Tensor,
):
    robot_root_pos_w_expand = robot_root_pos_w[:, None, :]
    robot_root_quat_w_expand = robot_root_quat_w[:, None, :]
    robot_root_quat_w_expand_inv = quat_conjugate(robot_root_quat_w_expand)
    ref_root_pos_future_b = quat_apply_inverse(
        robot_root_quat_w_expand,
        ref_root_pos_future_w - robot_root_pos_w_expand,
    )
    ref_root_quat_future_b = quat_mul(
        robot_root_quat_w_expand_inv.expand_as(ref_root_quat_future_w),
        ref_root_quat_future_w,
    )
    ref_root_mat_future_b = matrix_from_quat(ref_root_quat_future_b)
    return ref_root_pos_future_b, ref_root_mat_future_b


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_current_tracking_state(
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    robot_body_link_pos_w: torch.Tensor,
    robot_body_link_quat_w: torch.Tensor,
):
    ref_anchor_pos_w_z0 = ref_anchor_pos_w.clone()
    ref_anchor_pos_w_z0[..., 2] = 0.0
    robot_anchor_pos_w_z0 = robot_anchor_pos_w.clone()
    robot_anchor_pos_w_z0[..., 2] = 0.0

    # ref_anchor_yaw_quat_w = yaw_quat(ref_anchor_quat_w)
    # robot_anchor_yaw_quat_w = yaw_quat(robot_anchor_quat_w)
    ref_anchor_yaw_quat_w = ref_anchor_quat_w
    robot_anchor_yaw_quat_w = robot_anchor_quat_w
    
    ref_anchor_yaw_quat_conj_w = quat_conjugate(ref_anchor_yaw_quat_w)
    robot_anchor_yaw_quat_conj_w = quat_conjugate(robot_anchor_yaw_quat_w)

    ref_body_pos_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w[:, None],
        ref_body_pos_w - ref_anchor_pos_w_z0[:, None],
    )
    ref_body_quat_local = quat_mul(
        ref_anchor_yaw_quat_conj_w[:, None].expand_as(ref_body_quat_w),
        ref_body_quat_w,
    )

    robot_body_pos_local = quat_apply_inverse(
        robot_anchor_yaw_quat_w[:, None],
        robot_body_link_pos_w - robot_anchor_pos_w_z0[:, None],
    )
    robot_body_quat_local = quat_mul(
        robot_anchor_yaw_quat_conj_w[:, None].expand_as(robot_body_link_quat_w),
        robot_body_link_quat_w,
    )
    return ref_body_pos_local, ref_body_quat_local, robot_body_pos_local, robot_body_quat_local


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_tracking_errors(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    ref_body_pos_local: torch.Tensor,
    ref_body_quat_local: torch.Tensor,
    robot_body_link_pos_w: torch.Tensor,
    robot_body_link_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    robot_body_pos_local: torch.Tensor,
    robot_body_quat_local: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    ref_joint_vel: torch.Tensor,
    robot_joint_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
):
    body_pos_error = (ref_body_pos_w - robot_body_link_pos_w).norm(dim=-1)
    body_pos_error_local = (ref_body_pos_local - robot_body_pos_local).norm(dim=-1)

    body_quat_diff = quat_mul(
        quat_conjugate(ref_body_quat_w),
        robot_body_link_quat_w,
    )
    body_ori_error = quat_angle_magnitude(body_quat_diff)

    body_quat_local_diff = quat_mul(
        quat_conjugate(ref_body_quat_local),
        robot_body_quat_local,
    )
    body_ori_error_local = quat_angle_magnitude(body_quat_local_diff)

    body_lin_vel_error = (ref_body_lin_vel_w - robot_body_lin_vel_w).norm(dim=-1)
    body_ang_vel_error = (ref_body_ang_vel_w - robot_body_ang_vel_w).norm(dim=-1)

    joint_pos_error = (ref_joint_pos - robot_joint_pos).abs()
    joint_vel_error = (ref_joint_vel - robot_joint_vel).abs()

    return (
        body_pos_error,
        body_pos_error_local,
        body_ori_error,
        body_ori_error_local,
        body_lin_vel_error,
        body_ang_vel_error,
        joint_pos_error,
        joint_vel_error,
    )


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_motion_local_obs(
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    ref_body_pos_future_w: torch.Tensor,
    ref_body_quat_future_w: torch.Tensor,
):
    ref_anchor_pos_w_z0 = ref_anchor_pos_w.clone()
    ref_anchor_pos_w_z0[..., 2] = 0.0
    ref_anchor_pos_w_z0_future = ref_anchor_pos_w_z0[:, None, None, :]

    # ref_anchor_yaw_quat_w = yaw_quat(ref_anchor_quat_w)
    ref_anchor_yaw_quat_w = ref_anchor_quat_w
    ref_anchor_yaw_quat_w_future = ref_anchor_yaw_quat_w[:, None, None, :]
    ref_anchor_yaw_quat_conj_w_future = quat_conjugate(ref_anchor_yaw_quat_w_future)

    ref_body_pos_future_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w_future,
        ref_body_pos_future_w - ref_anchor_pos_w_z0_future,
    )
    ref_body_quat_future_local = quat_mul(
        ref_anchor_yaw_quat_conj_w_future.expand_as(ref_body_quat_future_w),
        ref_body_quat_future_w,
    )
    ref_body_ori_future_local_matrix = matrix_from_quat(ref_body_quat_future_local)

    return ref_body_pos_future_local, ref_body_ori_future_local_matrix


@torch.compile(mode="max-autotune-no-cudagraphs")
def _compute_body_local_diff_obs(
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    ref_body_pos_future_w: torch.Tensor,
    ref_body_lin_vel_future_w: torch.Tensor,
    ref_body_ang_vel_future_w: torch.Tensor,
    ref_body_quat_future_w: torch.Tensor,
    robot_body_link_pos_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    robot_body_link_quat_w: torch.Tensor,
):
    ref_anchor_pos_w_z0 = ref_anchor_pos_w.clone()
    ref_anchor_pos_w_z0[..., 2] = 0.0
    robot_anchor_pos_w_z0 = robot_anchor_pos_w.clone()
    robot_anchor_pos_w_z0[..., 2] = 0.0
    ref_anchor_pos_w_z0_future = ref_anchor_pos_w_z0[:, None, None, :]
    robot_anchor_pos_w_z0_body = robot_anchor_pos_w_z0[:, None, :]

    # ref_anchor_yaw_quat_w = yaw_quat(ref_anchor_quat_w)
    # robot_anchor_yaw_quat_w = yaw_quat(robot_anchor_quat_w)
    ref_anchor_yaw_quat_w = ref_anchor_quat_w
    robot_anchor_yaw_quat_w = robot_anchor_quat_w
    ref_anchor_yaw_quat_w_future = ref_anchor_yaw_quat_w[:, None, None, :]
    robot_anchor_yaw_quat_w_body = robot_anchor_yaw_quat_w[:, None, :]
    ref_anchor_yaw_quat_conj_w_future = quat_conjugate(ref_anchor_yaw_quat_w_future)
    robot_anchor_yaw_quat_conj_w_body = quat_conjugate(robot_anchor_yaw_quat_w_body)

    ref_body_pos_future_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w_future,
        ref_body_pos_future_w - ref_anchor_pos_w_z0_future,
    )
    ref_body_lin_vel_future_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w_future,
        ref_body_lin_vel_future_w,
    )
    ref_body_ang_vel_future_local = quat_apply_inverse(
        ref_anchor_yaw_quat_w_future,
        ref_body_ang_vel_future_w,
    )
    ref_body_quat_future_local = quat_mul(
        ref_anchor_yaw_quat_conj_w_future.expand_as(ref_body_quat_future_w),
        ref_body_quat_future_w,
    )

    robot_body_pos_local = quat_apply_inverse(
        robot_anchor_yaw_quat_w_body,
        robot_body_link_pos_w - robot_anchor_pos_w_z0_body,
    )
    robot_body_lin_vel_local = quat_apply_inverse(
        robot_anchor_yaw_quat_w_body,
        robot_body_lin_vel_w,
    )
    robot_body_ang_vel_local = quat_apply_inverse(
        robot_anchor_yaw_quat_w_body,
        robot_body_ang_vel_w,
    )
    robot_body_quat_local = quat_mul(
        robot_anchor_yaw_quat_conj_w_body.expand_as(robot_body_link_quat_w),
        robot_body_link_quat_w,
    )
    robot_body_quat_local_conj = quat_conjugate(robot_body_quat_local)

    diff_body_quat_future = quat_mul(
        robot_body_quat_local_conj.unsqueeze(1).expand_as(ref_body_quat_future_local),
        ref_body_quat_future_local,
    )
    diff_body_ori_future_local_matrix = matrix_from_quat(diff_body_quat_future)

    return (
        ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1),
        ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1),
        diff_body_ori_future_local_matrix,
        ref_body_ang_vel_future_local - robot_body_ang_vel_local.unsqueeze(1),
    )


@dataclass
class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    # mode: Literal["ghost", "frames"] = "frames"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)


class RobotTracking(Command, namespace="hdmi"):
    def __init__(
        self,
        env,
        data_path: List[str] | str,
        tracking_keypoint_names: List[str],
        tracking_joint_names: List[str],
        # reset parameters
        # will be offloaded to a dedicated randomization module in the future
        root_body_name: str = "pelvis",
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        },
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        anchor_body_name: str = "torso_link",
        call_update: bool = True,
        replay_motion: bool = False,
        record_motion: bool = False,
        rewind_prob: float = 0.0,
        rewind_steps_range: Tuple[int, int] = (25, 125),
        viz: VizCfg | Dict | None = None,
    ):
        from . import observations
        from . import rewards
        from . import terminations

        super().__init__(env)
        self.dataset = MotionDataset.create_from_path(
            data_path,
            asset_joint_names=self.asset.joint_names,
            target_fps=int(1 / self.env.step_dt),
        ).to(self.device)

        # Set tracking body and joint names for observation and termination
        tracking_body_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.tracking_body_names = to_simulation_body_order(
            tracking_body_names,
            self.asset.cfg,
        )
        self.tracking_body_indices_motion = [
            self.dataset.body_names.index(name) for name in self.tracking_body_names
        ]
        self.tracking_body_indices_asset = [
            self.asset.body_names.index(name) for name in self.tracking_body_names
        ]

        tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_names = to_simulation_joint_order(
            tracking_joint_names,
            self.asset.cfg,
        )
        self.tracking_joint_indices_motion = [
            self.dataset.joint_names.index(name) for name in self.tracking_joint_names
        ]
        self.tracking_joint_indices_asset = [
            self.asset.joint_names.index(name) for name in self.tracking_joint_names
        ]

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        future_steps = sorted(future_steps)
        assert 0 in future_steps, "future_steps must include 0 to compute current observation"
        assert 1 in future_steps, "future_steps must include 1 to compute current reward"
        self.obs_current_step_index = future_steps.index(0)
        self.reward_current_step_index = future_steps.index(1)

        self.anchor_body_name = anchor_body_name
        self.anchor_body_idx_motion = self.dataset.body_names.index(anchor_body_name)
        self.anchor_body_idx_asset = self.asset.body_names.index(anchor_body_name)

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_starts = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_ends = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)
            self.replay_motion_t = torch.zeros(self.num_envs, dtype=torch.long)

        # get root body and joint indices in motion for reset
        self.root_body_name = root_body_name
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        self.asset_joint_idx_motion = [
            self.dataset.joint_names.index(joint_name)
            for joint_name in self.asset.joint_names
        ]

        pose_range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        velocity_range_list = [
            velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.rewind_prob = rewind_prob
        self.rewind_steps_range = list(rewind_steps_range)
        assert self.rewind_steps_range[0] >= 0
        assert self.rewind_steps_range[1] > self.rewind_steps_range[0]

        self.first_sample_motion = True
        self.replay_motion = replay_motion
        self.record_motion = record_motion

        self.all_env_ids = torch.arange(self.num_envs, device=self.device)

        if self.replay_motion:
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0

        if self.record_motion:
            assert self.num_envs == 1, "record_motion only supports num_envs=1"
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0

        if call_update:
            self._read_current_robot_state()
            self._refresh_future_buffers()
            self.update()
            if self.record_motion:
                self.motion_frames = []

        # TODO: simplify viz config
        if isinstance(viz, dict):
            viz = VizCfg(**viz)
        self.viz = viz or VizCfg()
        self._ghost_model = None

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        num_resets = len(env_ids)

        if self.replay_motion:
            if self.first_sample_motion:
                sampled_frame_ids = torch.randint(
                    0, self.dataset.num_steps, size=(num_resets,), device=self.device
                )
                motion_ids = self.dataset.data.motion_id[sampled_frame_ids].long()
                self.motion_ids[env_ids] = motion_ids
                self.motion_len[env_ids] = self.dataset.lengths[motion_ids]
                self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
                self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
                self.first_sample_motion = False

            motion_len = self.motion_len[env_ids]
            self.replay_motion_t[env_ids] = (
                self.replay_motion_t[env_ids] + 1
            ) % motion_len
            self.t[env_ids] = self.replay_motion_t[env_ids]
            return

        if not self.env.training and not self.record_motion:
            if self.first_sample_motion:
                sampled_frame_ids = torch.randint(
                    0, self.dataset.num_steps, size=(num_resets,), device=self.device
                )
                motion_ids = self.dataset.data.motion_id[sampled_frame_ids].long()
                self.motion_ids[env_ids] = motion_ids
                self.motion_len[env_ids] = self.dataset.lengths[motion_ids]
                self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
                self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
                self.first_sample_motion = False

            self.t[env_ids] = 0
            return

        # Sample uniformly on the flattened dataset timeline, then recover the
        # owning motion and local step from that frame.
        sampled_frame_ids = torch.randint(
            0, self.dataset.num_steps, size=(num_resets,), device=self.device
        )
        sampled_motion_ids = self.dataset.data.motion_id[sampled_frame_ids].long()
        sampled_start_t = self.dataset.data.step[sampled_frame_ids].long()
        sampled_motion_len = self.dataset.lengths[sampled_motion_ids]
        sampled_motion_starts = self.dataset.starts[sampled_motion_ids]
        sampled_motion_ends = self.dataset.ends[sampled_motion_ids]

        terminated_t = self.t[env_ids]
        rewind_mask = torch.rand(num_resets, device=self.device) < self.rewind_prob
        if self.first_sample_motion:
            rewind_mask.fill_(False)
        rewind_steps = torch.randint(
            *self.rewind_steps_range, (num_resets,), device=self.device
        )
        rewind_t = torch.clamp(terminated_t - rewind_steps, min=0)

        motion_ids = torch.where(rewind_mask, self.motion_ids[env_ids], sampled_motion_ids)
        motion_len = torch.where(rewind_mask, self.motion_len[env_ids], sampled_motion_len)
        motion_starts = torch.where(
            rewind_mask, self.motion_starts[env_ids], sampled_motion_starts
        )
        motion_ends = torch.where(
            rewind_mask, self.motion_ends[env_ids], sampled_motion_ends
        )
        start_t = torch.where(rewind_mask, rewind_t, sampled_start_t)

        self.motion_ids[env_ids] = motion_ids
        self.motion_len[env_ids] = motion_len
        self.motion_starts[env_ids] = motion_starts
        self.motion_ends[env_ids] = motion_ends
        self.t[env_ids] = start_t
        self.first_sample_motion = False

    def sample_init(self, env_ids: torch.Tensor) -> None:
        self._sample_motions(env_ids)

        # reset root state and joint position/velocity from motion
        self._motion_reset: MotionData = self.dataset.get_slice(
            self.motion_ids[env_ids], self.t[env_ids], 1
        ).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]

        motion = self._motion_reset
        init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
        init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        # poses
        pose_rand_samples = sample_uniform(
            self.pose_range[:, 0],
            self.pose_range[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        if not self.env.training:
            pose_rand_samples.fill_(0.0)
        positions = (
            init_root_pos
            + self.env.scene.env_origins.to(self.device)[env_ids]
            + pose_rand_samples[:, 0:3]
        )
        orientations_delta = quat_from_euler_xyz(
            pose_rand_samples[:, 3], pose_rand_samples[:, 4], pose_rand_samples[:, 5]
        )
        orientations = quat_mul(init_root_quat, orientations_delta)

        # velocities
        vel_rand_samples = sample_uniform(
            self.velocity_range[:, 0],
            self.velocity_range[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        if not self.env.training:
            vel_rand_samples.fill_(0.0)
        velocities = (
            torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + vel_rand_samples
        )

        self.asset.write_root_link_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        self._write_root_com_velocity(velocities, env_ids)
        # self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = sample_uniform(
            -1, 1, init_joint_pos.shape, device=self.device
        )
        joint_vel_noise = sample_uniform(
            -1, 1, init_joint_vel.shape, device=self.device
        )

        init_joint_pos += joint_pos_noise * self.init_joint_pos_noise
        init_joint_vel += joint_vel_noise * self.init_joint_vel_noise

        # joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        # init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        # if hasattr(self.asset.data, "soft_joint_vel_limits"):
        #     joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
        #     init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.asset.write_joint_state_to_sim(
            init_joint_pos, init_joint_vel, env_ids=env_ids
        )

        if self.record_motion:
            if len(self.motion_frames) > 0:
                self._save_motion()
                self.motion_frames = []

    def _write_root_com_velocity(
        self, root_com_velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        if self.env.backend == "isaac":
            self.asset.write_root_com_velocity_to_sim(
                root_com_velocity, env_ids=env_ids
            )
        elif self.env.backend == "mjlab":
            asset_data = self.asset.data
            quat_w = asset_data.data.qpos[
                env_ids[:, None], asset_data.indexing.free_joint_q_adr[3:7]
            ]
            com_offset_b = asset_data.model.body_ipos[
                env_ids, asset_data.indexing.root_body_id
            ]
            com_offset_w = quat_rotate(quat_w, com_offset_b)

            ang_vel_w = root_com_velocity[:, 3:]
            lin_vel_link = root_com_velocity[:, :3] - torch.cross(
                ang_vel_w, com_offset_w, dim=-1
            )
            link_velocity = torch.cat([lin_vel_link, ang_vel_w], dim=-1)
            self.asset.write_root_link_velocity_to_sim(link_velocity, env_ids=env_ids)

    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names,
            "fps": int(1 / self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os, json

        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        breakpoint()

    def _read_current_robot_state(self):
        self.robot_body_link_pos_w = self.asset.data.body_link_pos_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_lin_vel_w = self.asset.data.body_com_lin_vel_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_link_quat_w = self.asset.data.body_link_quat_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_ang_vel_w = self.asset.data.body_com_ang_vel_w[
            :, self.tracking_body_indices_asset
        ]

        self.robot_joint_pos = self.asset.data.joint_pos[
            :, self.tracking_joint_indices_asset
        ]
        self.robot_joint_vel = self.asset.data.joint_vel[
            :, self.tracking_joint_indices_asset
        ]

        self.robot_root_pos_w = self.asset.data.root_link_pos_w
        self.robot_root_quat_w = self.asset.data.root_link_quat_w

        self.robot_anchor_pos_w = self.asset.data.body_link_pos_w[
            :, self.anchor_body_idx_asset
        ]
        self.robot_anchor_quat_w = self.asset.data.body_link_quat_w[
            :, self.anchor_body_idx_asset
        ]

    def _refresh_future_buffers(self):
        # `self.t` anchors the future-motion buffer used by observations.
        self.obs_motion_t = self.t.clone()
        self.future_ref_motion = self.dataset.get_slice(
            self.motion_ids, self.t, steps=self.future_steps
        )
        env_origins = self.env.scene.env_origins

        self.ref_body_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :]
            + env_origins[:, None, None, :]
        )
        self.ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[
            ..., self.tracking_body_indices_motion, :
        ]
        self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.tracking_body_indices_motion, :
        ]
        self.ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[
            ..., self.tracking_body_indices_motion, :
        ]

        self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[
            ..., self.tracking_joint_indices_motion
        ]
        self.ref_joint_vel_future_ = self.future_ref_motion.joint_vel[
            ..., self.tracking_joint_indices_motion
        ]

        self.ref_root_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :]
            + env_origins[:, None, :]
        )
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.root_body_idx_motion, :
        ]

        self.ref_anchor_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.anchor_body_idx_motion, :]
            + env_origins[:, None, :]
        )
        self.ref_anchor_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.anchor_body_idx_motion, :
        ]

        # root_diff_obs
        (
            self.ref_root_pos_future_b,
            self.ref_root_ori_future_b_matrix,
        ) = _compute_root_diff_obs(
            self.robot_root_pos_w,
            self.robot_root_quat_w,
            self.ref_root_pos_future_w,
            self.ref_root_quat_future_w,
        )

        # motion_local_obs
        (
            self.ref_body_pos_future_local,
            self.ref_body_ori_future_local_matrix,
        ) = _compute_motion_local_obs(
            self.ref_anchor_pos_future_w[:, self.obs_current_step_index],
            self.ref_anchor_quat_future_w[:, self.obs_current_step_index],
            self.ref_body_pos_future_w,
            self.ref_body_quat_future_w,
        )

        # body_local_diff_obs
        (
            self.diff_body_pos_future_local,
            self.diff_body_lin_vel_future_local,
            self.diff_body_ori_future_local_matrix,
            self.diff_body_ang_vel_future_local,
        ) = _compute_body_local_diff_obs(
            self.ref_anchor_pos_future_w[:, self.obs_current_step_index],
            self.ref_anchor_quat_future_w[:, self.obs_current_step_index],
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.ref_body_pos_future_w,
            self.ref_body_lin_vel_future_w,
            self.ref_body_ang_vel_future_w,
            self.ref_body_quat_future_w,
            self.robot_body_link_pos_w,
            self.robot_body_lin_vel_w,
            self.robot_body_ang_vel_w,
            self.robot_body_link_quat_w,
        )

    def step(self):
        self._refresh_future_buffers()
        self.t += 1

    def update(self):
        refresh_future_buffers = not hasattr(self, "future_ref_motion")
        if self.replay_motion:
            self.sample_init(self.all_env_ids)
            refresh_future_buffers = True

        if hasattr(self, "motion_frames"):
            motion_frame = {}
            motion_frame["body_pos_w"] = self.asset.data.body_link_pos_w.cpu()
            motion_frame["body_quat_w"] = self.asset.data.body_link_quat_w.cpu()
            motion_frame["body_lin_vel_w"] = self.asset.data.body_com_lin_vel_w.cpu()
            motion_frame["body_ang_vel_w"] = self.asset.data.body_com_ang_vel_w.cpu()
            motion_frame["joint_pos"] = self.asset.data.joint_pos.cpu()
            motion_frame["joint_vel"] = self.asset.data.joint_vel.cpu()
            self.motion_frames.append(TensorDict(motion_frame, batch_size=[1]))

        self._read_current_robot_state()
        if refresh_future_buffers:
            self._refresh_future_buffers()

        # Reward / termination: consume the current frame from the previously
        # prepared future-motion buffer.
        self.current_ref_motion = self.future_ref_motion[:, self.reward_current_step_index]
        self.ref_body_pos_w = self.ref_body_pos_future_w[:, self.reward_current_step_index]
        self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, self.reward_current_step_index]
        self.ref_body_quat_w = self.ref_body_quat_future_w[:, self.reward_current_step_index]
        self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, self.reward_current_step_index]
        self.ref_joint_pos = self.ref_joint_pos_future_[:, self.reward_current_step_index]
        self.ref_joint_vel = self.ref_joint_vel_future_[:, self.reward_current_step_index]
        self.ref_anchor_pos_w = self.ref_anchor_pos_future_w[:, self.reward_current_step_index]
        self.ref_anchor_quat_w = self.ref_anchor_quat_future_w[:, self.reward_current_step_index]

        (
            self.ref_body_pos_local,
            self.ref_body_quat_local,
            self.robot_body_pos_local,
            self.robot_body_quat_local,
        ) = _compute_current_tracking_state(
            self.ref_anchor_pos_w,
            self.ref_anchor_quat_w,
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.ref_body_pos_w,
            self.ref_body_quat_w,
            self.robot_body_link_pos_w,
            self.robot_body_link_quat_w,
        )

        (
            self.body_pos_error,
            self.body_pos_error_local,
            self.body_ori_error,
            self.body_ori_error_local,
            self.body_lin_vel_error,
            self.body_ang_vel_error,
            self.joint_pos_error,
            self.joint_vel_error,
        ) = _compute_tracking_errors(
            self.ref_body_pos_w,
            self.ref_body_quat_w,
            self.ref_body_lin_vel_w,
            self.ref_body_ang_vel_w,
            self.ref_body_pos_local,
            self.ref_body_quat_local,
            self.robot_body_link_pos_w,
            self.robot_body_link_quat_w,
            self.robot_body_lin_vel_w,
            self.robot_body_ang_vel_w,
            self.robot_body_pos_local,
            self.robot_body_quat_local,
            self.ref_joint_pos,
            self.ref_joint_vel,
            self.robot_joint_pos,
            self.robot_joint_vel,
        )

        # print(
        #     f"body lin vel error: {self.body_lin_vel_error.norm(dim=-1)}"
        # )
        # print(
        #     f"body ang vel error: {self.body_ang_vel_error.norm(dim=-1)}"
        # )

    def debug_draw(self):
        if not hasattr(self, "current_ref_motion"):
            return

        viewer = getattr(self.env.sim, "viewer", None)
        if viewer is None:
            return
        scene: "ViserMujocoScene" | None = getattr(viewer, "scene", None)
        if scene is None:
            return

        if self.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self.env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self.viz.ghost_color

            indexing = self.asset.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            if scene.show_all_envs or self.num_envs == 1:
                env_ids = range(self.num_envs)
            else:
                env_ids = [int(scene.env_idx)]

            for env_idx in env_ids:
                qpos = np.zeros(self.env.sim.mj_model.nq)
                # for time_index in [self.obs_current_step_index, -1]:
                for time_index in [self.obs_current_step_index]:
                    qpos[free_joint_q_adr[0:3]] = (
                        self.ref_root_pos_future_w[env_idx, time_index].cpu().numpy()
                    )
                    qpos[free_joint_q_adr[3:7]] = (
                        self.ref_root_quat_future_w[env_idx, time_index].cpu().numpy()
                    )
                    qpos[joint_q_adr] = (
                        self.future_ref_motion.joint_pos[
                            env_idx, time_index, self.asset_joint_idx_motion
                        ]
                        .cpu()
                        .numpy()
                    )

                    scene.add_ghost_mesh(
                        qpos,
                        model=self._ghost_model,
                        label=f"env_{env_idx}",
                    )
        elif self.viz.mode == "frames":
            for env_idx in range(self.num_envs):
                desired_body_pos = self.ref_body_pos_w[env_idx].cpu().numpy()
                desired_body_quat = self.ref_body_quat_w[env_idx]
                desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

                current_body_pos = self.robot_body_link_pos_w[env_idx].cpu().numpy()
                current_body_quat = self.robot_body_link_quat_w[env_idx]
                current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

                for i, body_name in enumerate(self.tracking_body_names):
                    scene.add_frame(
                        position=desired_body_pos[i],
                        rotation_matrix=desired_body_rotm[i],
                        scale=0.08,
                        label=f"desired_{body_name}_env_{env_idx}",
                        axis_colors=_DESIRED_FRAME_COLORS,
                    )
                    scene.add_frame(
                        position=current_body_pos[i],
                        rotation_matrix=current_body_rotm[i],
                        scale=0.12,
                        label=f"current_{body_name}_env_{env_idx}",
                    )
