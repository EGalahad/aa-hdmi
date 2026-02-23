from active_adaptation.envs.mdp.base import Command
from hdmi.hdmi_tasks.motion import MotionDataset, MotionData

from dataclasses import dataclass
from typing import List, Dict, Tuple, TYPE_CHECKING, Literal
import copy

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer

import torch
import numpy as np

from active_adaptation.utils.math import (
    sample_uniform as _sample_uniform,
    quat_from_euler_xyz as _quat_from_euler_xyz,
    quat_mul,
    matrix_from_quat,
)
from tensordict import TensorDict


_DESIRED_FRAME_COLORS = (
    (0.9, 0.3, 0.3, 0.9),
    (0.3, 0.9, 0.3, 0.9),
    (0.3, 0.3, 0.9, 0.9),
)


def sample_uniform(low, high, size, device):
    return _sample_uniform(size=size, low=low, high=high, device=device)


def quat_from_euler_xyz(roll, pitch, yaw):
    return _quat_from_euler_xyz(torch.stack([roll, pitch, yaw], dim=-1))


@dataclass
class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    # mode: Literal["ghost", "frames"] = "frames"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)


class RobotTracking(Command):
    def __init__(
        self,
        env,
        data_path: List[str] | str,
        tracking_keypoint_names: List[str],
        tracking_joint_names: List[str],
        # reset parameters
        root_body_name: str = "pelvis",
        anchor_body_name: str = "torso_link",
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
        call_update: bool = True,
        sample_motion: bool = False,
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

        # Set tracking keypoint and joint names for observation and termination
        self.tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[
            1
        ]
        self.tracking_body_indices_motion = [
            self.dataset.body_names.index(name) for name in self.tracking_keypoint_names
        ]
        self.tracking_body_indices_asset = [
            self.asset.body_names.index(name) for name in self.tracking_keypoint_names
        ]

        self.tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = [
            self.dataset.joint_names.index(name) for name in self.tracking_joint_names
        ]
        self.tracking_joint_indices_asset = [
            self.asset.joint_names.index(name) for name in self.tracking_joint_names
        ]

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        # get root body and joint indices in motion for reset
        self.root_body_name = root_body_name
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        self.anchor_body_name = anchor_body_name
        self.anchor_body_idx_motion = self.dataset.body_names.index(anchor_body_name)
        self.anchor_body_idx_asset = self.asset.body_names.index(anchor_body_name)

        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [
            self.dataset.joint_names.index(joint_name)
            for joint_name in asset_joint_names
        ]

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_starts = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_ends = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)
            self.replay_motion_t = torch.zeros(self.num_envs, dtype=torch.long)

            self.eval_t = torch.randint(
                0, self.dataset.lengths[0], (self.num_envs,), device=self.device
            )

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
        self.sample_motion = sample_motion
        self.replay_motion = replay_motion
        self.record_motion = record_motion

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
            self.update()
            if self.record_motion:
                self.motion_frames = []

        if isinstance(viz, dict):
            viz = VizCfg(**viz)
        self.viz = viz or VizCfg()
        self._ghost_model = None

    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        if self.sample_motion or self.first_sample_motion:
            # sample motion id and start time for each env
            motion_ids = torch.randint(
                0, self.dataset.num_motions, size=(len(env_ids),), device=self.device
            )
            for i, motion_id in enumerate(motion_ids):
                # print(f"Sampling motion {self.dataset.motion_paths[motion_id.item()]} for env {env_ids[i].item()}:")
                pass
            self.motion_ids[env_ids] = motion_ids
            self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
            self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
            self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            self.first_sample_motion = False
        else:
            motion_len = self.motion_len[env_ids]

        max_len = motion_len - self.future_steps[-1]
        start_phase = torch.rand(len(env_ids), device=self.device)
        start_t = (start_phase * max_len).long()

        terminated_t = self.t[env_ids]
        rewind_mask = torch.rand(len(env_ids), device=self.device) < self.rewind_prob
        rewind_steps = torch.randint(
            *self.rewind_steps_range, (len(env_ids),), device=self.device
        )
        rewind_t = torch.clamp(terminated_t - rewind_steps, min=0)  # , max=max_len - 1)
        start_t = torch.where(rewind_mask, rewind_t, start_t)

        if not self.env.training or self.record_motion:
            start_t.fill_(0)

        if self.replay_motion:
            self.replay_motion_t[env_ids] = (
                self.replay_motion_t[env_ids] + 1
            ) % motion_len
            start_t = self.replay_motion_t[env_ids]

        self.t[env_ids] = start_t

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
        rand_samples = sample_uniform(
            self.pose_range[:, 0],
            self.pose_range[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        if not self.env.training:
            rand_samples.fill_(0.0)
        positions = (
            init_root_pos + self.env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        )
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        orientations = quat_mul(init_root_quat, orientations_delta)

        # velocities
        rand_samples = sample_uniform(
            self.velocity_range[:, 0],
            self.velocity_range[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        if not self.env.training:
            rand_samples.fill_(0.0)
        velocities = (
            torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples
        )

        self.asset.write_root_link_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        self.asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = (
            sample_uniform(
                -1,
                1,
                (init_joint_pos.shape[0], init_joint_pos.shape[1]),
                device=self.device,
            )
            * self.init_joint_pos_noise
        )
        joint_vel_noise = (
            sample_uniform(
                -1,
                1,
                (init_joint_vel.shape[0], init_joint_vel.shape[1]),
                device=self.device,
            )
            * self.init_joint_vel_noise
        )

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        if hasattr(self.asset.data, "soft_joint_vel_limits"):
            joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
            init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.asset.write_joint_state_to_sim(
            init_joint_pos, init_joint_vel, env_ids=env_ids
        )

        if self.record_motion:
            if len(self.motion_frames) > 0:
                self._save_motion()
                self.motion_frames = []

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

    @property
    def success(self):
        return (self.t >= self.motion_len - 1).unsqueeze(1)

    @property
    def finished(self):
        if self.replay_motion:
            return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
        return (self.t >= self.motion_len).unsqueeze(1)

    def update(self):
        if hasattr(self, "motion_frames"):
            motion_frame = {}
            motion_frame["body_pos_w"] = self.asset.data.body_link_pos_w.cpu()
            motion_frame["body_quat_w"] = self.asset.data.body_link_quat_w.cpu()
            motion_frame["body_lin_vel_w"] = self.asset.data.body_com_lin_vel_w.cpu()
            motion_frame["body_ang_vel_w"] = self.asset.data.body_com_ang_vel_w.cpu()
            motion_frame["joint_pos"] = self.asset.data.joint_pos.cpu()
            motion_frame["joint_vel"] = self.asset.data.joint_vel.cpu()
            self.motion_frames.append(TensorDict(motion_frame, batch_size=[1]))

        # future ref motion for actor observation
        self.future_ref_motion = self.dataset.get_slice(
            self.motion_ids, self.t, steps=self.future_steps
        )
        # shape: [num_envs, len(future_steps), num_bodies/num_joints, 3/4/...]

        # Observations: future ref body and joint states
        self.ref_body_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :]
            + self.env.scene.env_origins[:, None, None, :]
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
            + self.env.scene.env_origins[:, None, :]
        )
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.root_body_idx_motion, :
        ]

        self.ref_anchor_pos_future_w = (
            self.future_ref_motion.body_pos_w[..., self.anchor_body_idx_motion, :]
            + self.env.scene.env_origins[:, None, :]
        )
        self.ref_anchor_quat_future_w = self.future_ref_motion.body_quat_w[
            ..., self.anchor_body_idx_motion, :
        ]

        # Reward: current robot body and joint states
        self.robot_body_link_pos_w = self.asset.data.body_link_pos_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_com_lin_vel_w = self.asset.data.body_com_lin_vel_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_link_quat_w = self.asset.data.body_link_quat_w[
            :, self.tracking_body_indices_asset
        ]
        self.robot_body_com_ang_vel_w = self.asset.data.body_com_ang_vel_w[
            :, self.tracking_body_indices_asset
        ]

        self.robot_joint_pos = self.asset.data.joint_pos[
            :, self.tracking_joint_indices_asset
        ]
        self.robot_joint_vel = self.asset.data.joint_vel[
            :, self.tracking_joint_indices_asset
        ]

        self.robot_root_link_pos_w = self.asset.data.root_link_pos_w
        self.robot_root_link_quat_w = self.asset.data.root_link_quat_w

        self.robot_anchor_link_pos_w = self.asset.data.body_link_pos_w[
            :, self.anchor_body_idx_asset
        ]
        self.robot_anchor_link_quat_w = self.asset.data.body_link_quat_w[
            :, self.anchor_body_idx_asset
        ]

        # Reward: current ref body and joint states
        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
        self.ref_body_link_pos_w = self.ref_body_pos_future_w[:, 0]
        self.ref_body_com_lin_vel_w = self.ref_body_lin_vel_future_w[:, 0]
        self.ref_body_link_quat_w = self.ref_body_quat_future_w[:, 0]
        self.ref_body_com_ang_vel_w = self.ref_body_ang_vel_future_w[:, 0]
        self.ref_joint_pos = self.ref_joint_pos_future_[:, 0]
        self.ref_joint_vel = self.ref_joint_vel_future_[:, 0]
        self.ref_root_link_pos_w = self.ref_root_pos_future_w[:, 0]
        self.ref_root_link_quat_w = self.ref_root_quat_future_w[:, 0]
        self.ref_anchor_link_pos_w = self.ref_anchor_pos_future_w[:, 0]
        self.ref_anchor_link_quat_w = self.ref_anchor_quat_future_w[:, 0]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, xxx]

        self.t += 1

    def debug_draw(self):
        if not hasattr(self, "current_ref_motion"):
            return

        visualizer: "DebugVisualizer" = getattr(self.env, "visualizer", None)
        if visualizer is None:
            return

        env_idx = visualizer.env_idx
        if self.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self.env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self.viz.ghost_color

            indexing = self.asset.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            qpos = np.zeros(self.env.sim.mj_model.nq)
            qpos[free_joint_q_adr[0:3]] = (
                self.ref_root_pos_future_w[env_idx, 0].cpu().numpy()
            )
            qpos[free_joint_q_adr[3:7]] = (
                self.ref_root_quat_future_w[env_idx, 0].cpu().numpy()
            )
            qpos[joint_q_adr] = (
                self.current_ref_motion.joint_pos[env_idx, self.asset_joint_idx_motion]
                .cpu()
                .numpy()
            )

            visualizer.add_ghost_mesh(qpos, model=self._ghost_model)
        elif self.viz.mode == "frames":
            desired_body_pos = self.ref_body_link_pos_w[env_idx].cpu().numpy()
            desired_body_quat = self.ref_body_link_quat_w[env_idx]
            desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

            current_body_pos = self.robot_body_link_pos_w[env_idx].cpu().numpy()
            current_body_quat = self.robot_body_link_quat_w[env_idx]
            current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

            for i, body_name in enumerate(self.tracking_keypoint_names):
                visualizer.add_frame(
                    position=desired_body_pos[i],
                    rotation_matrix=desired_body_rotm[i],
                    scale=0.08,
                    label=f"desired_{body_name}",
                    axis_colors=_DESIRED_FRAME_COLORS,
                )
                visualizer.add_frame(
                    position=current_body_pos[i],
                    rotation_matrix=current_body_rotm[i],
                    scale=0.12,
                    label=f"current_{body_name}",
                )
