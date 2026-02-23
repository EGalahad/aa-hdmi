import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from tensordict import TensorClass, MemoryMappedTensor
from typing import List, Union
from scipy.spatial.transform import Rotation as sRot, Slerp

try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names

unitree_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

unitree_body_names = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def lerp(ts_target, ts_source, x):
    return np.stack(
        [np.interp(ts_target, ts_source, x[:, i]) for i in range(x.shape[1])], axis=-1
    )


def slerp(ts_target, ts_source, quat):
    # time dim: 0
    # batch dim: 1:-1
    # quat dim: -1
    # for each batch dim, do the slerp
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]

    steps_target = ts_target.shape[0]
    steps_source = ts_source.shape[0]

    quat = quat.reshape(steps_source, -1, quat_dim)

    batch_size = int(np.prod(batch_shape, initial=1))
    out = np.empty((steps_target, batch_size, quat_dim))
    for i in range(batch_size):
        s = Slerp(
            ts_source, sRot.from_quat(quat[:, i, [1, 2, 3, 0]])
        )  # quat first to quat last
        out[:, i, :] = s(ts_target).as_quat()[
            ..., [3, 0, 1, 2]
        ]  # quat last to quat first
    out = out.reshape(steps_target, *batch_shape, quat_dim)
    return out


def interpolate(motion, source_fps: int, target_fps: int):
    if source_fps != target_fps:
        in_keys = [
            "body_pos_w",
            "body_lin_vel_w",
            "body_quat_w",
            "body_ang_vel_w",
            "joint_pos",
            "joint_vel",
        ]
        extra_keys = set(motion.keys()) - set(in_keys)
        if extra_keys:
            raise NotImplementedError(
                f"interpolation is not fully implemented for keys: {extra_keys}"
            )
        T = motion["joint_pos"].shape[0]
        ts_source = np.linspace(0, T, T)
        ts_target = np.linspace(0, T, int(T / source_fps * target_fps))
        motion["body_pos_w"] = lerp(
            ts_target, ts_source, motion["body_pos_w"].reshape(T, -1)
        ).reshape(len(ts_target), -1, 3)
        motion["body_lin_vel_w"] = lerp(
            ts_target, ts_source, motion["body_lin_vel_w"].reshape(T, -1)
        ).reshape(len(ts_target), -1, 3)
        motion["body_quat_w"] = slerp(ts_target, ts_source, motion["body_quat_w"])
        motion["body_ang_vel_w"] = lerp(
            ts_target, ts_source, motion["body_ang_vel_w"].reshape(T, -1)
        ).reshape(len(ts_target), -1, 3)
        motion["joint_pos"] = lerp(ts_target, ts_source, motion["joint_pos"])
        motion["joint_vel"] = lerp(ts_target, ts_source, motion["joint_vel"])
    return motion


def quat_to_angular_velocity(quat: torch.Tensor, fps: float) -> torch.Tensor:
    """Convert quaternion sequence to angular velocities using finite differences.

    Args:
        quat: Quaternion sequence of shape [T, ..., 4] where ... represents arbitrary batch dimensions
        fps: Frame rate for computing the time derivative

    Returns:
        Angular velocities of shape [T-1, ..., 3]
    """
    dt = 1.0 / fps

    # Get q1 and q2 for consecutive timesteps
    q1 = quat[:-1]  # [T-1, ..., 4]
    q2 = quat[1:]  # [T-1, ..., 4]

    # Compute angular velocities using the formula
    # ω = 2/dt * [q1w*q2x - q1x*q2w - q1y*q2z + q1z*q2y,
    #             q1w*q2y + q1x*q2z - q1y*q2w - q1z*q2x,
    #             q1w*q2z - q1x*q2y + q1y*q2x - q1z*q2w]

    ang_vel = (2.0 / dt) * torch.stack(
        [
            q1[..., 0] * q2[..., 1]
            - q1[..., 1] * q2[..., 0]
            - q1[..., 2] * q2[..., 3]
            + q1[..., 3] * q2[..., 2],
            q1[..., 0] * q2[..., 2]
            + q1[..., 1] * q2[..., 3]
            - q1[..., 2] * q2[..., 0]
            - q1[..., 3] * q2[..., 1],
            q1[..., 0] * q2[..., 3]
            - q1[..., 1] * q2[..., 2]
            + q1[..., 2] * q2[..., 1]
            - q1[..., 3] * q2[..., 0],
        ],
        dim=-1,
    )

    return ang_vel


class MotionData(TensorClass):
    motion_id: torch.Tensor
    step: torch.Tensor
    body_pos_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class MotionDataset:
    def __init__(
        self,
        body_names: List[str],
        joint_names: List[str],
        motion_paths: List[Path],
        starts: List[int],
        ends: List[int],
        data: MotionData,
    ):
        self.body_names = body_names
        self.joint_names = joint_names
        self.motion_paths = motion_paths
        self.starts = torch.as_tensor(starts)
        self.ends = torch.as_tensor(ends)
        self.lengths = self.ends - self.starts
        self.data = data
        self.device = data.device

    def to(self, device: torch.device):
        self.data = self.data.to(device)
        self.starts = self.starts.to(device)
        self.ends = self.ends.to(device)
        self.lengths = self.lengths.to(device)
        self.device = device
        return self

    @classmethod
    def create_from_path(
        cls,
        root_path: str | List[str],
        asset_joint_names: List[str] | None = None,
        target_fps: int = 50,
        memory_mapped: bool = False,
    ):
        import active_adaptation

        base_dir = Path(active_adaptation.__file__).parent.parent
        root = base_dir / root_path
        if root.is_file() and root.suffix == ".npz":
            motion_paths = [root]
        else:
            motion_paths = list(root.rglob("motion.npz"))
        if not motion_paths:
            raise RuntimeError(f"No motions found in {root_path}")
        motion_paths = [motion_path.parent for motion_path in motion_paths]

        print(f"Matched {len(motion_paths)} motions under {root_path}")

        if not motion_paths:
            raise RuntimeError(f"No motions matched the given patterns")

        # First read and verify all meta.json files are identical
        metas = []
        for path in motion_paths:
            meta_path = path / "meta.json"
            with open(meta_path, "r") as f:
                meta = json.load(f)
                meta.pop("length", None)
                metas.append(meta)

        # Compare all metas to the first one
        for i, meta in enumerate(metas[1:], 1):
            if meta != metas[0]:
                breakpoint()
                raise ValueError(
                    f"meta.json in {motion_paths[i]} differs from {motion_paths[0]}"
                )
        meta = metas[0]

        motion_paths = [path / "motion.npz" for path in motion_paths]

        motions = []
        total_length = 0
        for i, motion_path in enumerate(tqdm(motion_paths)):
            motion = dict(np.load(motion_path, allow_pickle=True))
            motion = interpolate(motion, source_fps=meta["fps"], target_fps=target_fps)
            total_length += motion["body_pos_w"].shape[0]
            motions.append(motion)

        if asset_joint_names is not None:
            asset_joint_names_list = list(asset_joint_names)
            share_joint_names = [
                name for name in meta["joint_names"] if name in asset_joint_names_list
            ]
            src_joint_indices = [
                meta["joint_names"].index(name) for name in share_joint_names
            ]
            dst_joint_indices = [
                asset_joint_names_list.index(name) for name in share_joint_names
            ]

            more_joint_names = [
                name
                for name in meta["joint_names"]
                if name not in asset_joint_names_list
            ]
            src_more_joint_indices = [
                meta["joint_names"].index(name) for name in more_joint_names
            ]
            dst_more_joint_indices = [
                len(asset_joint_names_list) + i for i in range(len(more_joint_names))
            ]

            joint_names = asset_joint_names_list + more_joint_names
            src_joint_indices = src_joint_indices + src_more_joint_indices
            dst_joint_indices = dst_joint_indices + dst_more_joint_indices

            for motion in motions:
                joint_pos = np.zeros((motion["joint_pos"].shape[0], len(joint_names)))
                joint_vel = np.zeros((motion["joint_vel"].shape[0], len(joint_names)))
                joint_pos[:, dst_joint_indices] = motion["joint_pos"][
                    :, src_joint_indices
                ]
                joint_vel[:, dst_joint_indices] = motion["joint_vel"][
                    :, src_joint_indices
                ]
                motion["joint_pos"] = joint_pos
                motion["joint_vel"] = joint_vel
            meta["joint_names"] = joint_names

        TensorClass = MemoryMappedTensor if memory_mapped else torch

        step: torch.Tensor = TensorClass.empty(total_length, dtype=int)
        motion_id: torch.Tensor = TensorClass.empty(total_length, dtype=int)
        body_pos_w: torch.Tensor = TensorClass.empty(
            total_length, len(meta["body_names"]), 3
        )
        body_lin_vel_w: torch.Tensor = TensorClass.empty(
            total_length, len(meta["body_names"]), 3
        )
        body_quat_w: torch.Tensor = TensorClass.empty(
            total_length, len(meta["body_names"]), 4
        )
        body_ang_vel_w: torch.Tensor = TensorClass.empty(
            total_length, len(meta["body_names"]), 3
        )
        joint_pos: torch.Tensor = TensorClass.empty(
            total_length, len(meta["joint_names"])
        )
        joint_vel: torch.Tensor = TensorClass.empty(
            total_length, len(meta["joint_names"])
        )

        start_idx = 0

        starts = []
        ends = []

        for i, motion in enumerate(motions):
            motion_length = motion["body_pos_w"].shape[0]
            step[start_idx : start_idx + motion_length] = torch.arange(motion_length)
            motion_id[start_idx : start_idx + motion_length] = i

            # Body and joint positions
            body_pos_w[start_idx : start_idx + motion_length] = torch.as_tensor(
                motion["body_pos_w"]
            )
            body_lin_vel_w[start_idx : start_idx + motion_length] = torch.as_tensor(
                motion["body_lin_vel_w"]
            )
            body_quat_w[start_idx : start_idx + motion_length] = torch.as_tensor(
                motion["body_quat_w"]
            )
            body_ang_vel_w[start_idx : start_idx + motion_length] = torch.as_tensor(
                motion["body_ang_vel_w"]
            )
            joint_pos[start_idx : start_idx + motion_length] = torch.as_tensor(
                motion["joint_pos"]
            )
            joint_vel[start_idx : start_idx + motion_length] = torch.as_tensor(
                motion["joint_vel"]
            )

            starts.append(start_idx)
            start_idx += motion_length
            ends.append(start_idx)

        kwargs = {
            "motion_id": motion_id,
            "step": step,
            "body_pos_w": body_pos_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_quat_w": body_quat_w,
            "body_ang_vel_w": body_ang_vel_w,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "batch_size": [total_length],
        }

        data = MotionData(**kwargs)

        return cls(
            body_names=meta["body_names"],
            joint_names=meta["joint_names"],
            motion_paths=motion_paths,
            starts=starts,
            ends=ends,
            data=data,
        )

    @property
    def num_motions(self):
        return len(self.starts)

    @property
    def num_steps(self):
        return len(self.data)

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: Union[int, torch.Tensor] = 1,
    ) -> MotionData:
        if isinstance(steps, int):
            steps = torch.arange(steps, device=self.device)
        idx = (self.starts[motion_ids] + starts).unsqueeze(1) + steps.unsqueeze(0)
        idx.clamp_max_(self.ends.unsqueeze(1)[motion_ids] - 1)
        return self.data[idx]  # shape: [len(motion_ids), len(steps), ...]

    def find_joints(self, joint_names, preserve_order: bool = False):
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names, preserve_order: bool = False):
        return resolve_matching_names(body_names, self.body_names, preserve_order)
