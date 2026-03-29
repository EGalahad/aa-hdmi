from __future__ import annotations

import torch
import numpy as np
import json
import hashlib
import os
import shutil
import time
import mujoco
from mjhub import resolve_mjcf_reference
from tqdm import tqdm
from pathlib import Path
from tensordict import TensorClass, MemoryMappedTensor, TensorDict
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

QPOS_CACHE_VERSION = 1
QPOS_CACHE_SUBDIR = ".cache/motion/qpos_fk"
QPOS_CACHE_INDEX_NAME = "motion_index.json"
QPOS_CACHE_META_NAME = "cache_meta.json"
QPOS_CACHE_READY_NAME = "ready.flag"
QPOS_CACHE_LOOKUP_SUBDIR = "lookup"
ANY4HDMI_MANIFEST_NAME = "manifest.json"


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
        ts_source = np.arange(0, (T - 1) * target_fps + 1, target_fps)
        ts_target = np.arange(0, (T - 1) * target_fps + 1, source_fps)
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


def _resolve_input_paths(base_dir: Path, root_path: str | List[str]) -> list[Path]:
    if isinstance(root_path, (str, Path)):
        raw_paths = [Path(root_path)]
    else:
        raw_paths = [Path(path) for path in root_path]
    resolved_paths = []
    for path in raw_paths:
        path = path.expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved_paths.append(path.resolve())
    return resolved_paths


def _find_any4hdmi_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ANY4HDMI_MANIFEST_NAME).is_file():
            return candidate
    return None


def _load_any4hdmi_manifest(dataset_root: Path) -> dict:
    manifest_path = dataset_root / ANY4HDMI_MANIFEST_NAME
    with open(manifest_path, "r") as f:
        return json.load(f)


def _resolve_any4hdmi_dataset_context(input_paths: list[Path]) -> tuple[Path, dict]:
    dataset_root: Path | None = None
    dataset_manifest: dict | None = None

    for input_path in input_paths:
        current_root = _find_any4hdmi_root(input_path)
        if current_root is None:
            raise RuntimeError(f"Could not find {ANY4HDMI_MANIFEST_NAME} above {input_path}")
        if dataset_root is None:
            dataset_root = current_root
            dataset_manifest = _load_any4hdmi_manifest(current_root)
        elif current_root != dataset_root:
            raise ValueError(
                f"All any4hdmi inputs must belong to one dataset root, got {dataset_root} and {current_root}"
            )

    if dataset_root is None or dataset_manifest is None:
        raise RuntimeError("Failed to resolve any4hdmi dataset root")
    return dataset_root, dataset_manifest


def _resolve_any4hdmi_motion_paths(
    input_paths: list[Path],
) -> tuple[Path, dict, list[Path]]:
    dataset_root, dataset_manifest = _resolve_any4hdmi_dataset_context(input_paths)
    motion_paths: set[Path] = set()

    for input_path in input_paths:
        motions_root = dataset_root / dataset_manifest.get("motions_subdir", "motions")

        if input_path.is_file():
            if input_path.suffix != ".npz":
                raise ValueError(f"Expected a .npz motion file under any4hdmi root, got {input_path}")
            motion_paths.add(input_path)
            continue

        scan_root = motions_root if input_path == dataset_root else input_path
        for motion_path in scan_root.rglob("*.npz"):
            motion_paths.add(motion_path.resolve())

    if not motion_paths:
        motions_subdir = dataset_manifest.get("motions_subdir", "motions")
        motion_paths.update((dataset_root / motions_subdir).rglob("*.npz"))
    motion_paths_list = sorted(path.resolve() for path in motion_paths)
    if not motion_paths_list:
        raise RuntimeError(f"No qpos motions found under {dataset_root}")
    return dataset_root, dataset_manifest, motion_paths_list


def _resolve_legacy_motion_paths(input_paths: list[Path]) -> list[Path]:
    motion_paths: set[Path] = set()
    for input_path in input_paths:
        if input_path.is_file() and input_path.suffix == ".npz":
            motion_paths.add(input_path.resolve())
        elif input_path.is_dir():
            motion_paths.update(path.resolve() for path in input_path.rglob("motion.npz"))
    motion_paths_list = sorted(motion_paths)
    if not motion_paths_list:
        raise RuntimeError(f"No motions found in {input_paths}")
    return motion_paths_list


def _load_legacy_meta(motion_dirs: list[Path]) -> dict:
    metas = []
    for path in motion_dirs:
        meta_path = path / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
            meta.pop("length", None)
            metas.append(meta)
    for i, meta in enumerate(metas[1:], 1):
        if meta != metas[0]:
            raise ValueError(
                f"meta.json in {motion_dirs[i]} differs from {motion_dirs[0]}"
            )
    return metas[0]


def _body_names_from_model(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    ]


def _hinge_joint_info(model: mujoco.MjModel) -> tuple[list[str], np.ndarray, np.ndarray]:
    joint_names: list[str] = []
    joint_qpos_addrs: list[int] = []
    joint_dof_addrs: list[int] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        joint_qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_addrs.append(int(model.jnt_dofadr[joint_id]))
    return (
        joint_names,
        np.asarray(joint_qpos_addrs, dtype=np.int32),
        np.asarray(joint_dof_addrs, dtype=np.int32),
    )


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float32)
    if qpos.shape[0] <= 1:
        return qvel
    dt = 1.0 / fps
    work = np.zeros(model.nv, dtype=np.float64)
    for frame_idx in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(
            model, work, dt, qpos[frame_idx], qpos[frame_idx + 1]
        )
        qvel[frame_idx] = np.asarray(work, dtype=np.float32)
    qvel[-1] = qvel[-2]
    return qvel


# TODO: use mujoco warp for batched fk
def _run_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray,
    joint_qpos_addrs: np.ndarray,
    joint_dof_addrs: np.ndarray,
) -> dict[str, np.ndarray]:
    frames = int(qpos.shape[0])
    body_pos_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_lin_vel_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((frames, model.nbody, 4), dtype=np.float32)
    body_ang_vel_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    joint_pos = np.zeros((frames, joint_qpos_addrs.shape[0]), dtype=np.float32)
    joint_vel = np.zeros((frames, joint_dof_addrs.shape[0]), dtype=np.float32)

    for frame_idx in range(frames):
        data.qpos[:] = qpos[frame_idx]
        data.qvel[:] = qvel[frame_idx]
        mujoco.mj_forward(model, data)
        body_pos_w[frame_idx] = np.asarray(data.xpos, dtype=np.float32)
        body_lin_vel_w[frame_idx] = np.asarray(data.cvel[:, 3:6], dtype=np.float32)
        body_quat_w[frame_idx] = np.asarray(data.xquat, dtype=np.float32)
        body_ang_vel_w[frame_idx] = np.asarray(data.cvel[:, 0:3], dtype=np.float32)
        joint_pos[frame_idx] = np.asarray(data.qpos[joint_qpos_addrs], dtype=np.float32)
        joint_vel[frame_idx] = np.asarray(data.qvel[joint_dof_addrs], dtype=np.float32)

    return {
        "body_pos_w": body_pos_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_quat_w": body_quat_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def _apply_joint_mapping(
    motions: list[dict[str, np.ndarray]],
    joint_names: list[str],
    asset_joint_names: List[str] | None,
) -> list[str]:
    if asset_joint_names is None:
        return joint_names

    asset_joint_names_list = list(asset_joint_names)
    share_joint_names = [name for name in joint_names if name in asset_joint_names_list]
    src_joint_indices = [joint_names.index(name) for name in share_joint_names]
    dst_joint_indices = [asset_joint_names_list.index(name) for name in share_joint_names]

    more_joint_names = [name for name in joint_names if name not in asset_joint_names_list]
    src_more_joint_indices = [joint_names.index(name) for name in more_joint_names]
    dst_more_joint_indices = [
        len(asset_joint_names_list) + i for i in range(len(more_joint_names))
    ]

    remapped_joint_names = asset_joint_names_list + more_joint_names
    src_joint_indices = src_joint_indices + src_more_joint_indices
    dst_joint_indices = dst_joint_indices + dst_more_joint_indices

    for motion in motions:
        motion_length = motion["joint_pos"].shape[0]
        joint_pos = np.zeros((motion_length, len(remapped_joint_names)), dtype=np.float32)
        joint_vel = np.zeros((motion_length, len(remapped_joint_names)), dtype=np.float32)
        joint_pos[:, dst_joint_indices] = motion["joint_pos"][:, src_joint_indices]
        joint_vel[:, dst_joint_indices] = motion["joint_vel"][:, src_joint_indices]
        motion["joint_pos"] = joint_pos
        motion["joint_vel"] = joint_vel
    return remapped_joint_names


def _build_motion_data(
    motions: list[dict[str, np.ndarray]],
    body_names: list[str],
    joint_names: list[str],
    use_memory_mapped_tensor: bool,
) -> tuple[MotionData, list[int], list[int]]:
    total_length = sum(int(motion["body_pos_w"].shape[0]) for motion in motions)
    tensor_class = MemoryMappedTensor if use_memory_mapped_tensor else torch

    step: torch.Tensor = tensor_class.empty(total_length, dtype=int)
    motion_id: torch.Tensor = tensor_class.empty(total_length, dtype=int)
    body_pos_w: torch.Tensor = tensor_class.empty(total_length, len(body_names), 3)
    body_lin_vel_w: torch.Tensor = tensor_class.empty(total_length, len(body_names), 3)
    body_quat_w: torch.Tensor = tensor_class.empty(total_length, len(body_names), 4)
    body_ang_vel_w: torch.Tensor = tensor_class.empty(total_length, len(body_names), 3)
    joint_pos: torch.Tensor = tensor_class.empty(total_length, len(joint_names))
    joint_vel: torch.Tensor = tensor_class.empty(total_length, len(joint_names))

    starts = []
    ends = []
    start_idx = 0
    for motion_idx, motion in enumerate(motions):
        motion_length = int(motion["body_pos_w"].shape[0])
        end_idx = start_idx + motion_length
        step[start_idx:end_idx] = torch.arange(motion_length)
        motion_id[start_idx:end_idx] = motion_idx
        body_pos_w[start_idx:end_idx] = torch.as_tensor(motion["body_pos_w"])
        body_lin_vel_w[start_idx:end_idx] = torch.as_tensor(motion["body_lin_vel_w"])
        body_quat_w[start_idx:end_idx] = torch.as_tensor(motion["body_quat_w"])
        body_ang_vel_w[start_idx:end_idx] = torch.as_tensor(motion["body_ang_vel_w"])
        joint_pos[start_idx:end_idx] = torch.as_tensor(motion["joint_pos"])
        joint_vel[start_idx:end_idx] = torch.as_tensor(motion["joint_vel"])
        starts.append(start_idx)
        ends.append(end_idx)
        start_idx = end_idx

    data = MotionData(
        motion_id=motion_id,
        step=step,
        body_pos_w=body_pos_w,
        body_lin_vel_w=body_lin_vel_w,
        body_quat_w=body_quat_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        batch_size=[total_length],
    )
    return data, starts, ends


def _build_motion_data_from_fields(
    *,
    motion_id: torch.Tensor,
    step: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> MotionData:
    return MotionData(
        motion_id=motion_id,
        step=step,
        body_pos_w=body_pos_w,
        body_lin_vel_w=body_lin_vel_w,
        body_quat_w=body_quat_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        batch_size=[int(motion_id.shape[0])],
    )


def _cache_root(base_dir: Path) -> Path:
    cache_root = base_dir / QPOS_CACHE_SUBDIR
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _cache_lookup_path(cache_root: Path, lookup_key: str) -> Path:
    lookup_dir = cache_root / QPOS_CACHE_LOOKUP_SUBDIR
    lookup_dir.mkdir(parents=True, exist_ok=True)
    return lookup_dir / f"{lookup_key}.json"


def _stat_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _make_qpos_cache_key(
    *,
    dataset_root: Path,
    manifest: dict,
    motion_paths: list[Path],
    mjcf_path: Path,
    target_fps: int,
) -> str:
    payload = {
        "cache_version": QPOS_CACHE_VERSION,
        "dataset_root": str(dataset_root),
        "manifest": _stat_fingerprint(dataset_root / ANY4HDMI_MANIFEST_NAME),
        "mjcf": _stat_fingerprint(mjcf_path),
        "target_fps": int(target_fps),
        "motions": [],
    }
    for motion_path in motion_paths:
        entry = {"motion": _stat_fingerprint(motion_path)}
        sidecar_path = motion_path.with_suffix(".json")
        if sidecar_path.is_file():
            entry["sidecar"] = _stat_fingerprint(sidecar_path)
        payload["motions"].append(entry)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _make_qpos_cache_lookup_key(
    *,
    dataset_root: Path,
    input_paths: list[Path],
    mjcf_path: Path,
    target_fps: int,
) -> str:
    payload = {
        "cache_version": QPOS_CACHE_VERSION,
        "dataset_root": str(dataset_root),
        "input_paths": sorted(str(path) for path in input_paths),
        "manifest": _stat_fingerprint(dataset_root / ANY4HDMI_MANIFEST_NAME),
        "mjcf": _stat_fingerprint(mjcf_path),
        "target_fps": int(target_fps),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _acquire_cache_lock(lock_dir: Path, ready_flag: Path, timeout_s: float = 600.0) -> bool:
    start_time = time.monotonic()
    while True:
        if ready_flag.is_file():
            return False
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            return True
        except FileExistsError:
            if time.monotonic() - start_time > timeout_s:
                raise TimeoutError(f"Timed out waiting for cache lock {lock_dir}")
            time.sleep(0.5)


def _resolve_any4hdmi_mjcf_path(dataset_root: Path, manifest: dict) -> Path:
    mjcf_ref = manifest.get("mjcf")
    if mjcf_ref is not None:
        return resolve_mjcf_reference(mjcf_ref, local_root=dataset_root)

    mjcf_path_raw = manifest.get("mjcf_path")
    if mjcf_path_raw is None:
        raise KeyError(f"{ANY4HDMI_MANIFEST_NAME} is missing mjcf or mjcf_path")
    mjcf_path = Path(mjcf_path_raw).expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
    return mjcf_path


def _build_qpos_cache(
    *,
    dataset_root: Path,
    manifest: dict,
    motion_paths: list[Path],
    mjcf_path: Path,
    cache_entry_dir: Path,
    target_fps: int,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    body_names = _body_names_from_model(model)
    joint_names, joint_qpos_addrs, joint_dof_addrs = _hinge_joint_info(model)

    source_fps = float(manifest.get("fps", 0.0))
    if source_fps <= 0.0:
        timestep = float(manifest.get("timestep", 0.0))
        if timestep <= 0.0:
            raise ValueError("any4hdmi manifest must contain fps or timestep")
        source_fps = 1.0 / timestep

    motions = []
    total_length = 0
    for motion_path in tqdm(motion_paths, desc="Loading qpos motions"):
        payload = np.load(motion_path, allow_pickle=False)
        qpos = np.asarray(payload["qpos"], dtype=np.float32)
        qvel = _compute_qvel(model, qpos, source_fps)
        motion = _run_fk(
            model,
            data,
            qpos,
            qvel,
            joint_qpos_addrs=joint_qpos_addrs,
            joint_dof_addrs=joint_dof_addrs,
        )
        motion = interpolate(motion, source_fps=int(round(source_fps)), target_fps=target_fps)
        total_length += int(motion["body_pos_w"].shape[0])
        motions.append(motion)

    td = TensorDict({}, batch_size=[total_length])
    td["motion_id"] = torch.empty(total_length, dtype=torch.int32)
    td["step"] = torch.empty(total_length, dtype=torch.int32)
    td["body_pos_w"] = torch.empty(total_length, len(body_names), 3, dtype=torch.float32)
    td["body_lin_vel_w"] = torch.empty(total_length, len(body_names), 3, dtype=torch.float32)
    td["body_quat_w"] = torch.empty(total_length, len(body_names), 4, dtype=torch.float32)
    td["body_ang_vel_w"] = torch.empty(total_length, len(body_names), 3, dtype=torch.float32)
    td["joint_pos"] = torch.empty(total_length, len(joint_names), dtype=torch.float32)
    td["joint_vel"] = torch.empty(total_length, len(joint_names), dtype=torch.float32)

    starts = []
    ends = []
    start_idx = 0
    for motion_idx, motion in enumerate(motions):
        motion_length = int(motion["body_pos_w"].shape[0])
        end_idx = start_idx + motion_length
        td["motion_id"][start_idx:end_idx] = motion_idx
        td["step"][start_idx:end_idx] = torch.arange(motion_length, dtype=torch.int32)
        td["body_pos_w"][start_idx:end_idx] = torch.from_numpy(motion["body_pos_w"])
        td["body_lin_vel_w"][start_idx:end_idx] = torch.from_numpy(motion["body_lin_vel_w"])
        td["body_quat_w"][start_idx:end_idx] = torch.from_numpy(motion["body_quat_w"])
        td["body_ang_vel_w"][start_idx:end_idx] = torch.from_numpy(motion["body_ang_vel_w"])
        td["joint_pos"][start_idx:end_idx] = torch.from_numpy(motion["joint_pos"])
        td["joint_vel"][start_idx:end_idx] = torch.from_numpy(motion["joint_vel"])
        starts.append(start_idx)
        ends.append(end_idx)
        start_idx = end_idx

    td_dir = cache_entry_dir / "td"
    td.memmap(prefix=str(td_dir))

    index_payload = {
        "body_names": body_names,
        "joint_names": joint_names,
        "starts": starts,
        "ends": ends,
        "motion_paths": [str(path) for path in motion_paths],
        "source_fps": float(source_fps),
        "target_fps": int(target_fps),
        "total_length": total_length,
    }
    (cache_entry_dir / QPOS_CACHE_INDEX_NAME).write_text(
        json.dumps(index_payload, indent=2),
        encoding="utf-8",
    )
    cache_meta = {
        "cache_version": QPOS_CACHE_VERSION,
        "dataset_root": str(dataset_root),
        "manifest_path": str((dataset_root / ANY4HDMI_MANIFEST_NAME).resolve()),
        "mjcf_path": str(mjcf_path),
        "target_fps": int(target_fps),
    }
    (cache_entry_dir / QPOS_CACHE_META_NAME).write_text(
        json.dumps(cache_meta, indent=2),
        encoding="utf-8",
    )
    (cache_entry_dir / QPOS_CACHE_READY_NAME).write_text("ready\n", encoding="utf-8")


def _load_qpos_cache_entry(
    cls,
    *,
    cache_entry_dir: Path,
    motion_paths: list[Path] | None,
    asset_joint_names: List[str] | None,
):
    td = TensorDict.load_memmap(cache_entry_dir / "td")
    index_payload = json.loads((cache_entry_dir / QPOS_CACHE_INDEX_NAME).read_text())
    joint_names = list(index_payload["joint_names"])
    joint_pos = td["joint_pos"]
    joint_vel = td["joint_vel"]
    if asset_joint_names is not None:
        asset_joint_names_list = list(asset_joint_names)
        if joint_names != asset_joint_names_list:
            share_joint_names = [
                name for name in joint_names if name in asset_joint_names_list
            ]
            src_joint_indices = [joint_names.index(name) for name in share_joint_names]
            dst_joint_indices = [
                asset_joint_names_list.index(name) for name in share_joint_names
            ]
            more_joint_names = [
                name for name in joint_names if name not in asset_joint_names_list
            ]
            src_joint_indices.extend(joint_names.index(name) for name in more_joint_names)
            dst_joint_indices.extend(
                len(asset_joint_names_list) + i for i in range(len(more_joint_names))
            )
            remapped_joint_names = asset_joint_names_list + more_joint_names
            remapped_joint_pos = torch.zeros(
                joint_pos.shape[0], len(remapped_joint_names), dtype=joint_pos.dtype
            )
            remapped_joint_vel = torch.zeros(
                joint_vel.shape[0], len(remapped_joint_names), dtype=joint_vel.dtype
            )
            remapped_joint_pos[:, dst_joint_indices] = joint_pos[:, src_joint_indices]
            remapped_joint_vel[:, dst_joint_indices] = joint_vel[:, src_joint_indices]
            joint_names = remapped_joint_names
            joint_pos = remapped_joint_pos
            joint_vel = remapped_joint_vel

    data = _build_motion_data_from_fields(
        motion_id=td["motion_id"],
        step=td["step"],
        body_pos_w=td["body_pos_w"],
        body_lin_vel_w=td["body_lin_vel_w"],
        body_quat_w=td["body_quat_w"],
        body_ang_vel_w=td["body_ang_vel_w"],
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    starts = list(index_payload["starts"])
    ends = list(index_payload["ends"])
    if motion_paths is None:
        motion_paths = [Path(path) for path in index_payload["motion_paths"]]
    return cls(
        body_names=index_payload["body_names"],
        joint_names=joint_names,
        motion_paths=motion_paths,
        starts=starts,
        ends=ends,
        data=data,
    )


def _load_cached_qpos_dataset(
    cls,
    *,
    dataset_root: Path,
    manifest: dict,
    input_paths: list[Path],
    asset_joint_names: List[str] | None,
    target_fps: int,
    base_dir: Path,
):
    mjcf_path = _resolve_any4hdmi_mjcf_path(dataset_root, manifest)
    cache_root = _cache_root(base_dir)
    lookup_key = _make_qpos_cache_lookup_key(
        dataset_root=dataset_root,
        input_paths=input_paths,
        mjcf_path=mjcf_path,
        target_fps=target_fps,
    )
    lookup_path = _cache_lookup_path(cache_root, lookup_key)

    if lookup_path.is_file():
        lookup_payload = json.loads(lookup_path.read_text())
        cache_entry_dir = cache_root / lookup_payload["cache_key"]
        ready_flag = cache_entry_dir / QPOS_CACHE_READY_NAME
        if ready_flag.is_file():
            return _load_qpos_cache_entry(
                cls,
                cache_entry_dir=cache_entry_dir,
                motion_paths=None,
                asset_joint_names=asset_joint_names,
            )

    dataset_root, manifest, motion_paths = _resolve_any4hdmi_motion_paths(input_paths)
    cache_key = _make_qpos_cache_key(
        dataset_root=dataset_root,
        manifest=manifest,
        motion_paths=motion_paths,
        mjcf_path=mjcf_path,
        target_fps=target_fps,
    )
    cache_entry_dir = cache_root / cache_key
    ready_flag = cache_entry_dir / QPOS_CACHE_READY_NAME
    lock_dir = cache_root / f"{cache_key}.lock"

    if not ready_flag.is_file():
        cache_root.mkdir(parents=True, exist_ok=True)
        owns_lock = _acquire_cache_lock(lock_dir, ready_flag)
        if owns_lock:
            tmp_entry_dir = cache_root / f"{cache_key}.tmp-{os.getpid()}-{time.time_ns()}"
            try:
                if tmp_entry_dir.exists():
                    shutil.rmtree(tmp_entry_dir)
                tmp_entry_dir.mkdir(parents=True, exist_ok=False)
                _build_qpos_cache(
                    dataset_root=dataset_root,
                    manifest=manifest,
                    motion_paths=motion_paths,
                    mjcf_path=mjcf_path,
                    cache_entry_dir=tmp_entry_dir,
                    target_fps=target_fps,
                )
                if cache_entry_dir.exists():
                    shutil.rmtree(tmp_entry_dir)
                else:
                    tmp_entry_dir.rename(cache_entry_dir)
            finally:
                if tmp_entry_dir.exists():
                    shutil.rmtree(tmp_entry_dir, ignore_errors=True)
                if lock_dir.exists():
                    shutil.rmtree(lock_dir, ignore_errors=True)
        elif not ready_flag.is_file():
            raise RuntimeError(f"Cache lock released but cache is not ready: {cache_entry_dir}")

    lookup_path.write_text(
        json.dumps({"cache_key": cache_key}, indent=2),
        encoding="utf-8",
    )
    return _load_qpos_cache_entry(
        cls,
        cache_entry_dir=cache_entry_dir,
        motion_paths=motion_paths,
        asset_joint_names=asset_joint_names,
    )


def _load_legacy_dataset(
    cls,
    *,
    motion_paths: list[Path],
    asset_joint_names: List[str] | None,
    target_fps: int,
    memory_mapped: bool,
):
    motion_dirs = [motion_path.parent for motion_path in motion_paths]
    print(f"Matched {len(motion_dirs)} motions under legacy motion.npz layout")
    meta = _load_legacy_meta(motion_dirs)

    motions = []
    for motion_path in tqdm(motion_paths):
        motion = dict(np.load(motion_path, allow_pickle=True))
        motion = interpolate(motion, source_fps=meta["fps"], target_fps=target_fps)
        motions.append(motion)

    joint_names = _apply_joint_mapping(motions, list(meta["joint_names"]), asset_joint_names)
    data, starts, ends = _build_motion_data(
        motions,
        body_names=list(meta["body_names"]),
        joint_names=joint_names,
        use_memory_mapped_tensor=memory_mapped,
    )
    return cls(
        body_names=list(meta["body_names"]),
        joint_names=joint_names,
        motion_paths=motion_paths,
        starts=starts,
        ends=ends,
        data=data,
    )


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
        input_paths = _resolve_input_paths(base_dir, root_path)
        is_any4hdmi = all(_find_any4hdmi_root(path) is not None for path in input_paths)
        if is_any4hdmi:
            dataset_root, manifest = _resolve_any4hdmi_dataset_context(input_paths)
            print(f"Matched any4hdmi dataset under {dataset_root}")
            return _load_cached_qpos_dataset(
                cls,
                dataset_root=dataset_root,
                manifest=manifest,
                input_paths=input_paths,
                asset_joint_names=asset_joint_names,
                target_fps=target_fps,
                base_dir=base_dir,
            )

        motion_paths = _resolve_legacy_motion_paths(input_paths)
        return _load_legacy_dataset(
            cls,
            motion_paths=motion_paths,
            asset_joint_names=asset_joint_names,
            target_fps=target_fps,
            memory_mapped=memory_mapped,
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
        idx.clamp_min_(self.starts.unsqueeze(1)[motion_ids])
        return self.data[idx]  # shape: [len(motion_ids), len(steps), ...]

    def find_joints(self, joint_names, preserve_order: bool = False):
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names, preserve_order: bool = False):
        return resolve_matching_names(body_names, self.body_names, preserve_order)
