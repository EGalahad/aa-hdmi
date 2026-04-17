from __future__ import annotations

import argparse
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
from tensordict import TensorClass, MemoryMappedTensor
from typing import Any, List
from any4hdmi import BaseDataset as Any4HDMIBaseDataset, load_any4hdmi_dataset, resolve_input_paths
from any4hdmi.dataset.base import MotionSample as Any4HDMIMotionSample
from any4hdmi.dataset.loading import find_any4hdmi_root, resolve_any4hdmi_dataset_context

try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names

QPOS_CACHE_VERSION = 1
QPOS_CACHE_SUBDIR = ".cache/motion/qpos_fk"
QPOS_CACHE_INDEX_NAME = "motion_index.json"
QPOS_CACHE_META_NAME = "cache_meta.json"
QPOS_CACHE_READY_NAME = "ready.flag"
QPOS_CACHE_LOOKUP_SUBDIR = "lookup"
ANY4HDMI_MANIFEST_NAME = "manifest.json"
QPOS_FK_WARP_BATCH_SIZE = 16384
QPOS_MOTION_BATCH_MAX_MOTIONS = 128
QPOS_SCAN_MAX_WORKERS = 8
QPOS_FINGERPRINT_MAX_WORKERS = 32
MOTION_DATASET_VALIDATE_CHUNK_SIZE = 131072
MOTION_DATASET_QUAT_NORM_ATOL = 1e-3


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def lerp(ts_target, ts_source, x):
    return np.stack(
        [np.interp(ts_target, ts_source, x[:, i]) for i in range(x.shape[1])], axis=-1
    )


def _lerp_torch(ts_target: torch.Tensor, ts_source: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    right_idx = torch.searchsorted(ts_source, ts_target, right=False)
    right_idx = right_idx.clamp(1, ts_source.numel() - 1)
    left_idx = right_idx - 1

    t_left = ts_source[left_idx]
    t_right = ts_source[right_idx]
    denom = torch.where(t_right > t_left, t_right - t_left, torch.ones_like(t_right))
    alpha = ((ts_target - t_left) / denom).unsqueeze(1)

    x0 = x[left_idx]
    x1 = x[right_idx]
    return (1.0 - alpha) * x0 + alpha * x1


def slerp(ts_target, ts_source, quat):
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]
    if quat_dim != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {quat.shape}")

    steps_target = ts_target.shape[0]
    steps_source = ts_source.shape[0]

    quat = np.asarray(quat, dtype=np.float64).reshape(steps_source, -1, quat_dim)
    ts_source = np.asarray(ts_source)
    ts_target = np.asarray(ts_target)

    if steps_source == 0:
        raise ValueError("Cannot interpolate empty quaternion sequence")
    if steps_source == 1:
        out = np.broadcast_to(quat[:1], (steps_target, *quat[:1].shape[1:])).copy()
        return out.reshape(steps_target, *batch_shape, quat_dim)

    right_idx = np.searchsorted(ts_source, ts_target, side="left")
    right_idx = np.clip(right_idx, 1, steps_source - 1)
    left_idx = right_idx - 1

    t_left = ts_source[left_idx]
    t_right = ts_source[right_idx]
    denom = np.where(t_right > t_left, t_right - t_left, 1.0)
    alpha = ((ts_target - t_left) / denom).astype(np.float64)[:, None, None]

    q0 = quat[left_idx]
    q1 = quat[right_idx]

    q0 /= np.linalg.norm(q0, axis=-1, keepdims=True).clip(min=1e-12)
    q1 /= np.linalg.norm(q1, axis=-1, keepdims=True).clip(min=1e-12)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip_mask = dot < 0.0
    q1 = np.where(flip_mask, -q1, q1)
    dot = np.where(flip_mask, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha

    safe_denom = np.where(sin_theta_0 > 1e-8, sin_theta_0, 1.0)
    s0 = np.sin(theta_0 - theta) / safe_denom
    s1 = np.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    # Near-identical quaternions are cheaper and more stable with normalized lerp.
    nlerp_out = (1.0 - alpha) * q0 + alpha * q1
    use_nlerp = dot > 0.9995
    out = np.where(use_nlerp, nlerp_out, slerp_out)
    out /= np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-12)
    return out.reshape(steps_target, *batch_shape, quat_dim)


def _slerp_torch(
    ts_target: torch.Tensor,
    ts_source: torch.Tensor,
    quat: torch.Tensor,
) -> torch.Tensor:
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]
    if quat_dim != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {tuple(quat.shape)}")

    steps_target = ts_target.shape[0]
    steps_source = ts_source.shape[0]

    quat = quat.reshape(steps_source, -1, quat_dim)
    if steps_source == 0:
        raise ValueError("Cannot interpolate empty quaternion sequence")
    if steps_source == 1:
        out = quat[:1].expand(steps_target, -1, -1).clone()
        return out.reshape(steps_target, *batch_shape, quat_dim)

    right_idx = torch.searchsorted(ts_source, ts_target, right=False)
    right_idx = right_idx.clamp(1, steps_source - 1)
    left_idx = right_idx - 1

    t_left = ts_source[left_idx]
    t_right = ts_source[right_idx]
    denom = torch.where(t_right > t_left, t_right - t_left, torch.ones_like(t_right))
    alpha = ((ts_target - t_left) / denom).view(-1, 1, 1)

    q0 = quat[left_idx]
    q1 = quat[right_idx]

    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    flip_mask = dot < 0.0
    q1 = torch.where(flip_mask, -q1, q1)
    dot = torch.where(flip_mask, -dot, dot).clamp(-1.0, 1.0)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * alpha

    safe_denom = torch.where(sin_theta_0 > 1e-8, sin_theta_0, torch.ones_like(sin_theta_0))
    s0 = torch.sin(theta_0 - theta) / safe_denom
    s1 = torch.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha) * q0 + alpha * q1
    out = torch.where(dot > 0.9995, nlerp_out, slerp_out)
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.reshape(steps_target, *batch_shape, quat_dim)


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
        joint_pos = motion["joint_pos"]
        if isinstance(joint_pos, torch.Tensor):
            device = joint_pos.device
            dtype = joint_pos.dtype
            ts_source = torch.arange(
                0, (T - 1) * target_fps + 1, target_fps, device=device, dtype=dtype
            )
            ts_target = torch.arange(
                0, (T - 1) * target_fps + 1, source_fps, device=device, dtype=dtype
            )
            motion["body_pos_w"] = _lerp_torch(
                ts_target, ts_source, motion["body_pos_w"].reshape(T, -1)
            ).reshape(len(ts_target), -1, 3)
            motion["body_lin_vel_w"] = _lerp_torch(
                ts_target, ts_source, motion["body_lin_vel_w"].reshape(T, -1)
            ).reshape(len(ts_target), -1, 3)
            motion["body_quat_w"] = _slerp_torch(
                ts_target, ts_source, motion["body_quat_w"]
            )
            motion["body_ang_vel_w"] = _lerp_torch(
                ts_target, ts_source, motion["body_ang_vel_w"].reshape(T, -1)
            ).reshape(len(ts_target), -1, 3)
            motion["joint_pos"] = _lerp_torch(ts_target, ts_source, motion["joint_pos"])
            motion["joint_vel"] = _lerp_torch(ts_target, ts_source, motion["joint_vel"])
        else:
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
        device=torch.device("cpu"),
        batch_size=[total_length],
    )
    return data, starts, ends


def _load_legacy_dataset(
    *,
    motion_paths: list[Path],
    target_fps: int,
    memory_mapped: bool,
    num_envs: int,
):
    motion_dirs = [motion_path.parent for motion_path in motion_paths]
    print(f"Matched {len(motion_dirs)} motions under legacy motion.npz layout")
    meta = _load_legacy_meta(motion_dirs)

    motions = []
    for motion_path in tqdm(motion_paths):
        motion = dict(np.load(motion_path, allow_pickle=True))
        motion = interpolate(motion, source_fps=meta["fps"], target_fps=target_fps)
        motions.append(motion)

    joint_names = list(meta["joint_names"])
    data, starts, ends = _build_motion_data(
        motions,
        body_names=list(meta["body_names"]),
        joint_names=joint_names,
        use_memory_mapped_tensor=memory_mapped,
    )
    return LegacyMotionDataset(
        body_names=list(meta["body_names"]),
        joint_names=joint_names,
        motion_paths=motion_paths,
        starts=starts,
        ends=ends,
        data=data,
        num_envs=num_envs,
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


class LegacyMotionDataset(Any4HDMIBaseDataset):
    def __init__(
        self,
        body_names: List[str],
        joint_names: List[str],
        motion_paths: List[Path],
        starts: List[int],
        ends: List[int],
        data: Any,
        num_envs: int,
    ):
        self.body_names = body_names
        self.joint_names = joint_names
        self.motion_paths = motion_paths
        self.data = data
        self.device = data.device
        self.starts = torch.as_tensor(starts, device=self.device, dtype=torch.long)
        self.ends = torch.as_tensor(ends, device=self.device, dtype=torch.long)
        self.lengths = self.ends - self.starts
        self._num_envs = int(num_envs)
        self._env_motion_id = torch.full((self._num_envs,), -1, device=self.device, dtype=torch.long)
        self._env_motion_len = torch.zeros((self._num_envs,), device=self.device, dtype=torch.long)

    def to(self, device: torch.device | str):
        target_device = torch.device(device)
        self.data = self.data.to(target_device)
        self.starts = self.starts.to(target_device)
        self.ends = self.ends.to(target_device)
        self.lengths = self.lengths.to(target_device)
        self._env_motion_id = self._env_motion_id.to(target_device)
        self._env_motion_len = self._env_motion_len.to(target_device)
        self.device = target_device
        return self

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
        *,
        profile_name: str | None = None,
    ) -> MotionData:
        del profile_name
        idx = (self.starts[motion_ids] + starts).unsqueeze(1) + steps.unsqueeze(0)
        idx.clamp_max_(self.ends.unsqueeze(1)[motion_ids] - 1)
        idx.clamp_min_(self.starts.unsqueeze(1)[motion_ids])
        return self.data[idx]

    def sample_motion(
        self,
        env_ids: torch.Tensor,
        *,
        terminated_t: torch.Tensor,
        rewind_mask: torch.Tensor,
        rewind_steps: torch.Tensor,
    ) -> Any4HDMIMotionSample:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        terminated_t = terminated_t.to(device=self.device, dtype=torch.long)
        rewind_mask = rewind_mask.to(device=self.device, dtype=torch.bool)
        rewind_steps = rewind_steps.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            empty = torch.empty((0,), device=self.device, dtype=torch.long)
            return Any4HDMIMotionSample(
                motion_id=empty,
                motion_len=empty,
                start_t=empty,
            )

        sampled_frame_ids = torch.randint(
            0,
            self.num_steps,
            size=(env_ids.numel(),),
            device=self.device,
        )
        sampled_motion_ids = self.data.motion_id[sampled_frame_ids].long()
        sampled_start_t = self.data.step[sampled_frame_ids].long()
        sampled_motion_len = self.lengths[sampled_motion_ids].long()

        if bool(torch.any(rewind_mask).item()):
            rewind_motion_ids = self._env_motion_id.index_select(0, env_ids)
            rewind_t = torch.clamp(terminated_t - rewind_steps, min=0)
            sampled_motion_ids = torch.where(rewind_mask, rewind_motion_ids, sampled_motion_ids)
            sampled_motion_len = torch.where(
                rewind_mask,
                self.lengths[rewind_motion_ids].long(),
                sampled_motion_len,
            )
            sampled_start_t = torch.where(rewind_mask, rewind_t, sampled_start_t)

        self._env_motion_id.index_copy_(0, env_ids, sampled_motion_ids)
        self._env_motion_len.index_copy_(0, env_ids, sampled_motion_len)
        return Any4HDMIMotionSample(
            motion_id=sampled_motion_ids,
            motion_len=sampled_motion_len,
            start_t=sampled_start_t,
        )

    def find_joints(self, joint_names, preserve_order: bool = False):
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names, preserve_order: bool = False):
        return resolve_matching_names(body_names, self.body_names, preserve_order)


def create_dataset_from_path(
    root_path: str | List[str],
    target_fps: int = 50,
    memory_mapped: bool = False,
    num_envs: int = 1,
    full_motion: bool = True,
) -> Any4HDMIBaseDataset:
    import active_adaptation

    base_dir = Path(active_adaptation.__file__).parent.parent
    input_paths = resolve_input_paths(base_dir, root_path)
    is_any4hdmi = all(find_any4hdmi_root(path) is not None for path in input_paths)
    if is_any4hdmi:
        dataset_root, _ = resolve_any4hdmi_dataset_context(input_paths)
        print(f"Matched any4hdmi dataset under {dataset_root}")
        any4hdmi_dataset = load_any4hdmi_dataset(
            input_paths=input_paths,
            target_fps=target_fps,
            base_dir=base_dir,
            num_envs=num_envs,
            full_motion=full_motion,
        )
        if not isinstance(any4hdmi_dataset, Any4HDMIBaseDataset):
            raise TypeError(
                f"Expected any4hdmi BaseDataset-compatible loader, got {type(any4hdmi_dataset)!r}"
            )
        return any4hdmi_dataset

    motion_paths = _resolve_legacy_motion_paths(input_paths)
    return _load_legacy_dataset(
        motion_paths=motion_paths,
        target_fps=target_fps,
        memory_mapped=memory_mapped,
        num_envs=num_envs,
    )


def _motion_dataset_total_steps(dataset: Any4HDMIBaseDataset) -> int:
    ends = dataset.ends
    if not isinstance(ends, torch.Tensor):
        raise TypeError(f"Expected dataset.ends to be a torch.Tensor, got {type(ends)!r}")
    if ends.numel() == 0:
        return 0
    return int(ends.max().item())


def _motion_dataset_field_chunk(
    dataset: Any4HDMIBaseDataset,
    *,
    field_name: str,
    start: int,
    end: int,
) -> torch.Tensor:
    if hasattr(dataset, "_storage_cpu"):
        storage = getattr(dataset, "_storage_cpu")
        if field_name not in storage:
            raise KeyError(f"Dataset storage does not contain field {field_name!r}")
        field = storage[field_name][start:end]
    elif hasattr(dataset, "data") and hasattr(dataset.data, field_name):
        field = getattr(dataset.data, field_name)[start:end]
    else:
        raise AttributeError(f"Dataset does not expose field {field_name!r} for validation")
    if not isinstance(field, torch.Tensor):
        field = torch.as_tensor(field)
    return field.detach().to(device="cpu")


def _max_finite_or_nan(x: torch.Tensor) -> float:
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.max().item())


def _validate_motion_dataset(dataset: Any4HDMIBaseDataset) -> None:
    total_steps = _motion_dataset_total_steps(dataset)
    if total_steps == 0:
        print("Motion dataset validation skipped: dataset is empty")
        return

    chunk_size = _env_int("HDMI_MOTION_VALIDATE_CHUNK_SIZE", MOTION_DATASET_VALIDATE_CHUNK_SIZE)
    quat_norm_atol = float(os.environ.get("HDMI_MOTION_VALIDATE_QUAT_NORM_ATOL", MOTION_DATASET_QUAT_NORM_ATOL))
    violation_counts = {
        "joint_pos_abs": 0,
        "joint_vel_abs": 0,
        "body_pos_z": 0,
        "body_lin_vel_norm": 0,
        "body_ang_vel_norm": 0,
        "body_quat_norm": 0,
        "any": 0,
    }
    max_stats = {
        "joint_pos_abs": float("-inf"),
        "joint_vel_abs": float("-inf"),
        "body_pos_z": float("-inf"),
        "body_lin_vel_norm": float("-inf"),
        "body_ang_vel_norm": float("-inf"),
        "body_quat_norm_error": float("-inf"),
    }

    chunk_starts = range(0, total_steps, chunk_size)
    for start in tqdm(chunk_starts, total=(total_steps + chunk_size - 1) // chunk_size, desc="Validating motion dataset", unit="chunk"):
        end = min(total_steps, start + chunk_size)
        joint_pos = _motion_dataset_field_chunk(dataset, field_name="joint_pos", start=start, end=end)
        joint_vel = _motion_dataset_field_chunk(dataset, field_name="joint_vel", start=start, end=end)
        body_pos_w = _motion_dataset_field_chunk(dataset, field_name="body_pos_w", start=start, end=end)
        body_lin_vel_w = _motion_dataset_field_chunk(dataset, field_name="body_lin_vel_w", start=start, end=end)
        body_ang_vel_w = _motion_dataset_field_chunk(dataset, field_name="body_ang_vel_w", start=start, end=end)
        body_quat_w = _motion_dataset_field_chunk(dataset, field_name="body_quat_w", start=start, end=end)

        joint_pos_abs = joint_pos.abs()
        joint_vel_abs = joint_vel.abs()
        body_pos_z = body_pos_w[..., 2]
        body_lin_vel_norm = torch.linalg.vector_norm(body_lin_vel_w, dim=-1)
        body_ang_vel_norm = torch.linalg.vector_norm(body_ang_vel_w, dim=-1)
        body_quat_norm = torch.linalg.vector_norm(body_quat_w, dim=-1)
        body_quat_norm_error = (body_quat_norm - 1.0).abs()

        joint_pos_bad = (~torch.isfinite(joint_pos_abs)) | (joint_pos_abs >= 3.0)
        joint_vel_bad = (~torch.isfinite(joint_vel_abs)) | (joint_vel_abs >= 20.0)
        body_pos_z_bad = (~torch.isfinite(body_pos_z)) | (body_pos_z >= 2.5)
        body_lin_vel_bad = (~torch.isfinite(body_lin_vel_norm)) | (body_lin_vel_norm >= 10.0)
        body_ang_vel_bad = (~torch.isfinite(body_ang_vel_norm)) | (body_ang_vel_norm >= 30.0)
        body_quat_bad = (~torch.isfinite(body_quat_norm_error)) | (body_quat_norm_error > quat_norm_atol)

        joint_pos_bad_frames = joint_pos_bad.any(dim=1)
        joint_vel_bad_frames = joint_vel_bad.any(dim=1)
        body_pos_z_bad_frames = body_pos_z_bad.any(dim=1)
        body_lin_vel_bad_frames = body_lin_vel_bad.any(dim=1)
        body_ang_vel_bad_frames = body_ang_vel_bad.any(dim=1)
        body_quat_bad_frames = body_quat_bad.any(dim=1)
        any_bad_frames = (
            joint_pos_bad_frames
            | joint_vel_bad_frames
            | body_pos_z_bad_frames
            | body_lin_vel_bad_frames
            | body_ang_vel_bad_frames
            | body_quat_bad_frames
        )

        violation_counts["joint_pos_abs"] += int(joint_pos_bad_frames.sum().item())
        violation_counts["joint_vel_abs"] += int(joint_vel_bad_frames.sum().item())
        violation_counts["body_pos_z"] += int(body_pos_z_bad_frames.sum().item())
        violation_counts["body_lin_vel_norm"] += int(body_lin_vel_bad_frames.sum().item())
        violation_counts["body_ang_vel_norm"] += int(body_ang_vel_bad_frames.sum().item())
        violation_counts["body_quat_norm"] += int(body_quat_bad_frames.sum().item())
        violation_counts["any"] += int(any_bad_frames.sum().item())

        max_stats["joint_pos_abs"] = max(max_stats["joint_pos_abs"], _max_finite_or_nan(joint_pos_abs))
        max_stats["joint_vel_abs"] = max(max_stats["joint_vel_abs"], _max_finite_or_nan(joint_vel_abs))
        max_stats["body_pos_z"] = max(max_stats["body_pos_z"], _max_finite_or_nan(body_pos_z))
        max_stats["body_lin_vel_norm"] = max(
            max_stats["body_lin_vel_norm"],
            _max_finite_or_nan(body_lin_vel_norm),
        )
        max_stats["body_ang_vel_norm"] = max(
            max_stats["body_ang_vel_norm"],
            _max_finite_or_nan(body_ang_vel_norm),
        )
        max_stats["body_quat_norm_error"] = max(
            max_stats["body_quat_norm_error"],
            _max_finite_or_nan(body_quat_norm_error),
        )

    if violation_counts["any"] > 0:
        raise RuntimeError(
            "Motion dataset validation failed: "
            f"{violation_counts['any']} invalid frames out of {total_steps}. "
            f"joint_pos abs < 2 violated on {violation_counts['joint_pos_abs']} frames (max={max_stats['joint_pos_abs']:.6g}); "
            f"joint_vel abs < 20 violated on {violation_counts['joint_vel_abs']} frames (max={max_stats['joint_vel_abs']:.6g}); "
            f"body_pos[..., 2] < 2.5 violated on {violation_counts['body_pos_z']} frames (max={max_stats['body_pos_z']:.6g}); "
            f"body_lin_vel norm < 10 violated on {violation_counts['body_lin_vel_norm']} frames (max={max_stats['body_lin_vel_norm']:.6g}); "
            f"body_ang_vel norm < 10 violated on {violation_counts['body_ang_vel_norm']} frames (max={max_stats['body_ang_vel_norm']:.6g}); "
            f"body_quat norm ~= 1 violated on {violation_counts['body_quat_norm']} frames "
            f"(max error={max_stats['body_quat_norm_error']:.6g}, atol={quat_norm_atol:.6g})"
        )

    print(
        "Motion dataset validation passed:",
        f"frames={total_steps}",
        f"max_joint_pos_abs={max_stats['joint_pos_abs']:.6g}",
        f"max_joint_vel_abs={max_stats['joint_vel_abs']:.6g}",
        f"max_body_pos_z={max_stats['body_pos_z']:.6g}",
        f"max_body_lin_vel_norm={max_stats['body_lin_vel_norm']:.6g}",
        f"max_body_ang_vel_norm={max_stats['body_ang_vel_norm']:.6g}",
        f"max_body_quat_norm_error={max_stats['body_quat_norm_error']:.6g}",
        f"quat_norm_atol={quat_norm_atol:.6g}",
    )


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a motion dataset and materialize the qpos cache when applicable."
    )
    parser.add_argument(
        "dataset_path",
        nargs="+",
        help="Dataset root, subdirectory, or motion file path to load.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=50,
        help="Target FPS used for interpolation and qpos cache keys.",
    )
    parser.add_argument(
        "--memory-mapped",
        action="store_true",
        help="Use MemoryMappedTensor for legacy motion.npz layouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    root_path: str | list[str]
    if len(args.dataset_path) == 1:
        root_path = args.dataset_path[0]
    else:
        root_path = args.dataset_path

    dataset = create_dataset_from_path(
        root_path=root_path,
        target_fps=args.target_fps,
        memory_mapped=args.memory_mapped,
    )
    _validate_motion_dataset(dataset)
    print(
        "Loaded motion dataset:",
        f"num_motions={dataset.num_motions}",
        f"num_steps={dataset.num_steps}",
        f"device={dataset.device}",
    )


if __name__ == "__main__":
    main()
