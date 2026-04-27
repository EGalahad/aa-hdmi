from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import torch

try:
    import isaaclab.utils.string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils


ActionBoundsConfig = Mapping[str, Sequence[float]]


def default_action_bounds() -> dict[str, list[float]]:
    return {
        "waist_yaw_joint|.*_ankle_roll_joint": [-1.0, 1.0],
        "waist_roll_joint": [-2.0, 2.0],
        ".*_hip_roll_joint|.*_hip_yaw_joint|waist_pitch_joint|.*_shoulder_yaw_joint": [-3.0, 3.0],
        ".*_hip_pitch_joint|.*_elbow_joint": [-5.0, 2.0],
        ".*_knee_joint": [-3.0, 6.0],
        ".*_ankle_pitch_joint": [-3.0, 5.0],
        ".*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_wrist_roll_joint": [-5.0, 5.0],
        ".*_wrist_pitch_joint": [-8.0, 12.0],
        ".*_wrist_yaw_joint": [-12.0, 15.0],
    }


def coerce_action_bounds_config(
    action_bounds: ActionBoundsConfig | None,
    *,
    action_min: float | None = None,
    action_max: float | None = None,
) -> dict[str, list[float]]:
    if action_min is not None or action_max is not None:
        if action_min is None or action_max is None:
            raise ValueError(
                "action_min and action_max must either both be set or both be omitted."
            )
        if action_bounds is not None and dict(action_bounds) != default_action_bounds():
            raise ValueError(
                "Use either action_bounds or action_min/action_max, not both."
            )
        warnings.warn(
            "action_min/action_max are deprecated; use action_bounds instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        action_bounds = {".*": [float(action_min), float(action_max)]}

    if action_bounds is None:
        action_bounds = default_action_bounds()

    normalized: dict[str, list[float]] = {}
    for pattern, bounds in dict(action_bounds).items():
        if len(bounds) != 2:
            raise ValueError(
                f"action_bounds[{pattern!r}] must have exactly two values, got {bounds!r}."
            )
        low = float(bounds[0])
        high = float(bounds[1])
        if not high > low:
            raise ValueError(
                f"action_bounds[{pattern!r}] must satisfy max > min, got {(low, high)!r}."
            )
        normalized[str(pattern)] = [low, high]
    return normalized


def resolve_action_bounds(
    action_bounds: ActionBoundsConfig,
    joint_names: Sequence[str],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_names = list(joint_names)
    normalized = coerce_action_bounds_config(action_bounds)

    min_cfg = {pattern: low for pattern, (low, _) in normalized.items()}
    max_cfg = {pattern: high for pattern, (_, high) in normalized.items()}

    _, min_names, min_values = string_utils.resolve_matching_names_values(
        min_cfg, joint_names
    )
    _, max_names, max_values = string_utils.resolve_matching_names_values(
        max_cfg, joint_names
    )

    missing_min = [name for name in joint_names if name not in set(min_names)]
    missing_max = [name for name in joint_names if name not in set(max_names)]
    if missing_min or missing_max:
        raise ValueError(
            "action_bounds must cover every controlled joint exactly once. "
            f"Missing min for {missing_min}, missing max for {missing_max}."
        )

    if list(min_names) != joint_names or list(max_names) != joint_names:
        raise ValueError(
            "Resolved action_bounds order does not match policy joint order."
        )

    action_min = torch.tensor(min_values, device=device, dtype=torch.float32)
    action_max = torch.tensor(max_values, device=device, dtype=torch.float32)
    return action_min, action_max
