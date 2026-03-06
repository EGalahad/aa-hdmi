from hdmi.hdmi_tasks.command import (
    RobotTracking,
    _DESIRED_FRAME_COLORS,
)
from active_adaptation.envs.mdp.base import Reward as BaseReward
from active_adaptation.utils.math import (
    batchify,
    quat_rotate_inverse as quat_apply_inverse,
    quat_rotate as quat_apply,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    yaw_quat,
    matrix_from_quat,
)

from typing import List, Dict, TYPE_CHECKING
from omegaconf import DictConfig
try:
    from isaaclab.utils.string import (
        resolve_matching_names,
        resolve_matching_names_values,
    )
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import (
        resolve_matching_names,
        resolve_matching_names_values,
    )

quat_apply_inverse = batchify(quat_apply_inverse)
quat_apply = batchify(quat_apply)
quat_mul = batchify(quat_mul)

import torch

if TYPE_CHECKING:
    from mjlab.sensor import ContactSensor
    from mjlab.viewer.debug_visualizer import DebugVisualizer

TrackReward = BaseReward[RobotTracking]


def align_pos_between_roots(
    src_points: torch.Tensor,
    src_root_pos: torch.Tensor,
    src_root_quat: torch.Tensor,
    dst_root_pos: torch.Tensor,
    dst_root_quat: torch.Tensor,
) -> torch.Tensor:
    """
    Transform points expressed in the source root frame into the destination root frame,
    then return them in world coordinates.
    Shapes (required):
        src_points: [N, K, 3]
        src_root_pos / dst_root_pos: [N, 3]
        src_root_quat / dst_root_quat: [N, 4]
    """
    assert src_points.dim() == 3, "align_pos_between_roots expects src_points ndim=3"

    delta_quat = quat_mul(dst_root_quat, quat_conjugate(src_root_quat))  # [N,4]
    delta_quat = delta_quat.unsqueeze(1).expand(-1, src_points.shape[1], -1)

    aligned = dst_root_pos.unsqueeze(1) + quat_apply(
        delta_quat, src_points - src_root_pos.unsqueeze(1)
    )

    # delta_quat = quat_mul(quat_conjugate(dst_root_quat), src_root_quat)  # [N,4]
    # delta_quat = delta_quat.unsqueeze(1).expand(-1, src_points.shape[1], -1)
    # aligned = dst_root_pos.unsqueeze(1) + quat_apply_inverse(
    #     delta_quat, src_points - src_root_pos.unsqueeze(1)
    # )

    return aligned


def align_quat_between_roots(
    src_quat: torch.Tensor,
    src_root_quat: torch.Tensor,
    dst_root_quat: torch.Tensor,
) -> torch.Tensor:
    """
    Transform orientations expressed in the source root frame into the destination root frame.
    Shapes (required):
        src_quat: [N, K, 4]
        src_root_quat / dst_root_quat: [N, 4]
    """
    assert src_quat.dim() == 3, "align_quats_between_roots expects src_quat ndim=3"

    src_root = src_root_quat.unsqueeze(1).expand(-1, src_quat.shape[1], -1)
    dst_root = dst_root_quat.unsqueeze(1).expand(-1, src_quat.shape[1], -1)

    aligned = quat_mul(
        dst_root,
        quat_mul(quat_conjugate(src_root), src_quat),
    )
    return aligned


class _tracking_keypoint(TrackReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        sigma: float = 0.03,
        tolerance: float | Dict[str, float] = 0.0,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_body_names

        self.sigma = sigma
        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert (
            set(matched_names) == set(matched_names_motion) == set(matched_names_asset)
        ), "body names in motion dataset and robot not matched"
        assert set(matched_names) <= set(
            self.command_manager.tracking_body_names
        ), "Some body names in motion dataset not found in tracking body names"

        self.body_indices_motion = []
        self.body_indices_asset = []
        self.body_names = list(sorted(matched_names))
        self.num_bodies = len(self.body_names)
        for body_name in self.body_names:
            body_idx_motion = self.command_manager.tracking_body_names.index(
                body_name
            )
            body_idx_asset = self.command_manager.asset.body_names.index(body_name)

            self.body_indices_motion.append(body_idx_motion)
            self.body_indices_asset.append(body_idx_asset)

        self.tolerance = torch.zeros(len(self.body_names), device=self.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = (
                resolve_matching_names_values(tolerance, self.body_names)
            )
            self.tolerance[tolerance_indices] = torch.tensor(
                tolerance_values, device=self.device
            )
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

    def compute(self):
        raise NotImplementedError


class keypoint_pos_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_pos_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]

        anchor_pos_asset = self.command_manager.robot_anchor_link_pos_w.clone()
        anchor_pos_motion = self.command_manager.ref_anchor_link_pos_w.clone()
        anchor_quat_asset = self.command_manager.robot_anchor_link_quat_w
        anchor_quat_motion = self.command_manager.ref_anchor_link_quat_w

        anchor_pos_asset[..., 2] = 0.0
        anchor_pos_motion[..., 2] = 0.0
        anchor_quat_asset = yaw_quat(anchor_quat_asset)
        anchor_quat_motion = yaw_quat(anchor_quat_motion)

        anchor_pos_asset = anchor_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        anchor_pos_motion = anchor_pos_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_asset = anchor_quat_asset.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_motion = anchor_quat_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )

        body_pos_asset_relative = quat_apply_inverse(
            anchor_quat_asset, body_pos_asset - anchor_pos_asset
        )
        body_pos_motion_relative = quat_apply_inverse(
            anchor_quat_motion, body_pos_motion - anchor_pos_motion
        )

        diff = body_pos_motion_relative - body_pos_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)

    def debug_draw(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]

        anchor_pos_asset = self.command_manager.robot_anchor_link_pos_w.clone()
        anchor_pos_motion = self.command_manager.ref_anchor_link_pos_w.clone()
        anchor_quat_asset = self.command_manager.robot_anchor_link_quat_w
        anchor_quat_motion = self.command_manager.ref_anchor_link_quat_w

        anchor_pos_asset[..., 2] = 0.0
        anchor_pos_motion[..., 2] = 0.0
        anchor_quat_asset = yaw_quat(anchor_quat_asset)
        anchor_quat_motion = yaw_quat(anchor_quat_motion)

        anchor_pos_asset = anchor_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        anchor_pos_motion = anchor_pos_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_asset = anchor_quat_asset.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_motion = anchor_quat_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )

        body_pos_asset_relative = quat_apply_inverse(
            anchor_quat_asset, body_pos_asset - anchor_pos_asset
        )
        body_pos_motion_relative = quat_apply_inverse(
            anchor_quat_motion, body_pos_motion - anchor_pos_motion
        )
        # self.env._debug_draw.vector(
        #     root_pos_asset,
        #     body_pos_asset_relative,
        #     color=(0.0, 1.0, 0.0),
        #     size=4.0,
        # )
        # self.env._debug_draw.vector(
        #     root_pos_motion,
        #     body_pos_motion_relative,
        #     color=(1.0, 0.0, 0.0),
        #     size=4.0,
        # )
        # self.env.debug_draw.point(
        #     body_pos_asset_relative.reshape(-1, 3),
        #     color=(0.0, 1.0, 0.0, 1.0),
        #     size=20,
        # )
        # self.env.debug_draw.point(
        #     body_pos_motion_relative.reshape(-1, 3),
        #     color=(1.0, 0.0, 0.0, 1.0),
        #     size=20,
        # )


class keypoint_pos_error(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_pos_error_local(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]

        anchor_pos_asset = self.command_manager.robot_anchor_link_pos_w.clone()
        anchor_pos_motion = self.command_manager.ref_anchor_link_pos_w.clone()
        anchor_quat_asset = self.command_manager.robot_anchor_link_quat_w
        anchor_quat_motion = self.command_manager.ref_anchor_link_quat_w

        anchor_pos_asset[..., 2] = 0.0
        anchor_pos_motion[..., 2] = 0.0
        anchor_quat_asset = yaw_quat(anchor_quat_asset)
        anchor_quat_motion = yaw_quat(anchor_quat_motion)

        anchor_pos_asset = anchor_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        anchor_pos_motion = anchor_pos_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_asset = anchor_quat_asset.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_motion = anchor_quat_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )

        body_pos_asset_relative = quat_apply_inverse(
            anchor_quat_asset, body_pos_asset - anchor_pos_asset
        )
        body_pos_motion_relative = quat_apply_inverse(
            anchor_quat_motion, body_pos_motion - anchor_pos_motion
        )

        diff = body_pos_motion_relative - body_pos_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[
            :, self.body_indices_asset
        ]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ori_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[
            :, self.body_indices_asset
        ]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]

        anchor_quat_asset = self.command_manager.robot_anchor_link_quat_w
        anchor_quat_motion = self.command_manager.ref_anchor_link_quat_w

        anchor_quat_asset = yaw_quat(anchor_quat_asset)
        anchor_quat_motion = yaw_quat(anchor_quat_motion)

        anchor_quat_asset = anchor_quat_asset.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_motion = anchor_quat_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )

        body_ori_asset_relative = quat_mul(
            quat_conjugate(anchor_quat_asset), body_ori_asset
        )
        body_ori_motion_relative = quat_mul(
            quat_conjugate(anchor_quat_motion), body_ori_motion
        )

        diff = quat_mul(
            quat_conjugate(body_ori_motion_relative), body_ori_asset_relative
        )
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ori_error(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[
            :, self.body_indices_asset
        ]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_error_local(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[
            :, self.body_indices_asset
        ]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]

        anchor_quat_asset = self.command_manager.robot_anchor_link_quat_w
        anchor_quat_motion = self.command_manager.ref_anchor_link_quat_w

        anchor_quat_asset = yaw_quat(anchor_quat_asset)
        anchor_quat_motion = yaw_quat(anchor_quat_motion)

        anchor_quat_asset = anchor_quat_asset.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )
        anchor_quat_motion = anchor_quat_motion.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )

        body_ori_asset_relative = quat_mul(
            quat_conjugate(anchor_quat_asset), body_ori_asset
        )
        body_ori_motion_relative = quat_mul(
            quat_conjugate(anchor_quat_motion), body_ori_motion
        )

        diff = quat_mul(
            quat_conjugate(body_ori_motion_relative), body_ori_asset_relative
        )
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_lin_vel_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[
            :, self.body_indices_asset
        ]
        body_lin_vel_motion = self.command_manager.ref_body_com_lin_vel_w[
            :, self.body_indices_motion
        ]
        diff = body_lin_vel_motion - body_lin_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ang_vel_tracking_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_ang_vel_asset = self.command_manager.asset.data.body_com_ang_vel_w[
            :, self.body_indices_asset
        ]
        body_ang_vel_motion = self.command_manager.ref_body_com_ang_vel_w[
            :, self.body_indices_motion
        ]
        diff = body_ang_vel_motion - body_ang_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_lin_vel_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[
            :, self.body_indices_asset
        ]
        body_lin_vel_motion = self.command_manager.ref_body_com_lin_vel_w[
            :, self.body_indices_motion
        ]

        body_lin_vel_asset_local = quat_apply_inverse(
            self.command_manager.robot_anchor_link_quat_w.unsqueeze(1),
            body_lin_vel_asset,
        )
        body_lin_vel_motion_local = quat_apply_inverse(
            self.command_manager.ref_anchor_link_quat_w.unsqueeze(1),
            body_lin_vel_motion,
        )

        diff = body_lin_vel_motion_local - body_lin_vel_asset_local
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class keypoint_ang_vel_tracking_local_product(_tracking_keypoint, namespace="hdmi"):
    def compute(self):
        body_ang_vel_asset = self.command_manager.asset.data.body_com_ang_vel_w[
            :, self.body_indices_asset
        ]
        body_ang_vel_motion = self.command_manager.ref_body_com_ang_vel_w[
            :, self.body_indices_motion
        ]

        body_ang_vel_asset_local = quat_apply_inverse(
            self.command_manager.robot_anchor_link_quat_w.unsqueeze(1),
            body_ang_vel_asset,
        )
        body_ang_vel_motion_local = quat_apply_inverse(
            self.command_manager.ref_anchor_link_quat_w.unsqueeze(1),
            body_ang_vel_motion,
        )

        diff = body_ang_vel_motion_local - body_ang_vel_asset_local
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class _tracking_joint(TrackReward, namespace="hdmi"):
    def __init__(
        self,
        env,
        joint_names: List[str] | str | None = None,
        sigma: float = 0.03,
        tolerance: float | Dict[str, float] = 0.0,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names

        self.sigma = sigma
        joint_indices_asset, matched_names_asset = resolve_matching_names(
            joint_names, self.command_manager.asset.joint_names
        )
        joint_indices_motion, matched_names_motion = resolve_matching_names(
            joint_names, self.command_manager.tracking_joint_names
        )

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert (
            set(matched_names) == set(matched_names_motion) == set(matched_names_asset)
        ), "joint names in motion dataset and robot not matched"
        assert set(matched_names) <= set(self.command_manager.tracking_joint_names), (
            "Some joint names in motion dataset not found in tracking joint names"
        )

        self.joint_indices_motion = []
        self.joint_indices_asset = []
        self.joint_names = list(sorted(matched_names))
        for joint_name in self.joint_names:
            joint_idx_motion = self.command_manager.tracking_joint_names.index(
                joint_name
            )
            joint_idx_asset = self.command_manager.asset.joint_names.index(joint_name)

            self.joint_indices_motion.append(joint_idx_motion)
            self.joint_indices_asset.append(joint_idx_asset)

        self.tolerance = torch.zeros(len(self.joint_names), device=self.env.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = (
                resolve_matching_names_values(tolerance, matched_names_motion)
            )
            self.tolerance[tolerance_indices] = torch.tensor(
                tolerance_values, device=self.env.device
            )
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")


class joint_pos_tracking_product(_tracking_joint, namespace="hdmi"):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[
            :, self.joint_indices_asset
        ]
        joint_pos_motion = self.command_manager.ref_joint_pos[
            :, self.joint_indices_motion
        ]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_joints]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class joint_pos_error(_tracking_joint, namespace="hdmi"):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[
            :, self.joint_indices_asset
        ]
        joint_pos_motion = self.command_manager.ref_joint_pos[
            :, self.joint_indices_motion
        ]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)


class joint_vel_tracking_product(_tracking_joint, namespace="hdmi"):
    def compute(self):
        joint_vel_asset = self.command_manager.asset.data.joint_vel[
            :, self.joint_indices_asset
        ]
        joint_vel_motion = self.command_manager.ref_joint_vel[
            :, self.joint_indices_motion
        ]
        diff = joint_vel_motion - joint_vel_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_joints]
        return torch.exp(-error.mean(dim=1) / self.sigma).unsqueeze(1)


class feet_air_time_ref(TrackReward, namespace="hdmi"):
    def __init__(self, env, body_names: List[str] | str, thres: float, **kwargs):
        super().__init__(env, **kwargs)
        self.thres = thres
        self.asset = self.command_manager.asset
        self.contact_sensor: "ContactSensor" = self.env.scene["feet_ground_contact"]

        # map body indices in motion & asset space
        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        matched_names = sorted(set(matched_names_motion) & set(matched_names_asset))
        assert matched_names, "feet_air_time_ref: no feet matched"
        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(n) for n in matched_names
        ]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(n) for n in matched_names
        ]
        sensor_ids, _ = self.contact_sensor.find_bodies(matched_names)
        self.sensor_body_ids = torch.tensor(sensor_ids, device=self.device)

        num_bodies = len(matched_names)
        self.reward_time = torch.zeros(self.num_envs, num_bodies, device=self.device)
        self.last_contact = torch.zeros(
            self.num_envs, num_bodies, dtype=bool, device=self.device
        )

        # height-dependent scaling
        self.h_low, self.h_high = 0.035, 0.12
        self.c_low, self.c_high = 0.5, 2.0
        self.exp_log_c_ratio = torch.log(
            torch.tensor(self.c_high / self.c_low, device=self.device)
        )

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.last_contact[env_ids] = False

    def compute(self):
        # current contact from sensor
        current_contact = (
            self.contact_sensor.data.current_contact_time[:, self.sensor_body_ids] > 0.0
        )
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact

        # reference stance: slow & low feet in the reference motion
        ref_vel = self.command_manager.ref_body_com_lin_vel_w[
            :, self.body_indices_motion
        ]
        ref_pos = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]
        ref_feet_standing = (ref_vel.norm(dim=-1) < 0.2) & (ref_pos[..., 2] < 0.15)

        # height-based scaling using current robot foot height
        feet_height = self.asset.data.body_link_pos_w[:, self.body_indices_asset, 2]
        t = (feet_height - self.h_low) / (self.h_high - self.h_low)
        t = torch.clamp(t, 0.0, 1.0)
        feet_height_coef = self.c_low * torch.exp(self.exp_log_c_ratio * t)

        contact_diff = ref_feet_standing ^ current_contact
        self.reward_time = self.reward_time + torch.where(
            contact_diff, -self.env.step_dt, self.env.step_dt * feet_height_coef
        )

        reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * first_contact,
            dim=1,
            keepdim=True,
        )

        # reset timer for feet that are on ground
        self.reward_time = self.reward_time * (~current_contact)
        return reward


# --------------------------------------------------------------------------- #
# Keypoint position tracking with root alignment & look-ahead buffer
# --------------------------------------------------------------------------- #
class keypoint_pos_tracking_aligned(TrackReward, namespace="hdmi"):
    """
    Align the reference motion root with the current robot root in xy + yaw and
    maintain a look-ahead buffer. Aligned future roots are pushed into the buffer,
    and the current step uses entry 0 in the buffer as the target.
    """

    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        sigma: float = 0.3,
        look_ahead: int = 50,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.sigma = sigma
        self.look_ahead = int(look_ahead)

        if body_names is None:
            body_names = self.command_manager.tracking_body_names

        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        matched_names = sorted(set(matched_names_motion) & set(matched_names_asset))
        assert matched_names, "keypoint_pos_tracking_aligned: no body matched"

        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(n) for n in matched_names
        ]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(n) for n in matched_names
        ]
        self.body_names = matched_names
        self.num_bodies = len(self.body_indices_asset)

        # buffers
        self.look_ahead_idx = torch.tensor(
            [self.look_ahead - 1], device=self.device, dtype=torch.long
        )
        self.look_ahead_indices = torch.arange(self.look_ahead, device=self.device)
        self.root_pos_buf = torch.zeros(
            self.num_envs, self.look_ahead, 3, device=self.device
        )
        self.root_quat_buf = torch.zeros(
            self.num_envs, self.look_ahead, 4, device=self.device
        )

    # --- lifecycle hooks --- #
    def reset(self, env_ids):
        future_ref_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids[env_ids],
            self.command_manager.t[env_ids],
            steps=self.look_ahead_indices,
        )
        ref_pos = future_ref_motion.body_pos_w[
            :, :, self.command_manager.root_body_idx_motion
        ] + self.command_manager.env.scene.env_origins[env_ids].unsqueeze(1)
        ref_quat = future_ref_motion.body_quat_w[
            :, :, self.command_manager.root_body_idx_motion
        ]
        self.root_pos_buf[env_ids] = ref_pos
        self.root_quat_buf[env_ids] = ref_quat

    def update(self):
        # Shift buffer left
        self.root_pos_buf[:, :-1] = self.root_pos_buf[:, 1:].clone()
        self.root_quat_buf[:, :-1] = self.root_quat_buf[:, 1:].clone()

        # Align current reference to robot (xy + yaw) and append new look-ahead frame
        ref_pos_t = self.command_manager.ref_root_link_pos_w.clone()  # [N,3]
        ref_quat_t = self.command_manager.ref_root_link_quat_w  # [N,4]
        robot_pos_t = self.command_manager.robot_root_link_pos_w.clone()  # [N,3]
        robot_quat_t = self.command_manager.robot_root_link_quat_w  # [N,4]

        ref_pos_t[:, 2] = 0.0
        robot_pos_t[:, 2] = 0.0
        ref_quat_t = yaw_quat(ref_quat_t)
        robot_quat_t = yaw_quat(robot_quat_t)

        look_ahead_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids,
            self.command_manager.t,
            steps=self.look_ahead_idx,
        )
        ref_pos_look_ahead = (
            look_ahead_motion.body_pos_w[
                :, 0, self.command_manager.root_body_idx_motion
            ]
            + self.command_manager.env.scene.env_origins
        )
        ref_quat_look_ahead = look_ahead_motion.body_quat_w[
            :, 0, self.command_manager.root_body_idx_motion
        ]

        aligned_root_pos_w = align_pos_between_roots(
            ref_pos_look_ahead.unsqueeze(1),
            ref_pos_t,
            ref_quat_t,
            robot_pos_t,
            robot_quat_t,
        ).squeeze(1)
        aligned_root_quat_w = align_quat_between_roots(
            ref_quat_look_ahead.unsqueeze(1), ref_quat_t, robot_quat_t
        ).squeeze(1)

        self.root_pos_buf[:, -1] = aligned_root_pos_w
        self.root_quat_buf[:, -1] = aligned_root_quat_w

    # --- reward --- #
    def compute(self):
        # Target root for this step
        target_root_pos = self.root_pos_buf[:, 0]
        target_root_quat = self.root_quat_buf[:, 0]

        # Reference body positions (current frame)
        ref_body_pos = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        ref_root_pos = self.command_manager.ref_root_link_pos_w
        ref_root_quat = self.command_manager.ref_root_link_quat_w

        # Align reference bodies to target root
        aligned_body_pos_w = align_pos_between_roots(
            ref_body_pos, ref_root_pos, ref_root_quat, target_root_pos, target_root_quat
        )

        # Matched key bodies
        robot_body_pos_w = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]

        error = (aligned_body_pos_w - robot_body_pos_w).norm(dim=-1)
        return torch.exp(-error.mean(dim=1, keepdim=True) / self.sigma)

    def debug_draw(self):
        visualizer: "DebugVisualizer" = getattr(self.env, "visualizer", None)
        if visualizer is None:
            return

        env_idx = visualizer.env_idx

        # Target root for this step (aligned reference root)
        target_root_pos = self.root_pos_buf[:, 0]
        target_root_quat = self.root_quat_buf[:, 0]

        # Reference body pose in the current frame
        ref_body_pos = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        ref_body_quat = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]
        ref_root_pos = self.command_manager.ref_root_link_pos_w
        ref_root_quat = self.command_manager.ref_root_link_quat_w

        # Align reference bodies to target root
        aligned_body_pos_w = align_pos_between_roots(
            ref_body_pos, ref_root_pos, ref_root_quat, target_root_pos, target_root_quat
        )

        # aligned_body_pos_w = target_root_pos.unsqueeze(1) + \
        #     quat_apply(
        #         quat_mul(target_root_quat, quat_conjugate(ref_root_quat)).unsqueeze(1),
        #         ref_body_pos - ref_root_pos.unsqueeze(1),
        #     )

        aligned_body_quat_w = align_quat_between_roots(
            ref_body_quat, ref_root_quat, target_root_quat
        )
        aligned_body_pos_np = aligned_body_pos_w[env_idx].cpu().numpy()
        aligned_body_rotm = matrix_from_quat(aligned_body_quat_w[env_idx]).cpu().numpy()

        for i, body_name in enumerate(self.body_names):
            visualizer.add_frame(
                position=aligned_body_pos_np[i],
                rotation_matrix=aligned_body_rotm[i],
                scale=0.08,
                label=f"aligned_{body_name}",
                axis_colors=_DESIRED_FRAME_COLORS,
            )


class keypoint_ori_tracking_aligned(TrackReward, namespace="hdmi"):
    """
    Same root alignment and look-ahead mechanism as keypoint_pos_tracking_aligned,
    but compares orientations (quaternions) of key bodies and uses axis-angle norm
    as the error metric.
    """

    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        sigma: float = 0.4,
        look_ahead: int = 50,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.sigma = sigma
        self.look_ahead = max(1, int(look_ahead))

        if body_names is None:
            body_names = self.command_manager.tracking_body_names

        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        matched_names = sorted(set(matched_names_motion) & set(matched_names_asset))
        assert matched_names, "keypoint_ori_tracking_aligned: no body matched"

        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(n) for n in matched_names
        ]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(n) for n in matched_names
        ]
        self.body_names = matched_names
        self.num_bodies = len(self.body_indices_asset)

        self.look_ahead_idx = torch.tensor(
            [self.look_ahead - 1], device=self.device, dtype=torch.long
        )
        self.look_ahead_indices = torch.arange(self.look_ahead, device=self.device)
        self.root_pos_buf = torch.zeros(
            self.num_envs, self.look_ahead, 3, device=self.device
        )
        self.root_quat_buf = torch.zeros(
            self.num_envs, self.look_ahead, 4, device=self.device
        )

    def reset(self, env_ids):
        future_ref_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids[env_ids],
            self.command_manager.t[env_ids],
            steps=self.look_ahead_indices,
        )
        ref_pos = future_ref_motion.body_pos_w[
            :, :, self.command_manager.root_body_idx_motion
        ] + self.command_manager.env.scene.env_origins[env_ids].unsqueeze(1)
        ref_quat = future_ref_motion.body_quat_w[
            :, :, self.command_manager.root_body_idx_motion
        ]
        self.root_pos_buf[env_ids] = ref_pos
        self.root_quat_buf[env_ids] = ref_quat

    def update(self):
        # Shift buffer left
        self.root_pos_buf[:, :-1] = self.root_pos_buf[:, 1:].clone()
        self.root_quat_buf[:, :-1] = self.root_quat_buf[:, 1:].clone()

        # Align current reference to robot (xy + yaw) and append new look-ahead frame
        ref_pos_t = self.command_manager.ref_root_link_pos_w.clone()  # [N,3]
        ref_quat_t = self.command_manager.ref_root_link_quat_w  # [N,4]
        robot_pos_t = self.command_manager.robot_root_link_pos_w.clone()  # [N,3]
        robot_quat_t = self.command_manager.robot_root_link_quat_w  # [N,4]

        ref_pos_t[:, 2] = 0.0
        robot_pos_t[:, 2] = 0.0
        ref_quat_t = yaw_quat(ref_quat_t)
        robot_quat_t = yaw_quat(robot_quat_t)

        look_ahead_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids,
            self.command_manager.t,
            steps=self.look_ahead_idx,
        )
        ref_pos_look_ahead = (
            look_ahead_motion.body_pos_w[
                :, 0, self.command_manager.root_body_idx_motion
            ]
            + self.command_manager.env.scene.env_origins
        )
        ref_quat_look_ahead = look_ahead_motion.body_quat_w[
            :, 0, self.command_manager.root_body_idx_motion
        ]

        aligned_root_pos_w = align_pos_between_roots(
            ref_pos_look_ahead.unsqueeze(1),
            ref_pos_t,
            ref_quat_t,
            robot_pos_t,
            robot_quat_t,
        ).squeeze(1)
        aligned_root_quat_w = align_quat_between_roots(
            ref_quat_look_ahead.unsqueeze(1), ref_quat_t, robot_quat_t
        ).squeeze(1)

        self.root_pos_buf[:, -1] = aligned_root_pos_w
        self.root_quat_buf[:, -1] = aligned_root_quat_w

    def compute(self):
        target_root_quat = self.root_quat_buf[:, 0]

        ref_body_quat = self.command_manager.ref_body_link_quat_w[
            :, self.body_indices_motion
        ]
        ref_root_quat = self.command_manager.ref_root_link_quat_w

        # Align reference bodies to target root
        aligned_body_quat_w = align_quat_between_roots(
            ref_body_quat, ref_root_quat, target_root_quat
        )

        robot_body_quat_w = self.command_manager.asset.data.body_link_quat_w[
            :, self.body_indices_asset
        ]
        diff = axis_angle_from_quat(
            quat_mul(quat_conjugate(aligned_body_quat_w), robot_body_quat_w)
        )
        error = diff.norm(dim=-1)
        return torch.exp(-error.mean(dim=1, keepdim=True) / self.sigma)

class keypoint_pos_tracking_pos_aligned(TrackReward, namespace="hdmi"):
    """
    Align the reference motion root with the current robot root in xy + yaw and
    maintain a look-ahead buffer. Aligned future roots are pushed into the buffer,
    and the current step uses entry 0 in the buffer as the target.
    """

    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        sigma: float = 0.3,
        look_ahead: int = 50,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.sigma = sigma
        self.look_ahead = int(look_ahead)

        if body_names is None:
            body_names = self.command_manager.tracking_body_names

        body_indices_motion, matched_names_motion = resolve_matching_names(
            body_names, self.command_manager.tracking_body_names
        )
        body_indices_asset, matched_names_asset = resolve_matching_names(
            body_names, self.command_manager.asset.body_names
        )
        matched_names = sorted(set(matched_names_motion) & set(matched_names_asset))
        assert matched_names, "keypoint_pos_tracking_aligned: no body matched"

        self.body_indices_motion = [
            self.command_manager.tracking_body_names.index(n) for n in matched_names
        ]
        self.body_indices_asset = [
            self.command_manager.asset.body_names.index(n) for n in matched_names
        ]
        self.body_names = matched_names
        self.num_bodies = len(self.body_indices_asset)

        # buffers
        self.look_ahead_idx = torch.tensor(
            [self.look_ahead - 1], device=self.device, dtype=torch.long
        )
        self.look_ahead_indices = torch.arange(self.look_ahead, device=self.device)
        self.root_pos_buf = torch.zeros(
            self.num_envs, self.look_ahead, 3, device=self.device
        )

    # --- lifecycle hooks --- #
    def reset(self, env_ids):
        future_ref_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids[env_ids],
            self.command_manager.t[env_ids],
            steps=self.look_ahead_indices,
        )
        ref_pos = future_ref_motion.body_pos_w[
            :, :, self.command_manager.root_body_idx_motion
        ] + self.command_manager.env.scene.env_origins[env_ids].unsqueeze(1)
        self.root_pos_buf[env_ids] = ref_pos

    def update(self):
        # Shift buffer left
        self.root_pos_buf[:, :-1] = self.root_pos_buf[:, 1:].clone()

        # Align current reference to robot (xy) and append new look-ahead frame
        ref_pos_t = self.command_manager.ref_root_link_pos_w.clone()  # [N,3]
        ref_quat_t = self.command_manager.ref_root_link_quat_w  # [N,4]
        robot_pos_t = self.command_manager.robot_root_link_pos_w.clone()  # [N,3]

        ref_pos_t[:, 2] = 0.0
        robot_pos_t[:, 2] = 0.0
        ref_quat_t = yaw_quat(ref_quat_t)

        look_ahead_motion = self.command_manager.dataset.get_slice(
            self.command_manager.motion_ids,
            self.command_manager.t,
            steps=self.look_ahead_idx,
        )
        ref_pos_look_ahead = (
            look_ahead_motion.body_pos_w[
                :, 0, self.command_manager.root_body_idx_motion
            ]
            + self.command_manager.env.scene.env_origins
        )

        aligned_root_pos_w = align_pos_between_roots(
            ref_pos_look_ahead.unsqueeze(1),
            ref_pos_t,
            ref_quat_t,
            robot_pos_t,
            ref_quat_t,
        ).squeeze(1)

        self.root_pos_buf[:, -1] = aligned_root_pos_w

    # --- reward --- #
    def compute(self):
        # Target root for this step
        target_root_pos = self.root_pos_buf[:, 0]

        # Reference body positions (current frame)
        ref_body_pos = self.command_manager.ref_body_link_pos_w[
            :, self.body_indices_motion
        ]
        ref_root_pos = self.command_manager.ref_root_link_pos_w
        ref_root_quat = self.command_manager.ref_root_link_quat_w

        # Align reference bodies to target root
        aligned_body_pos_w = align_pos_between_roots(
            ref_body_pos, ref_root_pos, ref_root_quat, target_root_pos, ref_root_quat
        )

        # Matched key bodies
        robot_body_pos_w = self.command_manager.asset.data.body_link_pos_w[
            :, self.body_indices_asset
        ]

        error = (aligned_body_pos_w - robot_body_pos_w).norm(dim=-1)
        return torch.exp(-error.mean(dim=1, keepdim=True) / self.sigma)
