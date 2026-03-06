from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
)
from active_adaptation.assets.humanoid import G1_WAIST_UNLOCKED_CFG
from active_adaptation.registry import Registry
from pathlib import Path

registry = Registry.instance()


# MJCF/USD paths.
G1_MJCF_DIR = Path(__file__).resolve().parent / "assets"
G1_USD = G1_MJCF_DIR / "g1_mjlab" / "g1_mjlab.usd"

G1_MJCF_BY_MODE = {
    5: G1_MJCF_DIR / "g1-mode_5_11.xml",
    11: G1_MJCF_DIR / "g1-mode_5_11.xml",
    13: G1_MJCF_DIR / "g1-mode_13_15.xml",
    15: G1_MJCF_DIR / "g1-mode_13_15.xml",
}


def reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    """Compute reflected inertia of a two-stage planetary gearbox.

    Formula matches mjlab.utils.actuator.reflected_inertia_from_two_stage_planetary.
    """
    assert gear_ratio[0] == 1
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    r3 = rotor_inertia[2]
    return r1 + r2 + r3


# Motor specs (from Unitree/mjlab constants + 5010 spec from mode table).
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)

ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)

ROTOR_INERTIAS_5010 = (0.084e-4, 0.015e-4, 0.068e-4)
GEARS_5010 = (1, 4, 4)

ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5020, GEARS_5020)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_14, GEARS_7520_14)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_22, GEARS_7520_22)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_4010, GEARS_4010)
ARMATURE_5010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5010, GEARS_5010)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _motor_actuator(joint_names_expr: str, effort: float, velocity: float, armature: float) -> ActuatorCfg:
    return ActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=effort,
        velocity_limit=velocity,
        stiffness=_stiffness(armature),
        damping=_damping(armature),
        friction=0.01,
        armature=armature,
    )


# Shared actuator groups.
G1_ACTUATOR_5020_UPPER = _motor_actuator(
    joint_names_expr=(
        ".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|"
        ".*_shoulder_yaw_joint|.*_wrist_roll_joint"
    ),
    effort=25.0,
    velocity=37.0,
    armature=ARMATURE_5020,
)

G1_ACTUATOR_HIP_PITCH_7520_14 = _motor_actuator(
    joint_names_expr=".*_hip_pitch_joint",
    effort=88.0,
    velocity=32.0,
    armature=ARMATURE_7520_14,
)

G1_ACTUATOR_HIP_PITCH_7520_22 = _motor_actuator(
    joint_names_expr=".*_hip_pitch_joint",
    effort=139.0,
    velocity=20.0,
    armature=ARMATURE_7520_22,
)

G1_ACTUATOR_HIP_YAW_7520_14 = _motor_actuator(
    joint_names_expr=".*_hip_yaw_joint|waist_yaw_joint",
    effort=88.0,
    velocity=32.0,
    armature=ARMATURE_7520_14,
)

G1_ACTUATOR_HIP_ROLL_KNEE_7520_22 = _motor_actuator(
    joint_names_expr=".*_hip_roll_joint|.*_knee_joint",
    effort=139.0,
    velocity=20.0,
    armature=ARMATURE_7520_22,
)

G1_ACTUATOR_WRIST_4010 = _motor_actuator(
    joint_names_expr=".*_wrist_pitch_joint|.*_wrist_yaw_joint",
    effort=5.0,
    velocity=22.0,
    armature=ARMATURE_4010,
)

G1_ACTUATOR_WRIST_5010 = _motor_actuator(
    joint_names_expr=".*_wrist_pitch_joint|.*_wrist_yaw_joint",
    effort=13.4,
    velocity=27.0,
    armature=ARMATURE_5010,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2x5020 equivalent at joint level.
G1_ACTUATOR_WAIST = _motor_actuator(
    joint_names_expr="waist_pitch_joint|waist_roll_joint",
    effort=50.0,
    velocity=37.0,
    armature=2.0 * ARMATURE_5020,
)

G1_ACTUATOR_ANKLE = _motor_actuator(
    joint_names_expr=".*_ankle_pitch_joint|.*_ankle_roll_joint",
    effort=50.0,
    velocity=37.0,
    armature=2.0 * ARMATURE_5020,
)


KNEES_BENT_KEYFRAME = InitialStateCfg(
    pos=(0.0, 0.0, 0.76),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)


def _build_g1_cfg(mode: int) -> AssetCfg:
    if mode not in (5, 11, 13, 15):
        raise ValueError(f"Unsupported mode: {mode}")

    hip_pitch_act = G1_ACTUATOR_HIP_PITCH_7520_14 if mode in (5, 13) else G1_ACTUATOR_HIP_PITCH_7520_22
    wrist_act = G1_ACTUATOR_WRIST_4010 if mode in (5, 11) else G1_ACTUATOR_WRIST_5010

    return AssetCfg(
        mjcf_path=G1_MJCF_BY_MODE[mode],
        usd_path=G1_USD,
        init_state=KNEES_BENT_KEYFRAME,
        self_collisions=True,
        actuators={
            "g1_5020_upper": G1_ACTUATOR_5020_UPPER,
            "g1_hip_pitch": hip_pitch_act,
            "g1_hip_yaw_waist_yaw": G1_ACTUATOR_HIP_YAW_7520_14,
            "g1_hip_roll_knee": G1_ACTUATOR_HIP_ROLL_KNEE_7520_22,
            "g1_wrist_pitch_yaw": wrist_act,
            "g1_waist": G1_ACTUATOR_WAIST,
            "g1_ankle": G1_ACTUATOR_ANKLE,
        },
        sensors_isaaclab=[
            ContactSensorCfg(
                name="contact_forces",
                # g1_mjlab.usd nests rigid links under Robot/pelvis/*
                primary="pelvis/.*",
                secondary=[],
                track_air_time=True,
                history_length=4,
            ),
        ],
        sensors_mjlab=[
            ContactSensorCfg(
                name="contact_forces",
                primary=".*",
                secondary=[],
                track_air_time=True,
                history_length=4
            ),
            ContactSensorCfg(
                name="self_collision",
                primary="pelvis",
                secondary="pelvis",
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
        ],
        joint_names_simulation=G1_WAIST_UNLOCKED_CFG.joint_names_simulation,
        body_names_simulation=G1_WAIST_UNLOCKED_CFG.body_names_simulation,
        joint_symmetry_mapping=G1_WAIST_UNLOCKED_CFG.joint_symmetry_mapping,
        spatial_symmetry_mapping=G1_WAIST_UNLOCKED_CFG.spatial_symmetry_mapping,
    )


G1_MODE_5_CFG = _build_g1_cfg(5)
G1_MODE_11_CFG = _build_g1_cfg(11)
G1_MODE_13_CFG = _build_g1_cfg(13)
G1_MODE_15_CFG = _build_g1_cfg(15)

registry.register("asset", "g1-hdmi-mode_5", G1_MODE_5_CFG)
registry.register("asset", "g1-hdmi-mode_11", G1_MODE_11_CFG)
registry.register("asset", "g1-hdmi-mode_13", G1_MODE_13_CFG)
registry.register("asset", "g1-hdmi-mode_15", G1_MODE_15_CFG)
