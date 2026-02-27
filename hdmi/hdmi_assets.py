from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
)
from active_adaptation.assets.humanoid import G1_WAIST_UNLOCKED_CFG
from active_adaptation.registry import Registry
from active_adaptation.assets import ASSET_DIR

registry = Registry.instance()

##
# MJCF/USD.
##

# G1_MJCF = ASSET_DIR / "G1" / "mjcf" / "g1.xml"
G1_MJCF = ASSET_DIR / "G1" / "mjcf" / "g1_mjlab.xml"

# G1_USD = ASSET_DIR / "G1" / "waist_unlocked.usd"
G1_USD = ASSET_DIR / "G1" / "mjcf" / "g1_mjlab" / "g1_mjlab.usd"
# G1_USD = ASSET_DIR / "G1" / "unitree" / "g1_29dof_rev_1_0.usd"
# G1_USD = ASSET_DIR / "G1" / "g1_29dof_nohand" / "g1_29dof_nohand.usd"

##
# Actuator constants (aligned with mjlab g1_constants style).
##

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = ActuatorCfg(
    joint_names_expr=[
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
    ],
    effort_limit={".*": 25.0},
    velocity_limit={".*": 37.0},
    stiffness={".*": STIFFNESS_5020},
    damping={".*": DAMPING_5020},
    friction={".*": 0.01},
    armature={".*": ARMATURE_5020},
)
G1_ACTUATOR_7520_14 = ActuatorCfg(
    joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"],
    effort_limit={".*": 88.0},
    velocity_limit={".*": 32.0},
    stiffness={".*": STIFFNESS_7520_14},
    damping={".*": DAMPING_7520_14},
    friction={".*": 0.01},
    armature={".*": ARMATURE_7520_14},
)
G1_ACTUATOR_7520_22 = ActuatorCfg(
    joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
    effort_limit={".*": 139.0},
    velocity_limit={".*": 20.0},
    stiffness={".*": STIFFNESS_7520_22},
    damping={".*": DAMPING_7520_22},
    friction={".*": 0.01},
    armature={".*": ARMATURE_7520_22},
)
G1_ACTUATOR_4010 = ActuatorCfg(
    joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
    effort_limit={".*": 5.0},
    velocity_limit={".*": 22.0},
    stiffness={".*": STIFFNESS_4010},
    damping={".*": DAMPING_4010},
    friction={".*": 0.01},
    armature={".*": ARMATURE_4010},
)
G1_ACTUATOR_WAIST = ActuatorCfg(
    joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
    effort_limit={".*": 50.0},
    velocity_limit={".*": 37.0},
    stiffness={".*": 2.0 * STIFFNESS_5020},
    damping={".*": 2.0 * DAMPING_5020},
    friction={".*": 0.01},
    armature={".*": 2.0 * ARMATURE_5020},
)
G1_ACTUATOR_ANKLE = ActuatorCfg(
    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    effort_limit={".*": 50.0},
    velocity_limit={".*": 37.0},
    stiffness={".*": 2.0 * STIFFNESS_5020},
    damping={".*": 2.0 * DAMPING_5020},
    friction={".*": 0.01},
    armature={".*": 2.0 * ARMATURE_5020},
)

##
# Keyframe.
##

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

##
# Final config.
##

G1_CFG = AssetCfg(
    mjcf_path=G1_MJCF,
    usd_path=G1_USD,
    init_state=KNEES_BENT_KEYFRAME,
    self_collisions=True,
    actuators={
        "g1_5020": G1_ACTUATOR_5020,
        "g1_7520_14": G1_ACTUATOR_7520_14,
        "g1_7520_22": G1_ACTUATOR_7520_22,
        "g1_4010": G1_ACTUATOR_4010,
        "g1_waist": G1_ACTUATOR_WAIST,
        "g1_ankle": G1_ACTUATOR_ANKLE,
    },
    sensors_isaaclab=[
        ContactSensorCfg(
            name="contact_forces",
            # g1_mjlab.usd nests rigid links under Robot/pelvis/*
            primary="pelvis/.*",
            # primary=".*",
            secondary=[],
            track_air_time=True,
            history_length=3,
        ),
    ],
    sensors_mjlab=G1_WAIST_UNLOCKED_CFG.sensors_mjlab,
    joint_names_isaac=G1_WAIST_UNLOCKED_CFG.joint_names_isaac,
    joint_names_mjlab=G1_WAIST_UNLOCKED_CFG.joint_names_mjlab,
    joint_names_simulation=G1_WAIST_UNLOCKED_CFG.joint_names_simulation,
    body_names_isaac=G1_WAIST_UNLOCKED_CFG.body_names_isaac,
    body_names_mjlab=G1_WAIST_UNLOCKED_CFG.body_names_mjlab,
    body_names_simulation=G1_WAIST_UNLOCKED_CFG.body_names_simulation,
    joint_symmetry_mapping=G1_WAIST_UNLOCKED_CFG.joint_symmetry_mapping,
    spatial_symmetry_mapping=G1_WAIST_UNLOCKED_CFG.spatial_symmetry_mapping,
)

registry.register("asset", "g1-hdmi", G1_CFG)
