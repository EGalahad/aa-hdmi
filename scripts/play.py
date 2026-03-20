"""
Play and export policy for HDMI project.

This script mirrors the full export behavior from HDMI/scripts/play.py:
- export traced policy as .pt
- export ONNX as .onnx
- export deploy config as .yaml
"""

from __future__ import annotations

import copy
import itertools
import os
import re
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
from tensordict.nn import TensorDictSequential
from torchrl.envs.transforms import VecNorm as TorchRLVecNorm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from active_adaptation.utils.profiling import ScopedTimer

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.torchrl import ObsNorm
from active_adaptation.utils.wandb import parse_checkpoint_path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def _find_vecnorm(env) -> TorchRLVecNorm | None:
    transform = getattr(env, "transform", None)
    if transform is None:
        return None

    transforms = getattr(transform, "transforms", None)
    if transforms is None:
        transforms = [transform]

    for t in transforms:
        if isinstance(t, TorchRLVecNorm):
            return t
    return None


def _get_asset_meta(asset) -> dict:
    meta = {
        "joint_names": list(getattr(asset, "joint_names", [])),
        "joint_kp": {},
        "joint_kd": {},
        "default_joint_pos": {},
    }

    cfg = getattr(asset, "cfg", None)
    init_state = getattr(cfg, "init_state", None)
    if init_state is not None and hasattr(init_state, "joint_pos"):
        joint_pos = init_state.joint_pos
        if isinstance(joint_pos, dict):
            meta["default_joint_pos"] = dict(joint_pos)

    actuators = getattr(asset, "actuators", [])
    for actuator in actuators:
        acfg = getattr(actuator, "cfg", None)
        if acfg is None:
            continue

        names = (
            getattr(acfg, "target_names_expr", None)
            or getattr(acfg, "joint_names_expr", None)
            or []
        )
        stiffness = getattr(acfg, "stiffness", None)
        damping = getattr(acfg, "damping", None)

        for joint_name in names:
            if stiffness is not None:
                meta["joint_kp"][joint_name] = float(stiffness)
            if damping is not None:
                meta["joint_kd"][joint_name] = float(damping)

    return meta


def _checkpoint_tags(checkpoint_path: str | None) -> tuple[str, str]:
    wandb_run_id = "unknown"
    checkpoint_num = "unknown"

    if checkpoint_path is None:
        return wandb_run_id, checkpoint_num

    state_dict = torch.load(checkpoint_path, weights_only=False)
    if "wandb" in state_dict and "id" in state_dict["wandb"]:
        wandb_run_id = state_dict["wandb"]["id"]

    filename = os.path.basename(checkpoint_path)
    match = re.search(r"checkpoint_(\d+)", filename)
    if match:
        checkpoint_num = match.group(1)
    elif filename.endswith("_final.pt"):
        checkpoint_num = "final"

    return wandb_run_id, checkpoint_num



@VecNorm.freeze()
def export_policy(cfg: DictConfig, env: "_EnvBase", policy) -> None:
    checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
    wandb_run_id, checkpoint_num = _checkpoint_tags(checkpoint_path)

    deploy_policy: ModBase = copy.deepcopy(policy.get_rollout_policy("deploy"))

    vecnorm = _find_vecnorm(env)
    if vecnorm is not None:
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
        export_module = TensorDictSequential(obs_norm, deploy_policy).cpu()
    else:
        export_module = deploy_policy.cpu()

    fake_input = env.observation_spec[0].rand().cpu()

    export_dir = FILE_PATH / "exports" / str(cfg.task.name)
    export_dir.mkdir(parents=True, exist_ok=True)
    base = export_dir / f"policy-{wandb_run_id}-{checkpoint_num}"

    onnx_path = str(base.with_suffix(".onnx"))
    yaml_path = str(base.with_suffix(".yaml"))

    meta = {}
    export_onnx(export_module, fake_input, onnx_path, meta)

    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    policy_config = {}

    obs_cfg = {}
    task_obs_cfg = dict_cfg["task"]["observation"]
    for k in deploy_policy.in_keys:
        if k in task_obs_cfg:
            obs_cfg[k] = task_obs_cfg[k]
    policy_config["observation"] = obs_cfg

    asset = env.scene.articulations["robot"]
    asset_meta = _get_asset_meta(asset)
    policy_config["joint_names_simulation"] = asset.cfg.joint_names_simulation
    policy_config["body_names_simulation"] = asset.cfg.body_names_simulation
    policy_config["joint_kp"] = asset_meta["joint_kp"]
    policy_config["joint_kd"] = asset_meta["joint_kd"]
    policy_config["default_joint_pos"] = asset_meta["default_joint_pos"]

    # Make joint observation order explicit for sim2real consumers.
    action_manager = env.action_manager
    policy_config["policy_joint_names"] = list(
        getattr(action_manager, "joint_names", [])
    )
    action_scale = getattr(action_manager, "action_scaling", None)
    if action_scale is not None:
        policy_config["action_scale"] = action_scale.detach().cpu().tolist()
    else:
        policy_config["action_scale"] = []

    command = env.command_manager
    cmd_key = "command" if "command" in policy_config["observation"] else "command_"
    command_obs = policy_config["observation"].get(cmd_key)

    if isinstance(command_obs, dict) and command is not None:
        from hdmi.tasks.command import RobotTracking

        command: RobotTracking
        for obs_key in command_obs:
            if (
                "task" in dict_cfg
                and "command" in dict_cfg["task"]
                and "data_path" in dict_cfg["task"]["command"]
            ):
                command_obs[obs_key]["motion_path"] = dict_cfg["task"]["command"][
                    "data_path"
                ]
            command_obs[obs_key]["future_steps"] = command.future_steps.tolist()
            command_obs[obs_key]["body_names"] = command.tracking_body_names
            command_obs[obs_key]["joint_names"] = command.tracking_joint_names
            command_obs[obs_key]["root_body_name"] = command.root_body_name
            command_obs[obs_key]["anchor_body_name"] = command.anchor_body_name

    import yaml

    with open(yaml_path, "w") as f:
        yaml.dump(policy_config, f, sort_keys=False)

    print(f"Exported deploy config to {yaml_path}")


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    from active_adaptation.helpers import make_env_policy

    checkpoint_path = parse_checkpoint_path(cfg.get("checkpoint_path", None))
    if checkpoint_path is not None:
        cfg.checkpoint_path = checkpoint_path

    env, policy = make_env_policy(cfg)

    if cfg.get("export_policy", False):
        export_policy(cfg, env, policy)

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")

    env.base_env.eval()
    carry = env.reset()

    assert not env.base_env.training

    timer = Timer(env.step_dt)
    fps_window_start = time.perf_counter()
    fps_window_frames = 0

    # with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            with ScopedTimer("inference", sync=False):
                carry = rollout_policy(carry)
            with ScopedTimer("env_step", sync=False):
                td, carry = env.step_and_maybe_reset(carry)
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())

            fps_window_frames += 1
            window_elapsed = time.perf_counter() - fps_window_start
            if window_elapsed >= 1.0:
                print(
                    f"Loop FPS: {fps_window_frames} frames in "
                    f"{window_elapsed:.2f}s"
                )
                # ScopedTimer.print_summary(clear=True)
                fps_window_start = time.perf_counter()
                fps_window_frames = 0

            timer.sleep()

    env.close()


if __name__ == "__main__":
    main()
