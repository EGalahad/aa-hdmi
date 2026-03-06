import datetime
import logging
import time
from collections import OrderedDict
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import active_adaptation as aa
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.profiling import ScopedTimer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def _feature_tensor(
    global_env_ids: torch.Tensor,
    step_id: int,
    feature_shape: tuple[int, ...],
    salt: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    n = global_env_ids.shape[0]
    flat = int(np.prod(feature_shape)) if len(feature_shape) > 0 else 1
    env_axis = global_env_ids.to(torch.float32).unsqueeze(1)
    feat_axis = torch.arange(flat, device=global_env_ids.device, dtype=torch.float32).unsqueeze(0)
    x = env_axis * 0.013 + feat_axis * 0.007 + float(step_id) * 0.019 + salt
    y = torch.sin(x) + 0.5 * torch.cos(x * 1.7)
    return y.reshape(n, *feature_shape).to(dtype)


def _set_if_present(td: TensorDict, key: str, value: torch.Tensor) -> None:
    if key in td.keys(True, True):
        td.set(key, value)


def _fill_float_key(td: TensorDict, key: str, global_env_ids: torch.Tensor, step_id: int, salt: float) -> None:
    if key not in td.keys(True, True):
        return
    x = td.get(key)
    if not torch.is_tensor(x) or not torch.is_floating_point(x):
        return
    _set_if_present(td, key, _feature_tensor(global_env_ids, step_id, tuple(x.shape[1:]), salt, x.dtype))


def _build_dummy_transition(
    obs_template_single: TensorDict,
    rollout_policy,
    policy: PPOBase,
    global_env_ids: torch.Tensor,
    iter_idx: int,
    step_idx: int,
    reward_dim: int,
) -> TensorDict:
    step_id = iter_idx * policy.cfg.train_every + step_idx
    next_step_id = step_id + 1

    local_envs = global_env_ids.shape[0]
    td = obs_template_single.expand(local_envs).clone()
    next_td = obs_template_single.expand(local_envs).clone()

    # Deterministic synthetic observations/features.
    for key, salt in [
        ("policy", 0.10),
        ("priv", 0.20),
        ("command", 0.30),
        ("command_", 0.40),
        ("ref_joint_pos_", 0.50),
        ("object", 0.60),
        ("adapt_hx", 0.70),
    ]:
        _fill_float_key(td, key, global_env_ids, step_id, salt)
        _fill_float_key(next_td, key, global_env_ids, next_step_id, salt)

    # Deterministic per-env step bookkeeping.
    n = global_env_ids.shape[0]
    done = ((global_env_ids + step_id) % 113 == 0).unsqueeze(-1)
    terminated = ((global_env_ids + step_id) % 197 == 0).unsqueeze(-1)
    truncated = done & (~terminated)
    is_init = ((global_env_ids + step_id) % 59 == 0).unsqueeze(-1)
    episode_id = global_env_ids + iter_idx * 100_000 + step_idx

    _set_if_present(td, "done", done)
    _set_if_present(td, "terminated", terminated)
    _set_if_present(td, "truncated", truncated)
    _set_if_present(td, "is_init", is_init)
    _set_if_present(td, "episode_id", episode_id)
    _set_if_present(td, "step_count", torch.full((n, 1), step_id + 8, device=global_env_ids.device, dtype=torch.long))

    next_done = ((global_env_ids + next_step_id) % 113 == 0).unsqueeze(-1)
    next_terminated = ((global_env_ids + next_step_id) % 197 == 0).unsqueeze(-1)
    next_truncated = next_done & (~next_terminated)
    next_is_init = ((global_env_ids + next_step_id) % 59 == 0).unsqueeze(-1)

    _set_if_present(next_td, "done", next_done)
    _set_if_present(next_td, "terminated", next_terminated)
    _set_if_present(next_td, "truncated", next_truncated)
    _set_if_present(next_td, "is_init", next_is_init)
    _set_if_present(next_td, "episode_id", episode_id + 1)

    discount = (~next_done).to(torch.float32)
    reward = _feature_tensor(global_env_ids, next_step_id, (reward_dim,), 1.25, torch.float32)

    next_td.set("discount", discount)
    next_td.set("reward", reward)

    # Use current policy to produce action/logprob/loc/scale from synthetic observations.
    with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
        td = rollout_policy(td)

    td.set("next", next_td)
    return td


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    from active_adaptation.helpers import make_env_policy

    env, policy = make_env_policy(cfg)
    policy: PPOBase

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_iters = cfg.get("total_iters", None)
    if total_iters is None:
        total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
        total_frames = total_frames // frames_per_batch * frames_per_batch
        total_iters = total_frames // frames_per_batch

    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
        run_dir = Path(run.dir)
        ckpt_path = run_dir / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()

        torch.save(state_dict, ckpt_path)
        if upload_to_wandb:
            run.save(str(ckpt_path), policy="now", base_path=run.dir)

        latest_link = run_dir / "checkpoint_latest.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
        logging.info(f"Saved checkpoint to {ckpt_path}" + (" (wandb)" if upload_to_wandb else ""))
        return str(ckpt_path)

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i % checkpoint_interval == 0 or i % upload_interval == 0

    ckpt_path = None
    obs_template = env.reset()
    obs_template_single = obs_template[:1].clone()

    env_frames = 0
    local_rank = aa.get_local_rank()
    world_size = aa.get_world_size()
    local_envs = env.num_envs
    global_env_ids = (
        torch.arange(local_envs, device=env.device, dtype=torch.long) + local_rank * local_envs
    )
    reward_dim = len(policy.reward_groups) if hasattr(policy, "reward_groups") else 1

    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)

    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()
        run.config["dummy_data"] = True

        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        run_idx = run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

        run_dir = Path(run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        run.save(str(cfg_save_path), policy="now")
        run.save(str(run_dir / "config.yaml"), policy="now")

    for stage in stages:
        policy.on_stage_start(stage)
        rollout_policy = policy.get_rollout_policy("train")

        progress = range(total_iters)
        if aa.is_main_process():
            progress = tqdm(progress, desc=stage)

        start_iter = getattr(env, "current_iter", 0)
        for i in progress:
            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
                    if hasattr(env, "set_progress"):
                        env.set_progress(start_iter + i)

                    data_buf = None
                    for step in range(cfg.algo.train_every):
                        td = _build_dummy_transition(
                            obs_template_single=obs_template_single,
                            rollout_policy=rollout_policy,
                            policy=policy,
                            global_env_ids=global_env_ids,
                            iter_idx=i,
                            step_idx=step,
                            reward_dim=reward_dim,
                        )
                        if data_buf is None:
                            data_buf = td.unsqueeze(-1).expand(local_envs, cfg.algo.train_every).clone()
                        data_buf[:, step] = td

                    if data_buf is None:
                        raise RuntimeError("Failed to build dummy data buffer.")
                    carry = data_buf[:, -1]["next"].clone(False)

                    policy.critic(data_buf)
                    values = data_buf["state_value"]
                    data_buf["next", "state_value"] = torch.where(
                        data_buf["next", "done"],
                        values,
                        torch.cat(
                            [
                                values[:, 1:],
                                policy.compute_value(carry.copy())["state_value"].unsqueeze(1),
                            ],
                            dim=1,
                        ),
                    )

            rollout_time = rollout_timer.last_time
            env_frames += local_envs * cfg.algo.train_every

            info = {}
            with ScopedTimer("training") as training_timer:
                info.update(policy.train_op(data_buf))
            training_time = training_timer.last_time

            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)

            info["env_frames"] = env_frames * world_size
            info["performance/rollout_fps"] = (
                local_envs * cfg.algo.train_every / rollout_time * world_size
            )
            info["performance/rollout_time"] = rollout_time
            info["performance/training_time"] = training_time
            info["performance/iter_time"] = time.perf_counter() - rollout_start

            if should_save(i):
                should_upload = i % upload_interval == 0
                checkpoint_name = f"checkpoint_{i}" if should_upload else "checkpoint_temp"
                ckpt_path = save(policy, checkpoint_name, upload_to_wandb=should_upload)
                print(f"Latest checkpoint: {ckpt_path}")

            if aa.is_main_process():
                run.log(info)

    if aa.is_main_process():
        ckpt_path = save(policy, "checkpoint_final")
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
    exit(0)


if __name__ == "__main__":
    main()
