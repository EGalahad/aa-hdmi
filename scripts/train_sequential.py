from __future__ import annotations

import datetime
import logging
import multiprocessing
import sys
import time
from collections import OrderedDict
from pathlib import Path

import hydra
import torch
import wandb
from hydra import compose
from omegaconf import DictConfig, OmegaConf
from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

import active_adaptation as aa
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.learning.ppo.ppo_base import PPOBase

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


def run_training_stage(
    cfg: DictConfig, return_queue: multiprocessing.Queue | None = None
) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    from active_adaptation.helpers import make_env_policy, evaluate
    from active_adaptation.utils.helpers import EpisodeStats

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

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    run = None
    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()

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

    def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
        if not aa.is_main_process():
            return None
        assert run is not None
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
        logging.info(
            f"Saved checkpoint to {ckpt_path}"
            + (" (wandb)" if upload_to_wandb else "")
        )
        return str(ckpt_path)

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i % checkpoint_interval == 0 or i % upload_interval == 0

    carry = env.reset()
    next_saved_keys = [
        # "command",
        # "command_",
        # "policy",
        # "priv",
        "done",
        "terminated",
        "truncated",
        "discount",
        "reward",
        "stats",
        "is_init",
        "adapt_hx",
        "episode_id",
    ]

    env_frames = 0

    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)

    for stage in stages:
        policy.on_stage_start(stage)
        rollout_policy = policy.get_rollout_policy("train")

        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            tmp_carry = rollout_policy(carry.clone(False))
            tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
            tmp_td["next"] = tmp_td["next"].select(*next_saved_keys, strict=False)

        data_buf: TensorDict = tmp_td.unsqueeze(-1).expand(
            env.num_envs, cfg.algo.train_every
        ).clone()

        progress = range(total_iters)
        if aa.is_main_process():
            progress = tqdm(progress, desc=stage_name(stage, cfg.algo.name))

        start_iter = getattr(env, "current_iter", 0)
        for i in progress:
            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
                    if hasattr(env, "set_progress"):
                        env.set_progress(start_iter + i)
                    for step in range(cfg.algo.train_every):
                        carry = rollout_policy(carry)
                        td, carry = env.step_and_maybe_reset(carry)
                        td["next"] = td["next"].select(*next_saved_keys, strict=False)
                        data_buf[:, step] = td

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

            episode_stats.add(data_buf)
            env_frames += data_buf.numel()

            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()

            with ScopedTimer("training") as training_timer:
                info.update(policy.train_op(data_buf))
            training_time = training_timer.last_time

            info.update(env.extra)
            info.update(env.stats_ema)

            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)

            info["env_frames"] = env_frames * aa.get_world_size()
            info["performance/rollout_fps"] = (
                data_buf.numel() / rollout_time * aa.get_world_size()
            )
            info["performance/rollout_time"] = rollout_time
            info["performance/training_time"] = training_time
            info["performance/iter_time"] = time.perf_counter() - rollout_start

            if should_save(i):
                should_upload = i % upload_interval == 0
                checkpoint_name = (
                    f"checkpoint_{i}" if should_upload else "checkpoint_temp"
                )
                ckpt_path = save(
                    policy, checkpoint_name, upload_to_wandb=should_upload
                )
                if ckpt_path is not None:
                    print(f"Latest checkpoint: {ckpt_path}")

            if aa.is_main_process() and run is not None:
                run.log(info)

    run_path = None
    if aa.is_main_process() and run is not None:
        final_ckpt = save(policy, "checkpoint_final")
        policy_eval = policy.get_rollout_policy("eval")
        info, _, _ = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
        info["env_frames"] = env_frames
        run.log(info)

        run_path = f"{run.entity}/{run.project}/{run.id}"
        print(f"Final checkpoint: {final_ckpt}")
        print(f"Run path: {run_path}")
        wandb.finish()

    if return_queue is not None:
        return_queue.put(run_path)
        return_queue.close()
        return_queue.join_thread()
    return


def stage_name(stage: str, fallback: str) -> str:
    if stage:
        return stage
    return fallback


@hydra.main(
    config_path=str(CONFIG_PATH), config_name="train_sequential", version_base=None
)
def main(cfg: DictConfig):
    cli_overrides = []
    script_name = Path(__file__).name
    for arg in sys.argv[1:]:
        if arg.startswith("hydra.") or script_name in arg:
            continue
        if not arg.startswith("stages="):
            cli_overrides.append(arg)

    print("=" * 80)
    print("Detected command-line overrides applied to all stages:")
    if cli_overrides:
        for ov in cli_overrides:
            print(f"  - {ov}")
    else:
        print("  - None")
    print("=" * 80)

    previous_run_path = None

    for i, stage in enumerate(cfg.stages):
        if isinstance(stage, DictConfig):
            stage_name_value = stage.get("name", f"stage-{i + 1}")
            stage_specific_overrides = list(stage.get("overrides", []))
            load_from_previous = bool(stage.get("load_checkpoint_from_previous", i > 0))
        else:
            # Backward compatibility: plain string stage means algo override.
            stage_name_value = str(stage)
            stage_specific_overrides = [f"algo={stage_name_value}"]
            load_from_previous = i > 0

        print("\n" + "=" * 80)
        print(f"Preparing stage {i + 1}/{len(cfg.stages)}: {stage_name_value}")
        print("=" * 80)

        stage_overrides = cli_overrides.copy()
        stage_overrides.extend(stage_specific_overrides)
        if previous_run_path and load_from_previous:
            stage_overrides.append(f"checkpoint_path=run:{previous_run_path}")
            print(f"Loading checkpoint from previous run: {previous_run_path}")

        stage_cfg = compose(config_name="train", overrides=stage_overrides)
        OmegaConf.resolve(stage_cfg)

        return_queue = multiprocessing.Queue(1)
        process = multiprocessing.Process(
            target=run_training_stage,
            kwargs={"cfg": stage_cfg, "return_queue": return_queue},
        )

        print(f"Starting child process for stage '{stage_name_value}'")
        process.start()
        process.join()
        print(f"Child process for stage '{stage_name_value}' finished")

        if process.exitcode != 0:
            print(
                f"ERROR: Child process for stage '{stage_name_value}' exited with code {process.exitcode}"
            )
            break

        try:
            current_run_path = return_queue.get(timeout=10)
        except Exception as exc:
            print(
                f"ERROR: Failed to retrieve run path for stage '{stage_name_value}': {exc}"
            )
            return_queue.close()
            return_queue.join_thread()
            break

        if not current_run_path:
            print(
                f"ERROR: Empty run path returned from stage '{stage_name_value}'. Stop remaining stages."
            )
            return_queue.close()
            return_queue.join_thread()
            break

        return_queue.close()
        return_queue.join_thread()
        previous_run_path = current_run_path
        print(f"Completed stage {i + 1}/{len(cfg.stages)}: {stage_name_value}")
        print(f"Run path: {current_run_path}")
        print("=" * 80)

    print("\nAll training stages finished")


if __name__ == "__main__":
    main()
