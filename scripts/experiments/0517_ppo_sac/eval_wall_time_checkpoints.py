#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT = "elijahgalahad/hdmi"
DEFAULT_OUTPUT_DIR = Path("outputs/eval/wall_time_checkpoint_rollout")
WAND_B_CSV_DIR = Path("outputs/wandb/0517_ppo_sac_scale_metrics/csv")
CHECKPOINTS = (1000, 2000, 3000, 4000)

PPO_WALL_TIME_RUNS = {
    "8x1024": {1: "ggh7r3as", 2: "4dtllhi4", 3: "l7b8boa2"},
    "8x2048": {1: "cmk28549", 2: "cv5lz2ua", 3: "ieoix5xq"},
    "8x4096": {1: "hmvc597l", 2: "oqimvqfi", 3: "yjzauwnj"},
    "8x8192": {1: "431jq1ef", 2: "jt9xxe99", 3: "i7ppgx6o"},
    "8x16384": {1: "sppel7at", 2: "oyyvto2j", 3: "wwun9k41"},
    "8x16384 huge": {1: "z2r3plvo", 2: "euf6es6i", 3: "tevenyu0"},
}


@dataclass(frozen=True)
class EvalSpec:
    label: str
    seed: int
    run_id: str
    checkpoint: int

    @property
    def key(self) -> str:
        clean_label = self.label.replace(" ", "_")
        return f"{clean_label}_seed{self.seed}_checkpoint{self.checkpoint}"


def import_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required to read checkpoint wall times.") from exc
    return wandb


def build_specs(
    seeds: set[int],
    labels: set[str] | None = None,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
) -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    for label, runs in PPO_WALL_TIME_RUNS.items():
        if labels is not None and label not in labels:
            continue
        for seed, run_id in runs.items():
            if seed not in seeds:
                continue
            for checkpoint in checkpoints:
                specs.append(EvalSpec(label=label, seed=seed, run_id=run_id, checkpoint=checkpoint))
    return specs


def checkpoint_wall_time_hours(run: Any, checkpoint: int) -> float:
    target_step = max(0, checkpoint - 1)
    best_step = -1
    best_runtime: float | None = None
    for row in run.scan_history(keys=["_step", "_runtime", "_timestamp"]):
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)
        if step > target_step:
            continue
        runtime = row.get("_runtime")
        if runtime is None:
            continue
        if step >= best_step:
            best_step = step
            best_runtime = float(runtime)
    if best_runtime is None:
        raise RuntimeError(f"No runtime found for {run.path} checkpoint {checkpoint}")
    return best_runtime / 3600.0


def cached_wall_time_hours(spec: EvalSpec) -> float | None:
    target_step = max(0, spec.checkpoint - 1)
    label = spec.label.replace(" ", "_")
    csv_path = WAND_B_CSV_DIR / f"ppo_{label}_seed{spec.seed}.csv"
    if not csv_path.exists():
        return None
    best_step = -1
    best_wall_time: float | None = None
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_value = row.get("_step")
            wall_time_value = row.get("derived/wall_time_hours")
            if not step_value or not wall_time_value:
                continue
            step = int(float(step_value))
            if step > target_step:
                continue
            if step >= best_step:
                best_step = step
                best_wall_time = float(wall_time_value)
    return best_wall_time


def command_for(spec: EvalSpec, output_path: Path, task: str) -> list[str]:
    cmd = [
        "uv",
        "--project",
        "venv/mjlab",
        "run",
        "--no-sync",
        "python",
        str(Path.cwd() / "projects/hdmi/scripts/eval.py"),
        "+exp=ppo/train",
        f"task={task}",
        "backend=mjlab",
        "task.num_envs=512",
        "task.termination.root_pos_error.enabled=false",
        "headless=true",
        f"checkpoint_path=run:{PROJECT}/runs/{spec.run_id}:{spec.checkpoint}",
        "eval_steps=1000",
        f"eval_output={output_path.with_suffix('.pt').resolve()}",
        f"eval_summary_output={output_path.resolve()}",
    ]
    if spec.label.endswith("huge"):
        cmd.append("algo/ppo/module=huge")
    else:
        cmd.append("algo/ppo/module=large")
    return cmd


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "seed",
        "run_id",
        "checkpoint",
        "task",
        "wall_time_hours",
        "success_rate",
        "joint_pos",
        "body_pos",
        "body_ori",
        "num_envs",
        "steps",
        "num_finished_episodes",
        "output",
    ]
    merged: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    if summary_path.exists():
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged[(row["label"], int(row["seed"]), int(row["checkpoint"]), row.get("task", ""))] = row
    for row in rows:
        merged[(row["label"], int(row["seed"]), int(row["checkpoint"]), row["task"])] = row
    ordered_rows = sorted(
        merged.values(),
        key=lambda row: (str(row["label"]), int(row["seed"]), int(row["checkpoint"])),
    )
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered_rows)
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout-evaluate PPO wall-time checkpoints.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels to evaluate, for example: --labels '8x16384 huge'",
    )
    parser.add_argument("--checkpoints", nargs="+", type=int, default=list(CHECKPOINTS))
    parser.add_argument("--task", default="lafan_100style_real")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-missing-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = None
    run_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")

    labels = set(args.labels) if args.labels is not None else None
    checkpoints = tuple(args.checkpoints)
    for spec in build_specs(set(args.seeds), labels=labels, checkpoints=checkpoints):
        output_path = args.output_dir / spec.label.replace(" ", "_") / f"seed{spec.seed}" / f"checkpoint_{spec.checkpoint}.json"
        wall_time_hours = cached_wall_time_hours(spec)
        if wall_time_hours is None:
            if api is None:
                wandb = import_wandb()
                api = wandb.Api()
            if spec.run_id not in run_cache:
                run_cache[spec.run_id] = api.run(f"{PROJECT}/{spec.run_id}")
            wall_time_hours = checkpoint_wall_time_hours(run_cache[spec.run_id], spec.checkpoint)

        if args.force or not output_path.exists():
            cmd = command_for(spec, output_path, args.task)
            print("[run]", " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            if args.only_missing_summary:
                raise FileNotFoundError(output_path)
            start = time.time()
            subprocess.run(cmd, check=True, env=env)
            print(f"[done] {spec.key} elapsed_sec={time.time() - start:.1f}", flush=True)
        else:
            print(f"[skip] {output_path}")

        if output_path.exists():
            data = json.loads(output_path.read_text())
            summary = data["summary"]
            rows.append(
                {
                    "label": spec.label,
                    "seed": spec.seed,
                    "run_id": spec.run_id,
                    "checkpoint": spec.checkpoint,
                    "task": args.task,
                    "wall_time_hours": wall_time_hours,
                    "success_rate": summary["success_rate"],
                    "joint_pos": summary["joint_pos"],
                    "body_pos": summary["body_pos"],
                    "body_ori": summary["body_ori"],
                    "num_envs": summary["num_envs"],
                    "steps": summary["steps"],
                    "num_finished_episodes": summary.get("num_finished_episodes", 0),
                    "output": str(output_path),
                }
            )

    if rows:
        write_summary(args.output_dir, rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
