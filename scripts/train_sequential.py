#!/usr/bin/env python3
from __future__ import annotations

import secrets
import shlex
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import active_adaptation as aa

FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"
ROOT_PATH = FILE_PATH.parents[2]
ORCHESTRATOR_KEYS = {
    "stages",
    "nproc_per_node",
    "script",
    "random_suffix_length",
    "tag",
}


def _resolve_script_path(script: str) -> str:
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = ROOT_PATH / script_path
    return str(script_path.resolve())


def _random_suffix(length: int) -> str:
    token_bytes = max(1, (length + 1) // 2)
    return secrets.token_hex(token_bytes)[:length]


def _sanitize_tag(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


def _stage_run_dir(stage_name: str, index: int, base_dir: Path | None = None) -> str:
    if base_dir is None:
        base_dir = Path.cwd()
    stage_slug = _sanitize_tag(stage_name)
    stage_dir = (base_dir / "stages" / f"{index + 1:02d}-{stage_slug}").resolve()
    return str(stage_dir)


def _collect_cli_overrides() -> list[str]:
    cli_overrides = []
    script_name = Path(__file__).name
    for arg in sys.argv[1:]:
        if arg.startswith("hydra.") or script_name in arg:
            continue
        top_level_key = arg.split("=", 1)[0].split(".", 1)[0]
        if top_level_key in ORCHESTRATOR_KEYS:
            continue
        cli_overrides.append(arg)
    return cli_overrides


def _normalize_stage(stage, index: int) -> dict[str, object]:
    if isinstance(stage, DictConfig):
        return {
            "name": str(stage.get("name", f"stage-{index + 1}")),
            "overrides": [str(item) for item in stage.get("overrides", [])],
            "load_checkpoint_from_previous": bool(
                stage.get("load_checkpoint_from_previous", index > 0)
            ),
        }
    stage_name = str(stage)
    return {
        "name": stage_name,
        "overrides": [f"algo={stage_name}"],
        "load_checkpoint_from_previous": index > 0,
    }


def _to_stage_items(stages_cfg) -> list[dict[str, object]]:
    stages = list(stages_cfg)
    if not stages:
        raise ValueError("No stages configured.")
    return [_normalize_stage(stage, index) for index, stage in enumerate(stages)]


def _has_future_checkpoint_consumer(
    stages: list[dict[str, object]], current_index: int
) -> bool:
    return any(
        bool(stage["load_checkpoint_from_previous"])
        for stage in stages[current_index + 1 :]
    )


def _run_command(command: list[str]) -> None:
    print(f">>> {shlex.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT_PATH)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


@hydra.main(
    config_path=str(CONFIG_PATH), config_name="train_sequential", version_base=None
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    cli_overrides = _collect_cli_overrides()
    stages = _to_stage_items(cfg.stages)
    tag = _sanitize_tag(str(cfg.tag))
    suffix = _random_suffix(int(cfg.random_suffix_length))
    script_path = _resolve_script_path(str(cfg.script))
    sequential_run_dir = Path.cwd()

    print("=" * 80)
    print("Detected command-line overrides applied to all stages:")
    if cli_overrides:
        for ov in cli_overrides:
            print(f"  - {ov}")
    else:
        print("  - None")
    print("=" * 80)

    previous_checkpoint_path = None

    for i, stage in enumerate(stages):
        stage_name_value = str(stage["name"])
        stage_specific_overrides = list(stage["overrides"])
        load_from_previous = bool(stage["load_checkpoint_from_previous"])

        print("\n" + "=" * 80)
        print(f"Preparing stage {i + 1}/{len(stages)}: {stage_name_value}")
        print("=" * 80)

        stage_run_id = f"{suffix}_{tag}_{stage_name_value}"
        stage_run_dir = _stage_run_dir(
            stage_name_value,
            i,
            base_dir=sequential_run_dir,
        )
        stage_overrides = cli_overrides.copy()
        stage_overrides.extend(stage_specific_overrides)
        stage_overrides.append(f"hydra.run.dir={stage_run_dir}")

        if previous_checkpoint_path and load_from_previous:
            stage_overrides.append(f"checkpoint_path={previous_checkpoint_path}")
            print(f"Loading checkpoint from previous stage: {previous_checkpoint_path}")

        stage_overrides.append(f"wandb.id={stage_run_id}")
        print(f"Stage run dir: {stage_run_dir}")

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={cfg.nproc_per_node}",
            script_path,
            *stage_overrides,
        ]

        print(f"Starting child process for stage '{stage_name_value}'")
        _run_command(command)
        print(f"Child process for stage '{stage_name_value}' finished")

        if _has_future_checkpoint_consumer(stages, i):
            previous_checkpoint_path = str(Path(stage_run_dir) / "checkpoint_latest.pt")
            print(
                "Recorded local checkpoint for downstream stages: "
                f"{previous_checkpoint_path}"
            )
        else:
            previous_checkpoint_path = None

        print(f"Completed stage {i + 1}/{len(stages)}: {stage_name_value}")
        print(f"Run ID: {stage_run_id}")
        print("=" * 80)

    print("\nAll training stages finished")


if __name__ == "__main__":
    main()
