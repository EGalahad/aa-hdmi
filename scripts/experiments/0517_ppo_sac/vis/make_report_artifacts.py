#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from _plot_common import (
    PLOT_METRIC_ORDER,
    VariantSpec,
    WALL_TIME_METRIC,
    aggregate_metrics,
    load_or_download_rows,
    normalize_project,
    plot_metric_grid,
    try_import_matplotlib,
)


PROJECT = "elijahgalahad/hdmi"
RUN_FAMILY_TAG = "0517_ppo_sac_scale"
SCALE_OUTPUT_DIR = Path("outputs/wandb/0517_ppo_sac_scale_metrics")
SECTION_ROOT = Path("projects/hdmi/scripts/experiments/0517_ppo_sac/sections")
LEGACY_EVAL_OUTPUT_DIR = Path("outputs/eval/final_policy_metrics")
BAR_FIG_HEIGHT_SCALE = 0.9
TRAINING_FIG_WIDTH = 13.6
FINAL_POLICY_BASE_WIDTH = 9.0
BAR_LAYOUT_RECT = (0, 0.11, 1, 0.925)
BAR_LEGEND_ANCHOR = (0.5, 0.035)
BAR_TITLE_Y = 0.96
REPORT_OUTPUT_DIR = Path("outputs/report/final_0517")
REPORT_DIRS = (
    Path("/home/elijah/Documents/projects/simple-tracking/course/reports/final/chinese/figures"),
    Path("/home/elijah/Documents/projects/simple-tracking/course/reports/final/english/figures"),
)

GROUP_4096 = ((1, 4096), (2, 2048), (4, 1024), (8, 512))
GROUP_8GPU = ((8, 1024), (8, 2048), (8, 4096), (8, 8192), (8, 16384))
GROUP_WALL_TIME = ((8, 1024), (8, 2048), (8, 4096), (8, 8192), (8, 16384))
PPO_TRAIN_EVERY = 32
ENV_FRAMES_EXTRA_SPECS = (
    VariantSpec(
        key="ppo_8x8192_8k",
        label="8x8192 8k",
        algo="ppo",
        tags=(
            RUN_FAMILY_TAG,
            "0517_ppo_sac_scale_8k",
            "algo_ppo",
            "module_large",
            "nproc_8",
            "num_envs_8192",
            "total_iters_8000",
        ),
        color="#8b55df",
        linestyle="--",
    ),
)
WALL_TIME_EXTRA_SPECS = (
    VariantSpec(
        key="ppo_8x16384_huge",
        label="8x16384 huge",
        algo="ppo",
        tags=(
            RUN_FAMILY_TAG,
            "algo_ppo",
            "module_huge",
            "nproc_8",
            "num_envs_16384",
        ),
        color="#6f2dbd",
        linestyle="--",
    ),
)
PPO_MODULES = ("small", "base", "base_deep", "large", "large_deep", "huge", "residual")
PPO_MODULE_WIDTH = ("small", "base", "large", "huge")
PPO_MODULE_DEPTH_RESIDUAL = ("base", "base_deep", "large", "large_deep", "residual")
SETTING_COLORS = {
    (1, 4096): "#e15759",
    (2, 2048): "#e3b505",
    (4, 1024): "#4caf50",
    (8, 512): "#57c26e",
    (8, 1024): "#57c26e",
    (8, 2048): "#45b4a6",
    (8, 4096): "#4c94d8",
    (8, 8192): "#8b55df",
    (8, 16384): "#b64ac7",
}
MODULE_COLORS = {
    "small": "#2ca02c",
    "base": "#1f77b4",
    "base_deep": "#5fa8ff",
    "large": "#ff7f0e",
    "large_deep": "#ffad59",
    "huge": "#d62728",
    "residual": "#8b55df",
}
DATA_COLORS = {
    "lafan": "#4c78a8",
    "lafan_100style": "#59a14f",
    "lafan_100style_real": "#f28e2b",
}
MODULE_CURVE_COLORS = {
    "base_deep": MODULE_COLORS["base"],
    "large_deep": MODULE_COLORS["large"],
}
MODULE_CURVE_LINESTYLES = {
    "base_deep": "--",
    "large_deep": "--",
}

FINAL_METRICS = (
    ("joint_pos", "关节位置误差"),
    ("body_pos", "身体位置误差"),
    ("body_ori", "身体姿态误差"),
)
FINAL_POLICY_TASKS = ("lafan", "100style", "sonic-subset")
FINAL_POLICY_TASK_LABELS = {
    "lafan": "LAFAN",
    "100style": "100STYLE",
    "sonic-subset": "SONIC 子集",
}
DATA_POLICY_TASKS = ("lafan", "100style", "sonic-subset")
COLLECT_TASKS = ("lafan", "100style", "sonic-subset")
LEGACY_COLLECT_TASKS = ("lafan", "100style", "sonic-subset")
COMPARISON_TASKS = ("lafan", "100style", "sonic-subset")
COMPARISON_TASK_LABELS = {
    "lafan": "LAFAN",
    "100style": "100STYLE",
    "sonic-subset": "SONIC 子集",
}
POLICY_COLORS = {
    "ppo": "#8b55df",
    "sonic": "#76B900",
    "sac": "#d62728",
}
SONIC_COMPARISON = {
    "lafan": (0.388, 1131.1),
    "100style": (0.933, 441.8),
    "sonic-subset": (1.000, 62.6),
}

TITLE_ZH = {
    "PPO compute ablation: total environments = 4096": "固定总环境数为 4096 时的 PPO 训练曲线",
    "PPO compute ablation: 8 GPUs with increasing environments": "固定 8 张图形处理器时的 PPO 训练曲线",
    "PPO env-frames performance comparison": "固定 8 张图形处理器时按环境帧对齐的 PPO 训练曲线",
    "PPO environment scaling by wall-clock time": "固定 8 张图形处理器时按训练时间对齐的 PPO 评测曲线",
    "PPO width ablation at 8x8192": "固定 8x8192 时的 PPO 宽度消融训练曲线",
    "PPO depth and residual ablation at 8x8192": "固定 8x8192 时的 PPO 深度与残差消融训练曲线",
    "Final policy metrics: total environments = 4096": "固定总环境数为 4096 时的离线跟踪精度对比",
    "Final policy metrics: 8 GPUs with increasing environments": "固定 8 张图形处理器时的离线跟踪精度对比",
    "Final policy metrics: PPO width ablation": "PPO 宽度消融的离线跟踪精度对比",
    "Final policy metrics: PPO depth and residual ablation": "PPO 深度与残差消融的离线跟踪精度对比",
    "Final policy metrics: PPO vs SAC at 8x8192": "PPO 与 SAC 的离线跟踪精度对比",
    "Final policy metrics: PPO data ablation": "数据选择消融的离线跟踪精度对比",
    "Offline evaluation under matched termination": "PPO、SAC 与 SONIC 的离线评测对比",
}

LABEL_ZH = {
    "lafan": "仅 LAFAN",
    "lafan_100style": "LAFAN+100STYLE",
    "lafan_100style_real": "LAFAN+100STYLE+Real",
    "ppo 8x8192": "PPO",
    "sac 8x8192": "SAC",
    "8x8192 8k": "8x8192 加长",
}

X_LABEL_ZH = {
    "training step": "训练迭代",
    "env frames": "环境交互帧数",
    "wall-clock time (hours)": "训练时间（小时）",
}


def _zh_title(title: str) -> str:
    return TITLE_ZH.get(title, title)


def _zh_label(label: str) -> str:
    return LABEL_ZH.get(label, label.replace("_", "-"))


def _legend_label(label: str, group: str) -> str:
    if group == "module":
        return label.replace("_", "-")
    return _zh_label(label)

def build_scale_specs(settings: tuple[tuple[int, int], ...]) -> list[VariantSpec]:
    specs = []
    for nproc, num_envs in settings:
        label = f"{nproc}x{num_envs}"
        specs.append(
            VariantSpec(
                key=f"ppo_{label}",
                label=label,
                algo="ppo",
                tags=(
                    RUN_FAMILY_TAG,
                    "algo_ppo",
                    "module_large",
                    f"nproc_{nproc}",
                    f"num_envs_{num_envs}",
                ),
                color=SETTING_COLORS[(nproc, num_envs)],
                linestyle="-",
            )
        )
    return specs


def _env_frames_per_step(spec: VariantSpec) -> int | None:
    nproc = None
    num_envs = None
    for tag in spec.tags:
        if tag.startswith("nproc_"):
            nproc = int(tag.removeprefix("nproc_"))
        elif tag.startswith("num_envs_"):
            num_envs = int(tag.removeprefix("num_envs_"))
    if nproc is None or num_envs is None:
        return None
    return nproc * num_envs * PPO_TRAIN_EVERY


def _with_derived_env_frames(
    rows_by_variant: dict[str, list[list[dict[str, float]]]],
    specs: list[VariantSpec],
) -> dict[str, list[list[dict[str, float]]]]:
    frame_steps = {spec.key: _env_frames_per_step(spec) for spec in specs}
    result: dict[str, list[list[dict[str, float]]]] = {}
    for variant_key, runs_rows in rows_by_variant.items():
        frames_per_step = frame_steps.get(variant_key)
        patched_runs = []
        for rows in runs_rows:
            patched_rows = []
            for row in rows:
                patched = dict(row)
                if "env_frames" not in patched and frames_per_step is not None and "_step" in patched:
                    patched["env_frames"] = (patched["_step"] + 1.0) * frames_per_step
                patched_rows.append(patched)
            patched_runs.append(patched_rows)
        result[variant_key] = patched_runs
    return result


def plot_training_curves() -> list[Path]:
    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")

    output_paths: list[Path] = []
    for name, settings, title in (
        ("ppo_compute_product4096_training", GROUP_4096, "PPO compute ablation: total environments = 4096"),
        ("ppo_compute_8gpu_training", GROUP_8GPU, "PPO compute ablation: 8 GPUs with increasing environments"),
    ):
        specs = build_scale_specs(settings)
        rows_by_variant, metric_names = load_or_download_rows(
            project=normalize_project(PROJECT),
            run_family_tag=RUN_FAMILY_TAG,
            variant_specs=specs,
            seeds={1, 2, 3},
            output_dir=SCALE_OUTPUT_DIR,
            force_refresh=False,
        )
        metric_names = [metric for metric in PLOT_METRIC_ORDER if metric in metric_names]
        aggregated = aggregate_metrics(rows_by_variant, metric_names, "_step")
        stem = REPORT_OUTPUT_DIR / name
        plot_metric_grid(
            plt=plt,
            variant_specs=specs,
            aggregated=aggregated,
            metric_names=metric_names,
            x_label="训练迭代",
            title=_zh_title(title),
            output_stem=stem,
        )
        output_paths.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])

    specs = [*build_scale_specs(GROUP_8GPU), *ENV_FRAMES_EXTRA_SPECS]
    rows_by_variant, metric_names = load_or_download_rows(
        project=normalize_project(PROJECT),
        run_family_tag=RUN_FAMILY_TAG,
        variant_specs=specs,
        seeds={1, 2, 3},
        output_dir=SCALE_OUTPUT_DIR,
        force_refresh=False,
    )
    metric_names = [metric for metric in PLOT_METRIC_ORDER if metric in metric_names]
    rows_by_variant = _with_derived_env_frames(rows_by_variant, specs)
    aggregated = aggregate_metrics(rows_by_variant, metric_names, "env_frames")
    stem = REPORT_OUTPUT_DIR / "ppo_compute_env_frames_training"
    plot_metric_grid(
        plt=plt,
        variant_specs=specs,
        aggregated=aggregated,
        metric_names=metric_names,
        x_label="环境交互帧数",
        title=_zh_title("PPO env-frames performance comparison"),
        output_stem=stem,
    )
    output_paths.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    return output_paths


def plot_env_metrics_by_wall_time() -> list[Path]:
    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")

    base_specs = build_scale_specs(GROUP_WALL_TIME)
    specs = [*base_specs, *WALL_TIME_EXTRA_SPECS]
    summary_path = Path("outputs/eval/wall_time_checkpoint_rollout/summary.csv")
    if not summary_path.exists():
        print(f"Missing rollout eval summary for wall-time plot: {summary_path}")
        return []

    label_to_key = {spec.label: spec.key for spec in specs}
    label_to_key["8x16384 huge"] = "ppo_8x16384_huge"
    metric_map = {
        "success_rate": "derived/success_rate",
        "joint_pos": "reward.tracking_metrics/joint_pos",
        "body_pos": "reward.tracking_metrics/body_pos",
        "body_ori": "reward.tracking_metrics/body_ori",
    }
    grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            key = label_to_key.get(raw["label"])
            if key is None:
                continue
            checkpoint = int(raw["checkpoint"])
            item = {"wall_time_hours": float(raw["wall_time_hours"])}
            for raw_metric in metric_map:
                item[raw_metric] = float(raw[raw_metric])
            grouped[(key, checkpoint)].append(item)

    aggregated: dict[str, dict[str, dict[str, list[float]]]] = {}
    for spec in specs:
        metric_values = {
            metric_name: {"x": [], "means": [], "stds": []}
            for metric_name in metric_map.values()
        }
        for (key, _checkpoint), items in sorted(grouped.items()):
            if key != spec.key:
                continue
            x_values = [item["wall_time_hours"] for item in items]
            for raw_metric, metric_name in metric_map.items():
                values = [item[raw_metric] for item in items]
                value_mean = mean(values)
                metric_values[metric_name]["x"].append(mean(x_values))
                metric_values[metric_name]["means"].append(value_mean)
                metric_values[metric_name]["stds"].append(pstdev(values) if len(values) > 1 else 0.0)
        metric_values = {
            metric: values
            for metric, values in metric_values.items()
            if values["x"]
        }
        if metric_values:
            aggregated[spec.key] = metric_values

    metric_names = [metric for metric in PLOT_METRIC_ORDER if any(metric in item for item in aggregated.values())]
    if not metric_names:
        print(f"No rollout eval metrics found in {summary_path}")
        return []

    stem = REPORT_OUTPUT_DIR / "ppo_env_wall_time_training"
    plot_metric_grid(
        plt=plt,
        variant_specs=specs,
        aggregated=aggregated,
        metric_names=metric_names,
        x_label="训练时间（小时）",
        title=_zh_title("PPO environment scaling by wall-clock time"),
        output_stem=stem,
    )
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def plot_module_curves() -> list[Path]:
    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")

    output_paths: list[Path] = []
    for name, modules, title in (
        ("ppo_module_width_training", PPO_MODULE_WIDTH, "PPO width ablation at 8x8192"),
        (
            "ppo_module_depth_residual_training",
            PPO_MODULE_DEPTH_RESIDUAL,
            "PPO depth and residual ablation at 8x8192",
        ),
    ):
        specs = [
            VariantSpec(
                key=f"ppo_{module_name}",
                label=module_name.replace("_", "-"),
                algo="ppo",
                tags=(
                    "0517_ppo_sac_module",
                    "algo_ppo",
                    f"module_{module_name}",
                    "nproc_8",
                    "num_envs_8192",
                ),
                color=MODULE_CURVE_COLORS.get(module_name, MODULE_COLORS[module_name]),
                linestyle=MODULE_CURVE_LINESTYLES.get(module_name, "-"),
            )
            for module_name in modules
        ]
        rows_by_variant, metric_names = load_or_download_rows(
            project=normalize_project(PROJECT),
            run_family_tag="0517_ppo_sac_module",
            variant_specs=specs,
            seeds={1, 2, 3},
            output_dir=Path("outputs/wandb/0517_ppo_sac_module_metrics"),
            force_refresh=False,
        )
        metric_names = [metric for metric in PLOT_METRIC_ORDER if metric in metric_names]
        aggregated = aggregate_metrics(rows_by_variant, metric_names, "_step")
        stem = REPORT_OUTPUT_DIR / name
        plot_metric_grid(
            plt=plt,
            variant_specs=specs,
            aggregated=aggregated,
            metric_names=metric_names,
            x_label="训练迭代",
            title=_zh_title(title),
            output_stem=stem,
        )
        output_paths.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    return output_paths


def _read_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("summary", {})


def _condition_from_dir(path: Path) -> tuple[str, str, int | None] | None:
    name = path.name
    match = re.fullmatch(r"compute_ppo_(\dx\d+)_seed(\d+)", name)
    if match:
        return ("compute", match.group(1), int(match.group(2)))
    match = re.fullmatch(r"algorithm_sac_(\dx\d+)_seed(\d+)", name)
    if match:
        return ("algorithm", f"sac {match.group(1)}", int(match.group(2)))
    match = re.fullmatch(r"module_ppo_(.+)_seed(\d+)", name)
    if match:
        return ("module", match.group(1), int(match.group(2)))
    match = re.fullmatch(r"data_ppo_(.+?)(?:_seed(\d+))?", name)
    if match:
        seed = int(match.group(2)) if match.group(2) else None
        return ("data", match.group(1), seed)
    return None


def _iter_final_eval_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs: set[Path] = set()
    for summary_path in root.glob("**/*.json"):
        if "wall_time_checkpoints" in summary_path.parts:
            continue
        if summary_path.stem not in COLLECT_TASKS:
            continue
        if _condition_from_dir(summary_path.parent) is not None:
            dirs.add(summary_path.parent)
    return sorted(dirs)


def collect_final_rows(root: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_root = root if root is not None else SECTION_ROOT
    for path in _iter_final_eval_dirs(eval_root):
        parsed = _condition_from_dir(path)
        if parsed is None:
            continue
        group, label, seed = parsed
        for task in COLLECT_TASKS:
            summary = _read_summary(path / f"{task}.json")
            if summary is None:
                continue
            row = {
                "group": group,
                "label": label,
                "seed": seed,
                "task": task,
            }
            for key, _ in FINAL_METRICS:
                row[key] = float(summary[key])
            row["num_envs"] = int(summary.get("num_envs", 0))
            row["steps"] = int(summary.get("steps", 0))
            row["num_finished_episodes"] = int(summary.get("num_finished_episodes", 0))
            rows.append(row)
    return rows


def collect_legacy_progress_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LEGACY_EVAL_OUTPUT_DIR.iterdir() if LEGACY_EVAL_OUTPUT_DIR.exists() else []):
        if not path.is_dir():
            continue
        parsed = _condition_from_dir(path)
        if parsed is None:
            continue
        group, label, seed = parsed
        for task in LEGACY_COLLECT_TASKS:
            summary = _read_summary(path / f"{task}.json")
            if summary is None:
                continue
            rows.append(
                {
                    "group": group,
                    "label": label,
                    "seed": seed,
                    "task": task,
                    "progress_mean": float(summary["progress_mean"]),
                }
            )
    return rows


def collect_latest_algorithm_rows() -> list[dict[str, Any]]:
    eval_root = SECTION_ROOT / "06_1_eval_protocol" / "outputs" / "eval"
    rows: list[dict[str, Any]] = []
    for path in _iter_final_eval_dirs(eval_root):
        parsed = _condition_from_dir(path)
        if parsed is None:
            continue
        group, label, seed = parsed
        if group == "compute" and label == "8x8192":
            out_group = "algorithm"
            out_label = "ppo 8x8192"
        elif group == "algorithm" and label == "sac 8x8192":
            out_group = "algorithm"
            out_label = label
        else:
            continue
        for task in FINAL_POLICY_TASKS:
            summary = _read_summary(path / f"{task}.json")
            if summary is None:
                continue
            rows.append(
                {
                    "group": out_group,
                    "label": out_label,
                    "seed": seed,
                    "task": task,
                    "joint_pos": float(summary["joint_pos"]),
                    "body_pos": float(summary["body_pos"]),
                    "body_ori": float(summary["body_ori"]),
                    "num_envs": int(summary.get("num_envs", 0)),
                    "steps": int(summary.get("steps", 0)),
                    "num_finished_episodes": int(summary.get("num_finished_episodes", 0)),
                }
            )
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["label"], row["task"])].append(row)

    result = []
    for (group, label, task), items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "group": group,
            "label": label,
            "task": task,
            "num_runs": len(items),
        }
        for key, _ in FINAL_METRICS:
            values = [float(item[key]) for item in items]
            out[f"{key}_mean"] = mean(values)
            out[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out["num_envs_mean"] = mean(float(item["num_envs"]) for item in items)
        out["steps_mean"] = mean(float(item["steps"]) for item in items)
        out["num_finished_episodes_mean"] = mean(
            float(item["num_finished_episodes"]) for item in items
        )
        result.append(out)
    return result


def aggregate_legacy_progress_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["label"], row["task"])].append(row)

    result = []
    for (group, label, task), items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "group": group,
            "label": label,
            "task": task,
            "num_runs": len(items),
        }
        progress_values = [float(item["progress_mean"]) for item in items]
        out["progress_mean_mean"] = mean(progress_values)
        out["progress_mean_std"] = pstdev(progress_values) if len(progress_values) > 1 else 0.0
        result.append(out)
    return result


def _plot_final_group(
    plt: Any,
    rows: list[dict[str, Any]],
    group: str,
    labels: list[str],
    output_stem: Path,
    title: str,
    tasks: tuple[str, ...] = FINAL_POLICY_TASKS,
) -> list[Path]:
    filtered = [row for row in rows if row["group"] == group and row["label"] in labels]
    if not filtered:
        return []
    data = {(row["label"], row["task"]): row for row in filtered}
    base_height = 5.4 * BAR_FIG_HEIGHT_SCALE
    fig_width = TRAINING_FIG_WIDTH
    fig_height = fig_width * base_height / FINAL_POLICY_BASE_WIDTH
    fig, axes = plt.subplots(
        1,
        len(FINAL_METRICS),
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axes_flat = axes[0]
    dataset_labels = [FINAL_POLICY_TASK_LABELS.get(task, task.replace("-", " ")) for task in tasks]
    x = list(range(len(tasks)))
    width = min(0.8 / max(len(labels), 1), 0.34)
    for axis, (metric, metric_title) in zip(axes_flat, FINAL_METRICS):
        for label_idx, label in enumerate(labels):
            offset = (label_idx - (len(labels) - 1) / 2) * width
            means = [data.get((label, task), {}).get(f"{metric}_mean", math.nan) for task in tasks]
            stds = [data.get((label, task), {}).get(f"{metric}_std", 0.0) for task in tasks]
            axis.bar(
                [pos + offset for pos in x],
                means,
                yerr=stds,
                width=width,
                label=_legend_label(label, group),
                color=_color_for_label(label),
                alpha=0.88,
                capsize=2,
            )
        axis.set_title(metric_title, fontsize=20)
        axis.set_xticks(x)
        axis.set_xticklabels(dataset_labels, rotation=20, ha="right", fontsize=16)
        axis.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=BAR_LEGEND_ANCHOR,
            ncol=len(legend_labels),
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="#c8c8c8",
        )
    fig.suptitle(_zh_title(title), y=BAR_TITLE_Y)
    fig.tight_layout(rect=BAR_LAYOUT_RECT)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]


def _parse_setting(label: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)x(\d+)", label)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _color_for_label(label: str) -> str:
    lower = label.lower()
    if lower.startswith("ppo"):
        return POLICY_COLORS["ppo"]
    if lower.startswith("sac"):
        return POLICY_COLORS["sac"]
    setting = _parse_setting(label)
    if setting is not None:
        return SETTING_COLORS.get(setting, POLICY_COLORS["ppo"])
    if lower in MODULE_COLORS:
        return MODULE_COLORS[lower]
    if lower in DATA_COLORS:
        return DATA_COLORS[lower]
    return "#4c78a8"


def _legacy_comparison_value(
    rows: list[dict[str, Any]],
    policy: str,
    task: str,
) -> tuple[float, float]:
    if policy == "sonic":
        progress, _joint = SONIC_COMPARISON[task]
        return progress, 0.0

    label = "ppo 8x8192" if policy == "ppo" else "sac 8x8192"
    row = next(
        (
            item
            for item in rows
            if item["group"] == "algorithm" and item["label"] == label and item["task"] == task
        ),
        None,
    )
    if row is None:
        return math.nan, 0.0
    return float(row["progress_mean_mean"]), float(row["progress_mean_std"])


def _latest_joint_value(
    rows: list[dict[str, Any]],
    policy: str,
    task: str,
) -> tuple[float, float]:
    label = "ppo 8x8192" if policy == "ppo" else "sac 8x8192"
    row = next(
        (
            item
            for item in rows
            if item["group"] == "algorithm" and item["label"] == label and item["task"] == task
        ),
        None,
    )
    if row is None:
        return math.nan, 0.0
    return float(row["joint_pos_mean"]), float(row["joint_pos_std"])


def _remove_artifact_outputs(stem: Path) -> None:
    for suffix in (".pdf", ".png"):
        output_path = stem.with_suffix(suffix)
        if output_path.exists():
            output_path.unlink()
        for report_dir in REPORT_DIRS:
            report_path = report_dir / f"{stem.name}{suffix}"
            if report_path.exists():
                report_path.unlink()


def _has_complete_rows(
    rows: list[dict[str, Any]],
    group: str,
    labels: list[str],
    tasks: tuple[str, ...],
    required_runs: int,
) -> bool:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        if row["group"] == group and row["label"] in labels and row["task"] in tasks:
            counts[(row["label"], row["task"])] += 1
    return all(counts.get((label, task), 0) >= required_runs for label in labels for task in tasks)


def plot_sonic_comparison(
    legacy_progress_rows: list[dict[str, Any]],
    latest_algorithm_rows: list[dict[str, Any]],
) -> list[Path]:
    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")

    legacy_policies = ["ppo", "sac", "sonic"]
    latest_policies = ["ppo", "sac"]
    policy_labels = {"ppo": "PPO", "sac": "SAC", "sonic": "SONIC"}
    legacy_x = list(range(len(COMPARISON_TASKS)))
    latest_x = list(range(len(FINAL_POLICY_TASKS)))
    legacy_width = min(0.8 / len(legacy_policies), 0.28)
    latest_width = min(0.8 / len(latest_policies), 0.34)
    latest_aggregated_rows = aggregate_rows(latest_algorithm_rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 5.25 * BAR_FIG_HEIGHT_SCALE))
    progress_axis, joint_axis = axes

    for offset_idx, policy in enumerate(legacy_policies):
        offset = (offset_idx - (len(legacy_policies) - 1) / 2) * legacy_width
        heights = []
        yerr = []
        for task in COMPARISON_TASKS:
            value, std = _legacy_comparison_value(legacy_progress_rows, policy, task)
            heights.append(value)
            yerr.append(std)
        progress_axis.bar(
            [pos + offset for pos in legacy_x],
            heights,
            yerr=yerr,
            width=legacy_width,
            label=policy_labels[policy],
            color=POLICY_COLORS[policy],
            alpha=0.9,
            capsize=2,
        )
    progress_axis.set_title("平均完成进度", fontsize=20)
    progress_axis.set_ylabel("完成进度")
    progress_axis.set_xticks(legacy_x)
    progress_axis.set_xticklabels(
        [COMPARISON_TASK_LABELS[task] for task in COMPARISON_TASKS],
        rotation=20,
        ha="right",
    )
    progress_axis.grid(axis="y", alpha=0.25)
    progress_axis.set_ylim(0, 1.08)

    for offset_idx, policy in enumerate(latest_policies):
        offset = (offset_idx - (len(latest_policies) - 1) / 2) * latest_width
        heights = []
        yerr = []
        for task in FINAL_POLICY_TASKS:
            value, std = _latest_joint_value(latest_aggregated_rows, policy, task)
            heights.append(value)
            yerr.append(std)
        joint_axis.bar(
            [pos + offset for pos in latest_x],
            heights,
            yerr=yerr,
            width=latest_width,
            label=policy_labels[policy],
            color=POLICY_COLORS[policy],
            alpha=0.9,
            capsize=2,
        )
    joint_axis.set_title("关节位置跟踪误差", fontsize=20)
    joint_axis.set_ylabel("跟踪误差")
    joint_axis.set_xticks(latest_x)
    joint_axis.set_xticklabels(
        [FINAL_POLICY_TASK_LABELS[task] for task in FINAL_POLICY_TASKS],
        rotation=20,
        ha="right",
    )
    joint_axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=BAR_LEGEND_ANCHOR,
        ncols=len(labels),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#c8c8c8",
    )
    fig.suptitle(_zh_title("Offline evaluation under matched termination"), y=BAR_TITLE_Y)
    fig.tight_layout(rect=BAR_LAYOUT_RECT)
    stem = REPORT_OUTPUT_DIR / "sonic_policy_comparison"
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def plot_final_metrics(raw_rows: list[dict[str, Any]], aggregated_rows: list[dict[str, Any]]) -> list[Path]:
    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")
    paths = []
    expanded_rows = add_alias_rows(raw_rows)
    paths.extend(
        _plot_final_group(
            plt,
            aggregated_rows,
            "compute",
            [f"{nproc}x{num_envs}" for nproc, num_envs in GROUP_4096],
            REPORT_OUTPUT_DIR / "final_policy_compute_product4096",
            _zh_title("Final policy metrics: total environments = 4096"),
        )
    )
    paths.extend(
        _plot_final_group(
            plt,
            aggregated_rows,
            "compute",
            [f"{nproc}x{num_envs}" for nproc, num_envs in GROUP_8GPU],
            REPORT_OUTPUT_DIR / "final_policy_compute_8gpu",
            _zh_title("Final policy metrics: 8 GPUs with increasing environments"),
        )
    )
    paths.extend(
        _plot_final_group(
            plt,
            aggregated_rows,
            "module",
            list(PPO_MODULE_WIDTH),
            REPORT_OUTPUT_DIR / "final_policy_module_width",
            _zh_title("Final policy metrics: PPO width ablation"),
        )
    )
    depth_residual_stem = REPORT_OUTPUT_DIR / "final_policy_module_depth_residual"
    if _has_complete_rows(
        expanded_rows,
        "module",
        list(PPO_MODULE_DEPTH_RESIDUAL),
        FINAL_POLICY_TASKS,
        required_runs=3,
    ):
        paths.extend(
            _plot_final_group(
                plt,
                aggregated_rows,
                "module",
                list(PPO_MODULE_DEPTH_RESIDUAL),
                depth_residual_stem,
                _zh_title("Final policy metrics: PPO depth and residual ablation"),
            )
        )
    else:
        _remove_artifact_outputs(depth_residual_stem)
        print("Skipping final_policy_module_depth_residual: eval results are incomplete.")
    paths.extend(
        _plot_final_group(
            plt,
            aggregated_rows,
            "algorithm",
            ["ppo 8x8192", "sac 8x8192"],
            REPORT_OUTPUT_DIR / "final_policy_algorithm",
            _zh_title("Final policy metrics: PPO vs SAC at 8x8192"),
        )
    )
    paths.extend(
        _plot_final_group(
            plt,
            aggregated_rows,
            "data",
            ["lafan", "lafan_100style", "lafan_100style_real"],
            REPORT_OUTPUT_DIR / "final_policy_data",
            _zh_title("Final policy metrics: PPO data ablation"),
            tasks=DATA_POLICY_TASKS,
        )
    )
    return paths


def add_alias_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(rows)
    for row in rows:
        if row["group"] == "compute" and row["label"] == "8x8192":
            alias = dict(row)
            alias["group"] = "algorithm"
            alias["label"] = "ppo 8x8192"
            result.append(alias)
        if row["group"] == "compute" and row["label"] == "8x8192" and row.get("seed") == 1:
            alias = dict(row)
            alias["group"] = "data"
            alias["label"] = "lafan_100style_real"
            result.append(alias)
    return result


def copy_to_reports(paths: list[Path]) -> None:
    for report_dir in REPORT_DIRS:
        report_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.exists():
                shutil.copy2(path, report_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-wall-time",
        action="store_true",
        help="Do not regenerate the wall-clock checkpoint rollout plot.",
    )
    args = parser.parse_args()

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths = plot_training_curves()
    if not args.skip_wall_time:
        figure_paths.extend(plot_env_metrics_by_wall_time())
    figure_paths.extend(plot_module_curves())
    rows = collect_final_rows()
    write_rows_csv(REPORT_OUTPUT_DIR / "final_policy_metrics_raw.csv", rows)
    aggregated_rows = aggregate_rows(add_alias_rows(rows))
    write_rows_csv(REPORT_OUTPUT_DIR / "final_policy_metrics_summary.csv", aggregated_rows)
    legacy_rows = collect_legacy_progress_rows()
    write_rows_csv(REPORT_OUTPUT_DIR / "legacy_policy_comparison_raw.csv", legacy_rows)
    legacy_aggregated_rows = aggregate_legacy_progress_rows(add_alias_rows(legacy_rows))
    write_rows_csv(REPORT_OUTPUT_DIR / "legacy_policy_comparison_summary.csv", legacy_aggregated_rows)
    latest_comparison_rows = collect_latest_algorithm_rows()
    latest_comparison_aggregated_rows = aggregate_rows(latest_comparison_rows)
    write_rows_csv(
        REPORT_OUTPUT_DIR / "latest_policy_comparison_raw.csv",
        latest_comparison_rows,
    )
    write_rows_csv(
        REPORT_OUTPUT_DIR / "latest_policy_comparison_summary.csv",
        latest_comparison_aggregated_rows,
    )
    figure_paths.extend(
        plot_sonic_comparison(legacy_aggregated_rows, latest_comparison_aggregated_rows)
    )
    figure_paths.extend(plot_final_metrics(rows, aggregated_rows))
    copy_to_reports(figure_paths)
    print(f"Wrote report artifacts under {REPORT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
