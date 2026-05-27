#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


SCRIPT_DIR = Path(__file__).resolve().parent
SECTION_DIR = SCRIPT_DIR.parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[2]
REPORT_OUTPUT_DIR = Path("/home/elijah/Documents/projects/simple-tracking/active-adaptation/outputs/report/final_0517")
REPORT_DIRS = (
    Path("/home/elijah/Documents/projects/simple-tracking/course/reports/final/chinese/figures"),
    Path("/home/elijah/Documents/projects/simple-tracking/course/reports/final/english/figures"),
)
LATEST_EVAL_ROOT = SECTION_DIR / "outputs" / "eval"
BAR_FIG_HEIGHT_SCALE = 0.9
FIG_WIDTH = 13.6
BASE_COMPARISON_WIDTH = 8.8
BASE_FINAL_WIDTH = 9.0
BAR_LAYOUT_RECT = (0, 0.11, 1, 0.925)
BAR_LEGEND_ANCHOR = (0.5, 0.035)
BAR_TITLE_Y = 0.96


def try_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    windows_font = Path.home() / ".local" / "share" / "fonts" / "windows-cjk" / "simsun.ttc"
    if windows_font.exists():
        font_manager.fontManager.addfont(str(windows_font))
    cjk_font = _cjk_sc_font_path()
    if cjk_font is not None and cjk_font.exists():
        font_manager.fontManager.addfont(str(cjk_font))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["SimSun", "Noto Serif CJK SC", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 24,
        }
    )

    return plt


def _cjk_sc_font_path() -> Path | None:
    ttc_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    target_path = Path.home() / ".cache" / "matplotlib" / "NotoSansCJKSC-Regular.otf"
    if target_path.exists():
        return target_path
    if not ttc_path.exists():
        return None
    try:
        from fontTools.ttLib import TTCollection

        target_path.parent.mkdir(parents=True, exist_ok=True)
        TTCollection(str(ttc_path)).fonts[2].save(str(target_path))
    except Exception:
        return None
    return target_path


POLICY_COLORS = {
    "ppo": "#8b55df",
    "sac": "#d62728",
    "sonic": "#76B900",
}

PROGRESS_TASKS = ("lafan", "100style", "sonic-subset")
PROGRESS_TASK_LABELS = {
    "lafan": "LAFAN",
    "100style": "100STYLE",
    "sonic-subset": "SONIC 子集",
}
JOINT_TASKS = ("lafan", "100style", "sonic-subset")
JOINT_TASK_LABELS = {
    "lafan": "LAFAN",
    "100style": "100STYLE",
    "sonic-subset": "SONIC 子集",
}
FINAL_POLICY_TASKS = ("lafan", "100style", "sonic-subset")
FINAL_POLICY_TASK_LABELS = {
    "lafan": "LAFAN",
    "100style": "100STYLE",
    "sonic-subset": "SONIC 子集",
}
FINAL_METRICS = (
    ("joint_pos", "关节位置误差"),
    ("body_pos", "身体位置误差"),
    ("body_ori", "身体姿态误差"),
)
HARDCODED_PROGRESS = {
    "ppo": {
        "lafan": (0.821, 0.004),
        "100style": (0.987, 0.001),
        "sonic-subset": (1.0, 0.0),
    },
    "sac": {
        "lafan": (0.644, 0.020),
        "100style": (0.983, 0.002),
        "sonic-subset": (1.0, 0.0),
    },
    "sonic": {
        "lafan": (0.38754557380452753, 0.0),
        "100style": (0.933419604653588, 0.0),
        "sonic-subset": (1.000, 0.0),
    },
}
SONIC_ALIGNED_TERMINATION = {
    "lafan": {"mpjpe": 1642.2070607220035, "joint_pos": 0.11665833306640301},
    "100style": {"mpjpe": 457.7359136573471, "joint_pos": 0.13149512313537493},
    "sonic-subset": {"mpjpe": 62.626221981348166, "joint_pos": 0.09325201057846842},
}


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _read_json(path).get("summary", {})


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _copy_to_reports(paths: list[Path]) -> None:
    for report_dir in REPORT_DIRS:
        report_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.exists():
                shutil.copy2(path, report_dir / path.name)


def _aggregate(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["label"], row["task"])].append(row)

    result = []
    for (label, task), items in sorted(grouped.items()):
        out = {
            "label": label,
            "task": task,
            "num_runs": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        result.append(out)
    return result


def collect_latest_joint_rows() -> list[dict]:
    rows: list[dict] = []
    for label, key in (("ppo 8x8192", "compute_ppo_8x8192"), ("sac 8x8192", "algorithm_sac_8x8192")):
        for seed in (1, 2, 3):
            run_dir = LATEST_EVAL_ROOT / f"{key}_seed{seed}"
            for task in JOINT_TASKS:
                summary = _read_summary(run_dir / f"{task}.json")
                if summary is None:
                    continue
                rows.append(
                    {
                        "label": label,
                        "task": task,
                        "joint_pos": float(summary["joint_pos"]),
                    }
                )
    return rows


def collect_latest_algorithm_rows() -> list[dict]:
    rows: list[dict] = []
    for label, key in (("ppo 8x8192", "compute_ppo_8x8192"), ("sac 8x8192", "algorithm_sac_8x8192")):
        for seed in (1, 2, 3):
            run_dir = LATEST_EVAL_ROOT / f"{key}_seed{seed}"
            for task in FINAL_POLICY_TASKS:
                summary = _read_summary(run_dir / f"{task}.json")
                if summary is None:
                    continue
                rows.append(
                    {
                        "label": label,
                        "task": task,
                        "joint_pos": float(summary["joint_pos"]),
                        "body_pos": float(summary["body_pos"]),
                        "body_ori": float(summary["body_ori"]),
                    }
                )
    return rows


def plot_sonic_policy_comparison(
    latest_joint_rows: list[dict],
) -> list[Path]:
    plt = try_import_matplotlib()
    joint_summary = _aggregate(latest_joint_rows, ("joint_pos",))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH, FIG_WIDTH * (5.25 * BAR_FIG_HEIGHT_SCALE) / BASE_COMPARISON_WIDTH),
    )

    progress_axis, joint_axis = axes

    progress_policies = ["ppo", "sac", "sonic"]
    progress_labels = {"ppo": "PPO", "sac": "SAC", "sonic": "SONIC"}
    progress_width = min(0.8 / len(progress_policies), 0.34)
    progress_x = list(range(len(PROGRESS_TASKS)))
    for idx, policy in enumerate(progress_policies):
        offset = (idx - (len(progress_policies) - 1) / 2) * progress_width
        heights = []
        yerr = []
        for task in PROGRESS_TASKS:
            value, std = HARDCODED_PROGRESS[policy][task]
            heights.append(value)
            yerr.append(std)
        progress_axis.bar(
            [x + offset for x in progress_x],
            heights,
            yerr=yerr,
            width=progress_width,
            label=progress_labels[policy],
            color=POLICY_COLORS[policy],
            alpha=0.9,
            capsize=2,
        )
    progress_axis.set_title("平均完成进度", fontsize=20)
    progress_axis.set_ylabel("完成进度")
    progress_axis.set_xticks(progress_x)
    progress_axis.set_xticklabels([PROGRESS_TASK_LABELS[t] for t in PROGRESS_TASKS], rotation=20, ha="right")
    progress_axis.grid(axis="y", alpha=0.25)
    progress_axis.set_ylim(0, 1.08)

    joint_data = {(row["label"], row["task"]): row for row in joint_summary}
    joint_policies = ["ppo", "sac", "sonic"]
    joint_labels = {"ppo": "PPO", "sac": "SAC", "sonic": "SONIC"}
    joint_width = min(0.8 / len(joint_policies), 0.34)
    joint_x = list(range(len(JOINT_TASKS)))
    for idx, policy in enumerate(joint_policies):
        offset = (idx - (len(joint_policies) - 1) / 2) * joint_width
        heights = []
        yerr = []
        for task in JOINT_TASKS:
            if policy == "sonic":
                heights.append(SONIC_ALIGNED_TERMINATION[task]["joint_pos"])
                yerr.append(0.0)
            else:
                label = f"{policy} 8x8192"
                row = joint_data.get((label, task), {})
                heights.append(row.get("joint_pos_mean", math.nan))
                yerr.append(row.get("joint_pos_std", 0.0))
        joint_axis.bar(
            [x + offset for x in joint_x],
            heights,
            yerr=yerr,
            width=joint_width,
            label=joint_labels[policy],
            color=POLICY_COLORS[policy],
            alpha=0.9,
            capsize=2,
        )
    joint_axis.set_title("关节位置跟踪误差", fontsize=20)
    joint_axis.set_ylabel("跟踪误差")
    joint_axis.set_xticks(joint_x)
    joint_axis.set_xticklabels([JOINT_TASK_LABELS[t] for t in JOINT_TASKS], rotation=20, ha="right")
    joint_axis.grid(axis="y", alpha=0.25)

    handles0, labels0 = progress_axis.get_legend_handles_labels()
    handles1, labels1 = joint_axis.get_legend_handles_labels()
    seen = {}
    for h, l in [*zip(handles0, labels0), *zip(handles1, labels1)]:
        seen.setdefault(l, h)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="lower center",
        bbox_to_anchor=BAR_LEGEND_ANCHOR,
        ncols=len(seen),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#c8c8c8",
    )
    fig.suptitle("PPO、SAC 与 SONIC 的离线评测对比", y=BAR_TITLE_Y)
    fig.tight_layout(rect=BAR_LAYOUT_RECT)

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = REPORT_OUTPUT_DIR / "sonic_policy_comparison"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def plot_final_policy_algorithm(rows: list[dict]) -> list[Path]:
    plt = try_import_matplotlib()
    summary = _aggregate(rows, tuple(metric for metric, _ in FINAL_METRICS))
    data = {(row["label"], row["task"]): row for row in summary}
    labels = ["ppo 8x8192", "sac 8x8192"]
    x = list(range(len(FINAL_POLICY_TASKS)))
    width = min(0.8 / len(labels), 0.34)

    fig, axes = plt.subplots(
        1,
        len(FINAL_METRICS),
        figsize=(FIG_WIDTH, FIG_WIDTH * (5.4 * BAR_FIG_HEIGHT_SCALE) / BASE_FINAL_WIDTH),
        squeeze=False,
    )
    axes_flat = axes[0]
    for axis, (metric, title) in zip(axes_flat, FINAL_METRICS):
        for idx, label in enumerate(labels):
            offset = (idx - (len(labels) - 1) / 2) * width
            means = [data.get((label, task), {}).get(f"{metric}_mean", math.nan) for task in FINAL_POLICY_TASKS]
            stds = [data.get((label, task), {}).get(f"{metric}_std", 0.0) for task in FINAL_POLICY_TASKS]
            color = POLICY_COLORS["ppo"] if label.startswith("ppo") else POLICY_COLORS["sac"]
            axis.bar([pos + offset for pos in x], means, yerr=stds, width=width, label=label.split()[0].upper(), color=color, alpha=0.88, capsize=2)
        axis.set_title(title, fontsize=20)
        axis.set_xticks(x)
        axis.set_xticklabels([FINAL_POLICY_TASK_LABELS[t] for t in FINAL_POLICY_TASKS], rotation=20, ha="right", fontsize=16)
        axis.grid(axis="y", alpha=0.25)
    handles, labels_out = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        loc="lower center",
        bbox_to_anchor=BAR_LEGEND_ANCHOR,
        ncol=len(labels_out),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#c8c8c8",
    )
    fig.suptitle("PPO 与 SAC 的离线跟踪精度对比", y=BAR_TITLE_Y)
    fig.tight_layout(rect=BAR_LAYOUT_RECT)

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = REPORT_OUTPUT_DIR / "final_policy_algorithm"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def main() -> None:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    latest_joint_rows = collect_latest_joint_rows()
    latest_algorithm_rows = collect_latest_algorithm_rows()

    _write_rows_csv(REPORT_OUTPUT_DIR / "latest_policy_comparison_raw.csv", latest_joint_rows)
    _write_rows_csv(REPORT_OUTPUT_DIR / "final_policy_algorithm_raw.csv", latest_algorithm_rows)

    paths = []
    paths.extend(plot_sonic_policy_comparison(latest_joint_rows))
    paths.extend(plot_final_policy_algorithm(latest_algorithm_rows))
    _copy_to_reports(paths)


if __name__ == "__main__":
    main()
