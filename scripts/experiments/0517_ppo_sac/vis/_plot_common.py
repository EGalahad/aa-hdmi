from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


DEFAULT_PROJECT = "elijahgalahad/hdmi"
DEFAULT_SEEDS = (1, 2, 3)
SUCCESS_RATE_COMPONENTS = (
    "train/stats/termination/motion_timeout",
    "train/stats/termination/root_pos_error",
)
SUCCESS_RATE_METRIC = "derived/success_rate"
WALL_TIME_METRIC = "derived/wall_time_hours"
TRACKING_METRICS = (
    "reward.tracking_metrics/joint_pos",
    "reward.tracking_metrics/body_pos",
    "reward.tracking_metrics/body_ori",
)
RAW_METRICS_TO_DOWNLOAD = (*SUCCESS_RATE_COMPONENTS, *TRACKING_METRICS)
PLOT_METRIC_ORDER = (
    SUCCESS_RATE_METRIC,
    *TRACKING_METRICS,
)
REQUIRED_CACHE_METRICS = (*PLOT_METRIC_ORDER, WALL_TIME_METRIC)
PLOT_METRIC_TITLES = {
    SUCCESS_RATE_METRIC: "成功率",
    "reward.tracking_metrics/joint_pos": "关节位置误差",
    "reward.tracking_metrics/body_pos": "身体位置误差",
    "reward.tracking_metrics/body_ori": "身体姿态误差",
}
ALGO_COLORS = {
    "ppo": "#8b55df",
    "sac": "#d62728",
}
LINE_ALPHA = 0.82
LINE_WIDTH = 2.25
SHADE_ALPHA = 0.16


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    algo: str
    tags: tuple[str, ...]
    color: str
    linestyle: str = "-"


def normalize_project(project: str) -> str:
    if "wandb.ai" not in project:
        return project.strip("/")
    parsed = urlparse(project)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Could not parse W&B project from URL: {project}")
    return f"{parts[0]}/{parts[1]}"


def format_run_path(run: Any) -> str:
    return "/".join(str(part) for part in run.path)


def run_sort_key(run: Any) -> tuple[str, str]:
    return (getattr(run, "updated_at", None) or "", getattr(run, "created_at", None) or "")


def _run_name(run: Any) -> str:
    return str(getattr(run, "name", "") or "")


def extract_seed(run: Any) -> int | None:
    for tag in set(run.tags or []):
        if tag.startswith("seed_") and tag.removeprefix("seed_").isdigit():
            return int(tag.removeprefix("seed_"))

    cfg = getattr(run, "config", {}) or {}
    seed = cfg.get("seed")
    if isinstance(seed, int):
        return seed
    if isinstance(seed, str) and seed.isdigit():
        return int(seed)

    name = _run_name(run)
    marker = "_seed"
    if marker in name:
        suffix = name.rsplit(marker, 1)[1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return int(digits)
    return None


def coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def import_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required to download run data. Install it or use existing cached CSV files."
        ) from exc
    return wandb


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


def try_import_matplotlib() -> Any:
    try:
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
    except ImportError:
        return None
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


def collect_metric_rows(run: Any) -> tuple[list[str], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    metric_names: set[str] = set()
    requested_metrics = set(RAW_METRICS_TO_DOWNLOAD)
    base_timestamp: float | None = None

    for row in run.scan_history():
        filtered: dict[str, float] = {}
        step_value = coerce_float(row.get("_step"))
        if step_value is not None:
            filtered["_step"] = step_value

        runtime_value = coerce_float(row.get("_runtime"))
        timestamp_value = coerce_float(row.get("_timestamp"))
        if base_timestamp is None and timestamp_value is not None:
            base_timestamp = timestamp_value

        if runtime_value is not None:
            filtered[WALL_TIME_METRIC] = runtime_value / 3600.0
            metric_names.add(WALL_TIME_METRIC)
        elif timestamp_value is not None and base_timestamp is not None:
            filtered[WALL_TIME_METRIC] = max(0.0, (timestamp_value - base_timestamp) / 3600.0)
            metric_names.add(WALL_TIME_METRIC)

        for key in requested_metrics:
            value = coerce_float(row.get(key))
            if value is None:
                continue
            filtered[key] = value
            metric_names.add(key)

        success_terms = []
        for key in SUCCESS_RATE_COMPONENTS:
            value = filtered.get(key)
            if value is None:
                break
            success_terms.append(value)
        if len(success_terms) == len(SUCCESS_RATE_COMPONENTS):
            filtered[SUCCESS_RATE_METRIC] = sum(success_terms)
            metric_names.add(SUCCESS_RATE_METRIC)

        if len(filtered) > 1:
            rows.append(filtered)

    metrics = [metric for metric in PLOT_METRIC_ORDER if metric in metric_names]
    metrics.extend(sorted(metric_names.difference(metrics)))
    return metrics, rows


def write_csv(path: Path, metrics: list[str], rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["_step", *metrics], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        metrics = [field for field in reader.fieldnames if field != "_step"]
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                float_value = coerce_float(value)
                if float_value is not None:
                    parsed[key] = float_value
            if parsed:
                rows.append(parsed)
        return metrics, rows


def cache_has_required_metrics(metrics: Iterable[str]) -> bool:
    metric_set = set(metrics)
    return all(metric in metric_set for metric in REQUIRED_CACHE_METRICS)


def find_matching_runs(
    api: Any,
    project: str,
    run_family_tag: str,
    variant_specs: list[VariantSpec],
    seeds: set[int],
) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {spec.key: {} for spec in variant_specs}
    runs = list(api.runs(project, filters={"tags": {"$in": [run_family_tag]}}))
    runs.sort(key=run_sort_key, reverse=True)

    match_specs = sorted(variant_specs, key=lambda spec: len(spec.tags), reverse=True)
    for run in runs:
        run_tags = set(run.tags or [])
        seed = extract_seed(run)
        if seed is None or seed not in seeds:
            continue
        for spec in match_specs:
            if set(spec.tags).issubset(run_tags):
                grouped[spec.key].setdefault(seed, run)
                break
    return grouped


def load_or_download_rows(
    *,
    project: str,
    run_family_tag: str,
    variant_specs: list[VariantSpec],
    seeds: set[int],
    output_dir: Path,
    force_refresh: bool,
) -> tuple[dict[str, list[list[dict[str, float]]]], list[str]]:
    csv_dir = output_dir / "csv"
    rows_by_variant: dict[str, list[list[dict[str, float]]]] = {spec.key: [] for spec in variant_specs}
    all_metrics: set[str] = set()
    missing: list[tuple[VariantSpec, int, Path]] = []

    for spec in variant_specs:
        for seed in sorted(seeds):
            csv_path = csv_dir / f"{spec.key}_seed{seed}.csv"
            if not force_refresh and csv_path.exists():
                metrics, rows = read_csv(csv_path)
                if rows and cache_has_required_metrics(metrics):
                    rows_by_variant[spec.key].append(rows)
                    all_metrics.update(metrics)
                    continue
            missing.append((spec, seed, csv_path))

    manifest_lines = []
    if missing:
        wandb = import_wandb()
        api = wandb.Api()
        runs_by_variant = find_matching_runs(
            api,
            project,
            run_family_tag,
            variant_specs,
            seeds,
        )
        for spec, seed, csv_path in missing:
            run = runs_by_variant.get(spec.key, {}).get(seed)
            if run is None:
                manifest_lines.append(f"MISSING {spec.key} seed={seed}")
                continue
            metrics, rows = collect_metric_rows(run)
            if not rows:
                manifest_lines.append(f"EMPTY {spec.key} seed={seed} run={format_run_path(run)}")
                continue
            write_csv(csv_path, metrics, rows)
            rows_by_variant[spec.key].append(rows)
            all_metrics.update(metrics)
            manifest_lines.append(f"{spec.key} seed={seed} run={format_run_path(run)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    rows_by_variant = {key: rows for key, rows in rows_by_variant.items() if rows}
    return rows_by_variant, [metric for metric in PLOT_METRIC_ORDER if metric in all_metrics]


def aggregate_metrics(
    rows_by_variant: dict[str, list[list[dict[str, float]]]],
    metric_names: Iterable[str],
    x_key: str,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    aggregated = {}
    for variant_key, runs_rows in rows_by_variant.items():
        metric_to_step_values = {metric: defaultdict(list) for metric in metric_names}
        step_to_x_values = defaultdict(list)
        for rows in runs_rows:
            for row in rows:
                step = row.get("_step")
                if step is None:
                    continue
                x_value = row.get(x_key)
                if x_value is not None:
                    step_to_x_values[step].append(x_value)
                for metric in metric_names:
                    value = row.get(metric)
                    if value is not None:
                        metric_to_step_values[metric][step].append(value)

        variant_metrics = {}
        for metric, step_values in metric_to_step_values.items():
            if not step_values:
                continue
            x_values = []
            means = []
            stds = []
            for step in sorted(step_values):
                values = step_values[step]
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                x_source = step_to_x_values.get(step)
                x_values.append(sum(x_source) / len(x_source) if x_source else step)
                means.append(mean)
                stds.append(math.sqrt(variance))
            variant_metrics[metric] = {"x": x_values, "means": means, "stds": stds}
        if variant_metrics:
            aggregated[variant_key] = variant_metrics
    return aggregated


def plot_metric_grid(
    *,
    plt: Any,
    variant_specs: list[VariantSpec],
    aggregated: dict[str, dict[str, dict[str, list[float]]]],
    metric_names: list[str],
    x_label: str,
    title: str,
    output_stem: Path,
) -> None:
    if not metric_names:
        print("No metrics available to plot.")
        return

    num_cols = min(2, len(metric_names))
    num_rows = math.ceil(len(metric_names) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.8 * num_cols, 5.7 * num_rows), squeeze=False)
    axes_flat = [axis for row in axes for axis in row]

    for axis, metric in zip(axes_flat, metric_names):
        for spec in variant_specs:
            variant = aggregated.get(spec.key)
            if variant is None or metric not in variant:
                continue
            values = variant[metric]
            x_values = values["x"]
            means = values["means"]
            stds = values["stds"]
            lower = [mean - std for mean, std in zip(means, stds)]
            upper = [mean + std for mean, std in zip(means, stds)]
            axis.plot(
                x_values,
                means,
                label=spec.label,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA,
            )
            axis.fill_between(x_values, lower, upper, color=spec.color, alpha=SHADE_ALPHA)
        axis.set_title(PLOT_METRIC_TITLES.get(metric, metric))
        axis.set_xlabel(x_label)
        axis.grid(True, alpha=0.25)

    for axis in axes_flat[len(metric_names) :]:
        axis.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="#c8c8c8",
        )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
