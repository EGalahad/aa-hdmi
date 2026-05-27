#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(EXPERIMENT_ROOT / "vis"))

from make_report_artifacts import (  # noqa: E402
    FINAL_POLICY_TASKS,
    GROUP_4096,
    GROUP_8GPU,
    REPORT_OUTPUT_DIR,
    SECTION_ROOT,
    _plot_final_group,
    aggregate_rows,
    collect_final_rows,
    copy_to_reports,
    try_import_matplotlib,
)


def main() -> None:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_final_rows(SECTION_ROOT / "07_1_env_scaling" / "outputs" / "eval")
    summary = aggregate_rows(rows)
    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")
    paths = []
    paths.extend(
        _plot_final_group(
            plt=plt,
            rows=summary,
            group="compute",
            labels=[f"{nproc}x{num_envs}" for nproc, num_envs in GROUP_4096],
            output_stem=REPORT_OUTPUT_DIR / "final_policy_compute_product4096",
            title="Final policy metrics: total environments = 4096",
            tasks=FINAL_POLICY_TASKS,
        )
    )
    paths.extend(
        _plot_final_group(
            plt=plt,
            rows=summary,
            group="compute",
            labels=[f"{nproc}x{num_envs}" for nproc, num_envs in GROUP_8GPU],
            output_stem=REPORT_OUTPUT_DIR / "final_policy_compute_8gpu",
            title="Final policy metrics: 8 GPUs with increasing environments",
            tasks=FINAL_POLICY_TASKS,
        )
    )
    copy_to_reports(paths)


if __name__ == "__main__":
    main()
