#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(EXPERIMENT_ROOT / "vis"))

from make_report_artifacts import (  # noqa: E402
    FINAL_POLICY_TASKS,
    PPO_MODULE_DEPTH_RESIDUAL,
    PPO_MODULE_WIDTH,
    REPORT_OUTPUT_DIR,
    SECTION_ROOT,
    _has_complete_rows,
    _plot_final_group,
    _remove_artifact_outputs,
    aggregate_rows,
    collect_final_rows,
    copy_to_reports,
    try_import_matplotlib,
)


def main() -> None:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_final_rows(SECTION_ROOT / "07_2_module_scaling" / "outputs" / "eval")
    summary = aggregate_rows(rows)
    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")
    paths = []
    paths.extend(
        _plot_final_group(
            plt=plt,
            rows=summary,
            group="module",
            labels=list(PPO_MODULE_WIDTH),
            output_stem=REPORT_OUTPUT_DIR / "final_policy_module_width",
            title="Final policy metrics: PPO width ablation",
            tasks=FINAL_POLICY_TASKS,
        )
    )
    depth_stem = REPORT_OUTPUT_DIR / "final_policy_module_depth_residual"
    if _has_complete_rows(rows, "module", list(PPO_MODULE_DEPTH_RESIDUAL), FINAL_POLICY_TASKS, 3):
        paths.extend(
            _plot_final_group(
                plt=plt,
                rows=summary,
                group="module",
                labels=list(PPO_MODULE_DEPTH_RESIDUAL),
                output_stem=depth_stem,
                title="Final policy metrics: PPO depth and residual ablation",
                tasks=FINAL_POLICY_TASKS,
            )
        )
    else:
        _remove_artifact_outputs(depth_stem)
    copy_to_reports(paths)


if __name__ == "__main__":
    main()
