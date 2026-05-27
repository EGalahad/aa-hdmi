#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(EXPERIMENT_ROOT / "vis"))

from make_report_artifacts import (  # noqa: E402
    FINAL_POLICY_TASKS,
    REPORT_OUTPUT_DIR,
    SECTION_ROOT,
    _plot_final_group,
    aggregate_rows,
    collect_final_rows,
    copy_to_reports,
    try_import_matplotlib,
    write_rows_csv,
)


def main() -> None:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_final_rows(SECTION_ROOT / "06_2_data_selection" / "outputs" / "eval")
    summary = aggregate_rows(rows)
    write_rows_csv(REPORT_OUTPUT_DIR / "final_policy_data_raw.csv", rows)
    write_rows_csv(REPORT_OUTPUT_DIR / "final_policy_data_summary.csv", summary)

    plt = try_import_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for plotting.")
    paths = _plot_final_group(
        plt=plt,
        rows=summary,
        group="data",
        labels=["lafan", "lafan_100style", "lafan_100style_real"],
        output_stem=REPORT_OUTPUT_DIR / "final_policy_data",
        title="Final policy metrics: PPO data ablation",
        tasks=FINAL_POLICY_TASKS,
    )
    copy_to_reports(paths)


if __name__ == "__main__":
    main()
