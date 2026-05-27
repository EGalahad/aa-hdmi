#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(EXPERIMENT_ROOT / "vis"))

from make_report_artifacts import REPORT_OUTPUT_DIR, copy_to_reports, plot_env_metrics_by_wall_time  # noqa: E402


def main() -> None:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_to_reports(plot_env_metrics_by_wall_time())


if __name__ == "__main__":
    main()
