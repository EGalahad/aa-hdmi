#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_ROOT="/home/elijah/Documents/projects/simple-tracking/course/reports"
OUTPUT_DIR="$SCRIPT_DIR/../outputs/figures"

mkdir -p "$OUTPUT_DIR"
(
    cd "$REPORT_ROOT"
    /home/elijah/Documents/projects/simple-tracking/active-adaptation/venv/mjlab/.venv/bin/python3 scripts/plot_orin_data_selection.py
)

for suffix in pdf png; do
    src="$REPORT_ROOT/final/chinese/figures/orin_data_selection.${suffix}"
    if [[ -f "$src" ]]; then
        cp "$src" "$OUTPUT_DIR/"
        echo "[copy] $src -> $OUTPUT_DIR/"
    else
        echo "[missing] $src" >&2
    fi
done
