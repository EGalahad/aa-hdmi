#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../../../../../../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval/wall_time_checkpoints"

cd "$WORKSPACE_ROOT"
uv --project venv/mjlab run --no-sync python "$WORKSPACE_ROOT/projects/hdmi/scripts/experiments/0517_ppo_sac/eval_wall_time_checkpoints.py" \
    --output-dir "$OUTPUT_DIR" \
    --seeds 1 2 3
