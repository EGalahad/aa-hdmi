#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../shared/plot_helpers.sh"

(
    cd "$(cd "$SCRIPT_DIR/../../../../../../../.." && pwd)"
    uv --project venv/mjlab run --no-sync python "$SCRIPT_DIR/plot_training_curves.py"
)
copy_report_artifacts "$SCRIPT_DIR/../outputs/figures" \
    ppo_compute_product4096_training \
    ppo_compute_8gpu_training \
    ppo_compute_env_frames_training
