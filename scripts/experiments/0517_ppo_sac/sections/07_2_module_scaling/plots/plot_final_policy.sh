#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../shared/plot_helpers.sh"

(
    cd "$(cd "$SCRIPT_DIR/../../../../../../../.." && pwd)"
    uv --project venv/mjlab run --no-sync python "$SCRIPT_DIR/plot_final_policy.py"
)
for stem in final_policy_module_width final_policy_module_depth_residual; do
    rm -f "$SCRIPT_DIR/../outputs/figures/${stem}.pdf" "$SCRIPT_DIR/../outputs/figures/${stem}.png"
done
copy_report_artifacts "$SCRIPT_DIR/../outputs/figures" \
    final_policy_module_width \
    final_policy_module_depth_residual
