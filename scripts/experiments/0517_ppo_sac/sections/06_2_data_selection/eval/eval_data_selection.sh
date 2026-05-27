#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="${TASKS:-lafan 100style sonic-subset}"
source "$SCRIPT_DIR/../../../shared/eval_helpers.sh"

OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval"

for item in \
    "data_ppo_lafan 0c350l2x large" \
    "data_ppo_lafan_100style ouss5fi3 large" \
    "data_ppo_lafan_100style_real_seed1 431jq1ef large"
do
    read -r key run_id module_name <<<"$item"
    eval_run_all_tasks ppo "$run_id" "$OUTPUT_DIR" "$key" "$module_name"
done
