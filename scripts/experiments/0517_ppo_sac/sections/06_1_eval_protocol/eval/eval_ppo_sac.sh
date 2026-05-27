#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../shared/eval_helpers.sh"

OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval"

for item in \
    "ppo compute_ppo_8x8192_seed1 431jq1ef large" \
    "ppo compute_ppo_8x8192_seed2 jt9xxe99 large" \
    "ppo compute_ppo_8x8192_seed3 i7ppgx6o large" \
    "sac algorithm_sac_8x8192_seed1 e01agg3e large" \
    "sac algorithm_sac_8x8192_seed2 jren57hl large" \
    "sac algorithm_sac_8x8192_seed3 7f50bnzh large"
do
    read -r algo key run_id module_name <<<"$item"
    eval_run_all_tasks "$algo" "$run_id" "$OUTPUT_DIR" "$key" "$module_name"
done
