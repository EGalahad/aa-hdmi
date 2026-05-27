#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../shared/eval_helpers.sh"

OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval/8gpu"

for item in \
    "compute_ppo_8x1024_seed1 ggh7r3as" \
    "compute_ppo_8x1024_seed2 4dtllhi4" \
    "compute_ppo_8x1024_seed3 l7b8boa2" \
    "compute_ppo_8x2048_seed1 cmk28549" \
    "compute_ppo_8x2048_seed2 cv5lz2ua" \
    "compute_ppo_8x2048_seed3 ieoix5xq" \
    "compute_ppo_8x4096_seed1 hmvc597l" \
    "compute_ppo_8x4096_seed2 oqimvqfi" \
    "compute_ppo_8x4096_seed3 yjzauwnj" \
    "compute_ppo_8x8192_seed1 431jq1ef" \
    "compute_ppo_8x8192_seed2 jt9xxe99" \
    "compute_ppo_8x8192_seed3 i7ppgx6o" \
    "compute_ppo_8x16384_seed1 sppel7at" \
    "compute_ppo_8x16384_seed2 oyyvto2j" \
    "compute_ppo_8x16384_seed3 wwun9k41"
do
    read -r key run_id <<<"$item"
    eval_run_all_tasks ppo "$run_id" "$OUTPUT_DIR" "$key" large
done
