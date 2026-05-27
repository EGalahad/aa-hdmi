#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../shared/eval_helpers.sh"

OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval/product4096"

for item in \
    "compute_ppo_1x4096_seed1 wa2iiasl" \
    "compute_ppo_1x4096_seed2 fdvnw2xa" \
    "compute_ppo_1x4096_seed3 qsn7iiab" \
    "compute_ppo_2x2048_seed1 kozjgwks" \
    "compute_ppo_2x2048_seed2 20ujgh3w" \
    "compute_ppo_2x2048_seed3 js5i0g8x" \
    "compute_ppo_4x1024_seed1 qp63ks16" \
    "compute_ppo_4x1024_seed2 86ox36ei" \
    "compute_ppo_4x1024_seed3 8nk0v3dc" \
    "compute_ppo_8x512_seed1 sqvoda0m" \
    "compute_ppo_8x512_seed2 ky923ol7" \
    "compute_ppo_8x512_seed3 lnjghqlt"
do
    read -r key run_id <<<"$item"
    eval_run_all_tasks ppo "$run_id" "$OUTPUT_DIR" "$key" large
done
