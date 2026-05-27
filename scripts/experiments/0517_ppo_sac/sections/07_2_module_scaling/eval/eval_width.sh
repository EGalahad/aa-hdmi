#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../shared/eval_helpers.sh"

OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval/width"

for item in \
    "module_ppo_small_seed1 2mea5v9o small" \
    "module_ppo_small_seed2 4v2w2rpm small" \
    "module_ppo_small_seed3 hy9wly18 small" \
    "module_ppo_base_seed1 u5bz60zm base" \
    "module_ppo_base_seed2 jrmeqnw0 base" \
    "module_ppo_base_seed3 2r1v4c1e base" \
    "module_ppo_large_seed1 zhichz7i large" \
    "module_ppo_large_seed2 08i48quf large" \
    "module_ppo_large_seed3 axn7rl8z large" \
    "module_ppo_huge_seed1 eesh6z82 huge" \
    "module_ppo_huge_seed2 qz4spf2m huge" \
    "module_ppo_huge_seed3 hrqyspit huge"
do
    read -r key run_id module_name <<<"$item"
    eval_run_all_tasks ppo "$run_id" "$OUTPUT_DIR" "$key" "$module_name"
done
