#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../shared/eval_helpers.sh"

OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval/depth_residual"
WIDTH_OUTPUT_DIR="$SCRIPT_DIR/../outputs/eval/width"

mkdir -p "$OUTPUT_DIR"
for shared_module in base large; do
    for seed in 1 2 3; do
        src="$WIDTH_OUTPUT_DIR/module_ppo_${shared_module}_seed${seed}"
        dst="$OUTPUT_DIR/module_ppo_${shared_module}_seed${seed}"
        if [[ -d "$src" ]]; then
            mkdir -p "$dst"
            for artifact in "$src"/*; do
                name="$(basename "$artifact")"
                if [[ ! -e "$dst/$name" ]]; then
                    cp -a "$artifact" "$dst/$name"
                    echo "[copy] $artifact -> $dst/$name"
                fi
            done
        fi
    done
done

for item in \
    "module_ppo_base_seed1 u5bz60zm base" \
    "module_ppo_base_seed2 jrmeqnw0 base" \
    "module_ppo_base_seed3 2r1v4c1e base" \
    "module_ppo_base_deep_seed1 ktp2prpm base_deep" \
    "module_ppo_base_deep_seed2 btoxxla7 base_deep" \
    "module_ppo_base_deep_seed3 as3zjgbl base_deep" \
    "module_ppo_large_seed1 zhichz7i large" \
    "module_ppo_large_seed2 08i48quf large" \
    "module_ppo_large_seed3 axn7rl8z large" \
    "module_ppo_large_deep_seed1 8jmjyutm large_deep" \
    "module_ppo_large_deep_seed2 jq2tdskz large_deep" \
    "module_ppo_large_deep_seed3 xprnrl8m large_deep" \
    "module_ppo_residual_seed1 gwe7ou3y residual" \
    "module_ppo_residual_seed2 zwr06xvs residual" \
    "module_ppo_residual_seed3 9zb2hgx6 residual"
do
    read -r key run_id module_name <<<"$item"
    eval_run_all_tasks ppo "$run_id" "$OUTPUT_DIR" "$key" "$module_name"
done
