#!/usr/bin/env bash

set -euo pipefail

HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$HELPER_DIR/../../../../../.." && pwd)"
ARTIFACT_DIR="$WORKSPACE_ROOT/outputs/report/final_0517"

copy_report_artifacts() {
    local output_dir="$1"
    shift

    mkdir -p "$output_dir"
    local stem
    local suffix
    for stem in "$@"; do
        for suffix in pdf png; do
            local src="$ARTIFACT_DIR/${stem}.${suffix}"
            if [[ -f "$src" ]]; then
                cp "$src" "$output_dir/"
                echo "[copy] $src -> $output_dir/"
            else
                echo "[missing] $src" >&2
            fi
        done
    done
}
