#!/usr/bin/env bash

set -euo pipefail

HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$HELPER_DIR/../../../../../.." && pwd)"
WANDB_PROJECT="${WANDB_PROJECT:-elijahgalahad/hdmi}"
NUM_ENVS="${NUM_ENVS:-512}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
TASKS=(${TASKS:-lafan 100style sonic-subset})

run_hdmi_eval() {
    local algo="$1"
    local run_id="$2"
    local output_dir="$3"
    local key="$4"
    local task="$5"
    local module_name="${6:-}"
    local checkpoint="${7:-}"

    local exp_override="${algo}/train"
    local checkpoint_path="run:${WANDB_PROJECT}/${run_id}"
    if [[ -n "$checkpoint" ]]; then
        checkpoint_path="run:${WANDB_PROJECT}/runs/${run_id}:${checkpoint}"
    fi

    local json_path="${output_dir}/${key}/${task}.json"
    local pt_path="${output_dir}/${key}/${task}.pt"
    if [[ "$FORCE" != "1" && -f "$json_path" && -f "$pt_path" ]]; then
        echo "[skip] ${json_path}"
        return 0
    fi

    mkdir -p "$(dirname "$json_path")"
    local -a cmd=(
        uv --project venv/mjlab run --no-sync python "$WORKSPACE_ROOT/projects/hdmi/scripts/eval.py"
        "+exp=${exp_override}"
        "task=${task}"
        backend=mjlab
        "task.num_envs=${NUM_ENVS}"
        task.termination.root_pos_error.enabled=false
        headless=true
        "checkpoint_path=${checkpoint_path}"
        "eval_steps=${EVAL_STEPS}"
        "eval_output=${pt_path}"
        "eval_summary_output=${json_path}"
    )
    if [[ -n "$module_name" ]]; then
        cmd+=("algo/${algo}/module=${module_name}")
    fi

    (
        cd "$WORKSPACE_ROOT"
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry-run] ${cmd[*]}"
            return 0
        fi
        echo "[run] ${cmd[*]}"
        "${cmd[@]}"
    )
}

eval_run_all_tasks() {
    local algo="$1"
    local run_id="$2"
    local output_dir="$3"
    local key="$4"
    local module_name="${5:-}"
    local checkpoint="${6:-}"

    for task in "${TASKS[@]}"; do
        run_hdmi_eval "$algo" "$run_id" "$output_dir" "$key" "$task" "$module_name" "$checkpoint"
    done
}
