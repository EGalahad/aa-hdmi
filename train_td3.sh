#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
HEADLESS="${HEADLESS:-true}"

uv --project venv/mjlab run python -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  projects/hdmi/scripts/train-off_policy.py \
  task=lafan-single \
  +exp=fast-td3-train \
  backend=mjlab \
  headless="${HEADLESS}" \
  task.num_envs=512 \
  algo.buffer_size=100 \
  algo.replay_batch_size=4096 \
  algo.actor_hidden_dim=512 \
  algo.critic_hidden_dim=1024 \
  algo.actor_lr=3e-4 \
  algo.critic_lr=3e-4 \
  algo.tau=0.05 \
  algo.updates_per_step=2 \
  algo.policy_frequency=2 \
  algo.gamma=0.99 \
  algo.use_cdq=true \
  algo.use_layer_norm=false \
  algo.log_std_min=-5.0 \
  algo.log_std_max=-1.0 \
  algo.policy_noise=0.001 \
  algo.noise_clip=0.5 \
  algo.critic_type=distributional \
  "$@"
