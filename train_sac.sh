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
  +exp=fast-sac-train \
  backend=mjlab \
  headless="${HEADLESS}" \
  task.num_envs=2048 \
  +task.command.init_start_t_zero=true \
  task.command.rewind_prob=0.0 \
  task.command.init_joint_pos_noise=0.0 \
  task.command.init_joint_vel_noise=0.0 \
  task.command.pose_range.x='[0.0,0.0]' \
  task.command.pose_range.y='[0.0,0.0]' \
  task.command.pose_range.z='[0.0,0.0]' \
  task.command.pose_range.roll='[0.0,0.0]' \
  task.command.pose_range.pitch='[0.0,0.0]' \
  task.command.pose_range.yaw='[0.0,0.0]' \
  task.command.velocity_range.x='[0.0,0.0]' \
  task.command.velocity_range.y='[0.0,0.0]' \
  task.command.velocity_range.z='[0.0,0.0]' \
  task.command.velocity_range.roll='[0.0,0.0]' \
  task.command.velocity_range.pitch='[0.0,0.0]' \
  task.command.velocity_range.yaw='[0.0,0.0]' \
  task.observation.policy.root_ang_vel_history.noise_std=0.0 \
  task.observation.policy.projected_gravity_history.noise_std=0.0 \
  task.observation.policy.joint_pos_history.noise_std=0.0 \
  task.observation.policy.joint_vel_history.noise_std=0.0 \
  algo.replay_batch_size=4096 \
  algo.gamma=0.99 \
  algo.tau=0.05 \
  algo.vecnorm=false \
  algo.updates_per_step=4 \
  algo.policy_frequency=2 \
  algo.target_entropy_ratio=0.5 \
  algo.num_atoms=501 \
  algo.v_min=-50.0 \
  algo.v_max=200.0 \
  algo.log_std_min=-5.0 \
  algo.max_grad_norm=0.0 \
  task.reward.tracking.root_pos.weight=1.0 \
  task.reward.tracking.root_ori.weight=0.5 \
  task.reward.tracking.root_linvel.weight=1.0 \
  task.reward.tracking.root_angvel.weight=1.0 \
  task.reward.tracking.root_angvel.sigma=3.14 \
  task.reward.tracking.body_pos.weight=2.0 \
  task.reward.tracking.body_ori.weight=1.0 \
  task.reward.tracking.body_linvel.weight=1.0 \
  task.reward.tracking.body_angvel.weight=1.0 \
  task.reward.tracking.body_angvel.sigma=3.14 \
  task.reward.tracking.joint_pos.weight=0.0 \
  task.reward.tracking.joint_vel.weight=0.0 \
  task.reward.loco.feet_air_time.enabled=false \
  task.reward.loco.survival.enabled=false \
  task.reward.loco.joint_vel_l2.weight=0.0 \
  task.reward.loco.self_collisions.weight=-0.1 \
  "$@"
