# 0517 PPO/SAC Plots: W&B Runs, Eval Data, and Commands

This folder is organized by report section. Each section owns its eval scripts,
plot scripts, and outputs. Shared shell helpers live in `shared/`.

Fixed-step offline eval uses `projects/hdmi/scripts/eval.py` directly:

- task is one of `lafan`, `100style`, or `sonic-subset`;
- `task.num_envs=512`;
- `eval_steps=1000`;
- `headless=true`;
- `task.termination.root_pos_error.enabled=false` is passed on the command line;
- metrics come from reward statistics in the rollout;
- each eval writes a TensorDict `.pt` plus a summary `.json`.

Example eval command:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
uv --project venv/mjlab run --no-sync python /home/elijah/Documents/projects/simple-tracking/active-adaptation/projects/hdmi/scripts/eval.py \
  +exp=ppo/train task=sonic-subset backend=mjlab \
  task.num_envs=512 eval_steps=1000 \
  task.termination.root_pos_error.enabled=false headless=true \
  checkpoint_path=run:elijahgalahad/hdmi/<run_id> \
  eval_output=projects/hdmi/scripts/experiments/0517_ppo_sac/sections/<section>/outputs/eval/<case>/<task>.pt \
  eval_summary_output=projects/hdmi/scripts/experiments/0517_ppo_sac/sections/<section>/outputs/eval/<case>/<task>.json
```

Do not modify `projects/hdmi/cfg/task/tracking-base.yaml` for eval.

## Folder Structure

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/
  shared/
    eval_helpers.sh
    plot_helpers.sh
  sections/
    06_1_eval_protocol/
      eval/eval_ppo_sac.sh
      plots/plot_policy_comparison.sh
      outputs/eval/
      outputs/figures/
    06_2_data_selection/
      eval/eval_data_selection.sh
      plots/plot_offline_data.sh
      plots/plot_orin_deployment.sh
      outputs/eval/
      outputs/figures/
    07_1_env_scaling/
      eval/eval_product4096.sh
      eval/eval_8gpu.sh
      eval/eval_wall_time_checkpoints.sh
      plots/plot_training_curves.sh
      plots/plot_final_policy.sh
      plots/plot_wall_time.sh
      outputs/eval/
      outputs/figures/
      outputs/wandb/
    07_2_module_scaling/
      eval/eval_width.sh
      eval/eval_depth_residual.sh
      plots/plot_training_curves.sh
      plots/plot_final_policy.sh
      outputs/eval/
      outputs/figures/
      outputs/wandb/
```

To inspect eval commands without launching rollouts, prefix any section eval
script with `DRY_RUN=1`, for example:

```bash
DRY_RUN=1 TASKS=lafan projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_1_eval_protocol/eval/eval_ppo_sac.sh
```

## 06.1 Evaluation Protocol: PPO vs SAC vs SONIC

Plots:

```text
sonic_policy_comparison.pdf
final_policy_algorithm.pdf
```

W&B runs:

| Policy | Setting | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| PPO | 8x8192 large | `431jq1ef` | `jt9xxe99` | `i7ppgx6o` |
| SAC | 8x8192 large | `e01agg3e` | `jren57hl` | `7f50bnzh` |
| SONIC | offline baseline constants | none | none | none |

Eval script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_1_eval_protocol/eval/eval_ppo_sac.sh
```

Plot script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_1_eval_protocol/plots/plot_policy_comparison.sh
```

Eval outputs:

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_1_eval_protocol/outputs/eval/{compute_ppo_8x8192_seed*,algorithm_sac_8x8192_seed*}/{lafan,100style,sonic-subset}.{pt,json}
```

Each generated command calls `projects/hdmi/scripts/eval.py` with the protocol at
the top of this file.

Hardcoded progress values used by `sonic_policy_comparison` left panel:

```text
PPO 8x8192 progress:
  LAFAN: 0.821 +- 0.004
  100STYLE: 0.987 +- 0.001
  SONIC subset: 1.0

SAC 8x8192 progress:
  LAFAN: 0.644 +- 0.020
  100STYLE: 0.983 +- 0.002
  SONIC subset: 1.0

SONIC release progress (aligned termination):
  LAFAN: 0.38754557380452753
  100STYLE: 0.933419604653588
  SONIC subset: 1.0
```

SONIC release aligned-termination metrics copied from:

```text
/home/elijah/Documents/projects/simple-tracking/GR00T-WholeBodyControl/sonic_release/eval_metrics_lafan_aligned_termination/metrics_eval.json
/home/elijah/Documents/projects/simple-tracking/GR00T-WholeBodyControl/sonic_release/eval_metrics_100style_aligned_termination/metrics_eval.json
/home/elijah/Documents/projects/simple-tracking/GR00T-WholeBodyControl/sonic_release/eval_metrics_sonic_subset_aligned_termination/metrics_eval.json
```

Hardcoded SONIC release values referenced by the plot script:

```text
LAFAN:
  progress_rate = 0.38754557380452753
  mpjpe_pre_terminate = 1642.2070607220035
  joint_pos_error = 0.11665833306640301

100STYLE:
  progress_rate = 0.933419604653588
  mpjpe_pre_terminate = 457.7359136573471
  joint_pos_error = 0.13149512313537493

SONIC subset:
  progress_rate = 1.0
  mpjpe_pre_terminate = 62.626221981348166
  joint_pos_error = 0.09325201057846842
```

## 06.2 Data Selection: Offline Policy Evaluation

Plot:

```text
final_policy_data.pdf
```

Evaluation datasets:

```text
LAFAN, 100STYLE, SONIC subset
```

W&B runs:

| Data setting | Report label | W&B run |
|---|---|---|
| LAFAN | `lafan` | `0c350l2x` |
| LAFAN+100STYLE | `lafan_100style` | `ouss5fi3` |
| LAFAN+100STYLE+Real | `lafan_100style_real` / `ppo_best` | `431jq1ef` |

Eval script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_2_data_selection/eval/eval_data_selection.sh
```

This section overrides the shared eval task list and runs:

```bash
TASKS="lafan 100style sonic-subset"
```

Each generated command uses `task.num_envs=512`, `eval_steps=1000`,
`headless=true`, and `task.termination.root_pos_error.enabled=false`.

Plot script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_2_data_selection/plots/plot_offline_data.sh
```

Eval outputs:

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_2_data_selection/outputs/eval/data_ppo_{lafan,lafan_100style,lafan_100style_real_seed1}/{lafan,100style,sonic-subset}.{pt,json}
```

## 06.2 Data Selection: Orin Deployment

Plot:

```text
orin_data_selection.pdf
```

W&B runs: none directly. This plot uses deployed Orin tracking logs.

Data source:

```text
/home/elijah/Documents/projects/simple-tracking/sim2real/outputs-orin/deploy_2026_05_18/{ppo_data_lafan,ppo_data_lafan_100style,ppo_best}/run_{1,2,3}/macarena-tracking_error.npz
```

Plot script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/06_2_data_selection/plots/plot_orin_deployment.sh
```

## 07.1 Environment Count Ablation: Product = 4096

Plots:

```text
ppo_compute_product4096_training.pdf
final_policy_compute_product4096.pdf
```

W&B runs:

| Setting | seed1 | seed2 | seed3 |
|---|---|---|---|
| 1x4096 | `wa2iiasl` | `fdvnw2xa` | `qsn7iiab` |
| 2x2048 | `kozjgwks` | `20ujgh3w` | `js5i0g8x` |
| 4x1024 | `qp63ks16` | `86ox36ei` | `8nk0v3dc` |
| 8x512 | `sqvoda0m` | `ky923ol7` | `lnjghqlt` |

Eval script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/eval/eval_product4096.sh
```

Plot scripts:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/plots/plot_training_curves.sh
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/plots/plot_final_policy.sh
```

Eval outputs:

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/outputs/eval/product4096/compute_ppo_{1x4096,2x2048,4x1024,8x512}_seed*/{lafan,100style,sonic-subset}.{pt,json}
```

Training-history cache:

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/outputs/wandb/
```

## 07.1 Environment Count Ablation: 8 GPUs

Plots:

```text
ppo_compute_8gpu_training.pdf
ppo_compute_env_frames_training.pdf
final_policy_compute_8gpu.pdf
```

W&B runs:

| Setting | seed1 | seed2 | seed3 |
|---|---|---|---|
| 8x1024 | `ggh7r3as` | `4dtllhi4` | `l7b8boa2` |
| 8x2048 | `cmk28549` | `cv5lz2ua` | `ieoix5xq` |
| 8x4096 | `hmvc597l` | `oqimvqfi` | `yjzauwnj` |
| 8x8192 | `431jq1ef` | `jt9xxe99` | `i7ppgx6o` |
| 8x8192 8k | `n4111eub` | `rb4tbdq5` | `p56ez0i9` |
| 8x16384 | `sppel7at` | `oyyvto2j` | `wwun9k41` |

Eval script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/eval/eval_8gpu.sh
```

Plot scripts:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/plots/plot_training_curves.sh
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/plots/plot_final_policy.sh
```

`ppo_compute_env_frames_training.pdf` is the compute-ablation conclusion figure
used in the paper. It is a training-history curve rather than a checkpoint eval
curve: the x-axis is `env frames`, recovered as
`(_step + 1) * nproc * num_envs * 32` for these PPO runs. It compares the
8-GPU large runs at the same environment-frame scale.

Paper usage:

- Chinese report figure: `course/reports/final/chinese/sections/07_compute_ablation.tex`,
  `fig:ppo-compute-env-frames-training`;
- included asset: `course/reports/final/chinese/figures/ppo_compute_env_frames_training.pdf`;
- caption intent: compare different environment counts and model capacity at
  the same environment-frame scale under 8 GPUs;
- plotted settings: `8x1024`, `8x2048`, `8x4096`, `8x8192`,
  `8x8192 8k`, and `8x16384`, all with `module_large`;
- plotted metrics: training reward statistics for success rate, joint-position
  error, body-position error, and body-orientation error;
- reason for env-frame x-axis: removes the misleading x-axis shift from
  different per-iteration sample counts, so compute scaling is compared at the
  same amount of environment interaction rather than the same optimizer step;
- paper interpretation: from `8x1024` to `8x8192`, larger parallel sampling
  improves policy quality at the same environment-frame count; from `8x8192`
  to `8x16384`, final-policy gains are limited and same-sample performance can
  degrade, while GPU throughput is already saturated enough that equal-sample
  runs take roughly the same wall-clock time.

Eval outputs:

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/outputs/eval/8gpu/compute_ppo_{8x1024,8x2048,8x4096,8x8192,8x16384}_seed*/{lafan,100style,sonic-subset}.{pt,json}
```

## 07.1 Environment Count Ablation: Env-Frames Performance Comparison

Plot:

```text
ppo_compute_env_frames_training.pdf
```

This is the paper conclusion plot for the environment-count ablation. The
x-axis is environment frames, so `8x8192`, `8x8192 8k`, and `8x16384` can be
compared at matched interaction counts instead of matched optimizer iteration
counts. The `8x8192 8k` curve uses the same large PPO policy/config as the
regular `8x8192` run, but extends `total_iters` from 4000 to 8000 through
`projects/hdmi/cfg/exp/ppo/train-long.yaml`.

W&B runs:

| Setting | seed1 | seed2 | seed3 |
|---|---|---|---|
| 8x1024 large | `ggh7r3as` | `4dtllhi4` | `l7b8boa2` |
| 8x2048 large | `cmk28549` | `cv5lz2ua` | `ieoix5xq` |
| 8x4096 large | `hmvc597l` | `oqimvqfi` | `yjzauwnj` |
| 8x8192 large | `431jq1ef` | `jt9xxe99` | `i7ppgx6o` |
| 8x8192 large 8k | `n4111eub` | `rb4tbdq5` | `p56ez0i9` |
| 8x16384 large | `sppel7at` | `oyyvto2j` | `wwun9k41` |

8k launch assignment:

| Seed | Host | Command |
|---:|---|---|
| 1 | `rp-4090-2` | `GPU_IDS=0,1,2,3,4,5,6,7 bash projects/hdmi/scripts/experiments/0517_ppo_sac/launch/ppo_scale_8x8192_long.sh 1` |
| 2 | `rp-4090-3` | `GPU_IDS=0,1,2,3,4,5,6,7 bash projects/hdmi/scripts/experiments/0517_ppo_sac/launch/ppo_scale_8x8192_long.sh 2` |
| 3 | `rp-4090-5` | `GPU_IDS=0,1,2,3,4,5,6,7 bash projects/hdmi/scripts/experiments/0517_ppo_sac/launch/ppo_scale_8x8192_long.sh 3` |

8k launch script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/launch/ppo_scale_8x8192_long.sh
```

Plot script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_1_env_scaling/plots/plot_training_curves.sh
```

Training-history cache:

```text
outputs/wandb/0517_ppo_sac_scale_metrics/csv/ppo_8x8192_8k_seed{1,2,3}.csv
```

## 07.2 Parameter-Size Ablation: PPO Width

Plots:

```text
ppo_module_width_training.pdf
final_policy_module_width.pdf
```

W&B runs:

| Module | seed1 | seed2 | seed3 |
|---|---|---|---|
| small | `2mea5v9o` | `4v2w2rpm` | `hy9wly18` |
| base | `u5bz60zm` | `jrmeqnw0` | `2r1v4c1e` |
| large | `zhichz7i` | `08i48quf` | `axn7rl8z` |
| huge | `eesh6z82` | `qz4spf2m` | `hrqyspit` |

Eval script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/eval/eval_width.sh
```

Plot scripts:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/plots/plot_training_curves.sh
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/plots/plot_final_policy.sh
```

Eval outputs:

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/outputs/eval/width/module_ppo_{small,base,large,huge}_seed*/{lafan,100style,sonic-subset}.{pt,json}
```

## 07.2 Parameter-Size Ablation: PPO Depth and Residual

Plots:

```text
ppo_module_depth_residual_training.pdf
final_policy_module_depth_residual.pdf
```

W&B runs:

| Module | seed1 | seed2 | seed3 |
|---|---|---|---|
| base | `u5bz60zm` | `jrmeqnw0` | `2r1v4c1e` |
| base_deep | `ktp2prpm` | `btoxxla7` | `as3zjgbl` |
| large | `zhichz7i` | `08i48quf` | `axn7rl8z` |
| large_deep | `8jmjyutm` | `jq2tdskz` | `xprnrl8m` |
| residual | `gwe7ou3y` | `zwr06xvs` | `9zb2hgx6` |

Eval script:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/eval/eval_depth_residual.sh
```

Plot scripts:

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/plots/plot_training_curves.sh
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/plots/plot_final_policy.sh
```

Eval outputs:

```text
projects/hdmi/scripts/experiments/0517_ppo_sac/sections/07_2_module_scaling/outputs/eval/depth_residual/module_ppo_{base,base_deep,large,large_deep,residual}_seed*/{lafan,100style,sonic-subset}.{pt,json}
```
