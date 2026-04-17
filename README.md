# Simple-Tracking

## Setup

Clone the `active-adaptation` repository and the HDMI project:

```bash
git clone -b dev/hdmi https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone https://github.com/EGalahad/aa-hdmi projects/aa-hdmi
```

Refresh project discovery:

```bash
uv --project venv/mjlab run aa-discover-projects
```

This command generates the project registry at `.cache/projects.json`. Open that file and make sure both HDMI entries are enabled:

## Train

Run the sequential training pipeline with the Lafan dataset:

```bash
uv --project venv/mjlab run projects/hdmi/scripts/train_sequential.py \
  nproc_per_node=8 \
  task=lafan \
  stages=normal \
  +algo/module=large \
  backend=mjlab \
  task.num_envs=8192 \
  +task/patches=root_pos_timeout
```

To play your checkpoints, run:

```bash
uv --project venv/mjlab run projects/hdmi/scripts/play.py \
    task=lafan +exp=train \
    +algo/module=large \
    backend=mjlab \
    task.num_envs=16 \
    task.max_episode_steps=4000 \
    checkpoint_path=run:elijahgalahad/hdmi/runs/8c72248f_exp1_ppo_scale_512x8_seed1_train
```

## Troubleshooting

If IsaacLab picks up Isaac Sim's bundled Warp instead of the venv-installed
`warp-lang`, clear the cached Omni Warp extensions and retry:

```bash
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.warp*
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.warp*
```
