# Simple-Tracking

## Setup

Clone the `active-adaptation` repository and the HDMI project:

```bash
git clone -b dev/hdmi https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone https://github.com/EGalahad/aa-hdmi projects/hdmi
```

Setup uv venv directories and install dependencies:

```bash
mkdir -p venv/mjlab
cp projects/hdmi/pyproject-mjlab.toml venv/mjlab/pyproject.toml

mkdir -p venv/isaaclab
cp projects/hdmi/pyproject-isaaclab.toml venv/isaaclab/pyproject.toml
```

The repository should now look like this:

```text
active-adaptation/
├── venv/
│   ├── mjlab/
│   │   └── pyproject.toml
│   └── isaaclab/
│       └── pyproject.toml
├── active_adaptation/
├── projects/
│   └── hdmi/
└─ scripts/
```

Refresh project discovery:

```bash
uv --project venv/mjlab run aa-discover-projects
```

This command generates the project registry at `.cache/projects.json`. Open that file and make sure both HDMI entries are enabled:

Set up environment variables:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```

## Train

Run the new single-stage HDMI PPO:

```bash
uv --project venv/mjlab run torchrun --nproc_per_node=8 projects/hdmi/scripts/train.py \
  task=lafan_100style_real +exp=train-hdmi backend=mjlab
```

Run the sequential ROA training pipeline:

```bash
uv --project venv/mjlab run projects/hdmi/scripts/train_sequential.py \
  nproc_per_node=8 task=lafan_100style_real stages=normal backend=mjlab
```

Run fast sac training:

```bash
uv --project venv/mjlab run projects/hdmi/scripts/train.py \
  task=lafan-single +exp=fast-sac-train backend=mjlab algo.vecnorm=false
```

To play checkpoints:

```bash
uv --project venv/mjlab run projects/hdmi/scripts/play.py \
  task=lafan task/command/future_steps=long \
  algo=ppo_roa_finetune backend=mjlab \
  checkpoint_path=run:elijahgalahad/hdmi/runs/a03f770b_lafan_100style_finetune

uv --project venv/mjlab run projects/hdmi/scripts/play.py \
  task=lafan \
  +exp=train-hdmi backend=mjlab \
  checkpoint_path=run:elijahgalahad/hdmi/runs/kr0vxm4n

uv --project venv/mjlab run projects/hdmi/scripts/play.py \
  task=lafan-single +exp=fast-sac-train backend=mjlab algo.vecnorm=false \
  checkpoint_path=run:elijahgalahad/hdmi/runs/ffjg0e9k task.num_envs=4
```

## Troubleshooting

### IsaacLab Warp cache

If IsaacLab picks up Isaac Sim's bundled Warp instead of the venv-installed
`warp-lang`, clear the cached Omni Warp extensions and retry:

```bash
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.warp*
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.warp*
```

### mjlab MuJoCo compatibility

If mjlab training fails with an error like `mujoco.mjtEnableBit.mjENBL_MULTICCD` missing while importing `mujoco_warp`, your environment likely resolved `mujoco>=3.8`. Pin `mujoco<3.8` and resync the environment:

```bash
uv --project venv/mjlab add 'mujoco<3.8'
uv --project venv/mjlab sync
```
