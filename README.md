# Simple-Tracking

## Setup

Clone the `active-adaptation` repository and the HDMI project:

```bash
git clone -b dev/hdmi https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone https://github.com/EGalahad/aa-hdmi projects/hdmi
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
  nproc_per_node=8 task=lafan_100style_real stages=normal backend=mjlab
```

To play your checkpoints, run:

```bash
uv --project venv/mjlab run projects/hdmi/scripts/play.py \
    task=lafan_100style_real algo=ppo_roa_finetune backend=mjlab \
    checkpoint_path=run:elijahgalahad/hdmi/runs/a03f770b_lafan_100style_finetune
```

## Troubleshooting

If IsaacLab picks up Isaac Sim's bundled Warp instead of the venv-installed
`warp-lang`, clear the cached Omni Warp extensions and retry:

```bash
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.warp*
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.warp*
```
