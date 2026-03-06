Apply mjlab patch

```bash
patch -N -d venv/mjlab -p0 < projects/hdmi/mjlab_local.patch
```

Test actor_std discrepancy

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 uv --project venv/mjlab run torchrun --nproc_per_node 4 projects/hdmi/scripts/train_dummy.py task=lafan-single +exp=train-single-test +algo/module=base backend=mjlab task.num_envs=1024 algo.desired_kl=null


CUDA_VISIBLE_DEVICES=0 uv --project venv/mjlab run projects/hdmi/scripts/train_dummy.py task=lafan-single +exp=train-single-test +algo/module=base backend=mjlab task.num_envs=4096 algo.desired_kl=null
```