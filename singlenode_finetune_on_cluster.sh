#!/bin/bash

NUM_GPUS=${2-1}
VRAM_PER_GPU=${3-20g}
echo "Selected configuration: $1, GPUs: $NUM_GPUS, $VRAM_PER_GPU VRAM/GPU"

# 5 days is the upper limit on Euler (before PartitionTimeLimit kicks in)

# N.B.
# 1. For multiple GPUs, runs will be shown as a group on WandB (not as a single run, which is typically nicer)
# 2. If you set the job name to `bash` or `interactive`, Lightning’s SLURM auto-detection
# will get bypassed and it can launch processes normally. This is apparently needed for single node multi-GPU runs...
sbatch --job-name="bash" \
  --time=1-00:00:00 \
  --nodes=1 \
  --ntasks="$NUM_GPUS" \
  --ntasks-per-node="$NUM_GPUS" \
  --gpus="$NUM_GPUS" \
  --cpus-per-task=4 \
  --mem-per-cpu=10000 \
  --gres=gpumem:"$VRAM_PER_GPU" \
  --output "finetune_$(date "+%F-%T").log" \
  --wrap="./finetune_all_tasks.sh $1"
