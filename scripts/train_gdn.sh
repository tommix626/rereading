#!/bin/bash
# Single-GPU GDN training on the CSAIL CL Slurm cluster.
#   Submit from kennel:  sbatch scripts/train_gdn.sh [CONFIG]
#   Default CONFIG = config/ar_gdn.py (MQAR associative-recall demo).
#
# Pinned to a6000 on purpose: our torch (cu124) + triton 3.2 stack does NOT
# support b300/Blackwell (akita) -- it dies with `triton.set_allocator`.
# a100/h100 also work; b300 needs a newer CUDA/triton build.
#
# To scale up later: switch --partition to standard/long, request more GPUs
# (e.g. --gres=gpu:a6000:8) and launch with
#   torchrun --standalone --nproc_per_node=8 train.py <config>
# pointed at a real Megatron dataset instead of the synthetic bins.
#SBATCH --job-name=gdn
#SBATCH --partition=debug
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/%x-%j.out
#SBATCH --requeue

set -euo pipefail

PROJECT=/data/cl/u/xwang397/GDN-minimal-BlueVela

# Home/caches on NFS, never AFS (AFS Kerberos tickets expire mid-job).
export HOME=/data/cl/u/xwang397
export TRITON_CACHE_DIR=$HOME/.triton
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor
export HF_HOME=$HOME/.cache/huggingface

# Activate the conda env directly from NFS (don't rely on AFS ~/.bashrc).
source /data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh
conda activate gdn

cd "$PROJECT"
CONFIG="${1:-config/ar_gdn.py}"
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES config=$CONFIG"

# single GPU, no DDP
srun python -u train.py "$CONFIG"
