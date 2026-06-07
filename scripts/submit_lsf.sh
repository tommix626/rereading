#!/bin/bash
# Submit a training run on LSF (single node × 8 GPUs).
#
# Usage:
#   bash scripts/submit_lsf.sh                                  # 200M GDN (default)
#   bash scripts/submit_lsf.sh RUN_NAME                         # override run name
#   bash scripts/submit_lsf.sh RUN_NAME config/train_gdn.py     # explicit config
#   bash scripts/submit_lsf.sh 200M-transformer config/train_transformer.py
#
# Wall: 12h (~190M @ 524k tok/step × 19k ≈ 3-4h on 8×H100, safety margin).

set -euo pipefail

RUN_NAME="${1:-200M-gdn}"
CONFIG="${2:-config/train_gdn.py}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_ABS="$(readlink -f "$ROOT/$CONFIG")"

mkdir -p "$ROOT/lsf"

GIT_SHA=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "no-git")
GIT_BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no-branch")
GIT_DIRTY=$(git -C "$ROOT" diff --quiet 2>/dev/null || echo "-dirty")

bsub -G grp_models <<EOF
#BSUB -q normal
#BSUB -J ${RUN_NAME//\//-}
#BSUB -n 8
#BSUB -W 12:00
#BSUB -gpu "num=8:mode=exclusive_process"
#BSUB -R "rusage[mem=800GB]"
#BSUB -oo ${ROOT}/lsf/out.%J
#BSUB -eo ${ROOT}/lsf/err.%J

# WandB key + cluster env (matches lm-engine's setup)
source /u/osieberl/dynFLA/nanoGPT/.env
# Reuse the existing nanoGPTLead venv (must have flash-linear-attention installed in it).
source /u/osieberl/futureLM/nanoGPTLead/.venv/bin/activate
cd ${ROOT}

# WandB metadata: run name from arg, notes/tags include git provenance + LSF jobid
export WANDB_NAME="${RUN_NAME}"
export WANDB_NOTES="LSF \${LSB_JOBID} | git ${GIT_BRANCH}@${GIT_SHA}${GIT_DIRTY} | ${CONFIG_ABS}"
export WANDB_TAGS="git-${GIT_SHA}${GIT_DIRTY}"

torchrun --standalone --nproc_per_node=8 ${ROOT}/train.py ${CONFIG_ABS}
EOF
