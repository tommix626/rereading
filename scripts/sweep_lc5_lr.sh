#!/usr/bin/env bash
# LR sweep for the softmax MQAR debug: our model, S=16, constant LR (no decay).
# Relies on config/lc5_transformer.py having decay_lr=False -> LR is held constant
# from step 0 (get_lr, incl. warmup, is bypassed when decay_lr=False).
#
# Usage:
#   SLURM=1 bash scripts/sweep_lc5_lr.sh                 # default LR grid, softmax, S=16
#   SLURM=1 bash scripts/sweep_lc5_lr.sh 1e-3 3e-3       # custom LRs
#   MIXER=gdn S=16 SLURM=1 bash scripts/sweep_lc5_lr.sh  # sweep GDN instead
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then LRS=("$@"); else LRS=(1e-4 3e-4 6e-4 1e-3 3e-3 1e-2); fi
S="${S:-16}"
MIXER="${MIXER:-softmax}"
VOCAB="${VOCAB:-2048}"
DATASET="mqar_s${S}_q1_lc5"

if [[ "$MIXER" == "gdn" ]]; then CONFIG="config/lc5_gdn.py"; else CONFIG="config/lc5_transformer.py"; fi

PROJECT="$ROOT"
CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=16G --cpus-per-task=4 --requeue}"
TIME="${TIME:-02:00:00}"

if [[ ! -d "data/${DATASET}" ]]; then echo "missing data/${DATASET}"; exit 1; fi
mkdir -p slurm

for LR in "${LRS[@]}"; do
  tag="${LR}"
  log="slurm/lrsweep_${MIXER}_s${S}_lr${tag}.log"
  out_dir="out/lrsweep_${MIXER}_s${S}_lr${tag}"
  jobname="lr-${MIXER}-s${S}-${tag}"
  echo "=== ${MIXER} S=${S} lr=${LR} (constant, decay_lr=False) -> ${log} ==="
  inner="cd ${PROJECT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} \
  --dataset=${DATASET} --block_size=${S} --mqar_masked=True --vocab_size=${VOCAB} \
  --learning_rate=${LR} --batch_size=256 --compile=False --out_dir=${out_dir} \
  > ${PROJECT}/${log} 2>&1"
  if [[ -n "${SLURM:-}" ]]; then
    sbatch $SBATCH_EXTRA --job-name="$jobname" --time="$TIME" \
      --output="slurm/${jobname}-%j.out" --wrap="bash -lc $(printf '%q' "$inner")"
  else
    bash -lc "$inner" | tee "$log"
  fi
done
echo "LR sweep launched (${MIXER}, S=${S}). Check: squeue -u \$USER | grep lr-"