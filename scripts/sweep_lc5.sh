#!/usr/bin/env bash
# lc5: small-model length-sweep MQAR (Q=1), GDN vs softmax.
# vs lc4: small backbone (2 layers, n_embd=64 — see config/lc5_*.py), vocab back
# to 2048, and 10x more (fully online) data. lr stays 3e-4 (baked into the configs).
# Uses dedicated config/lc5_{gdn,transformer}.py (not the shared lc_len_*.py).
#
# Prereq: bash scripts/gen_lc5_data.sh
#
# Usage:
#   bash scripts/sweep_lc5.sh                  # all S, both mixers, foreground
#   bash scripts/sweep_lc5.sh 32 64            # subset
#   SLURM=1 bash scripts/sweep_lc5.sh          # submit via sbatch (one job per run)
#   MIXER=gdn SLURM=1 bash scripts/sweep_lc5.sh
#
# Env:
#   LOG_PREFIX=lc5   log names slurm/${LOG_PREFIX}_{gdn,softmax}_s{S}_q1.log
#   BATCH_SIZE=256   fixed batch (default 256)
#   VOCAB=2048       vocab size (must match the generated data)
#   DATASET_TAG=_lc5 dataset dir suffix (must match gen_lc5_data.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  SEQS=("$@")
else
  SEQS=(16 32 64 128)
fi
Q=1
BATCH="${BATCH_SIZE:-256}"
PREFIX="${LOG_PREFIX:-lc5}"
VOCAB="${VOCAB:-2048}"
TAG="${DATASET_TAG:-_lc5}"
MIXERS=("${MIXER:+${MIXER}}")
if [[ ${#MIXERS[@]} -eq 0 || -z "${MIXERS[0]}" ]]; then
  MIXERS=(gdn softmax)
fi

PROJECT="$ROOT"
CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=32G --cpus-per-task=4 --requeue}"
TIME="${TIME:-08:00:00}"

run_one() {
  local mixer="$1" S="$2"
  local config log jobname inner dataset out_dir
  dataset="mqar_s${S}_q${Q}${TAG}"
  log="slurm/${PREFIX}_${mixer}_s${S}_q${Q}.log"
  jobname="${PREFIX}-${mixer}-s${S}"
  out_dir="out/${PREFIX}_${mixer}_s${S}_q${Q}"

  if [[ "$mixer" == "gdn" ]]; then
    config="config/lc5_gdn.py"
  else
    config="config/lc5_transformer.py"
  fi

  if [[ ! -d "data/${dataset}" ]]; then
    echo "missing data/${dataset}; run: bash scripts/gen_lc5_data.sh $S"
    exit 1
  fi

  echo "=== $mixer S=$S (2L d=64) vocab=$VOCAB batch=$BATCH -> $log ==="

  inner="cd ${PROJECT} && \
export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${config} \
  --dataset=${dataset} --block_size=${S} --mqar_masked=True --vocab_size=${VOCAB} \
  --batch_size=${BATCH} --compile=False --out_dir=${out_dir} \
  > ${PROJECT}/${log} 2>&1"

  if [[ -n "${SLURM:-}" ]]; then
    # shellcheck disable=SC2086
    sbatch $SBATCH_EXTRA \
      --job-name="$jobname" \
      --time="$TIME" \
      --output="slurm/${jobname}-%j.out" \
      --wrap="bash -lc $(printf '%q' "$inner")"
  else
    bash -lc "$inner" | tee "$log"
  fi
}

mkdir -p slurm
for S in "${SEQS[@]}"; do
  for mixer in "${MIXERS[@]}"; do
    run_one "$mixer" "$S"
  done
done
echo "Sweep launched (${PREFIX}). Check: squeue -u \$USER | grep ${PREFIX}"
