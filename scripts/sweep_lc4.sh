#!/usr/bin/env bash
# lc4: length-sweep MQAR (Q=1), GDN vs softmax. lc3 with three fixes:
#   - learning_rate 2e-3 -> 3e-4   (LR env, applied to both mixers)
#   - vocab_size    2048 -> 8192   (VOCAB env; must match gen_lc4_data.sh)
#   - N_TRAIN       400k -> 5.12M  (~1 pass / no epoch reuse; see gen_lc4_data.sh)
# Configs (config/lc_len_{gdn,transformer}.py) are shared with lc3/mq4 and left
# untouched; lr/vocab are overridden on the CLI instead.
#
# Prereq: bash scripts/gen_lc4_data.sh
#
# Usage:
#   bash scripts/sweep_lc4.sh                  # all S, both mixers, foreground
#   bash scripts/sweep_lc4.sh 32 64            # subset
#   SLURM=1 bash scripts/sweep_lc4.sh          # submit via sbatch (one job per run)
#   MIXER=gdn SLURM=1 bash scripts/sweep_lc4.sh
#
# Env:
#   LOG_PREFIX=lc4   log names slurm/${LOG_PREFIX}_{gdn,softmax}_s{S}_q1.log
#   BATCH_SIZE=256   fixed batch (default 256)
#   LR=3e-4          learning rate for both mixers
#   VOCAB=8192       vocab size (must match the generated data)
#   DATASET_TAG=_lc4 dataset dir suffix (must match gen_lc4_data.sh)
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
PREFIX="${LOG_PREFIX:-lc4}"
LR="${LR:-3e-4}"
VOCAB="${VOCAB:-8192}"
TAG="${DATASET_TAG:-_lc4}"
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
    config="config/lc_len_gdn.py"
  else
    config="config/lc_len_transformer.py"
  fi

  if [[ ! -d "data/${dataset}" ]]; then
    echo "missing data/${dataset}; run: bash scripts/gen_lc4_data.sh $S"
    exit 1
  fi

  echo "=== $mixer S=$S lr=$LR vocab=$VOCAB batch=$BATCH -> $log ==="

  inner="cd ${PROJECT} && \
export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${config} \
  --dataset=${dataset} --block_size=${S} --mqar_masked=True --vocab_size=${VOCAB} \
  --learning_rate=${LR} \
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
