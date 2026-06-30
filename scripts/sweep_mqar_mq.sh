#!/usr/bin/env bash
# Multi-query MQAR sweep: GDN vs softmax.
#
# S = context KV pairs (user notation); Q = S/4; block_size derived from layout.
#
# Prereq: bash scripts/gen_mqar_mq_sweep.sh
#
# Usage:
#   SLURM=1 bash scripts/sweep_mqar_mq.sh
#   MIXER=gdn SLURM=1 bash scripts/sweep_mqar_mq.sh 64
#
# Env:
#   LOG_PREFIX=mq4   (default)
#   BATCH_SIZE=256
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  CTX_PAIRS=("$@")
else
  CTX_PAIRS=(16 32 64 128)
fi
BATCH="${BATCH_SIZE:-256}"
PREFIX="${LOG_PREFIX:-mq4}"
MIXERS=("${MIXER:+${MIXER}}")
if [[ ${#MIXERS[@]} -eq 0 || -z "${MIXERS[0]}" ]]; then
  MIXERS=(gdn softmax)
fi

PROJECT="$ROOT"
CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=32G --cpus-per-task=4 --requeue}"
TIME="${TIME:-08:00:00}"

seq_len_for() {
  local S="$1" Q=$((S / 4)) SEQ=$((2 * S + 1 + 2 * Q))
  if (( SEQ % 2 == 1 )); then SEQ=$((SEQ + 1)); fi
  echo "$SEQ"
}

run_one() {
  local mixer="$1" S="$2"
  local Q=$((S / 4))
  local block_size dataset log jobname inner out_dir
  block_size="$(seq_len_for "$S")"
  dataset="mqar_ctx${S}_q${Q}"
  log="slurm/${PREFIX}_${mixer}_ctx${S}_q${Q}.log"
  jobname="${PREFIX}-${mixer}-ctx${S}-q${Q}"
  out_dir="out/${PREFIX}_${mixer}_ctx${S}_q${Q}"

  if [[ "$mixer" == "gdn" ]]; then
    config="config/lc_len_gdn.py"
  else
    config="config/lc_len_transformer.py"
  fi

  if [[ ! -d "data/${dataset}" ]]; then
    echo "missing data/${dataset}; run: bash scripts/gen_mqar_mq_sweep.sh $S"
    exit 1
  fi

  echo "=== $mixer ctx_pairs=$S Q=$Q seq_len=$block_size batch=$BATCH -> $log ==="

  inner="cd ${PROJECT} && \
export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${config} \
  --dataset=${dataset} --block_size=${block_size} --mqar_masked=True --vocab_size=2048 \
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
for S in "${CTX_PAIRS[@]}"; do
  for mixer in "${MIXERS[@]}"; do
    run_one "$mixer" "$S"
  done
done
echo "Sweep launched (${PREFIX}). Check: squeue -u \$USER | grep ${PREFIX}"
