#!/usr/bin/env bash
# Debug: isolate batch-size vs dataset-size for softmax S=16.
#
# Reference runs:
#   lc2: 40k train, batch=512, 20k iters  → 10.24M example draws, ~256 epochs
#   lc3: 400k train, batch=256, 20k iters → 5.12M example draws, ~13 epochs
#
# This script submits two softmax S=16 jobs on 40k train (data/mqar_s16_q1_n40k):
#   lc_dbg_exp  batch=256, 20k iters — same draws/epoch regime as lc3, lc2-sized train
#   lc_dbg_bs   batch=256, 40k iters — same total draws as lc2 (10.24M), tests batch only
#
# Usage:
#   bash scripts/debug_lc_exposure.sh
#   SLURM=1 bash scripts/debug_lc_exposure.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

S=16
Q=1
BATCH=256
DATASET="mqar_s${S}_q${Q}_n40k"
CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
PROJECT="$ROOT"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=32G --cpus-per-task=4 --requeue}"
TIME="${TIME:-04:00:00}"

echo "=== generating ${DATASET} (40k train / 2k val) ==="
SEQ_LEN="$S" NUM_QUERIES="$Q" N_TRAIN=40000 N_VAL=2000 DATASET_TAG=_n40k \
  python data/mqar/gen_fixed.py

run_one() {
  local prefix="$1" max_iters="$2"
  local log="slurm/${prefix}_softmax_s${S}_q${Q}.log"
  local jobname="${prefix}-softmax-s${S}"
  local out_dir="out/${prefix}_softmax_s${S}_q${Q}"

  echo "=== ${prefix}: dataset=${DATASET} batch=${BATCH} max_iters=${max_iters} -> ${log} ==="

  inner="cd ${PROJECT} && \
export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py config/lc_len_transformer.py \
  --dataset=${DATASET} --block_size=${S} --mqar_masked=True --vocab_size=2048 \
  --batch_size=${BATCH} --max_iters=${max_iters} --compile=False --out_dir=${out_dir} \
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
# Match lc3 exposure (5.12M draws) on 40k train — if this hits ~100%, 400k data was the issue.
run_one lc_dbg_exp 20000
# Match lc2 total draws (10.24M) with batch=256 — if this hits ~100%, batch=512 wasn't required.
run_one lc_dbg_bs 40000
echo "Debug jobs launched. Compare to lc2/lc3 softmax S=16 logs."
