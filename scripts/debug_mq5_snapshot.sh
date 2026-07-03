#!/usr/bin/env bash
# Re-run the mq5 D=32, lr=1e-3, softmax config with full (resumable) checkpoint
# snapshots ckpt_iter{N}.pt every SNAP steps (default 500, ~7.7MB each) so the
# grokking trajectory can be replayed for per-position / per-token error analysis.
# 20k steps @ 500 -> 40 files, ~0.3GB.
#
# Deterministic (same seeds/config as the original run) → reproduces the same bumps
# (~80% @ step 3-4k, ~88% @ 5-7k, ~92% @ 13-15k).
#
# Usage:
#   SLURM=1 bash scripts/debug_mq5_snapshot.sh          # D=32 lr=1e-3 (default)
#   D=32 LR=1e-3 SLURM=1 bash scripts/debug_mq5_snapshot.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

D="${D:-32}"
LR="${LR:-1e-3}"
SNAP="${SNAP:-500}"
BATCH="${BATCH:-256}"
NTRAIN="${NTRAIN:-5120000}"
VOCAB="${VOCAB:-8192}"
MAX_ITERS="${MAX_ITERS:-$(( NTRAIN / BATCH ))}"

N=$(( 6 * D )); (( N % 2 == 1 )) && N=$(( N + 1 ))
# DATASET_TAG (e.g. "_rnq") selects a variant dataset dir + tags the out_dir/log,
# so e.g. the random-non-queries (filler) run doesn't clobber the clean run.
DTAG="${DATASET_TAG:-}"
DATASET="mqar_zoo_d${D}_n${N}${DTAG}"
CONFIG="config/mq5_transformer.py"
if [[ ! -d "data/${DATASET}" ]]; then echo "missing data/${DATASET}"; exit 1; fi

PROJECT="$ROOT"
CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=24G --cpus-per-task=4 --requeue}"
TIME="${TIME:-08:00:00}"
LOGDIR="slurm/mq5"; mkdir -p "$LOGDIR"

# RUN_TAG tags the out_dir/log/jobname (e.g. "_rope") so variant runs on the SAME
# dataset don't clobber each other. EXTRA_ARGS passes arbitrary train.py overrides
# (e.g. "--use_rope=True --use_learned_pos_emb=False" for the RoPE variant).
RUN_TAG="${RUN_TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TAG="${DTAG}${RUN_TAG}"
out_dir="out/mq5_softmax_d${D}_lr${LR}${TAG}_snap"
log="${LOGDIR}/mq5_softmax_d${D}_lr${LR}${TAG}_snap.log"
jobname="mq5snap-d${D}-${LR}${TAG}"
echo "=== snapshot run: softmax D=${D} N=${N} lr=${LR} iters=${MAX_ITERS} extra='${EXTRA_ARGS}' -> ${log} ==="
inner="cd ${PROJECT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} \
  --dataset=${DATASET} --block_size=${N} --mqar_masked=True --vocab_size=${VOCAB} \
  --learning_rate=${LR} --batch_size=${BATCH} --max_iters=${MAX_ITERS} \
  --snapshot_checkpoints=True --snapshot_interval=${SNAP} --compile=False --out_dir=${out_dir} \
  ${EXTRA_ARGS} \
  > ${PROJECT}/${log} 2>&1"
if [[ -n "${SLURM:-}" ]]; then
  sbatch $SBATCH_EXTRA --job-name="$jobname" --time="$TIME" \
    --output="${LOGDIR}/${jobname}-%j.out" --wrap="bash -lc $(printf '%q' "$inner")"
else
  bash -lc "$inner" | tee "$log"
fi
echo "launched. snapshots -> ${out_dir}/ckpt_iter{N}.pt ; check: squeue -u \$USER | grep mq5snap-"
