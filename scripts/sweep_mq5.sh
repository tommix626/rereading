#!/usr/bin/env bash
# mq5: Zoology-faithful MQAR sweep. Softmax (learned pos-emb) or GDN-no-shortconv.
#   - D = num KV pairs (mq4 axis); seq_len N = 6D; all D queried.
#   - LR SWEEP (constant LR) since 3e-4 is a softmax dead-zone (grokking finding).
#   - SINGLE EPOCH: max_iters = NTRAIN // batch  (each example seen once, no repeat).
#
# Prereq: gen data, e.g.  KV_PAIRS=16 N_TRAIN=5120000 python data/mqar/gen_zoology.py
#
# Usage:
#   MIXER=softmax SLURM=1 bash scripts/sweep_mq5.sh 16              # LR sweep, D=16
#   MIXER=softmax LRS="6e-4 3e-3" SLURM=1 bash scripts/sweep_mq5.sh 16 32 64 128
#   MIXER=gdn LRS="6e-4" SLURM=1 bash scripts/sweep_mq5.sh 16 32 64 128
#
# Env: MIXER=softmax|gdn  LRS="..."  BATCH=256  NTRAIN=5120000  VOCAB=8192
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then DS=("$@"); else DS=(16 32 64 128); fi
MIXER="${MIXER:-softmax}"
LRS_STR="${LRS:-1e-4 3e-4 6e-4 1e-3 3e-3 1e-2}"
read -r -a LRS <<< "$LRS_STR"
BATCH="${BATCH:-256}"
NTRAIN="${NTRAIN:-5120000}"
VOCAB="${VOCAB:-8192}"
MAX_ITERS="${MAX_ITERS:-$(( NTRAIN / BATCH ))}"   # single epoch unless MAX_ITERS is set

if [[ "$MIXER" == "gdn" ]]; then CONFIG="config/mq5_gdn_noconv.py"; else CONFIG="config/mq5_transformer.py"; fi

PROJECT="$ROOT"
CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=24G --cpus-per-task=4 --requeue}"
TIME="${TIME:-08:00:00}"
LOGDIR="slurm/mq5"; mkdir -p "$LOGDIR"

for D in "${DS[@]}"; do
  N=$(( 6 * D )); (( N % 2 == 1 )) && N=$(( N + 1 ))
  DATASET="mqar_zoo_d${D}_n${N}"
  if [[ ! -d "data/${DATASET}" ]]; then
    echo "missing data/${DATASET}; run: KV_PAIRS=${D} N_TRAIN=${NTRAIN} python data/mqar/gen_zoology.py"
    exit 1
  fi
  for LR in "${LRS[@]}"; do
    log="${LOGDIR}/mq5_${MIXER}_d${D}_lr${LR}.log"
    out_dir="out/mq5_${MIXER}_d${D}_lr${LR}"
    jobname="mq5-${MIXER}-d${D}-${LR}"
    echo "=== ${MIXER} D=${D} N=${N} lr=${LR} batch=${BATCH} iters=${MAX_ITERS} (1 epoch) -> ${log} ==="
    inner="cd ${PROJECT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} \
  --dataset=${DATASET} --block_size=${N} --mqar_masked=True --vocab_size=${VOCAB} \
  --learning_rate=${LR} --batch_size=${BATCH} --max_iters=${MAX_ITERS} \
  --compile=False --out_dir=${out_dir} \
  > ${PROJECT}/${log} 2>&1"
    if [[ -n "${SLURM:-}" ]]; then
      sbatch $SBATCH_EXTRA --job-name="$jobname" --time="$TIME" \
        --output="${LOGDIR}/${jobname}-%j.out" --wrap="bash -lc $(printf '%q' "$inner")"
    else
      bash -lc "$inner" | tee "$log"
    fi
  done
done
echo "mq5 sweep launched (${MIXER}). Check: squeue -u \$USER | grep mq5-"