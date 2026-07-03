#!/usr/bin/env bash
# rt = regression testing: walk from the FAILING setup (rt1 = lc5) to the
# SUCCEEDING one (rtX = mq5), flipping bundles/factors to find the culprit.
# Every cell: 2L d=64 softmax, constant LR, full LR sweep, 50k iters.
#
#   rt1  OLD model (RoPE,2h,wd1.0) + OLD data (lc5 single-query, vocab2048)   [old baseline, expect FAIL]
#   rt2  OLD model                 + NEW data (Zoology multi-query, vocab8192) [data-only swap]
#   rt3  NEW model (learned-abs,1h,wd0.1) + OLD data                          [model/HP-only swap]
#   rtX  NEW model                 + NEW data                                 [new baseline = mq5, expect SUCCEED]
#
# vocab follows the data. Note: 50k*256=12.8M > 5.12M data -> ~2.5 epochs on the
# existing datasets (set NTRAIN + regenerate for strict single-epoch).
#
# Usage:
#   SLURM=1 bash scripts/sweep_rt.sh rt1 rt2 rt3 rtX
#   LRS="6e-4 1e-3" SLURM=1 bash scripts/sweep_rt.sh rt2
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CELLS=("$@"); [[ ${#CELLS[@]} -eq 0 ]] && CELLS=(rt1 rt2 rt3 rtX)
LRS_STR="${LRS:-1e-4 3e-4 6e-4 1e-3 3e-3 1e-2}"; read -r -a LRS <<< "$LRS_STR"
BATCH="${BATCH:-256}"
MAX_ITERS="${MAX_ITERS:-50000}"

CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=24G --cpus-per-task=4 --requeue}"
TIME="${TIME:-12:00:00}"
mkdir -p slurm

OLD_MODEL=config/lc5_transformer.py   # RoPE, 2 heads, wd=1.0
NEW_MODEL=config/mq5_transformer.py   # learned-abs pos-emb, 1 head, wd=0.1
# 12.8M-example datasets so 50k*256 steps = exactly 1 epoch (no repeats).
OLD_DATA=mqar_s16_q1_rt;      OLD_VOCAB=2048; OLD_BLOCK=16   # lc5-style: single-query, distractors, sep
NEW_DATA=mqar_zoo_d16_n96_rt; NEW_VOCAB=8192; NEW_BLOCK=96   # Zoology: multi-query, no distractors

cell_spec() {
  case "$1" in
    rt1) CONFIG=$OLD_MODEL; DATASET=$OLD_DATA; VOCAB=$OLD_VOCAB; BLOCK=$OLD_BLOCK ;;
    rt2) CONFIG=$OLD_MODEL; DATASET=$NEW_DATA; VOCAB=$NEW_VOCAB; BLOCK=$NEW_BLOCK ;;
    rt3) CONFIG=$NEW_MODEL; DATASET=$OLD_DATA; VOCAB=$OLD_VOCAB; BLOCK=$OLD_BLOCK ;;
    rtX) CONFIG=$NEW_MODEL; DATASET=$NEW_DATA; VOCAB=$NEW_VOCAB; BLOCK=$NEW_BLOCK ;;
    *) echo "unknown cell $1 (use rt1|rt2|rt3|rtX)"; exit 1 ;;
  esac
}

for CELL in "${CELLS[@]}"; do
  cell_spec "$CELL"
  [[ -d "data/${DATASET}" ]] || { echo "missing data/${DATASET}"; exit 1; }
  for LR in "${LRS[@]}"; do
    mkdir -p "slurm/${CELL}"
    log="slurm/${CELL}/${CELL}_lr${LR}.log"; out_dir="out/${CELL}_lr${LR}"; jobname="${CELL}-${LR}"
    echo "=== ${CELL}: ${CONFIG##*/} on ${DATASET} vocab=${VOCAB} lr=${LR} iters=${MAX_ITERS} -> ${log} ==="
    inner="cd ${ROOT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} \
  --dataset=${DATASET} --block_size=${BLOCK} --mqar_masked=True --vocab_size=${VOCAB} \
  --learning_rate=${LR} --batch_size=${BATCH} --max_iters=${MAX_ITERS} \
  --compile=False --out_dir=${out_dir} > ${ROOT}/${log} 2>&1"
    if [[ -n "${SLURM:-}" ]]; then
      sbatch $SBATCH_EXTRA --job-name="$jobname" --time="$TIME" \
        --output="slurm/${CELL}/${jobname}-%j.out" --wrap="bash -lc $(printf '%q' "$inner")"
    else
      bash -lc "$inner" | tee "$log"
    fi
  done
done
echo "rt launched (${CELLS[*]}). Check: squeue -u \$USER | grep -E 'rt[0-9X]-'"