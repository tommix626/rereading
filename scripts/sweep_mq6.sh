#!/usr/bin/env bash
# mq6: faithful Zoology-Figure-2 RECIPE in OUR repo (our train.py/model.py).
# Data = actual zoology generator (data/mqar/gen_from_zoology.py -> mqar_zoo6_*).
# Training = Figure-2 protocol, DIFFERENT from single-epoch mq5:
#   - 100k train / 3k val examples, ~64 epochs (max_iters = 64*100000/batch)
#   - batch schedule: seqlen<=128 -> 512, 256 -> 256, 512 -> 128
#   - COSINE LR -> 0 (warmup 0), AdamW wd=0.1, vocab 8192
#   - 4-point LR sweep np.logspace(-4,-2,4); we VISUALIZE ALL curves (not the max)
# Models: softmax (mq5_transformer: learned-abs, 1 head) vs GDN (mq5_gdn_noconv: short-conv OFF).
#
# Usage:
#   SLURM=1 bash scripts/sweep_mq6.sh                       # 2 x 4 x 4 = 32 jobs
#   MIXERS=softmax SEQLENS="64 128" SLURM=1 bash scripts/sweep_mq6.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

MIXERS_STR="${MIXERS:-softmax gdn}"; read -r -a MIXERS_A <<< "$MIXERS_STR"
SEQLENS_STR="${SEQLENS:-64 128 256 512}"; read -r -a SEQLENS_A <<< "$SEQLENS_STR"
LRS_STR="${LRS:-1e-4 4.6416e-4 2.1544e-3 1e-2}"; read -r -a LRS_A <<< "$LRS_STR"

N_EPOCHS="${N_EPOCHS:-64}"; N_TRAIN=100000
declare -A KV=( [64]=4 [128]=8 [256]=16 [512]=64 )

CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=24G --cpus-per-task=4 --requeue}"
TIME="${TIME:-12:00:00}"
LOGDIR="slurm/mq6"; mkdir -p "$LOGDIR"

seqlen_params() {  # sets BATCH, ITERS, EVAL for a seqlen (Figure-2 batch schedule)
  case "$1" in
    64|128) BATCH=512 ;;
    256)    BATCH=256 ;;
    512)    BATCH=128 ;;
  esac
  ITERS=$(( N_EPOCHS * N_TRAIN / BATCH ))
  EVAL=$(( ITERS / 50 )); if (( EVAL < 100 )); then EVAL=100; fi
}

for MIXER in "${MIXERS_A[@]}"; do
  if [[ "$MIXER" == "gdn" ]]; then CONFIG="config/mq5_gdn_noconv.py"; POSEMB="";
  else CONFIG="config/mq5_transformer.py"; POSEMB="--use_rope=True --use_learned_pos_emb=False"; fi  # softmax: RoPE
  for SEQLEN in "${SEQLENS_A[@]}"; do
    D=${KV[$SEQLEN]}; DATASET="mqar_zoo6_d${D}_n${SEQLEN}"
    [[ -d "data/${DATASET}" ]] || { echo "missing data/${DATASET} (run data/mqar/gen_from_zoology.py)"; exit 1; }
    seqlen_params "$SEQLEN"
    for LR in "${LRS_A[@]}"; do
      tag="mq6_${MIXER}_s${SEQLEN}_lr${LR}"
      log="${LOGDIR}/${tag}.log"; out_dir="out/${tag}"; jobname="mq6-${MIXER}-s${SEQLEN}-${LR}"
      echo "=== ${tag}: ${CONFIG##*/} on ${DATASET} batch=${BATCH} iters=${ITERS} eval=${EVAL} lr=${LR} -> ${log} ==="
      inner="cd ${ROOT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} ${POSEMB} \
  --dataset=${DATASET} --block_size=${SEQLEN} --mqar_masked=True --vocab_size=8192 \
  --learning_rate=${LR} --batch_size=${BATCH} --max_iters=${ITERS} \
  --decay_lr=True --warmup_iters=0 --constant_iters=0 --lr_decay_iters=${ITERS} --lr_decay_factor=0.0 \
  --eval_interval=${EVAL} --eval_iters=50 --save_checkpoint=False \
  --compile=False --out_dir=${out_dir} > ${ROOT}/${log} 2>&1"
      if [[ -n "${SLURM:-}" ]]; then
        sbatch $SBATCH_EXTRA --job-name="$jobname" --time="$TIME" \
          --output="${LOGDIR}/${jobname}-%j.out" --wrap="bash -lc $(printf '%q' "$inner")"
      else
        bash -lc "$inner" | tee "$log"
      fi
    done
  done
done
echo "mq6 launched (${MIXERS_A[*]}). Check: squeue -u \$USER | grep mq6-"
