#!/usr/bin/env bash
# ol1 (online 1): longer, fully-ONLINE version of mq6 to isolate the training-TIME axis.
#   - infinite data: every train example generated on the fly by the upstream zoology
#     generator (each seen once). No dataset dir / no epochs.
#   - train up to 100k steps, but EARLY STOP once tail-mean val/acc >= 0.99 (3 evals).
#   - constant LR (no decay confound) so a failure at 100k is unambiguous, not "LR->0".
#   - same models/sizes as mq6: softmax (learned-abs,1 head) vs GDN (short-conv OFF), 2L d=64.
#   - same 4-pt LR sweep; same batch schedule as mq6 (512/512/256/128).
# Deprioritized (--nice) so mq6 / zf2 finish first; if mq6 already succeeds these can be cancelled.
#
# Usage:
#   SLURM=1 bash scripts/sweep_ol1.sh                 # 2 x 4 x 4 = 32 jobs
#   MIXERS=softmax SEQLENS=64 LRS=1e-2 SLURM=1 bash scripts/sweep_ol1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

MIXERS_STR="${MIXERS:-softmax gdn}"; read -r -a MIXERS_A <<< "$MIXERS_STR"
SEQLENS_STR="${SEQLENS:-64 128 256 512}"; read -r -a SEQLENS_A <<< "$SEQLENS_STR"
LRS_STR="${LRS:-1e-4 4.6416e-4 2.1544e-3 1e-2}"; read -r -a LRS_A <<< "$LRS_STR"

MAX_ITERS="${MAX_ITERS:-100000}"
EARLY_ACC="${EARLY_ACC:-0.99}"; EARLY_WIN="${EARLY_WIN:-3}"
declare -A KV=( [64]=4 [128]=8 [256]=16 [512]=64 )

CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=24G --cpus-per-task=4 --requeue}"
TIME="${TIME:-20:00:00}"
LOGDIR="slurm/ol1"; mkdir -p "$LOGDIR"

batch_for() { case "$1" in 64|128) echo 512 ;; 256) echo 256 ;; 512) echo 128 ;; esac; }

for MIXER in "${MIXERS_A[@]}"; do
  if [[ "$MIXER" == "gdn" ]]; then CONFIG="config/mq5_gdn_noconv.py"; POSEMB="";
  else CONFIG="config/mq5_transformer.py"; POSEMB="--use_rope=True --use_learned_pos_emb=False"; fi  # softmax: RoPE
  for SEQLEN in "${SEQLENS_A[@]}"; do
    D=${KV[$SEQLEN]}; BATCH=$(batch_for "$SEQLEN")
    for LR in "${LRS_A[@]}"; do
      tag="ol1_${MIXER}_s${SEQLEN}_lr${LR}"
      log="${LOGDIR}/${tag}.log"; out_dir="out/${tag}"; jobname="ol1-${MIXER}-s${SEQLEN}-${LR}"
      echo "=== ${tag}: ${CONFIG##*/} ONLINE kv=${D} seqlen=${SEQLEN} batch=${BATCH} max=${MAX_ITERS} lr=${LR} -> ${log} ==="
      inner="cd ${ROOT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} ${POSEMB} \
  --dataset=online_mqar --block_size=${SEQLEN} --mqar_masked=True --vocab_size=8192 \
  --online_mqar=True --online_num_kv_pairs=${D} --online_power_a=0.01 --online_random_non_queries=False \
  --learning_rate=${LR} --batch_size=${BATCH} --max_iters=${MAX_ITERS} --decay_lr=False \
  --early_stop_acc=${EARLY_ACC} --early_stop_window=${EARLY_WIN} \
  --eval_interval=1000 --eval_iters=50 --save_checkpoint=False \
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
echo "ol1 launched (${MIXERS_A[*]}). Check: squeue -u \$USER | grep ol1-"
