#!/usr/bin/env bash
# fnl1: final capacity suite on the compact K/Q MQAR format (data/mqar/gen_fnl1.py).
#   - softmax with RoPE (the setting shown to fix the clean-0-pad plateau; see
#     reports/mq5_ape_cleanpad_plateau.md) vs GDN (short-conv OFF). 2L, d=64.
#   - sweep K (recall load; Q=K, all pairs queried) x LR (zoology 4-pt grid).
#   - 400k train examples, 16 epochs, batch 256 -> 25k steps. constant LR, wd 0.1.
# Expect: attention solves all K; GDN falls off past its d=64 state capacity.
#
# Usage:
#   SLURM=1 bash scripts/sweep_fnl1.sh                    # 2 x 4 x 4 = 32 jobs
#   MIXERS=softmax KS="16 32" SLURM=1 bash scripts/sweep_fnl1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

MIXERS_STR="${MIXERS:-softmax gdn}"; read -r -a MIXERS_A <<< "$MIXERS_STR"
KS_STR="${KS:-16 32 64 128}"; read -r -a KS_A <<< "$KS_STR"
LRS_STR="${LRS:-1e-4 4.6416e-4 2.1544e-3 1e-2}"; read -r -a LRS_A <<< "$LRS_STR"
BATCH="${BATCH:-256}"; MAX_ITERS="${MAX_ITERS:-25000}"

CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=24G --cpus-per-task=4 --requeue}"
TIME="${TIME:-12:00:00}"
LOGDIR="slurm/fnl1"; mkdir -p "$LOGDIR"

for MIXER in "${MIXERS_A[@]}"; do
  if [[ "$MIXER" == "gdn" ]]; then CONFIG="config/mq5_gdn_noconv.py"; POSEMB="";
  else CONFIG="config/mq5_transformer.py"; POSEMB="--use_rope=True --use_learned_pos_emb=False"; fi  # softmax: RoPE
  for K in "${KS_A[@]}"; do
    Q=$K; SEQLEN=$(( 2*K + 2*Q )); DATASET="mqar_fnl1_k${K}_q${Q}"
    [[ -d "data/${DATASET}" ]] || { echo "missing data/${DATASET} (run: K=${K} python data/mqar/gen_fnl1.py)"; exit 1; }
    for LR in "${LRS_A[@]}"; do
      tag="fnl1_${MIXER}_k${K}_lr${LR}"
      log="${LOGDIR}/${tag}.log"; out_dir="out/${tag}"; jobname="fnl1-${MIXER}-k${K}-${LR}"
      echo "=== ${tag}: ${CONFIG##*/} K=${K} seqlen=${SEQLEN} batch=${BATCH} iters=${MAX_ITERS} lr=${LR} -> ${log} ==="
      inner="cd ${ROOT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} ${POSEMB} \
  --dataset=${DATASET} --block_size=${SEQLEN} --mqar_masked=True --vocab_size=8192 \
  --learning_rate=${LR} --batch_size=${BATCH} --max_iters=${MAX_ITERS} \
  --eval_interval=500 --eval_iters=50 --save_checkpoint=False \
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
echo "fnl1 launched (${MIXERS_A[*]}). Check: squeue -u \$USER | grep fnl1-"
