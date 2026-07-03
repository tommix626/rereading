#!/usr/bin/env bash
# 2x2 grid to test escaping the softmax "uniform-over-values" local optimum:
#   positional encoding  : RoPE (use_rope=True)  vs  NoPE (use_rope=False)
#   LR schedule          : cosine decay (decay_lr=True) vs constant (decay_lr=False)
# Softmax, S=16, longer training (MAX_ITERS, default 50000). LR fixed at 3e-4 so
# the two factors are isolated.
#
# Requires model.py `use_rope` switch. Uses config/lc5_transformer.py and overrides
# the grid factors on the CLI.
#
# Usage:
#   SLURM=1 bash scripts/sweep_nope_decay.sh
#   MAX_ITERS=100000 SLURM=1 bash scripts/sweep_nope_decay.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

S="${S:-16}"
LR="${LR:-3e-4}"
MAX_ITERS="${MAX_ITERS:-50000}"
VOCAB="${VOCAB:-2048}"
DATASET="mqar_s${S}_q1_lc5"
CONFIG="config/lc5_transformer.py"

PROJECT="$ROOT"
CONDA_SH="/data/cl/u/xwang397/miniforge3/etc/profile.d/conda.sh"
SBATCH_EXTRA="${SBATCH_EXTRA:---partition=standard --gres=gpu:a6000:1 --mem=16G --cpus-per-task=4 --requeue}"
TIME="${TIME:-08:00:00}"

if [[ ! -d "data/${DATASET}" ]]; then echo "missing data/${DATASET}"; exit 1; fi
mkdir -p slurm

submit_cell() {
  local pe="$1" decay="$2"
  local use_rope decay_lr extra tag log out_dir jobname
  if [[ "$pe" == "rope" ]]; then use_rope=True; else use_rope=False; fi
  if [[ "$decay" == "decayon" ]]; then
    decay_lr=True; extra="--decay_lr=True --lr_decay_iters=${MAX_ITERS}"
  else
    decay_lr=False; extra="--decay_lr=False"
  fi
  tag="${pe}_${decay}"
  log="slurm/grid_softmax_s${S}_${tag}.log"
  out_dir="out/grid_softmax_s${S}_${tag}"
  jobname="grid-s${S}-${tag}"
  echo "=== softmax S=${S} ${pe} ${decay} lr=${LR} iters=${MAX_ITERS} -> ${log} ==="
  # shellcheck disable=SC2086
  inner="cd ${PROJECT} && export HOME=/data/cl/u/xwang397 && \
export TRITON_CACHE_DIR=/data/cl/u/xwang397/.triton && \
export TORCHINDUCTOR_CACHE_DIR=/data/cl/u/xwang397/.inductor && \
export HF_HOME=/data/cl/u/xwang397/.cache/huggingface && \
source ${CONDA_SH} && conda activate gdn && \
srun --unbuffered python -u train.py ${CONFIG} \
  --dataset=${DATASET} --block_size=${S} --mqar_masked=True --vocab_size=${VOCAB} \
  --learning_rate=${LR} --use_rope=${use_rope} ${extra} \
  --max_iters=${MAX_ITERS} --batch_size=256 --compile=False --out_dir=${out_dir} \
  > ${PROJECT}/${log} 2>&1"
  if [[ -n "${SLURM:-}" ]]; then
    sbatch $SBATCH_EXTRA --job-name="$jobname" --time="$TIME" \
      --output="slurm/${jobname}-%j.out" --wrap="bash -lc $(printf '%q' "$inner")"
  else
    bash -lc "$inner" | tee "$log"
  fi
}

for pe in rope nope; do
  for decay in decayon decayoff; do
    submit_cell "$pe" "$decay"
  done
done
echo "Grid launched. Check: squeue -u \$USER | grep grid-"