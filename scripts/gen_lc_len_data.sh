#!/usr/bin/env bash
# Generate lc length-sweep datasets: fixed Q=1, varying sequence length S.
#
# Defaults (lc3 sweep):
#   vocab=2048, train=400k, val=2k, Q=1, Zoology masked labels + '?' separator
#   S in 16 32 64 128
#
# Usage (from repo root):
#   bash scripts/gen_lc_len_data.sh
#   bash scripts/gen_lc_len_data.sh 32 64
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  SEQS=("$@")
else
  SEQS=(16 32 64 128)
fi
Q="${NUM_QUERIES:-1}"
VOCAB="${VOCAB_SIZE:-2048}"
N_TRAIN="${N_TRAIN:-400000}"
N_VAL="${N_VAL:-2000}"

echo "Generating lc length-sweep data: Q=$Q vocab=$VOCAB train=$N_TRAIN val=$N_VAL"
for S in "${SEQS[@]}"; do
  echo "--- SEQ_LEN=$S ---"
  SEQ_LEN="$S" NUM_QUERIES="$Q" VOCAB_SIZE="$VOCAB" \
    N_TRAIN="$N_TRAIN" N_VAL="$N_VAL" \
    python data/mqar/gen_fixed.py
done
echo "Done. Datasets under data/mqar_s{S}_q${Q}/"
