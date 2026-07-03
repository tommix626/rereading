#!/usr/bin/env bash
# Generate lc4 length-sweep datasets: Q=1, bigger vocab, ~single-pass (online) data.
#
# lc4 = lc3 with three fixes (see scripts/sweep_lc4.sh):
#   - VOCAB_SIZE  2048 -> 8192     harder to memorize a small token->token table
#   - N_TRAIN     400k -> 5.12M    = batch 256 * 20k iters -> ~1 pass, no epoch reuse
#   - DATASET_TAG=_lc4             -> data/mqar_s{S}_q1_lc4/ (does not touch lc3 data)
#
# Zoology masked labels + '?' separator, Q=1, val=2k. S in 16 32 64 128.
#
# Usage (from repo root, in the `gdn` conda env):
#   bash scripts/gen_lc4_data.sh
#   bash scripts/gen_lc4_data.sh 32 64
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  SEQS=("$@")
else
  SEQS=(16 32 64 128)
fi
Q="${NUM_QUERIES:-1}"
VOCAB="${VOCAB_SIZE:-8192}"
N_TRAIN="${N_TRAIN:-5120000}"
N_VAL="${N_VAL:-2000}"
TAG="${DATASET_TAG:-_lc4}"

echo "Generating lc4 length-sweep data: Q=$Q vocab=$VOCAB train=$N_TRAIN val=$N_VAL tag=$TAG"
for S in "${SEQS[@]}"; do
  echo "--- SEQ_LEN=$S ---"
  SEQ_LEN="$S" NUM_QUERIES="$Q" VOCAB_SIZE="$VOCAB" \
    N_TRAIN="$N_TRAIN" N_VAL="$N_VAL" DATASET_TAG="$TAG" \
    python data/mqar/gen_fixed.py
done
echo "Done. Datasets under data/mqar_s{S}_q${Q}${TAG}/"
