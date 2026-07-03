#!/usr/bin/env bash
# Generate lc5 length-sweep datasets: Q=1, vocab back to 2048, online-size data.
#
# lc5 vs lc4:
#   - vocab      8192 -> 2048    (revert to smaller vocab)
#   - N_TRAIN    5.12M (= batch 256 * 20k iters -> ~1 pass, no epoch reuse; online)
#   - DATASET_TAG=_lc5           -> data/mqar_s{S}_q1_lc5/ (distinct from lc3's vocab-2048 data)
#
# Streamed in chunks (CHUNK_SIZE, default 2M) so peak RAM stays bounded.
# Zoology masked labels + '?' separator, Q=1, val=2k. S in 16 32 64 128.
#
# Usage (from repo root, in the `gdn` conda env):
#   bash scripts/gen_lc5_data.sh
#   bash scripts/gen_lc5_data.sh 32 64
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
N_TRAIN="${N_TRAIN:-5120000}"
N_VAL="${N_VAL:-2000}"
TAG="${DATASET_TAG:-_lc5}"

echo "Generating lc5 length-sweep data: Q=$Q vocab=$VOCAB train=$N_TRAIN val=$N_VAL tag=$TAG"
for S in "${SEQS[@]}"; do
  echo "--- SEQ_LEN=$S ---"
  SEQ_LEN="$S" NUM_QUERIES="$Q" VOCAB_SIZE="$VOCAB" \
    N_TRAIN="$N_TRAIN" N_VAL="$N_VAL" DATASET_TAG="$TAG" \
    python data/mqar/gen_fixed.py
done
echo "Done. Datasets under data/mqar_s{S}_q${Q}${TAG}/"
