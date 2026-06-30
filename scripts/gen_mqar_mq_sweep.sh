#!/usr/bin/env bash
# Generate multi-query MQAR sweep data.
#
# User notation: S = number of KV pairs in context (before '?' and queries).
#   Q = S / 4
#   seq_len = 2*S + 1 + 2*Q (padded to even)
#
# Defaults (mq4 sweep):
#   S (context pairs) in 16 32 64 128
#   → Q=4,8,16,32 and seq_len=42,82,162,322
#
# Usage:
#   bash scripts/gen_mqar_mq_sweep.sh
#   bash scripts/gen_mqar_mq_sweep.sh 64 128
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  CTX_PAIRS=("$@")
else
  CTX_PAIRS=(16 32 64 128)
fi
VOCAB="${VOCAB_SIZE:-2048}"
N_TRAIN="${N_TRAIN:-400000}"
N_VAL="${N_VAL:-2000}"

echo "Generating mq4 data: S=context pairs, Q=S/4 vocab=$VOCAB train=$N_TRAIN val=$N_VAL"
for S in "${CTX_PAIRS[@]}"; do
  Q=$((S / 4))
  SEQ=$((2 * S + 1 + 2 * Q))
  if (( SEQ % 2 == 1 )); then SEQ=$((SEQ + 1)); fi
  echo "--- ctx_pairs S=$S → Q=$Q seq_len=$SEQ ---"
  POOL_PAIRS="$S" VOCAB_SIZE="$VOCAB" N_TRAIN="$N_TRAIN" N_VAL="$N_VAL" \
    python data/mqar/gen_fixed.py
done
echo "Done. Datasets under data/mqar_ctx{S}_q{S/4}/"
