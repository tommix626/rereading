# GDN-minimal-BlueVela

Minimal harness for **GDN vs softmax attention** on MQAR (multi-query associative
recall). Same small GPT backbone (2 layers, `n_embd=64`, 1 head, SwiGLU, RMSNorm,
tied embeds); only the sequence mixer differs. The finalized experiment is the
**fnl2** capacity sweep.

## Layout

```
model.py, train.py, configurator.py     # backbone, training loop, CLI overrides
config/mq5_transformer.py               # softmax baseline (run with RoPE)
config/mq5_gdn_noconv.py                # GDN, short conv OFF
data/mqar/gen_fnl2.py                   # fnl2 data generator (compact K/Q MQAR, random fillers)
scripts/sweep_fnl2.sh                   # softmax vs GDN sweep over K and LR
scripts/plot_fnl2_curves.py             # capacity curve + all-LR learning curves
scripts/_watch_fnl2*.sh                 # Slurm completion watchers (auto-plot)
```

## The fnl2 task

Each length-`2K+2Q` example: `[k v]*K` context, then `[q r]*Q` queries (`r` = random
filler). The model, seeing a query key, must predict that key's value (masked CE over
the query positions; `-100` elsewhere). Keys ∈ `[1, V/2)`, values ∈ `[V/2, V)`, vocab
8192, K = Q. Sequence length = `4K`, so K sweeps recall load and context length together.

**Result:** softmax(RoPE) solves recall at every K (seqlen up to 2048+); GDN degrades
once K exceeds its fixed `d=64` recurrent-state capacity (solid to K≈64, collapses by K≈256).

**Note on random fillers:** clean **0-padding** (`RANDOM_NON_QUERIES=0`) is pathological
for softmax — it plateaus at exactly 1/K (copies a plausible in-context value without
matching), and RoPE does not rescue the compact layout. Random fillers force content-based
retrieval. See `reports/mq5_ape_cleanpad_plateau.md`.

## Quickstart

```bash
# 1. generate data for a few K (random fillers by default):
for K in 16 32 64 128; do K=$K python data/mqar/gen_fnl2.py; done

# 2. sweep softmax(RoPE) vs GDN over K x LR on Slurm:
KS="16 32 64 128 256 512" SLURM=1 bash scripts/sweep_fnl2.sh

# 3. plot capacity curve + learning curves -> slurm/plots/fnl2_*.png
python scripts/plot_fnl2_curves.py
```

Requires `fla` (flash-linear-attention), imported at model load even for softmax runs.
Pin `--gres=gpu:a6000:1` — `fla`/triton 3.2 breaks on newer GPUs.
