# GDN-minimal-BlueVela

Minimal harness for **GDN vs softmax** on Zoology-style MQAR. Same 4-layer d=256
GPT (SwiGLU, RMSNorm, tied embeds); only the sequence mixer differs.

## Layout

```
model.py, train.py, configurator.py
config/lc_len_gdn.py          GDN MQAR config
config/lc_len_transformer.py  param-matched softmax baseline
data/mqar/zoology.py          MQAR generator (masked labels, '?' separator)
data/mqar/gen_fixed.py        writes data/mqar_{...}/
scripts/gen_lc_len_data.sh    lc3 data (Q=1, seq len S)
scripts/sweep_lc_len.sh       lc3 training sweep
scripts/gen_mqar_mq_sweep.sh  mq4 data (context pairs S, Q=S/4)
scripts/sweep_mqar_mq.sh      mq4 training sweep
scripts/gen_lc4_data.sh       lc4 data (lc3 + bigger vocab + online-size)
scripts/sweep_lc4.sh          lc4 training sweep (lr 3e-4)
scripts/gen_lc5_data.sh       lc5 data (vocab 2048, 10x lc4 / fully online)
scripts/sweep_lc5.sh          lc5 training sweep (small model, 2L d=64)
config/lc5_gdn.py             lc5 small-model GDN config
config/lc5_transformer.py     lc5 small-model softmax config
scripts/plot_lc_curves.py     plot slurm/{lc3,lc4,lc5,mq4}_*.log
```

## Active experiments

### lc3 — length sweep, single query

| Setting | Value |
|---------|-------|
| S | total sequence length ∈ {16, 32, 64, 128} |
| Q | 1 |
| Train / val | 400k / 2k |
| batch | 256, max_iters 20k |
| Data | `data/mqar_s{S}_q1/` |

```bash
bash scripts/gen_lc_len_data.sh
SLURM=1 bash scripts/sweep_lc_len.sh
python scripts/plot_lc_curves.py lc3 16,32,64,128
```

### lc4 — lc3 with anti-overfit fixes

Same task/layout as lc3 (Q=1, S ∈ {16,32,64,128}) with three changes aimed at the
lc3 failures (softmax near-random; GDN collapsing at S=128):

| Setting | lc3 | lc4 |
|---------|-----|-----|
| learning_rate | 2e-3 | **3e-4** (both mixers) |
| vocab_size | 2048 | **8192** |
| N_TRAIN | 400k | **5.12M** (= batch·max_iters → ~1 pass, no epoch reuse) |
| Data | `data/mqar_s{S}_q1/` | `data/mqar_s{S}_q1_lc4/` |

lr/vocab are CLI overrides on the shared configs; nothing else changes.

```bash
bash scripts/gen_lc4_data.sh
SLURM=1 bash scripts/sweep_lc4.sh
python scripts/plot_lc_curves.py lc4 16,32,64,128
```

### lc5 — small model, reverted vocab, fully-online data

Same task as lc3/lc4 (Q=1, S ∈ {16,32,64,128}) but with a **small backbone** to hit
the capacity limit faster, plus a vocab revert and 10× more data:

| Setting | lc4 | lc5 |
|---------|-----|-----|
| n_layer / n_embd | 4 / 256 | **2 / 64** (heads: 2×32) |
| learning_rate | 3e-4 | 3e-4 |
| vocab_size | 8192 | **2048** (reverted) |
| N_TRAIN | 5.12M | **5.12M** (= consumed → ~1 pass, online) |
| Config | shared lc_len_*.py + CLI | dedicated `config/lc5_{gdn,transformer}.py` |
| Data | `data/mqar_s{S}_q1_lc4/` | `data/mqar_s{S}_q1_lc5/` |

```bash
bash scripts/gen_lc5_data.sh
SLURM=1 bash scripts/sweep_lc5.sh
python scripts/plot_lc_curves.py lc5 16,32,64,128
```

### mq4 — multi-query, context-pair sweep

| Setting | Value |
|---------|-------|
| S | **context KV pairs** (before `?`) ∈ {16, 32, 64, 128} |
| Q | S / 4 |
| seq_len | 2S + 1 + 2Q (padded to even) |
| Train / val | 400k / 2k |
| Data | `data/mqar_ctx{S}_q{Q}/` |

```bash
bash scripts/gen_mqar_mq_sweep.sh
SLURM=1 bash scripts/sweep_mqar_mq.sh
python scripts/plot_lc_curves.py mq4 16,32,64,128
```

## Model (~3.7M params)

4 layers, n_embd=256, 4 heads, vocab=2048, SwiGLU FFN=704.
Softmax path: causal MHA + RoPE (θ=10000). GDN path: `fla` GatedDeltaNet, no RoPE.

## Run (single GPU debug)

```bash
conda activate gdn   # needs flash-linear-attention + CUDA
python train.py config/lc_len_gdn.py \
  --dataset=mqar_s16_q1 --block_size=16 --mqar_masked=True --vocab_size=2048 \
  --batch_size=256 --compile=False --max_iters=100
```

## Tests

```bash
python data/mqar/test_zoology.py
```
