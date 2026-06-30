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
scripts/plot_lc_curves.py     plot slurm/{lc3,mq4}_*.log
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
