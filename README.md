# GDN-minimal-BlueVela

Minimal training setup for Gated DeltaNet (GDN) research. Forked from nanoGPT
(via [`futureLM-nanoGPTLead`](https://github.com/OliverSieberling/futureLM-nanoGPTLead)
@`linear-lead@467d8a0`) and stripped down to: one model file, one training
script, two configs (GDN and a param-matched softmax-attention baseline).

## Layout

```
model.py                       GPT with mixer={'gdn','softmax'} switch
train.py                       Training loop (DDP, fixed-set val, no checkpoints by default)
configurator.py                nanoGPT's CLI-override exec pattern
config/
  train_gdn.py                 Default GDN run (~190M, "200M-gdn")
  train_transformer.py         Param-matched softmax baseline ("200M-transformer")
data/
  megatron.py                  Megatron .bin/.idx reader
scripts/
  submit_lsf.sh                LSF launcher (single node × 8 GPUs)
```

## Default model

Both configs use the same shape (16 layers × 768 hidden, FFN 2048, block 4096,
SwiGLU + RMSNorm + tied embeds). Only the mixer differs:

| | GDN (default) | softmax baseline |
|---|---|---|
| `mixer` | `'gdn'` | `'softmax'` |
| Heads / head_dim | 6 × 128 | 6 × 128 (via `n_head=6`) |
| Key / value | `key_dim = val_dim = 768` (`expand_v=1`) | standard `d × d` |
| Gate | `use_gate=False` (no SiLU output gate) | n/a |
| Positional info | short-conv (k=4) + GDN recurrence | RoPE (`rope_theta=10000`) |
| Mixer params/layer | ~4.22·d² | 4.00·d² |

Param totals match to <0.5% — any loss delta is attributable to the mixer.

## Training recipe

| | value |
|---|---|
| Tokens / step | 524,288 (≈500k; `mbs=8`, `grad_accum=16` global, `world=8`, `block=4096`) |
| Iters | 19,000 → ~9.96B tokens total |
| Optimizer | AdamW, lr=3e-4, β=(0.9, 0.95), wd=0.1, **eps=1e-10** (not 1e-8), grad_clip=1.0 |
| LR schedule | warmup 1500 → cosine to 0.1·lr over 17,500 → hold |
| Validation | every 500 steps, fixed ~20M-token set, sharded across DDP ranks |
| Checkpoints | disabled by default (`save_checkpoint=False`) |
| Data | 50/50 blend of `/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_{0,1}` |
| Compile | `torch.compile` on |

## Run

```bash
# GDN (default)
bash scripts/submit_lsf.sh                                          # → 200M-gdn

# softmax baseline
bash scripts/submit_lsf.sh 200M-transformer config/train_transformer.py

# arbitrary run name / config
bash scripts/submit_lsf.sh <wandb-run-name> <config-path>
```

The submit script's first arg sets `WANDB_NAME` (run name). The second arg is
the config path; CLI overrides like `--lr=1e-4` work too via `configurator.py`.

Launch prerequisites (defined inside `scripts/submit_lsf.sh`):
- `/u/osieberl/dynFLA/nanoGPT/.env` — WandB key
- `/u/osieberl/futureLM/nanoGPTLead/.venv/` — Python env with `flash-linear-attention` installed

## Validation details

Custom fixed-set validation: `estimate_loss()` reseeds a dedicated
`_val_rng` to `val_seed + ddp_rank` at the start of every call, so every eval
draws the same positions. Per-rank mean losses are all-reduced into the global
mean. Default `val_seed=1234`. To resize the eval set, change `eval_iters` —
total tokens per eval is `world_size × eval_iters × batch_size × block_size`.
