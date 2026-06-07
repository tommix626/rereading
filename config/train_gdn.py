# Default GDN run (~190M, called "200M-gdn"). Param-matched at the mixer level
# against config/train_transformer.py — only the sequence mixer differs.
#
# Shape: no output gate, key_dim = val_dim = hidden. With use_gate=False the
# fla 0.75·hidden constraint disappears, so num_heads × head_dim = hidden
# naturally (6 × 128 = 768).

# ============================================================
# Data: 50/50 blend of granite-4 web-nemotron-cc-hq-p2_{0,1}
# ============================================================
megatron_train_paths = [
    '/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0',
    '/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1',
]
megatron_train_weights = [0.5, 0.5]
megatron_val_paths = [
    '/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0',
    '/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1',
]
megatron_val_weights = [0.5, 0.5]
vocab_size = 100352

# ============================================================
# Model: ~190M GDN (16 layers, hidden 768)
# ============================================================
n_layer = 16
n_embd = 768
ffn_intermediate_size = 2048   # 8/3 × 768 rounded to multiple of 64
block_size = 4096
norm_eps = 1e-5

mixer = 'gdn'
gdn_head_dim = 128
gdn_num_heads = 6              # 6 × 128 = 768 = hidden
gdn_num_v_heads = None         # → defaults to num_heads = 6
gdn_expand_v = 1.0             # key_dim == val_dim
gdn_mode = 'chunk'
gdn_use_gate = False           # no output SiLU gate → drops g_proj (~1.57M params/layer); also drops the 0.75·hidden constraint
gdn_use_short_conv = True
gdn_conv_size = 4
gdn_allow_neg_eigval = False

# ============================================================
# Optimizer: TorchAdamW, lr=3e-4, eps=1e-10 (NOT 1e-8)
# ============================================================
learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
eps = 1e-10
grad_clip = 1.0

# ============================================================
# LR schedule: warmup 1500 → cosine to 0.1·lr over next 17500 → hold
# (lr_decay_iters tracks max_iters so the cosine completes at end of training)
# ============================================================
decay_lr = True
warmup_iters = 1500
constant_iters = 0
lr_decay_iters = 19000
lr_decay_factor = 0.1
min_lr = None

# ============================================================
# Batching: micro=8, grad_accum=2 per-GPU × 8 GPUs → 524288 tok/step × 19k ≈ 9.96B
# (≈500k tok/step target; exact 500k isn't a clean integer factorization with
#  block=4096 and world=8, so we use the nearest power-of-two-friendly value.)
# ============================================================
max_iters = 19000
batch_size = 8
gradient_accumulation_steps = 2 * 8

# ============================================================
# Eval: fixed ~20M-token val set every 500 steps.
# 8 ranks × 76 iters × bs=8 × block=4096 ≈ 19.93M tokens per eval.
# val_rng is reseeded to (val_seed + ddp_rank) inside estimate_loss, so every
# eval samples the same positions deterministically across the run.
# ============================================================
eval_during_training = True
eval_interval = 500
eval_iters = 76
log_interval = 10

# ============================================================
# WandB
# ============================================================
wandb_log = True
wandb_project = 'keyExpansion'
wandb_run_name = '200M-gdn'

# ============================================================
# Precision / compile / checkpointing
# ============================================================
compile = True
save_checkpoint = False
always_save_checkpoint = False
init_from = 'scratch'
