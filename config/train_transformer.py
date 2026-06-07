# Param-matched softmax-attention baseline for config/train_gdn.py.
# Identical model shape (16L × 768d, ffn=2048), identical data, optimizer,
# LR schedule, batching, eval. Only the sequence mixer differs:
#   GDN: 6 heads × head_dim 128, expand_v=1, no gate → ~4.22·d²/layer mixer
#   MHA: 6 heads × head_dim 128, RoPE                → ~4.00·d²/layer mixer
# Mixer params differ by ~0.7% per layer (GDN's α/β + depthwise conv overhead);
# the rest of the model is identical, so total params land within <0.5%.

# ============================================================
# Data
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
# Model: ~190M softmax-attention transformer
# ============================================================
n_layer = 16
n_embd = 768
ffn_intermediate_size = 2048
block_size = 4096
norm_eps = 1e-5

mixer = 'softmax'
n_head = 6                      # head_dim = 768/6 = 128, matching GDN config
rope_theta = 10000.0

# ============================================================
# Optimizer
# ============================================================
learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
eps = 1e-10
grad_clip = 1.0

# ============================================================
# LR schedule
# ============================================================
decay_lr = True
warmup_iters = 1500
constant_iters = 0
lr_decay_iters = 19000
lr_decay_factor = 0.1
min_lr = None

# ============================================================
# Batching: matches GDN config — 524288 tok/step × 19k ≈ 9.96B
# ============================================================
max_iters = 19000
batch_size = 8
gradient_accumulation_steps = 2 * 8

# ============================================================
# Eval: fixed ~20M-token val set every 500 steps (same as GDN config)
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
wandb_run_name = '200M-transformer'

# ============================================================
# Precision / compile / checkpointing
# ============================================================
compile = True
save_checkpoint = False
always_save_checkpoint = False
init_from = 'scratch'
