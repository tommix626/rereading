# Tiny single-GPU GDN smoke test. Goal: watch loss drop on synthetic data.
# Param-matched at the mixer level against config/debug_transformer.py
# (4 heads x 64 = 256 = hidden for both).

# --- data: nanoGPT-native synthetic bins (see data/synthetic/prepare.py) ---
megatron_train_paths = []      # empty -> use native data/<dataset>/{train,val}.bin
megatron_val_paths = []
dataset = 'synthetic'
vocab_size = 512

# --- small model ---
n_layer = 4
n_embd = 256
ffn_intermediate_size = None   # -> 8/3 x 256 rounded to mult of 64
block_size = 256
norm_eps = 1e-5

mixer = 'gdn'
gdn_head_dim = 64
gdn_num_heads = 4              # 4 x 64 = 256 = hidden
gdn_num_v_heads = None
gdn_expand_v = 1.0
gdn_mode = 'chunk'
gdn_use_gate = False
gdn_use_short_conv = True
gdn_conv_size = 4
gdn_allow_neg_eigval = False

# --- optimizer ---
learning_rate = 1e-3           # a bit hot so the demo converges fast
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
eps = 1e-10
grad_clip = 1.0

# --- short schedule ---
decay_lr = True
warmup_iters = 50
constant_iters = 0
lr_decay_iters = 500
lr_decay_factor = 0.1
min_lr = None

# --- single GPU: no grad accumulation ---
max_iters = 500
batch_size = 16
gradient_accumulation_steps = 1

# --- eval / logging ---
eval_during_training = True
eval_interval = 100
eval_iters = 20
log_interval = 10

# --- off for a local debug run ---
wandb_log = False
compile = False
save_checkpoint = False
always_save_checkpoint = False
init_from = 'scratch'
