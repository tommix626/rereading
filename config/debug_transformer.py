# Tiny single-GPU softmax-attention smoke test. Param-matched against
# config/debug_gdn.py -- only the mixer differs (4 heads x 64 = 256 = hidden).

# --- data: nanoGPT-native synthetic bins (see data/synthetic/prepare.py) ---
megatron_train_paths = []      # empty -> use native data/<dataset>/{train,val}.bin
megatron_val_paths = []
dataset = 'synthetic'
vocab_size = 512

# --- small model ---
n_layer = 4
n_embd = 256
ffn_intermediate_size = None
block_size = 256
norm_eps = 1e-5

mixer = 'softmax'
n_head = 4                     # head_dim = 256/4 = 64, matching GDN config
rope_theta = 10000.0

# --- optimizer ---
learning_rate = 1e-3
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
