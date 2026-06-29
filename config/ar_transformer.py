# Softmax-attention baseline on MQAR. Param-matched vs config/ar_gdn.py
# (only the mixer differs: 4 heads x 64 = 256 = hidden). Expected to recall
# better than GDN because attention can copy the exact in-context binding.

# --- data: native MQAR bins, one self-contained example per block ---
megatron_train_paths = []
megatron_val_paths = []
dataset = 'mqar'
vocab_size = 512
block_size = 256
block_align = True

# --- small model ---
n_layer = 4
n_embd = 256
ffn_intermediate_size = None
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

# --- schedule ---
decay_lr = True
warmup_iters = 100
constant_iters = 0
lr_decay_iters = 2000
lr_decay_factor = 0.1
min_lr = None

# --- single GPU, no grad accumulation ---
max_iters = 2000
batch_size = 32
gradient_accumulation_steps = 1

# --- eval / logging ---
eval_during_training = True
eval_interval = 100
eval_iters = 40
log_interval = 25

wandb_log = False
compile = False
save_checkpoint = False
always_save_checkpoint = False
init_from = 'scratch'
