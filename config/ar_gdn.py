# GDN on Multi-Query Associative Recall (MQAR). See data/mqar/prepare.py.
# Param-matched mixer vs config/ar_transformer.py (4 heads x 64 = 256 = hidden).
# Intuition under test: softmax should out-recall GDN (fixed-state) here.

# --- data: native MQAR bins, one self-contained example per block ---
megatron_train_paths = []
megatron_val_paths = []
dataset = 'mqar'
vocab_size = 512               # NUM_KEYS(256) + NUM_VALS(256)
block_size = 256               # = example length L = 2*64 pairs + 2*64 queries
block_align = True             # snap each random window to one MQAR example

# --- small model ---
n_layer = 4
n_embd = 256
ffn_intermediate_size = None
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
