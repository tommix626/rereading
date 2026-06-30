# GDN on Zoology-style MQAR (answer-token-only CE; labels -100 elsewhere).
# Param-matched mixer vs config/ar_transformer.py.
# Generate data first:
#   NUM_PAIRS=64 INPUT_SEQ_LEN=256 VOCAB_SIZE=8192 python data/mqar/gen.py
#   dataset below must match the output dir name (mqar_s{S}_p{P}).

megatron_train_paths = []
megatron_val_paths = []
dataset = 'mqar_s258_p64'
mqar_masked = True
vocab_size = 8192
block_size = 258               # +2 for '?' separator after 128-token context at P=64
block_align = True             # one self-contained example per row in the bins

# --- small model (Zoology canonical d128; scale via CLI) ---
n_layer = 2
n_embd = 128
ffn_intermediate_size = None
norm_eps = 1e-5

mixer = 'gdn'
gdn_head_dim = 64
gdn_num_heads = 2              # 2 x 64 = 128 = hidden
gdn_num_v_heads = None
gdn_expand_v = 1.0
gdn_mode = 'chunk'
gdn_use_gate = False
gdn_use_short_conv = True
gdn_conv_size = 4
gdn_allow_neg_eigval = False

learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
eps = 1e-10
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
constant_iters = 0
lr_decay_iters = 2000
lr_decay_factor = 0.1
min_lr = None

max_iters = 2000
batch_size = 32
gradient_accumulation_steps = 1

eval_during_training = True
eval_interval = 100
eval_iters = 40
log_interval = 25

wandb_log = False
compile = False
save_checkpoint = False
always_save_checkpoint = False
init_from = 'scratch'
