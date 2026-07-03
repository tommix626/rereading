# lc5: small-model length-sweep MQAR (Q=1). GDN path.
# Differs from config/lc_len_gdn.py (lc3/lc4): 2 layers, n_embd=64, lr=3e-4.
# Data: gen_lc5_data.sh -> data/mqar_s{S}_q1_lc5/ (vocab 2048, ~51M examples).
# Kept in lockstep with config/lc5_transformer.py except mixer fields.

megatron_train_paths = []
megatron_val_paths = []
dataset = 'mqar_s16_q1_lc5'
mqar_masked = True
vocab_size = 2048
block_size = 16
block_align = True

# --- small model (see [[small-model-default]]): 2L, d=64 ---
n_layer = 2
n_embd = 64
ffn_intermediate_size = None
norm_eps = 1e-5

mixer = 'gdn'
gdn_head_dim = 32        # 2 heads * 32 = 64 = n_embd (param-matched to softmax)
gdn_num_heads = 2
gdn_num_v_heads = None
gdn_expand_v = 1.0
gdn_mode = 'chunk'
gdn_use_gate = False
gdn_use_short_conv = True
gdn_conv_size = 4
gdn_allow_neg_eigval = False

learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 1.0
eps = 1e-10
grad_clip = 1.0

decay_lr = True
warmup_iters = 200
constant_iters = 0
lr_decay_iters = 20000
lr_decay_factor = 0.1
min_lr = None

max_iters = 20000
batch_size = 256
gradient_accumulation_steps = 1

eval_during_training = True
eval_interval = 1000
eval_iters = 50
log_interval = 1000
train_loss_log_interval = 500

wandb_log = False
compile = False
save_checkpoint = True
always_save_checkpoint = True
init_from = 'scratch'
