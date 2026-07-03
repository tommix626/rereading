# mq5: Zoology-faithful MQAR, GDN with the short convolution REMOVED.
# Tests whether gated-delta recurrence alone (no local short-conv shift) can still
# do MQAR. 2 layers, 1 head, no positional encoding (GDN is implicitly positional).
# dataset / block_size / learning_rate / max_iters / batch_size set per-run by sweep.

megatron_train_paths = []
megatron_val_paths = []
dataset = 'mqar_zoo_d16_n96'
mqar_masked = True
vocab_size = 8192
block_size = 96
block_align = True

n_layer = 2
n_embd = 64
ffn_intermediate_size = None
norm_eps = 1e-5

mixer = 'gdn'
gdn_head_dim = 64            # 1 head * 64 = 64 = n_embd
gdn_num_heads = 1
gdn_num_v_heads = None
gdn_expand_v = 1.0
gdn_mode = 'chunk'
gdn_use_gate = False
gdn_use_short_conv = False   # <-- short conv REMOVED (the key mq5 GDN change)
gdn_conv_size = 4
gdn_allow_neg_eigval = False

learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
eps = 1e-10
grad_clip = 1.0

decay_lr = False
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
