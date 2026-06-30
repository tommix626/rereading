# Softmax-attention baseline on Zoology-style MQAR. Param-matched vs config/ar_gdn.py.
# Generate data: NUM_PAIRS=64 python data/mqar/gen.py  (see ar_gdn.py)

megatron_train_paths = []
megatron_val_paths = []
dataset = 'mqar_s258_p64'
mqar_masked = True
vocab_size = 8192
block_size = 258
block_align = True

n_layer = 2
n_embd = 128
ffn_intermediate_size = None
norm_eps = 1e-5

mixer = 'softmax'
n_head = 2                     # head_dim = 128/2 = 64, matching GDN config
rope_theta = 10000.0

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
