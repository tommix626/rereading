# Softmax baseline for length-sweep MQAR. Param-matched vs lc_len_gdn.py.

megatron_train_paths = []
megatron_val_paths = []
dataset = 'mqar_s256_q1'
mqar_masked = True
vocab_size = 2048
block_size = 256
block_align = True

n_layer = 4
n_embd = 256
ffn_intermediate_size = None
norm_eps = 1e-5

mixer = 'softmax'
n_head = 4
rope_theta = 10000.0

learning_rate = 2e-3
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
