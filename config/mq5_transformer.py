# mq5: Zoology-faithful MQAR, softmax attention baseline.
# Matches Zoology synthetic attention: 2 layers, 1 head, GPT-2 LEARNED absolute
# positional embeddings (no RoPE), weight_decay 0.1, large vocab.
# dataset / block_size / learning_rate / max_iters / batch_size are set per-run by
# scripts/sweep_mq5.sh (LR sweep, single-epoch sizing).

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

mixer = 'softmax'
n_head = 1                    # Zoology synthetics use exactly 1 attention head
rope_theta = 10000.0
use_rope = False              # Zoology uses learned absolute pos-emb, not RoPE
use_learned_pos_emb = True

learning_rate = 3e-4          # base; swept by scripts/sweep_mq5.sh
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1            # Zoology synthetic protocol
eps = 1e-10
grad_clip = 1.0

decay_lr = False             # constant LR (clean LR-sweep methodology)
warmup_iters = 200
constant_iters = 0
lr_decay_iters = 20000
lr_decay_factor = 0.1
min_lr = None

max_iters = 20000            # overridden: single-epoch = N_TRAIN // batch_size
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
