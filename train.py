"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# Logging / eval (ported from nanoGPTLead fb93fc5 / 2026-05-13).
# `train_loss_log_interval` controls the running-window train/loss metric
# (avg over ALL per-iter losses in the window, one loss per iter). The legacy
# 50-batch `estimate_loss` is gated by `eval_during_training` and renamed
# `train/loss_eval` / `val/loss_eval` to avoid colliding with the running avg.
train_loss_log_interval = 500
eval_during_training = False
# Validation seed: re-applied to a dedicated val RNG at the start of every
# estimate_loss() call so every eval samples the SAME positions. Mixed with
# ddp_rank so each rank covers a distinct shard, and the cross-rank union is
# the fixed val set. eval_iters should be sized so
# world_size * eval_iters * batch_size * block_size ≈ desired val tokens.
val_seed = 1234
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
# Megatron .bin/.idx — single path (legacy) or list of paths with weights (blended).
# When using a blend (matches lm-engine's BlendedDataset), each sample independently
# picks a file by weight then a uniform random offset within that file's token stream.
megatron_train_path = None
megatron_val_path = None
megatron_train_paths = []     # list[str] of path prefixes (no extension); overrides _path when non-empty
megatron_train_weights = []   # list[float] same length as _paths; will be normalized
megatron_val_paths = []
megatron_val_weights = []
# Explicit vocab size override (bypasses meta.pkl). Set when using Megatron data.
vocab_size = None
# Disable checkpoint writes entirely (for lead-generation runs).
save_checkpoint = True
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
block_align = False # if True, snap each random window to a multiple of block_size
                    # (for fixed-length self-contained examples, e.g. MQAR; native path only)
mqar_masked = False # Zoology MQAR: load {split}_inputs.bin + {split}_labels.bin;
                    # labels use -100 on non-answer positions; no extra NTP shift in get_batch
mqar_ignore_index = -100
ar_eval_acc = False # legacy fixed-position MQAR accuracy (pre-Zoology contiguous queries)
ar_num_queries = 0  # number of trailing (query,answer) pairs per example; answers are
                    # predicted at local positions [block_size-2*Q + 2*t for t in 0..Q-1]
# model (SwiGLU MLP, RMSNorm, no biases, tied embeds; mixer chosen below)
n_layer = 12
n_embd = 768
ffn_intermediate_size = None  # None → 8/3 × n_embd rounded to multiple of 64
norm_eps = 1e-5
# Sequence mixer: 'gdn' (default) or 'softmax' (CausalSelfAttention + RoPE)
mixer = 'gdn'
# GDN mixer config (used when mixer='gdn')
gdn_head_dim = 128
gdn_num_heads = 6
gdn_num_v_heads = None
gdn_expand_v = 1.0
gdn_mode = 'chunk'
gdn_use_gate = False
gdn_use_short_conv = True
gdn_conv_size = 4
gdn_allow_neg_eigval = False
# Softmax-attention config (used when mixer='softmax')
n_head = 12
rope_theta = 10000.0
# (no `bias` / `dropout` — the new model removes both)
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8 # AdamW eps (lm-engine 300m uses 1e-10, not torch default 1e-8)
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings — mirrors lm-engine's CosineScheduler (scheduler.py:90-106)
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # linear warmup from 0 to learning_rate over this many steps
constant_iters = 0 # steps held at learning_rate before cosine decay starts
lr_decay_iters = 600000 # warmup + constant + decay end (= total steps at which cosine reaches its floor)
lr_decay_factor = 0.1 # cosine floor as fraction of learning_rate (lm-engine default; LR ends at lr * 0.1)
# legacy nanoGPT min_lr (used only if you want to override lr_decay_factor implicitly via min_lr)
min_lr = None # if None, falls back to learning_rate * lr_decay_factor
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process and save_checkpoint:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# Normalize the megatron data config: either list-of-paths (preferred, supports blend)
# or legacy single path. Either side may be empty/None → fall back to nanoGPT native.
def _resolve_megatron(paths_list, weights_list, single_path):
    if paths_list:
        weights = list(weights_list) if weights_list else [1.0] * len(paths_list)
        assert len(weights) == len(paths_list), "megatron_*_paths and _weights must align"
        return list(paths_list), weights
    if single_path is not None:
        return [single_path], [1.0]
    return None, None
_train_paths, _train_weights = _resolve_megatron(megatron_train_paths, megatron_train_weights, megatron_train_path)
_val_paths, _val_weights = _resolve_megatron(megatron_val_paths, megatron_val_weights, megatron_val_path)
use_megatron = _train_paths is not None
if use_megatron:
    from data.megatron import open_megatron, normalize_weights
    _train_weights_np = normalize_weights(_train_weights)
    _val_weights_np = normalize_weights(_val_weights) if _val_paths is not None else None

def get_batch(split):
    # All random ops route through `data_rng` (a dedicated torch.Generator seeded
    # AFTER model init), so changing the architecture (and thus the number of RNG
    # draws consumed by model init) doesn't shift the data sequence. Every arch
    # sees identical batches at every iter for the same (seed, ddp_rank, split).
    # Route val draws through _val_rng (reseeded per estimate_loss call → fixed val set).
    rng = _val_rng if split == 'val' else data_rng
    if use_megatron:
        if split == 'train':
            paths, weights_np = _train_paths, _train_weights_np
        else:
            paths, weights_np = _val_paths, _val_weights_np
        streams = [open_megatron(p) for p in paths]
        if len(streams) == 1:
            ds_choices = [0] * batch_size
        else:
            w = torch.tensor(weights_np, dtype=torch.float64)
            ds_choices = torch.multinomial(w, batch_size, replacement=True, generator=rng).tolist()
        x_list, y_list = [], []
        for ds in ds_choices:
            stream = streams[ds]
            i = int(torch.randint(0, len(stream) - block_size - 1, (1,), generator=rng).item())
            x_list.append(torch.from_numpy(stream[i:i+block_size].astype(np.int64)))
            y_list.append(torch.from_numpy(stream[i+1:i+1+block_size].astype(np.int64)))
        x = torch.stack(x_list)
        y = torch.stack(y_list)
    elif mqar_masked:
        x_path = os.path.join(data_dir, f'{split}_inputs.bin')
        y_path = os.path.join(data_dir, f'{split}_labels.bin')
        x_data = np.memmap(x_path, dtype=np.int32, mode='r')
        y_data = np.memmap(y_path, dtype=np.int32, mode='r')
        n_examples = len(x_data) // block_size
        ix = torch.randint(n_examples, (batch_size,), generator=rng)
        x = torch.stack([
            torch.from_numpy(x_data[i * block_size:(i + 1) * block_size].astype(np.int64))
            for i in ix.tolist()
        ])
        y = torch.stack([
            torch.from_numpy(y_data[i * block_size:(i + 1) * block_size].astype(np.int64))
            for i in ix.tolist()
        ])
    elif split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=rng)
        if block_align:
            ix = (ix // block_size) * block_size  # snap to example boundaries
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=rng)
        if block_align:
            ix = (ix // block_size) * block_size  # snap to example boundaries
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size / mqar_masked from the dataset meta
meta_vocab_size = None
if vocab_size is None and not use_megatron:
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        if mqar_masked is False and meta.get('mqar_masked'):
            mqar_masked = True
        if meta.get('ignore_index') is not None:
            mqar_ignore_index = meta['ignore_index']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_embd=n_embd, block_size=block_size,
    vocab_size=None, ffn_intermediate_size=ffn_intermediate_size, norm_eps=norm_eps,
    mixer=mixer,
    gdn_head_dim=gdn_head_dim, gdn_num_heads=gdn_num_heads, gdn_num_v_heads=gdn_num_v_heads,
    gdn_expand_v=gdn_expand_v, gdn_mode=gdn_mode, gdn_use_gate=gdn_use_gate,
    gdn_use_short_conv=gdn_use_short_conv, gdn_conv_size=gdn_conv_size,
    gdn_allow_neg_eigval=gdn_allow_neg_eigval,
    n_head=n_head, rope_theta=rope_theta,
)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if vocab_size is not None:
        model_args['vocab_size'] = vocab_size
    elif meta_vocab_size is not None:
        model_args['vocab_size'] = meta_vocab_size
    else:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    for k in ['n_layer', 'n_embd', 'block_size', 'vocab_size', 'ffn_intermediate_size',
              'norm_eps', 'mixer',
              'gdn_head_dim', 'gdn_num_heads', 'gdn_num_v_heads',
              'gdn_expand_v', 'gdn_mode', 'gdn_use_gate', 'gdn_use_short_conv',
              'gdn_conv_size', 'gdn_allow_neg_eigval',
              'n_head', 'rope_theta']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Unsupported init_from={init_from!r}; expected 'scratch' or 'resume'")
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, eps=eps)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

def _checkpoint_model_state():
    state_dict = raw_model.state_dict()
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def save_training_checkpoint(tag=''):
    if not (master_process and save_checkpoint):
        return
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = {
        'model': _checkpoint_model_state(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    torch.save(checkpoint, ckpt_path)
    suffix = f' ({tag})' if tag else ''
    print(f'saved checkpoint to {ckpt_path}{suffix}')

# Fixed-set validation loss. _val_rng is reseeded at the start of every call so
# every eval iterates over the same (rank-sharded) sequence of val positions.
# Across DDP ranks the union covers world_size * eval_iters * batch_size * block_size
# tokens; the per-rank mean losses are all-reduced into the global mean.
@torch.no_grad()
def estimate_loss():
    _val_rng.manual_seed(val_seed + ddp_rank)
    model.eval()
    raw = model.module if ddp else model
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        X, Y = get_batch('val')
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss
    val_loss = losses.mean()
    if ddp:
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)
    out = {'val': val_loss.item()}
    # Zoology MQAR: masked CE is the main metric; also report answer-token accuracy.
    if mqar_masked:
        accs = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch('val')
            with ctx:
                accs[k] = raw.masked_answer_accuracy(X, Y, ignore_index=mqar_ignore_index)
        val_acc = accs.mean()
        if ddp:
            torch.distributed.all_reduce(val_acc, op=torch.distributed.ReduceOp.AVG)
        out['val_acc'] = val_acc.item()
    # Legacy contiguous-query MQAR layout (pre-Zoology bins).
    elif ar_eval_acc and ar_num_queries > 0:
        base = block_size - 2 * ar_num_queries
        positions = torch.arange(base, block_size - 1, 2, device=device)
        correct = torch.zeros((), device=device)
        total = torch.zeros((), device=device)
        for k in range(eval_iters):
            X, Y = get_batch('val')
            with ctx:
                lg = raw.logits_at(X, positions)
            pred = lg.argmax(dim=-1)
            tgt = Y[:, positions]
            correct += (pred == tgt).sum()
            total += tgt.numel()
        if ddp:
            torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        out['val_acc'] = (correct / total).item()
    model.train()
    return out

# learning rate schedule — mirrors lm-engine CosineScheduler (scheduler.py:90-106) exactly
def get_lr(it):
    lr_constant_boundary = warmup_iters + constant_iters
    lr_decay_boundary = lr_decay_iters
    # cosine floor: explicit min_lr overrides; else learning_rate * lr_decay_factor
    floor_factor = (min_lr / learning_rate) if min_lr is not None else lr_decay_factor
    if warmup_iters > 0 and it <= warmup_iters:
        factor = it / warmup_iters
    elif it <= lr_constant_boundary:
        factor = 1.0
    elif it <= lr_decay_boundary:
        x = it - lr_constant_boundary
        t = lr_decay_boundary - lr_constant_boundary
        factor = (1.0 - floor_factor) * (1.0 + math.cos(math.pi * x / t)) / 2 + floor_factor
    else:
        factor = floor_factor
    return learning_rate * factor

# logging
if wandb_log and master_process:
    import wandb
    # Prefer WANDB_NAME env var (set by submit_lsf.sh from its first arg) over the
    # config default — otherwise wandb.init(name=...) silently overrides the env var.
    run_name = os.environ.get('WANDB_NAME', wandb_run_name)
    wandb.init(project=wandb_project, name=run_name, config=config)

# Dedicated CPU torch.Generator for data sampling. Seeded AFTER model construction
# so the data sequence is fully decoupled from however many RNG draws each
# architecture's init consumed. Identical (seed, ddp_rank) → identical batches
# across architectures.
data_rng = torch.Generator()
data_rng.manual_seed(1337 + seed_offset)
# Dedicated val RNG — reseeded inside estimate_loss() to (val_seed + ddp_rank).
_val_rng = torch.Generator()

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
# rolling buffer of per-iter losses; averaged + flushed every train_loss_log_interval iters
running_loss_buf = []
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Fixed-set val loss every eval_interval steps. Runs on all ranks (each does its
    # shard); all-reduce inside estimate_loss gives the global mean. Logged from master.
    if eval_during_training and iter_num % eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            acc_str = f" val/acc {losses['val_acc']:.4f}" if 'val_acc' in losses else ""
            print(f"step {iter_num}: val/loss {losses['val']:.4f}{acc_str}")
            if wandb_log:
                log_d = {
                    "iter": iter_num,
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                }
                if 'val_acc' in losses:
                    log_d["val/acc"] = losses['val_acc']
                wandb.log(log_d, step=iter_num)
            if save_checkpoint and (losses['val'] < best_val_loss or always_save_checkpoint):
                improved = losses['val'] < best_val_loss
                if improved:
                    best_val_loss = losses['val']
                save_training_checkpoint(tag=f'iter {iter_num}')
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16.
    # We also accumulate the RAW per-micro-batch loss (detached, no autograd) so that the
    # rolling-window train/loss metric averages over all micro-batches in the iter, not just the last.
    loss_sum_for_metric = None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
        # Capture the raw (un-divided) loss for the rolling-window metric BEFORE we scale it for backward.
        if loss_sum_for_metric is None:
            loss_sum_for_metric = loss.detach()
        else:
            loss_sum_for_metric = loss_sum_for_metric + loss.detach()
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    # Per-iter loss for the rolling-window train/loss metric, averaged over ALL micro-batches
    # in this iter (not just the last). One CPU↔GPU sync per iter (the .item() call).
    if master_process:
        lossf = (loss_sum_for_metric / gradient_accumulation_steps).item()
        running_loss_buf.append(lossf)
    if iter_num % log_interval == 0 and master_process:
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/iter_loss": lossf,
                "lr": lr,
                "mfu": running_mfu * 100,
                "step_time_ms": dt * 1000,
                "tokens": iter_num * tokens_per_iter,
            }, step=iter_num)

    # === Rolling-window train/loss: avg of every per-iter loss over the last train_loss_log_interval iters ===
    if (train_loss_log_interval > 0 and iter_num > 0
            and iter_num % train_loss_log_interval == 0 and master_process and running_loss_buf):
        avg_loss = sum(running_loss_buf) / len(running_loss_buf)
        print(f"step {iter_num}: train/loss (avg over last {len(running_loss_buf)} iters) = {avg_loss:.4f}")
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": avg_loss}, step=iter_num)
        running_loss_buf = []

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        if master_process and save_checkpoint:
            save_training_checkpoint(tag='final')
        break

if ddp:
    destroy_process_group()
