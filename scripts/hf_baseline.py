"""Independent softmax-transformer baseline for the MQAR debug.

Trains a HuggingFace LlamaForCausalLM (tiny) on the SAME masked-MQAR .bin data,
with the SAME model size and optimization as our `mixer='softmax'` runs, to check
whether a trusted, off-the-shelf transformer can learn the task our hand-written
CausalSelfAttention struggles with.

Why Llama: same architecture family as our model (RoPE + SwiGLU + RMSNorm +
tied embeddings), so hidden_size/layers/heads/ffn match ours 1:1 — the only
difference is the (battle-tested) implementation.

CRITICAL: our .bin data is already causally shifted (inputs=raw[:-1],
labels=raw[1:]) and our GPT computes CE with no further shift. HF CausalLM shifts
internally when given `labels`, so we DO NOT pass labels to HF — we take .logits
and compute masked CE ourselves, exactly matching train.py's alignment.

Log format mirrors train.py so scripts/plot_lc_curves.py-style parsers work.

Usage (on a GPU, e.g. via sbatch):
  python scripts/hf_baseline.py --dataset mqar_s16_q1_lc5 --block_size 16 \
      --vocab_size 2048 --weight_decay 1.0 --max_iters 20000
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_split(data_dir, split, block_size):
    x = np.memmap(os.path.join(data_dir, f'{split}_inputs.bin'), dtype=np.int32, mode='r')
    y = np.memmap(os.path.join(data_dir, f'{split}_labels.bin'), dtype=np.int32, mode='r')
    n = len(x) // block_size
    return x, y, n


def get_batch(x, y, n, block_size, batch_size, device, gen):
    ix = torch.randint(n, (batch_size,), generator=gen)
    xb = torch.stack([torch.from_numpy(x[i * block_size:(i + 1) * block_size].astype(np.int64)) for i in ix.tolist()])
    yb = torch.stack([torch.from_numpy(y[i * block_size:(i + 1) * block_size].astype(np.int64)) for i in ix.tolist()])
    return xb.to(device), yb.to(device)


def get_lr(it, lr, warmup, decay_iters, decay_factor):
    if warmup > 0 and it <= warmup:
        return lr * it / warmup
    if it > decay_iters:
        return lr * decay_factor
    t = (it - warmup) / (decay_iters - warmup)
    return lr * (decay_factor + (1 - decay_factor) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--block_size', type=int, required=True)
    p.add_argument('--vocab_size', type=int, default=2048)
    p.add_argument('--n_layer', type=int, default=2)
    p.add_argument('--n_embd', type=int, default=64)
    p.add_argument('--n_head', type=int, default=2)
    p.add_argument('--ffn', type=int, default=192)        # matches our SwiGLU 8/3*64 -> 192
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--max_iters', type=int, default=20000)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1.0)
    p.add_argument('--warmup_iters', type=int, default=200)
    p.add_argument('--lr_decay_iters', type=int, default=20000)
    p.add_argument('--lr_decay_factor', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--eval_interval', type=int, default=1000)
    p.add_argument('--train_loss_log_interval', type=int, default=500)
    args = p.parse_args()

    device = 'cuda'
    torch.manual_seed(1337)
    data_dir = os.path.join(ROOT, 'data', args.dataset)

    cfg = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.n_embd,
        intermediate_size=args.ffn,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_head,
        hidden_act='silu',
        max_position_embeddings=max(args.block_size, 64),
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        attn_implementation='eager',   # reference attention (most trustworthy)
    )
    model = LlamaForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # tied embeddings are stored once; report like our harness
    print(f"[hf-llama] number of parameters: {n_params / 1e6:.2f}M "
          f"(L={args.n_layer} d={args.n_embd} h={args.n_head} ffn={args.ffn} "
          f"wd={args.weight_decay} lr={args.learning_rate})", flush=True)

    # optimizer: same split as model.configure_optimizers (decay dim>=2, no-decay dim<2)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': args.weight_decay},
         {'params': nodecay, 'weight_decay': 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-10, fused=True,
    )

    xtr, ytr, ntr = load_split(data_dir, 'train', args.block_size)
    xva, yva, nva = load_split(data_dir, 'val', args.block_size)
    gen = torch.Generator().manual_seed(1337)
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # full-val tensors (val is small, ~2k rows)
    Xva = torch.from_numpy(np.asarray(xva).astype(np.int64)).view(nva, args.block_size)
    Yva = torch.from_numpy(np.asarray(yva).astype(np.int64)).view(nva, args.block_size)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot_loss, tot_tok, tot_correct = 0.0, 0, 0
        for i in range(0, nva, args.batch_size):
            xb = Xva[i:i + args.batch_size].to(device)
            yb = Yva[i:i + args.batch_size].to(device)
            with ctx:
                logits = model(input_ids=xb).logits          # NO labels -> no HF shift
            V = logits.size(-1)
            mask = yb != -100
            loss = F.cross_entropy(logits.view(-1, V).float(), yb.view(-1), ignore_index=-100, reduction='sum')
            pred = logits.argmax(-1)
            tot_loss += loss.item()
            tot_tok += int(mask.sum().item())
            tot_correct += int((pred[mask] == yb[mask]).sum().item())
        model.train()
        return tot_loss / max(1, tot_tok), tot_correct / max(1, tot_tok)

    run_buf = []
    for it in range(args.max_iters + 1):
        lr = get_lr(it, args.learning_rate, args.warmup_iters, args.lr_decay_iters, args.lr_decay_factor)
        for g in optim.param_groups:
            g['lr'] = lr

        if it % args.eval_interval == 0:
            vl, va = evaluate()
            print(f"step {it}: val/loss {vl:.4f} val/acc {va:.4f}", flush=True)

        xb, yb = get_batch(xtr, ytr, ntr, args.block_size, args.batch_size, device, gen)
        with ctx:
            logits = model(input_ids=xb).logits
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), ignore_index=-100)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)

        run_buf.append(loss.item())
        if it > 0 and it % args.train_loss_log_interval == 0:
            print(f"step {it}: train/loss (avg over last {len(run_buf)} iters) = {sum(run_buf)/len(run_buf):.4f}", flush=True)
            run_buf = []

    print("done", flush=True)


if __name__ == '__main__':
    main()
