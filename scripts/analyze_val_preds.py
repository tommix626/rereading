"""Analyze MQAR validation argmax predictions (collapse / mode-check).

Usage:
  # After training with a checkpoint:
  python scripts/analyze_val_preds.py config/lc_len_transformer.py \\
    --dataset=mqar_s64_q1 --block_size=64 --ckpt=out/ckpt.pt

  # Reproduce softmax-S64 collapse then analyze (~20 min on GPU):
  python scripts/analyze_val_preds.py config/lc_len_transformer.py \\
    --dataset=mqar_s64_q1 --block_size=64 --train-iters=17000 --save-ckpt=out/diag_s64.pt
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import GPT, GPTConfig  # noqa: E402


def load_val_examples(data_dir: str, block_size: int, max_examples: int | None = None):
    x = np.fromfile(os.path.join(data_dir, 'val_inputs.bin'), dtype=np.int32)
    y = np.fromfile(os.path.join(data_dir, 'val_labels.bin'), dtype=np.int32)
    n = len(x) // block_size
    x = x[: n * block_size].reshape(n, block_size)
    y = y[: n * block_size].reshape(n, block_size)
    if max_examples is not None:
        x, y = x[:max_examples], y[:max_examples]
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


@torch.no_grad()
def collect_answer_preds(model, inputs, labels, ignore_index: int = -100, device: str = 'cuda'):
    model.eval()
    preds, tgts, qkeys = [], [], []
    bs = 64
    for i in range(0, len(inputs), bs):
        X = inputs[i:i + bs].to(device)
        Y = labels[i:i + bs].to(device)
        x = model.transformer.wte(X)
        for block in model.transformer.h:
            x = block(x)
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
        pred = logits.argmax(dim=-1)
        mask = Y != ignore_index
        # query key sits at the token before each supervised answer
        pos = mask.nonzero(as_tuple=False)
        for b, t in pos:
            preds.append(int(pred[b, t]))
            tgts.append(int(Y[b, t]))
            qkeys.append(int(X[b, t]))
    return np.array(preds), np.array(tgts), np.array(qkeys)


def get_train_batch(data_dir: str, block_size: int, batch_size: int, rng: torch.Generator):
    x_data = np.memmap(os.path.join(data_dir, 'train_inputs.bin'), dtype=np.int32, mode='r')
    y_data = np.memmap(os.path.join(data_dir, 'train_labels.bin'), dtype=np.int32, mode='r')
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
    return x, y


def maybe_train(args, device: str) -> GPT:
    """Optional training run to reproduce collapse when no ckpt exists."""
    data_dir = os.path.join(ROOT, 'data', args.dataset)
    rng = torch.Generator()
    rng.manual_seed(1337)

    cfg = GPTConfig(
        n_layer=4, n_embd=256, block_size=args.block_size, vocab_size=args.vocab_size,
        mixer='softmax', n_head=4, rope_theta=10000.0,
    )
    model = GPT(cfg).to(device)
    opt = model.configure_optimizers(1.0, 2e-3, (0.9, 0.95), device, eps=1e-10)

    model.train()
    X, Y = get_train_batch(data_dir, args.block_size, args.batch_size, rng)
    X, Y = X.to(device), Y.to(device)
    for it in range(1, args.train_iters + 1):
        _, loss = model(X, Y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        X, Y = get_train_batch(data_dir, args.block_size, args.batch_size, rng)
        X, Y = X.to(device), Y.to(device)
        if it % 1000 == 0:
            print(f'  train iter {it}: loss {loss.item():.4f}')

    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or '.', exist_ok=True)
        torch.save({'model': model.state_dict()}, args.save_ckpt)
        print(f'saved {args.save_ckpt}')
    return model


def plot_distribution(preds, tgts, out_path: str, title: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    uniq, counts = np.unique(preds, return_counts=True)
    top = np.argsort(-counts)[:20]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.bar(range(len(top)), counts[top], color='steelblue')
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([str(uniq[i]) for i in top], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('count')
    ax.set_title(f'Top-{len(top)} argmax predictions\n({len(uniq)} unique / {len(preds)} answers)')

    ax = axes[1]
    ax.hist(preds, bins=min(80, max(10, len(uniq))), color='coral', edgecolor='none')
    ax.set_xlabel('predicted token id')
    ax.set_ylabel('count')
    ax.set_title('Full prediction histogram')

    ax = axes[2]
    correct = preds == tgts
    ax.bar(['wrong', 'correct'], [(~correct).sum(), correct.sum()],
           color=['salmon', 'seagreen'])
    ax.set_ylabel('count')
    ax.set_title(f'Accuracy {correct.mean():.2%}')

    mode_frac = counts.max() / len(preds)
    fig.suptitle(f'{title}\nmode token={uniq[counts.argmax()]} ({mode_frac:.1%} of preds)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=130)
    print(f'wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', nargs='?', default='config/lc_len_transformer.py')
    p.add_argument('--dataset', default='mqar_s64_q1')
    p.add_argument('--block-size', type=int, default=64)
    p.add_argument('--vocab-size', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--train-iters', type=int, default=0,
                   help='if no ckpt, train this many iters first (reproduce collapse)')
    p.add_argument('--save-ckpt', default=None)
    p.add_argument('--max-val', type=int, default=None, help='cap val examples analyzed')
    p.add_argument('--out', default='slurm/val_pred_softmax_s64.png')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('warning: CPU only — training reproduction will be very slow')

    data_dir = os.path.join(ROOT, 'data', args.dataset)
    inputs, labels = load_val_examples(data_dir, args.block_size, args.max_val)

    if args.ckpt and os.path.exists(args.ckpt):
        cfg = GPTConfig(
            n_layer=4, n_embd=256, block_size=args.block_size, vocab_size=args.vocab_size,
            mixer='softmax', n_head=4,
        )
        model = GPT(cfg).to(device)
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck['model'] if 'model' in ck else ck)
        print(f'loaded {args.ckpt}')
    elif args.train_iters > 0:
        print(f'training {args.train_iters} iters to reproduce...')
        model = maybe_train(args, device)
    else:
        print('No checkpoint and --train-iters=0; analyzing random init (sanity only).')
        cfg = GPTConfig(
            n_layer=4, n_embd=256, block_size=args.block_size, vocab_size=args.vocab_size,
            mixer='softmax', n_head=4,
        )
        model = GPT(cfg).to(device)

    preds, tgts, qkeys = collect_answer_preds(model, inputs, labels, device=device)

    uniq, counts = np.unique(preds, return_counts=True)
    mode_tok = uniq[counts.argmax()]
    mode_frac = counts.max() / len(preds)
    acc = (preds == tgts).mean()

    print(f'\nval examples: {len(preds)}')
    print(f'unique argmax predictions: {len(uniq)} / {args.vocab_size - 1} value tokens')
    print(f'mode token: {mode_tok}  ({mode_frac:.2%} of all answers)')
    print(f'answer accuracy: {acc:.2%}')
    print(f'top-5 preds:')
    for i in np.argsort(-counts)[:5]:
        print(f'  token {uniq[i]:4d}: {counts[i]:5d} ({counts[i]/len(preds):.2%})')

    if len(uniq) == 1:
        print('\n>>> YES: argmax is ALWAYS the same token on validation.')
    else:
        print(f'\n>>> NO: {len(uniq)} distinct argmax tokens (but mode covers {mode_frac:.1%}).')

    plot_distribution(preds, tgts, os.path.join(ROOT, args.out),
                      f'softmax val argmax — {args.dataset} (S={args.block_size})')


if __name__ == '__main__':
    main()
