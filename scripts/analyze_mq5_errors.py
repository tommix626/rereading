"""Per-position / per-token error analysis for an mq5 (Zoology MQAR) softmax run.

Motivation: the D=32, lr=1e-3 softmax run groks in *bumps* (val acc ~0.80 -> ~0.88
-> ~0.92) and plateaus below 1.0. This script dissects the errors of each saved
checkpoint on the SAME fixed val set to answer:
  - Are failures concentrated at specific *positions* (query distance `gap`, or the
    binding's *slot* in the context prefix)?
  - Are failures concentrated at specific *value tokens*?
  - What kind of wrong token does the model emit (another value in the example =
    binding confusion, a key, or garbage)?
  - As the model bumps up in accuracy, *which* previously-wrong positions get fixed
    (by gap / slot), i.e. what does each grokking step buy?

Layout recap (Zoology MQAR, no separator, num_passes=1):
    context = k0 v0 k1 v1 ... k_{D-1} v_{D-1}     (positions 0..2D-1)
    then D queries: each is a re-occurrence of some key at a power-law gap; the
    label (the value) is supervised at the query-key position `j` (shifted bins).
  -> at a supervised index j: input[j] = query key, label[j] = its value.
     gap      = (j - 2D) / 2                 (query distance into the query region)
     key_slot = index of that key in the context prefix (0..D-1)

Usage (run on a GPU node, e.g. via srun):
    python scripts/analyze_mq5_errors.py --out_dir out/mq5_softmax_d32_lr1e-3_snap \
        --dataset mqar_zoo_d32_n192
    # single checkpoint (e.g. the existing final ckpt) to validate the pipeline:
    python scripts/analyze_mq5_errors.py --out_dir out/mq5_softmax_d32_lr1e-3 \
        --dataset mqar_zoo_d32_n192 --ckpts ckpt.pt

Writes:
    <out_dir>/error_analysis.npz   (raw per-position records across checkpoints)
    slurm/plots/mq5_errors_<tag>.png
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig  # noqa: E402

ITER_RE = re.compile(r'ckpt_iter(\d+)\.pt')


def load_val(dataset, block_size, max_examples=None):
    ddir = os.path.join(ROOT, 'data', dataset)
    with open(os.path.join(ddir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    T = block_size or meta['block_size']
    xi = np.memmap(os.path.join(ddir, 'val_inputs.bin'), dtype=np.int32, mode='r')
    yi = np.memmap(os.path.join(ddir, 'val_labels.bin'), dtype=np.int32, mode='r')
    n = len(xi) // T
    if max_examples:
        n = min(n, max_examples)
    inputs = np.asarray(xi[:n * T]).reshape(n, T).astype(np.int64)
    labels = np.asarray(yi[:n * T]).reshape(n, T).astype(np.int64)
    return inputs, labels, meta


def supervised_records(inputs, labels, D, ignore_index=-100):
    """Flatten all supervised positions with per-position tags. Every example has
    exactly D supervised positions (the D queries)."""
    n, T = inputs.shape
    context_size = 2 * D
    mask = labels != ignore_index
    per = mask.sum(axis=1)
    assert np.all(per == D), f"expected {D} supervised/example, got {sorted(set(per.tolist()))}"
    rows, cols = np.where(mask)                    # row-major -> D contiguous cols per row
    sup_j = cols.reshape(n, D)
    val_tok = labels[rows, cols].reshape(n, D)
    key_tok = inputs[rows, cols].reshape(n, D)
    gap = (sup_j - context_size) // 2
    # key_slot: position of the query key within the context prefix (keys at 0,2,...,2D-2)
    keys_in_ctx = inputs[:, 0:context_size:2]      # [n, D]
    eq = keys_in_ctx[:, None, :] == key_tok[:, :, None]   # [n, D(query), D(ctx)]
    key_slot = eq.argmax(axis=2)
    assert eq.any(axis=2).all(), "some query key not found in its context (data bug?)"
    ex_id = np.repeat(np.arange(n), D)
    return dict(
        n=n, T=T, D=D, context_size=context_size,
        sup_j=sup_j.reshape(-1), val_tok=val_tok.reshape(-1), key_tok=key_tok.reshape(-1),
        gap=gap.reshape(-1), key_slot=key_slot.reshape(-1), ex_id=ex_id,
        sup_j_2d=sup_j,                            # [n, D] for gather
    )


def find_ckpts(out_dir, explicit):
    if explicit:
        paths = [os.path.join(out_dir, c) for c in explicit]
    else:
        paths = sorted(glob.glob(os.path.join(out_dir, 'ckpt_iter*.pt')),
                       key=lambda p: int(ITER_RE.search(p).group(1)))
        if not paths and os.path.exists(os.path.join(out_dir, 'ckpt.pt')):
            paths = [os.path.join(out_dir, 'ckpt.pt')]
    iters = []
    for p in paths:
        m = ITER_RE.search(os.path.basename(p))
        iters.append(int(m.group(1)) if m else -1)
    return paths, np.array(iters)


@torch.no_grad()
def predict_supervised(ckpt_path, inputs, sup_j_2d, device, batch_size, dtype):
    """Return predicted token id at every supervised position -> [n, D]."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = GPT(GPTConfig(**ckpt['model_args']))
    sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
          for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    n, D = sup_j_2d.shape
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = (torch.amp.autocast(device_type='cuda', dtype=ptdtype)
           if device.startswith('cuda') else torch.autocast('cpu', enabled=False))
    x = torch.from_numpy(inputs).to(device)
    sj = torch.from_numpy(sup_j_2d).to(device)
    pred = torch.empty((n, D), dtype=torch.long, device=device)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        with ctx:
            h = model._backbone(x[s:e])
            logits = model.lm_head(h)               # [b, T, V]
        p = logits.argmax(dim=-1)                    # [b, T]
        pred[s:e] = torch.gather(p, 1, sj[s:e])
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return pred.cpu().numpy()


def compute(out_dir, dataset, block_size, explicit, device, batch_size, dtype, max_examples):
    paths, iters = find_ckpts(out_dir, explicit)
    if not paths:
        raise SystemExit(f"no checkpoints found in {out_dir}")
    D = int(re.search(r'_d(\d+)_', dataset).group(1)) if '_d' in dataset else None
    inputs, labels, meta = load_val(dataset, block_size, max_examples)
    if D is None:
        D = int(meta['num_kv_pairs'])
    rec = supervised_records(inputs, labels, D)
    print(f"{rec['n']} val examples x {D} queries = {rec['n']*D} supervised positions; "
          f"{len(paths)} checkpoints")
    correct = np.zeros((len(paths), rec['n'] * D), dtype=np.uint8)
    pred = np.zeros((len(paths), rec['n'] * D), dtype=np.int32)
    for i, p in enumerate(paths):
        pr = predict_supervised(p, inputs, rec['sup_j_2d'], device, batch_size, dtype)
        pred[i] = pr.reshape(-1)
        correct[i] = (pr.reshape(-1) == rec['val_tok']).astype(np.uint8)
        print(f"  [{iters[i]:>6}] {os.path.basename(p):<24} acc={correct[i].mean():.4f}", flush=True)
    npz = os.path.join(out_dir, 'error_analysis.npz')
    np.savez_compressed(
        npz, iters=iters, correct=correct, pred=pred,
        gap=rec['gap'], key_slot=rec['key_slot'], val_tok=rec['val_tok'],
        key_tok=rec['key_tok'], sup_j=rec['sup_j'], ex_id=rec['ex_id'],
        n=rec['n'], D=D, context_size=rec['context_size'],
    )
    print("wrote", npz)
    return npz


# ------------------------------- plotting ------------------------------------

def _grouped_acc(correct_row, key, nbins):
    """mean correctness per integer group in [0, nbins)."""
    acc = np.full(nbins, np.nan)
    for g in range(nbins):
        m = key == g
        if m.any():
            acc[g] = correct_row[m].mean()
    return acc


def plot(npz_path, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    iters, correct, pred = d['iters'], d['correct'], d['pred']
    gap, key_slot, val_tok = d['gap'], d['key_slot'], d['val_tok']
    key_tok, ex_id = d['key_tok'], d['ex_id']
    n, D = int(d['n']), int(d['D'])
    S = len(iters)
    overall = correct.mean(axis=1)

    ngap = int(gap.max()) + 1
    nslot = D
    # value buckets: values live in [V/2, V); bucket into 32
    vmin, vmax = int(val_tok.min()), int(val_tok.max())
    nvb = 32
    vbucket = np.clip(((val_tok - vmin) * nvb) // (vmax - vmin + 1), 0, nvb - 1)

    acc_gap = np.stack([_grouped_acc(correct[s], gap, ngap) for s in range(S)])
    acc_slot = np.stack([_grouped_acc(correct[s], key_slot, nslot) for s in range(S)])
    acc_vb = np.stack([_grouped_acc(correct[s], vbucket, nvb) for s in range(S)])

    # per-example value/key sets for error typing (using final checkpoint's preds)
    ex_vals = [set() for _ in range(n)]
    ex_keys = [set() for _ in range(n)]
    for v, k, e in zip(val_tok, key_tok, ex_id):
        ex_vals[e].add(int(v)); ex_keys[e].add(int(k))
    key_lo, key_hi = 1, vmin        # keys in [1, V/2), values in [V/2, V)

    def error_types(s):
        wrong = correct[s] == 0
        pr = pred[s]
        is_val = np.array([pr[i] in ex_vals[ex_id[i]] for i in range(len(pr))])
        in_key_range = (pr >= key_lo) & (pr < key_hi)
        conf_val = wrong & is_val                       # emitted another value in-context
        conf_key = wrong & (~is_val) & in_key_range     # emitted a key
        other = wrong & (~is_val) & (~in_key_range)     # garbage / oov value
        tot = len(pr)
        return (correct[s].mean(), conf_val.sum()/tot, conf_key.sum()/tot, other.sum()/tot)

    etypes = np.array([error_types(s) for s in range(S)])   # [S, 4]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.22)

    # (1) overall acc trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(iters, overall, 'o-', color='#2166ac')
    ax.set_title('overall val answer accuracy vs step')
    ax.set_xlabel('step'); ax.set_ylabel('acc'); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)

    # (2) error-type composition
    ax = fig.add_subplot(gs[0, 1])
    ax.stackplot(iters, etypes[:, 0], etypes[:, 1], etypes[:, 2], etypes[:, 3],
                 labels=['correct', 'wrong: other in-ctx value', 'wrong: a key', 'wrong: other'],
                 colors=['#4daf4a', '#377eb8', '#ff7f00', '#999999'], alpha=0.9)
    ax.set_title('prediction composition vs step')
    ax.set_xlabel('step'); ax.set_ylabel('fraction'); ax.set_ylim(0, 1)
    ax.legend(loc='center right', fontsize=7)

    # (3) acc vs query gap (heatmap: step x gap)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(acc_gap, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1,
                   extent=[0, ngap, 0, S])
    ax.set_yticks(np.arange(S) + 0.5); ax.set_yticklabels(iters, fontsize=6)
    ax.set_title('acc by query gap (distance) per checkpoint')
    ax.set_xlabel('query gap (positions into query region)'); ax.set_ylabel('step')
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (4) acc vs key slot in context
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(acc_slot, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1,
                   extent=[0, nslot, 0, S])
    ax.set_yticks(np.arange(S) + 0.5); ax.set_yticklabels(iters, fontsize=6)
    ax.set_title('acc by binding slot in context (0=first pair)')
    ax.set_xlabel('key slot in context prefix'); ax.set_ylabel('step')
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (5) acc vs value-token bucket
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(acc_vb, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1,
                   extent=[0, nvb, 0, S])
    ax.set_yticks(np.arange(S) + 0.5); ax.set_yticklabels(iters, fontsize=6)
    ax.set_title(f'acc by value-token bucket ({nvb} bins over [{vmin},{vmax}])')
    ax.set_xlabel('value-token bucket'); ax.set_ylabel('step')
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (6) transition: for the largest acc jumps, what gap/slot gets fixed
    ax = fig.add_subplot(gs[2, 1])
    jumps = np.diff(overall)
    bump_idx = np.argsort(jumps)[::-1][:3] if S > 1 else []
    bump_idx = sorted(int(i) for i in bump_idx if jumps[i] > 0.01)
    for bi in bump_idx:
        newly_fixed = (correct[bi] == 0) & (correct[bi + 1] == 1)
        acc_by_gap_fixed = _grouped_acc(newly_fixed.astype(np.uint8), gap, ngap)
        ax.plot(np.arange(ngap), acc_by_gap_fixed,
                label=f'{iters[bi]}->{iters[bi+1]} (+{jumps[bi]:.2f})')
    ax.set_title('fraction of positions NEWLY fixed at each bump, by query gap')
    ax.set_xlabel('query gap'); ax.set_ylabel('P(fixed at bump | gap)'); ax.grid(alpha=0.3)
    if bump_idx:
        ax.legend(fontsize=8)

    fig.suptitle(f'mq5 error analysis: {tag}', fontsize=14, y=0.995)
    outdir = os.path.join(ROOT, 'slurm', 'plots')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'mq5_errors_{tag}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print('wrote', out)

    # console summary
    print('\nper-checkpoint summary:')
    print(f'{"step":>7} {"acc":>6} {"conf_val":>9} {"conf_key":>9} {"other":>7}')
    for s in range(S):
        print(f'{iters[s]:>7} {etypes[s,0]:>6.3f} {etypes[s,1]:>9.3f} '
              f'{etypes[s,2]:>9.3f} {etypes[s,3]:>7.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--block_size', type=int, default=0, help='0 -> from meta.pkl')
    ap.add_argument('--ckpts', nargs='*', default=None,
                    help='explicit ckpt filenames within out_dir (else all ckpt_iter*.pt)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--max_examples', type=int, default=0, help='0 -> all val examples')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--plot_only', action='store_true', help='skip compute, plot existing npz')
    args = ap.parse_args()

    tag = args.tag or os.path.basename(args.out_dir.rstrip('/'))
    npz = os.path.join(args.out_dir, 'error_analysis.npz')
    if not args.plot_only:
        npz = compute(args.out_dir, args.dataset, args.block_size or None, args.ckpts,
                      args.device, args.batch_size, args.dtype,
                      args.max_examples or None)
    plot(npz, tag)


if __name__ == '__main__':
    main()
