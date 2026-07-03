"""Aligned slot-accuracy-over-time heatmap + val-acc learning curve for an mq5 run.

Reads the error_analysis.npz produced by scripts/analyze_mq5_errors.py and draws:
  - top:   heatmap of per-context-slot answer accuracy vs training step
           (y = binding slot in context prefix, x = step, color = accuracy)
  - bottom: overall val answer accuracy vs step, sharing the x-axis, so it is
            visually obvious which slots "turn on" at each grokking bump.

Pure numpy/matplotlib (no torch/fla) -> runs anywhere, incl. the login node.

Usage:
  python scripts/plot_mq5_slot_heatmap.py \
      --npz out/mq5_softmax_d32_lr1e-3_snap/error_analysis.npz --tag d32_lr1e-3
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def acc_by(row, key, n):
    return np.array([row[key == g].mean() if (key == g).any() else np.nan for g in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--tag', default='mq5')
    ap.add_argument('--bump_thresh', type=float, default=0.02,
                    help='min jump in overall acc between snapshots to mark as a bump')
    args = ap.parse_args()

    d = np.load(args.npz)
    iters = np.asarray(d['iters'])
    order = np.argsort(iters)
    iters = iters[order]
    C = d['correct'][order]
    slot = d['key_slot']
    D = int(d['D'])
    S = len(iters)

    acc_slot = np.stack([acc_by(C[s], slot, D) for s in range(S)], axis=1)  # [D, S]
    overall = C.mean(axis=1)

    # bump locations (between consecutive snapshots)
    jumps = np.diff(overall)
    bump_mid = [(iters[i] + iters[i + 1]) / 2 for i in range(S - 1) if jumps[i] >= args.bump_thresh]

    # x cell edges (snapshots are ~evenly spaced)
    if S > 1:
        mids = (iters[:-1] + iters[1:]) / 2
        half0 = iters[1] - iters[0]
        halfN = iters[-1] - iters[-2]
        xedges = np.concatenate([[iters[0] - half0 / 2], mids, [iters[-1] + halfN / 2]])
    else:
        xedges = np.array([iters[0] - 250, iters[0] + 250])
    yedges = np.arange(D + 1) - 0.5

    fig, (ax_h, ax_c) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw=dict(height_ratios=[3.2, 1.0], hspace=0.06))

    pcm = ax_h.pcolormesh(xedges, yedges, acc_slot, cmap='viridis', vmin=0, vmax=1)
    ax_h.set_ylabel('binding slot in context prefix\n(0 = first KV pair, %d = last before queries)' % (D - 1))
    ax_h.set_yticks(np.arange(0, D, 2))
    ax_h.set_title(f'mq5 {args.tag}: per-context-slot recall accuracy over training', fontsize=13)
    cb = fig.colorbar(pcm, ax=[ax_h, ax_c], fraction=0.035, pad=0.01)
    cb.set_label('answer accuracy at that slot')

    ax_c.plot(iters, overall, 'o-', color='#b2182b', ms=4, lw=1.8)
    ax_c.set_ylabel('overall\nval acc')
    ax_c.set_xlabel('training step')
    ax_c.set_ylim(-0.03, 1.03)
    ax_c.grid(alpha=0.3)

    for bx in bump_mid:
        for ax in (ax_h, ax_c):
            ax.axvline(bx, color='white' if ax is ax_h else 'gray', ls='--', lw=1.0, alpha=0.7)

    # annotate final stuck slots on the heatmap right edge
    final = acc_slot[:, -1]
    for sl in np.where(final < 0.6)[0]:
        ax_h.text(iters[-1], sl, f' {final[sl]:.2f}', va='center', ha='left',
                  fontsize=7, color='#b2182b')

    outdir = os.path.join(ROOT, 'slurm', 'plots')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'mq5_slot_heatmap_{args.tag}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('wrote', out)
    if bump_mid:
        print('bumps marked at steps ~', [int(b) for b in bump_mid])


if __name__ == '__main__':
    main()
