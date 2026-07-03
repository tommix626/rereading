"""Side-by-side slot-accuracy heatmaps + learning curves for several mq5 runs.

Each column = one run: top = per-context-slot accuracy over training (heatmap),
bottom = overall val acc, sharing that column's x-axis. Lets you compare, e.g.,
learned-APE-clean vs learned-APE-fillers vs RoPE at a glance.

Usage:
  python scripts/plot_mq5_slot_compare.py --tag d32_lr1e-3_compare \
    --runs \
      "out/mq5_softmax_d32_lr1e-3_snap/error_analysis.npz:APE + clean (0-pad)" \
      "out/mq5_softmax_d32_lr1e-3_rnq_snap/error_analysis.npz:APE + random fillers" \
      "out/mq5_softmax_d32_lr1e-3_rope_snap/error_analysis.npz:RoPE + clean (0-pad)"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_run(npz):
    d = np.load(npz)
    it = np.asarray(d['iters']); order = np.argsort(it)
    it = it[order]; C = d['correct'][order]; slot = d['key_slot']; D = int(d['D'])
    acc_slot = np.stack([np.array([C[s][slot == g].mean() if (slot == g).any() else np.nan
                                   for g in range(D)]) for s in range(len(it))], axis=1)
    return it, acc_slot, C.mean(axis=1), D


def xedges(it):
    if len(it) < 2:
        return np.array([it[0] - 250, it[0] + 250])
    mids = (it[:-1] + it[1:]) / 2
    return np.concatenate([[it[0] - (it[1] - it[0]) / 2], mids, [it[-1] + (it[-1] - it[-2]) / 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='"npz:label" per run')
    ap.add_argument('--tag', default='compare')
    ap.add_argument('--bump_thresh', type=float, default=0.02)
    args = ap.parse_args()

    runs = [r.split(':', 1) for r in args.runs]
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 7.2), squeeze=False,
                             gridspec_kw=dict(height_ratios=[3.0, 1.0], hspace=0.07, wspace=0.13))
    pcm = None
    for c, (npz, label) in enumerate(runs):
        it, acc_slot, overall, D = load_run(npz)
        xe = xedges(it); ye = np.arange(D + 1) - 0.5
        ax_h, ax_c = axes[0][c], axes[1][c]
        pcm = ax_h.pcolormesh(xe, ye, acc_slot, cmap='viridis', vmin=0, vmax=1)
        ax_h.set_title(f'{label}\nfinal val acc = {overall[-1]:.3f}', fontsize=11)
        if c == 0:
            ax_h.set_ylabel('binding slot in context prefix\n(0=first pair, %d=last before queries)' % (D - 1))
            ax_c.set_ylabel('overall\nval acc')
        ax_h.set_yticks(np.arange(0, D, 4))
        ax_c.plot(it, overall, 'o-', color='#b2182b', ms=3, lw=1.6)
        ax_c.set_ylim(-0.03, 1.03); ax_c.grid(alpha=0.3); ax_c.set_xlabel('training step')
        jumps = np.diff(overall)
        for i in range(len(it) - 1):
            if jumps[i] >= args.bump_thresh:
                bx = (it[i] + it[i + 1]) / 2
                ax_h.axvline(bx, color='white', ls='--', lw=0.9, alpha=0.7)
                ax_c.axvline(bx, color='gray', ls='--', lw=0.9, alpha=0.7)
        final = acc_slot[:, -1]
        for sl in np.where(final < 0.6)[0]:
            ax_h.text(it[-1], sl, f' {final[sl]:.2f}', va='center', ha='left', fontsize=7, color='#b2182b')

    cb = fig.colorbar(pcm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label('answer accuracy at that slot')
    fig.suptitle('mq5 D=32, lr=1e-3, 1-head softmax: per-slot recall over training — '
                 'the sub-1.0 plateau needs BOTH learned-APE and clean 0-padding', fontsize=13)
    out = os.path.join(ROOT, 'slurm', 'plots', f'mq5_slot_compare_{args.tag}.png')
    fig.savefig(out, dpi=135, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
