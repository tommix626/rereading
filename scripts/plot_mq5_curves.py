"""Plot mq5 sweep results: capacity curve, overview LR sweeps, and per-D detail.

Logs: slurm/mq5/mq5_{softmax,gdn}_d{D}_lr{LR}.log

Usage:
  python scripts/plot_mq5_curves.py
  python scripts/plot_mq5_curves.py 16,32,64,128,256,512
"""
import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, 'slurm', 'mq5')
OUTDIR = os.path.join(ROOT, 'slurm', 'plots')
DS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else
                       ['16', '32', '64', '128', '256', '512'])]
MIXERS = {
    'softmax': ('softmax', '#2166ac', '-'),
    'gdn': ('GDN-noconv', '#b2182b', '--'),
}
LOG_RE = re.compile(r'mq5_(softmax|gdn)_d(\d+)_lr(.+)\.log')
VAL_RE = re.compile(r'step (\d+): val/loss ([\d.]+) val/acc ([\d.]+)')
TRAIN_RE = re.compile(r'step (\d+): train/loss \(avg over last \d+ iters\) = ([\d.]+)')
VOCAB = 8192
NO_RECALL = np.log(VOCAB // 2)


def lr_sort_key(lr):
    return float(lr.replace('e', 'E'))


def discover_runs():
    """Return {d: {mixer: [lr, ...]}} from existing log files."""
    runs = {}
    for path in glob.glob(os.path.join(LOGDIR, 'mq5_*.log')):
        m = LOG_RE.search(os.path.basename(path))
        if not m:
            continue
        mixer, d, lr = m.group(1), int(m.group(2)), m.group(3)
        runs.setdefault(d, {}).setdefault(mixer, set()).add(lr)
    for d in runs:
        for mixer in runs[d]:
            runs[d][mixer] = sorted(runs[d][mixer], key=lr_sort_key)
    return runs


def parse(path):
    val_steps, val_loss, val_acc = [], [], []
    train_steps, train_loss = [], []
    if not os.path.exists(path):
        return dict(
            val_steps=np.array([]), val_loss=np.array([]), val_acc=np.array([]),
            train_steps=np.array([]), train_loss=np.array([]),
        )
    with open(path) as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                val_steps.append(int(m.group(1)))
                val_loss.append(float(m.group(2)))
                val_acc.append(float(m.group(3)))
                continue
            m = TRAIN_RE.search(line)
            if m:
                train_steps.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
    return dict(
        val_steps=np.array(val_steps),
        val_loss=np.array(val_loss),
        val_acc=np.array(val_acc),
        train_steps=np.array(train_steps),
        train_loss=np.array(train_loss),
    )


def log_path(mixer, d, lr):
    return os.path.join(LOGDIR, f'mq5_{mixer}_d{d}_lr{lr}.log')


def runs_for_d(d, all_runs):
    out = {}
    for mixer in MIXERS:
        out[mixer] = all_runs.get(d, {}).get(mixer, [])
    return out


def best_for(mixer, d, lrs):
    best_acc, best_lr = -1.0, None
    for lr in lrs:
        acc = parse(log_path(mixer, d, lr))['val_acc']
        if len(acc) and acc.max() > best_acc:
            best_acc = acc.max()
            best_lr = lr
    return best_acc, best_lr


def lr_colors(lrs):
    if not lrs:
        return {}
    uniq = sorted(set(lrs), key=lr_sort_key)
    cmap = plt.cm.viridis(np.linspace(0.12, 0.88, len(uniq)))
    return {lr: cmap[i] for i, lr in enumerate(uniq)}


def plot_capacity(all_runs):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mixer, (label, color, _) in MIXERS.items():
        bests, labels = [], []
        for d in DS:
            ba, blr = best_for(mixer, d, runs_for_d(d, all_runs)[mixer])
            bests.append(ba if ba >= 0 else np.nan)
            labels.append(f'{blr}' if blr else '')
        ax.plot(DS, bests, 'o-', color=color, lw=2.2, ms=8, label=label)
        for d, b, lr in zip(DS, bests, labels):
            if not np.isnan(b):
                ax.annotate(f'{b:.2f}\n({lr})', (d, b), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=7, color=color)

    ax.set_xscale('log', base=2)
    ax.set_xticks(DS)
    ax.set_xticklabels([str(d) for d in DS])
    ax.set_xlabel('KV pairs D (seq len N = 6D)')
    ax.set_ylabel('best val answer accuracy')
    ax.set_ylim(-0.05, 1.08)
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_title('mq5 capacity curve: best acc across LR sweep per D')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'curves_mq5_highD.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)
    return out


def plot_lrsweep(all_runs):
    n = len(DS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    all_lrs = sorted({lr for d in DS for mixer in MIXERS for lr in runs_for_d(d, all_runs)[mixer]},
                     key=lr_sort_key)
    colors = lr_colors(all_lrs)

    for idx, d in enumerate(DS):
        ax = axes[idx // ncols][idx % ncols]
        for mixer, (mlabel, _, ls) in MIXERS.items():
            for lr in runs_for_d(d, all_runs)[mixer]:
                dct = parse(log_path(mixer, d, lr))
                if len(dct['val_steps']):
                    ax.plot(dct['val_steps'], dct['val_acc'], color=colors[lr], lw=1.6,
                            ls=ls, alpha=0.85, label=f'{mlabel} lr={lr}' if idx == 0 else None)
        ax.set_title(f'D={d}')
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel('step')
        ax.set_ylabel('val acc')
        ax.grid(alpha=0.25)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles = [Line2D([0], [0], color=colors[lr], lw=2, label=f'lr={lr}') for lr in all_lrs]
    handles += [Line2D([0], [0], color='k', lw=2, ls='-', label='softmax'),
                Line2D([0], [0], color='k', lw=2, ls='--', label='GDN-noconv')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('mq5 LR sweep learning curves (solid=softmax, dashed=GDN)', fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(OUTDIR, 'curves_mq5_lrsweep.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('wrote', out)
    return out


def plot_per_d_detail(d, all_runs):
    """One figure per D: val acc, val loss, train loss — every run on the logs."""
    d_runs = runs_for_d(d, all_runs)
    all_lrs = sorted({lr for mixer in MIXERS for lr in d_runs[mixer]}, key=lr_sort_key)
    if not all_lrs:
        return None
    colors = lr_colors(all_lrs)
    max_step = 0
    for mixer in MIXERS:
        for lr in d_runs[mixer]:
            dct = parse(log_path(mixer, d, lr))
            if len(dct['val_steps']):
                max_step = max(max_step, int(dct['val_steps'][-1]))

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    ax_acc, ax_vloss, ax_tloss = axes
    n_seq = 6 * d
    if n_seq % 2 == 1:
        n_seq += 1

    for mixer, (mlabel, mcolor, ls) in MIXERS.items():
        for lr in d_runs[mixer]:
            path = log_path(mixer, d, lr)
            dct = parse(path)
            tag = f'{mlabel} lr={lr}'
            c = colors[lr]
            if len(dct['val_steps']):
                ax_acc.plot(dct['val_steps'], dct['val_acc'], color=c, ls=ls, lw=2.0,
                            alpha=0.9, label=tag)
                ax_vloss.plot(dct['val_steps'], dct['val_loss'], color=c, ls=ls, lw=2.0,
                              alpha=0.9, label=tag)
            if len(dct['train_steps']):
                ax_tloss.plot(dct['train_steps'], dct['train_loss'], color=c, ls=ls, lw=1.6,
                              alpha=0.85, label=tag)

    ax_acc.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.7)
    ax_acc.set_ylabel('val answer accuracy')
    ax_acc.set_ylim(-0.03, 1.03)
    ax_acc.set_title(f'mq5 D={d} (seq N={n_seq}): all runs')
    ax_acc.grid(alpha=0.25)

    ax_vloss.axhline(NO_RECALL, color='gray', ls='--', lw=1, alpha=0.7, label='no recall ln(V/2)')
    ax_vloss.axhline(0.0, color='gray', ls=':', lw=1, alpha=0.7, label='perfect recall')
    ax_vloss.set_ylabel('masked val CE (nats)')
    ax_vloss.grid(alpha=0.25)

    ax_tloss.set_ylabel('train loss (nats)')
    ax_tloss.set_xlabel('iteration')
    ax_tloss.grid(alpha=0.25)

    lr_handles = [Line2D([0], [0], color=colors[lr], lw=2.5, label=f'lr={lr}') for lr in all_lrs]
    mixer_handles = [
        Line2D([0], [0], color='k', lw=2.5, ls='-', label='softmax'),
        Line2D([0], [0], color='k', lw=2.5, ls='--', label='GDN-noconv'),
    ]
    ref_handles = [
        Line2D([0], [0], color='gray', lw=1.5, ls='--', label='no recall'),
        Line2D([0], [0], color='gray', lw=1.5, ls=':', label='perfect recall'),
    ]
    fig.legend(handles=lr_handles + mixer_handles + ref_handles, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    note = f'{len(all_lrs)} LRs x 2 mixers' if all(len(d_runs[m]) == len(all_lrs) for m in MIXERS) else \
        ', '.join(f'{MIXERS[m][0]}: {len(d_runs[m])} runs' for m in MIXERS)
    fig.suptitle(
        f'mq5 detail D={d} — val acc / val loss / train loss ({note}; color=LR, linestyle=mixer)',
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(OUTDIR, f'curves_mq5_d{d}_detail.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)
    plt.close(fig)
    return out


def plot_per_d_detail_grid(all_runs):
    """Combined 2x3 grid: each cell is val-acc-only for quick comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    all_lrs = sorted({lr for d in DS for mixer in MIXERS for lr in runs_for_d(d, all_runs)[mixer]},
                     key=lr_sort_key)
    colors = lr_colors(all_lrs)

    for idx, d in enumerate(DS):
        ax = axes[idx // 3][idx % 3]
        d_runs = runs_for_d(d, all_runs)
        for mixer, (mlabel, _, ls) in MIXERS.items():
            for lr in d_runs[mixer]:
                dct = parse(log_path(mixer, d, lr))
                if len(dct['val_steps']):
                    ax.plot(dct['val_steps'], dct['val_acc'], color=colors[lr], ls=ls,
                            lw=1.8, alpha=0.9)
        ax.set_title(f'D={d}')
        ax.set_xlabel('step')
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        if idx % 3 == 0:
            ax.set_ylabel('val acc')

    handles = [Line2D([0], [0], color=colors[lr], lw=2, label=f'lr={lr}') for lr in all_lrs]
    handles += [Line2D([0], [0], color='k', lw=2, ls='-', label='softmax'),
                Line2D([0], [0], color='k', lw=2, ls='--', label='GDN-noconv')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('mq5 val-acc overview per D (all runs)', fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = os.path.join(OUTDIR, 'curves_mq5_perD_acc_grid.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('wrote', out)
    return out


def print_summary(all_runs):
    print('\nmq5 summary (best val acc per mixer x D):')
    for d in DS:
        parts = []
        for mixer, (label, _, _) in MIXERS.items():
            ba, blr = best_for(mixer, d, runs_for_d(d, all_runs)[mixer])
            parts.append(f'{label}={ba:.4f}@{blr}' if ba >= 0 else f'{label}=NA')
        print(f'  D={d:<4d}  ' + '  '.join(parts))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    all_runs = discover_runs()
    plot_capacity(all_runs)
    plot_lrsweep(all_runs)
    plot_per_d_detail_grid(all_runs)
    for d in DS:
        plot_per_d_detail(d, all_runs)
    print_summary(all_runs)


if __name__ == '__main__':
    main()
