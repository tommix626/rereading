"""Plot rt regression-test learning curves and summary bar chart.

Logs: slurm/{rt1,rt2,rt3,rtX}/{cell}_lr{LR}.log

Usage:
  python scripts/plot_rt_curves.py
  python scripts/plot_rt_curves.py rt1,rt2,rt3,rtX
"""
import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'slurm', 'plots')
CELLS = (sys.argv[1].split(',') if len(sys.argv) > 1 else
         ['rt1', 'rt2', 'rt3', 'rtX'])
LRS = ['1e-4', '3e-4', '6e-4', '1e-3', '3e-3', '1e-2']

CELL_DESC = {
    'rt1': 'OLD model + OLD data\n(RoPE,2h,wd1.0 | lc5 Q=1)',
    'rt2': 'OLD model + NEW data\n(data-only swap)',
    'rt3': 'NEW model + OLD data\n(model/HP-only swap)',
    'rtX': 'NEW model + NEW data\n(mq5 baseline)',
}

VAL_RE = re.compile(r'step (\d+): val/loss ([\d.]+) val/acc ([\d.]+)')


def parse(path):
    steps, loss, acc = [], [], []
    if not os.path.exists(path):
        return steps, loss, acc
    with open(path) as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                loss.append(float(m.group(2)))
                acc.append(float(m.group(3)))
    return np.array(steps), np.array(loss), np.array(acc)


def log_path(cell, lr):
    return os.path.join(ROOT, 'slurm', cell, f'{cell}_lr{lr}.log')


def cell_stats(cell):
    finals, bests = [], []
    for lr in LRS:
        steps, _, acc = parse(log_path(cell, lr))
        if len(acc):
            finals.append(acc[-1])
            bests.append(acc.max())
        else:
            finals.append(np.nan)
            bests.append(np.nan)
    return finals, bests


def plot_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(LRS)))

    for ax, cell in zip(axes.flat, CELLS):
        for i, lr in enumerate(LRS):
            steps, _, acc = parse(log_path(cell, lr))
            if len(steps):
                ax.plot(steps, acc, color=cmap[i], lw=1.8, label=f'lr={lr}')
        ax.set_title(CELL_DESC.get(cell, cell))
        ax.set_xlabel('step')
        ax.set_ylabel('val answer accuracy')
        ax.set_ylim(-0.03, 1.03)
        ax.axhline(0.9, color='gray', ls=':', lw=1, alpha=0.6)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('rt regression series: LR sweep @ 50k steps (2L d=64 softmax variants)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUTDIR, 'curves_rt_lrsweep.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)
    return out


def plot_summary():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(LRS))
    width = 0.18
    colors = ['#4d4d4d', '#2166ac', '#b2182b', '#1b7837']

    for ci, cell in enumerate(CELLS):
        finals, bests = cell_stats(cell)
        off = (ci - 1.5) * width
        axes[0].bar(x + off, finals, width, label=cell, color=colors[ci], alpha=0.9)
        axes[1].bar(x + off, bests, width, label=cell, color=colors[ci], alpha=0.9)

    for ax, title in zip(axes, ['final val acc @50k', 'best val acc']):
        ax.set_xticks(x)
        ax.set_xticklabels(LRS, rotation=30, ha='right')
        ax.set_xlabel('learning rate')
        ax.set_ylabel('answer accuracy')
        ax.set_ylim(0, 1.08)
        ax.axhline(0.9, color='gray', ls=':', lw=1)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis='y')

    fig.suptitle('rt series summary: which factor fixes recall?', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUTDIR, 'curves_rt_summary.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)
    return out


def print_verdict():
    print('\nrt verdict (threshold acc >= 0.9):')
    for cell in CELLS:
        finals, bests = cell_stats(cell)
        good_final = sum(1 for f in finals if not np.isnan(f) and f >= 0.9)
        good_best = sum(1 for b in bests if not np.isnan(b) and b >= 0.9)
        best_lr = LRS[int(np.nanargmax(bests))] if any(not np.isnan(b) for b in bests) else 'NA'
        best_val = np.nanmax(bests)
        print(f'  {cell}: best={best_val:.4f}@{best_lr}  LRs>=0.9: {good_best}/6 final, {good_final}/6 @50k')

    _, rt2_best = cell_stats('rt2')
    _, rt3_best = cell_stats('rt3')
    rt2_ok = sum(1 for b in rt2_best if not np.isnan(b) and b >= 0.9)
    rt3_ok = sum(1 for b in rt3_best if not np.isnan(b) and b >= 0.9)
    print('\n  Interpretation:')
    if rt2_ok >= 4:
        print('  - rt2 (data-only) works broadly → multi-query DATA is a major enabler.')
    if rt3_ok <= 2:
        print('  - rt3 (model-only) works only at select LRs → MODEL/HP change alone is insufficient.')
    if rt2_ok > rt3_ok:
        print('  - DATA swap helps more than MODEL swap → culprit leans DATA (multi-query gradient density).')
    elif rt3_ok > rt2_ok:
        print('  - MODEL swap helps more → culprit leans MODEL/HP.')
    else:
        print('  - Both swaps comparable → mixed factors.')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plot_curves()
    plot_summary()
    print_verdict()


if __name__ == '__main__':
    main()
