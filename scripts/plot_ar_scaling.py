"""Parse the MQAR size-sweep logs and plot train + val loss curves.

Two color groups (GDN = reds, softmax = blues), shaded light->dark by model size.
Usage: python scripts/plot_ar_scaling.py
Reads slurm/sweep_{gdn,softmax}_d{128,256,512}.log, writes slurm/ar_scaling.png
"""
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, 'slurm')
PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'sweep'  # log/png filename prefix

ITER_RE = re.compile(r'iter (\d+): loss ([\d.]+)')
VAL_RE = re.compile(r'step (\d+): val/loss ([\d.]+)')
NPARAM_RE = re.compile(r'number of parameters: ([\d.]+)M')

VOCAB = 512
NUM_VALS = 256


def parse(path):
    iters, tloss, steps, vloss = [], [], [], []
    nparam = None
    with open(path) as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                iters.append(int(m.group(1))); tloss.append(float(m.group(2)))
            m = VAL_RE.search(line)
            if m:
                steps.append(int(m.group(1))); vloss.append(float(m.group(2)))
            m = NPARAM_RE.search(line)
            if m:
                nparam = float(m.group(1))
    return (np.array(iters), np.array(tloss), np.array(steps),
            np.array(vloss), nparam)


def smooth(y, k=9):
    if len(y) < 3:
        return y
    k = min(k | 1, len(y) | 1)   # force odd, <= len
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode='edge')
    ker = np.ones(k) / k
    return np.convolve(yp, ker, mode='valid')[:len(y)]


def main():
    sizes = [128, 256, 512]
    mixers = {
        'gdn':     ('GDN',     plt.cm.Reds),
        'softmax': ('softmax', plt.cm.Blues),
    }
    # light -> dark for small -> large
    shades = {128: 0.45, 256: 0.68, 512: 0.92}

    fig, (axT, axV) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for key, (label, cmap) in mixers.items():
        for d in sizes:
            path = os.path.join(LOGDIR, f'{PREFIX}_{key}_d{d}.log')
            if not os.path.exists(path):
                print(f'missing {path}; skipping')
                continue
            it, tl, st, vl, npar = parse(path)
            if len(it) == 0:
                print(f'no data in {path}; skipping')
                continue
            color = cmap(shades[d])
            tag = f'{label} d{d}' + (f' ({npar:.2f}M)' if npar else '')
            axT.plot(it, smooth(tl), color=color, lw=1.6, label=tag)
            axV.plot(st, vl, color=color, lw=2.0, marker='o', ms=3, label=tag)

    # reference lines
    for ax in (axT, axV):
        ax.axhline(np.log(NUM_VALS), color='gray', ls='--', lw=1,
                   label=f'recall baseline ln({NUM_VALS})={np.log(NUM_VALS):.2f}')
        ax.set_xlabel('iteration')
        ax.grid(alpha=0.25)
    axT.set_ylabel('loss (nats)')
    axT.set_title('Training loss (smoothed)')
    axV.set_title('Validation loss')
    axV.legend(fontsize=8, loc='upper right', ncol=1)
    fig.suptitle('MQAR associative recall: GDN vs softmax across model size\n'
                 '(reds=GDN, blues=softmax; light->dark = small->large)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(LOGDIR, f'ar_scaling_{PREFIX}.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)

    # print a quick final-loss table
    print('\nfinal val loss:')
    for key, (label, _) in mixers.items():
        for d in sizes:
            path = os.path.join(LOGDIR, f'{PREFIX}_{key}_d{d}.log')
            if os.path.exists(path):
                _, _, _, vl, npar = parse(path)
                if len(vl):
                    print(f'  {label:8s} d{d} ({npar}M): {vl[-1]:.4f}')


if __name__ == '__main__':
    main()
