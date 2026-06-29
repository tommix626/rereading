"""Long-context retrieval: recall accuracy vs retrieval count, for two sequence
lengths, GDN vs softmax.

Reads slurm/lc_{gdn,softmax}_s{S}_q{Q}.log (val/acc lines), writes slurm/lc_recall.png.
Left : final recall accuracy vs Q (color=mixer, solid=S256, dashed=S1024).
Right: accuracy training curves.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, 'slurm')
SEQS = [256, 1024]
QS = [1, 8, 64]
ACC_RE = re.compile(r'step (\d+): val/loss [\d.]+ val/acc ([\d.]+)')


def parse(path):
    steps, acc = [], []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                m = ACC_RE.search(line)
                if m:
                    steps.append(int(m.group(1))); acc.append(float(m.group(2)))
    return np.array(steps), np.array(acc)


def main():
    mixers = {'gdn': ('GDN', 'C3'), 'softmax': ('softmax', 'C0')}
    styles = {256: '-', 1024: '--'}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    for key, (label, c) in mixers.items():
        for S in SEQS:
            finals = []
            for Q in QS:
                st, ac = parse(os.path.join(LOGDIR, f'lc_{key}_s{S}_q{Q}.log'))
                finals.append(ac[-1] if len(ac) else np.nan)
                if len(ac):
                    axR.plot(st, ac, color=c, ls=styles[S], alpha=0.5 + 0.5 * (S == 1024),
                             lw=1.6)
            axL.plot(QS, finals, color=c, ls=styles[S], lw=2.2, marker='o', ms=7,
                     label=f'{label} S={S}')

    axL.axhline(1.0, color='gray', ls=':', lw=1)
    axL.set_xscale('log', base=2); axL.set_xticks(QS); axL.set_xticklabels(QS)
    axL.set_xlabel('retrieval count Q (queries)')
    axL.set_ylabel('final recall accuracy')
    axL.set_ylim(-0.03, 1.03)
    axL.set_title('Recall accuracy vs retrieval count\n(solid S=256, dashed S=1024)')
    axL.legend(); axL.grid(alpha=0.25)

    axR.set_xlabel('iteration'); axR.set_ylabel('recall accuracy')
    axR.set_ylim(-0.03, 1.03)
    axR.set_title('Accuracy curves (red=GDN, blue=softmax; dashed=S1024)')
    axR.grid(alpha=0.25)

    fig.suptitle('Long-context associative recall: attention vs GDN (d256)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(LOGDIR, 'lc_recall.png')
    fig.savefig(out, dpi=130); print('wrote', out)
    print('\nfinal recall accuracy:')
    for key, (label, _) in mixers.items():
        for S in SEQS:
            row = []
            for Q in QS:
                _, ac = parse(os.path.join(LOGDIR, f'lc_{key}_s{S}_q{Q}.log'))
                row.append(f'Q{Q}={ac[-1]:.3f}' if len(ac) else f'Q{Q}=NA')
            print(f'  {label:8s} S={S:<4d}: ' + '  '.join(row))


if __name__ == '__main__':
    main()
