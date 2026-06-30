"""Length sweep: recall accuracy vs sequence length (pool size), GDN vs softmax.

Reads slurm/lc2_{gdn,softmax}_s{S}_q1.log (val/acc), writes slurm/lc_len.png.
Left : final recall accuracy vs S (log x).
Right: accuracy training curves (light->dark = longer S).
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, 'slurm')
SEQS = [16, 64, 256, 1024]
ACC_RE = re.compile(r'step (\d+): val/loss [\d.]+ val/acc ([\d.]+)')


def parse(path):
    st, ac = [], []
    if not os.path.exists(path):
        return np.array(st), np.array(ac), 'missing'
    with open(path) as f:
        text = f.read()
    for line in text.splitlines():
        m = ACC_RE.search(line)
        if m:
            st.append(int(m.group(1))); ac.append(float(m.group(2)))
    if 'Killed' in text or 'CANCELLED' in text or 'Force Terminated' in text:
        status = 'killed' if len(ac) else 'revoked'
    elif len(ac) and st[-1] >= 19000:
        status = 'complete'
    elif len(ac):
        status = 'partial'
    else:
        status = 'empty'
    return np.array(st), np.array(ac), status


def main():
    mixers = {'gdn': ('GDN', 'C3', plt.cm.Reds), 'softmax': ('softmax', 'C0', plt.cm.Blues)}
    shades = {s: 0.4 + 0.5 * i / (len(SEQS) - 1) for i, s in enumerate(SEQS)}
    fig, (axL, axR, axZ) = plt.subplots(1, 3, figsize=(17, 5.2))

    for key, (label, c, cmap) in mixers.items():
        finals = []
        for S in SEQS:
            path = os.path.join(LOGDIR, f'lc2_{key}_s{S}_q1.log')
            st, ac, status = parse(path)
            finals.append(ac[-1] if len(ac) else np.nan)
            if len(ac):
                ls = '-' if status == 'complete' else '--'
                tag = '' if status == 'complete' else f' ({status}@{st[-1]})'
                axR.plot(st, ac, color=cmap(shades[S]), lw=2, ls=ls,
                         label=f'{label} S={S}{tag}')
                axZ.plot(st, ac, color=cmap(shades[S]), lw=2, ls=ls,
                         label=f'{label} S={S}{tag}')
        axL.plot(SEQS, finals, color=c, lw=2.4, marker='o', ms=8, label=label)

    axL.axhline(1.0, color='gray', ls=':', lw=1)
    axL.set_xscale('log', base=2); axL.set_xticks(SEQS); axL.set_xticklabels(SEQS)
    axL.set_xlabel('sequence length S  (pool ~ S/2 bindings)')
    axL.set_ylabel('final recall accuracy'); axL.set_ylim(-0.03, 1.03)
    axL.set_title('Recall accuracy vs context length')
    axL.legend(); axL.grid(alpha=0.25)

    axR.set_xlabel('iteration'); axR.set_ylabel('recall accuracy'); axR.set_ylim(-0.03, 1.03)
    axR.set_title('Full training curves')
    axR.legend(fontsize=7, ncol=2); axR.grid(alpha=0.25)

    axZ.set_xlim(0, 5000); axZ.set_ylim(-0.03, 1.05)
    axZ.set_xlabel('iteration'); axZ.set_ylabel('recall accuracy')
    axZ.set_title('Zoom: first 5k iters (learning phase)')
    axZ.legend(fontsize=7, ncol=2); axZ.grid(alpha=0.25)

    fig.suptitle('Associative recall vs context length: attention vs GDN (d256, Q=1)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(LOGDIR, 'lc_len.png')
    fig.savefig(out, dpi=130); print('wrote', out)
    print('\nfinal recall accuracy:')
    for key, (label, _, _) in mixers.items():
        row = []
        for S in SEQS:
            _, ac, status = parse(os.path.join(LOGDIR, f'lc2_{key}_s{S}_q1.log'))
            row.append(f'S{S}={ac[-1]:.3f} ({status})' if len(ac) else f'S{S}=NA ({status})')
        print(f'  {label:8s}: ' + '  '.join(row))


if __name__ == '__main__':
    main()
