"""Plot the Zoology MQAR difficulty sweep: masked val CE and answer accuracy.

Reads slurm/diff_{gdn,softmax}_p{P}.log (expects 'val/loss' and 'val/acc' lines).
Writes slurm/ar_difficulty.png.

Masked CE baselines (answer positions only, vocab half = num values):
  no recall   ~ ln(vocab_size // 2)
  perfect     ~ 0
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
PAIRS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['1', '8', '64'])]
VOCAB = int(sys.argv[2]) if len(sys.argv) > 2 else 8192

VAL_RE = re.compile(r'step (\d+): val/loss ([\d.]+)(?: val/acc ([\d.]+))?')
NO_RECALL = np.log(VOCAB // 2)
PERFECT = 0.0


def parse(path):
    steps, vloss, vacc = [], [], []
    if not os.path.exists(path):
        return np.array([]), np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                vloss.append(float(m.group(2)))
                if m.group(3) is not None:
                    vacc.append(float(m.group(3)))
    return np.array(steps), np.array(vloss), np.array(vacc)


def main():
    mixers = {'gdn': ('GDN', 'C3', plt.cm.Reds), 'softmax': ('softmax', 'C0', plt.cm.Blues)}
    shades = {p: 0.4 + 0.5 * i / max(1, len(PAIRS) - 1) for i, p in enumerate(PAIRS)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    for key, (label, line_c, cmap) in mixers.items():
        finals = []
        for p in PAIRS:
            st, vl, va = parse(os.path.join(LOGDIR, f'diff_{key}_p{p}.log'))
            finals.append(vl[-1] if len(vl) else np.nan)
            if len(vl):
                axR.plot(st, vl, color=cmap(shades[p]), lw=2, marker='o', ms=3,
                         label=f'{label} p{p}')
        axL.plot(PAIRS, finals, color=line_c, lw=2.2, marker='o', ms=7, label=label)
        for p, fv in zip(PAIRS, finals):
            if not np.isnan(fv):
                _, _, va = parse(os.path.join(LOGDIR, f'diff_{key}_p{p}.log'))
                ann = f'{va[-1]*100:.0f}%' if len(va) else ''
                axL.annotate(ann, (p, fv), textcoords='offset points',
                             xytext=(6, 6), fontsize=8, color=line_c)

    for y, t in [(NO_RECALL, 'no recall'), (PERFECT, 'perfect recall')]:
        axL.axhline(y, color='gray', ls='--', lw=1)
        axL.text(PAIRS[0], y + 0.05, t, fontsize=8, color='gray')
    axL.set_xscale('log', base=2)
    axL.set_xticks(PAIRS)
    axL.set_xticklabels(PAIRS)
    axL.set_xlabel('num_kv_pairs (task difficulty)')
    axL.set_ylabel('masked val CE (nats)')
    axL.set_title('Masked CE vs difficulty (annotated: answer accuracy %)')
    axL.legend()
    axL.grid(alpha=0.25)

    axR.axhline(NO_RECALL, color='gray', ls='--', lw=1)
    axR.axhline(PERFECT, color='gray', ls=':', lw=1)
    axR.set_xlabel('iteration')
    axR.set_ylabel('masked val CE (nats)')
    axR.set_title('Masked CE curves (reds=GDN, blues=softmax; light->dark=more pairs)')
    axR.legend(fontsize=8, ncol=2)
    axR.grid(alpha=0.25)

    fig.suptitle('Zoology MQAR difficulty sweep (masked CE, d128): GDN vs softmax', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(LOGDIR, 'ar_difficulty.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    main()
