"""Plot the MQAR difficulty sweep: recall quality vs number of key->value pairs.

Reads slurm/diff_{gdn,softmax}_p{P}.log, writes slurm/ar_difficulty.png.
Left  : final val loss vs #pairs (log x), with no-recall / perfect-recall lines.
Right : val-loss training curves, shaded light->dark by #pairs.

Loss scale (vocab 512 = 256 keys + 256 vals, 25% answer positions):
  no recall (but learned key/val ranges) ~ ln(256)      = 5.545
  perfect recall                         ~ 0.75*ln(256) = 4.158
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

VAL_RE = re.compile(r'step (\d+): val/loss ([\d.]+)')
NO_RECALL = np.log(256)
PERFECT = 0.75 * np.log(256)


def parse(path):
    steps, vloss = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                steps.append(int(m.group(1))); vloss.append(float(m.group(2)))
    return np.array(steps), np.array(vloss)


def recall_pct(vl):
    return float(np.clip(1 - (vl - PERFECT) / (NO_RECALL - PERFECT), 0, 1)) * 100


def main():
    mixers = {'gdn': ('GDN', 'C3', plt.cm.Reds), 'softmax': ('softmax', 'C0', plt.cm.Blues)}
    shades = {p: 0.4 + 0.5 * i / max(1, len(PAIRS) - 1) for i, p in enumerate(PAIRS)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    for key, (label, line_c, cmap) in mixers.items():
        finals = []
        for p in PAIRS:
            st, vl = parse(os.path.join(LOGDIR, f'diff_{key}_p{p}.log'))
            finals.append(vl[-1] if len(vl) else np.nan)
            if len(vl):
                axR.plot(st, vl, color=cmap(shades[p]), lw=2, marker='o', ms=3,
                         label=f'{label} p{p}')
        axL.plot(PAIRS, finals, color=line_c, lw=2.2, marker='o', ms=7, label=label)
        for p, fv in zip(PAIRS, finals):
            if not np.isnan(fv):
                axL.annotate(f'{recall_pct(fv):.0f}%', (p, fv), textcoords='offset points',
                             xytext=(6, 6), fontsize=8, color=line_c)

    for y, t in [(NO_RECALL, 'no recall (ln256)'), (PERFECT, 'perfect recall')]:
        axL.axhline(y, color='gray', ls='--', lw=1)
        axL.text(PAIRS[0], y + 0.02, t, fontsize=8, color='gray')
    axL.set_xscale('log', base=2)
    axL.set_xticks(PAIRS); axL.set_xticklabels(PAIRS)
    axL.set_xlabel('number of key->value pairs (task difficulty)')
    axL.set_ylabel('final val loss (nats)')
    axL.set_title('Recall vs difficulty (annotated: approx recall %)')
    axL.legend(); axL.grid(alpha=0.25)

    axR.axhline(NO_RECALL, color='gray', ls='--', lw=1)
    axR.axhline(PERFECT, color='gray', ls=':', lw=1)
    axR.set_xlabel('iteration'); axR.set_ylabel('val loss (nats)')
    axR.set_title('Val-loss curves (reds=GDN, blues=softmax; light->dark=more pairs)')
    axR.legend(fontsize=8, ncol=2); axR.grid(alpha=0.25)

    fig.suptitle('MQAR difficulty sweep at fixed model size (d256): GDN vs softmax', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(LOGDIR, 'ar_difficulty.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)
    print('\nfinal val loss (approx recall %):')
    for key, (label, _, _) in mixers.items():
        for p in PAIRS:
            _, vl = parse(os.path.join(LOGDIR, f'diff_{key}_p{p}.log'))
            if len(vl):
                print(f'  {label:8s} p{p:<3d}: {vl[-1]:.3f}  ({recall_pct(vl[-1]):.0f}% recall)')


if __name__ == '__main__':
    main()
