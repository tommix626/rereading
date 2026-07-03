"""Plot MQAR sweep training curves: masked val CE, train loss, answer accuracy.

Log naming:
  lc3: slurm/lc3_{gdn,softmax}_s{S}_q1.log          (S = total sequence length)
  mq4: slurm/mq4_{gdn,softmax}_ctx{S}_q{Q}.log      (S = context KV pairs, Q=S/4)

Usage:
  python scripts/plot_lc_curves.py
  python scripts/plot_lc_curves.py lc3 16,32,64,128
  python scripts/plot_lc_curves.py mq4 16,32,64,128
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
PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'lc3'
SEQS = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['16', '32', '64', '128'])]
# vocab only feeds the "no-recall" chance reference line; lc4 bumped it to 8192.
VOCAB = int(sys.argv[3]) if len(sys.argv) > 3 else (8192 if PREFIX == 'lc4' else 2048)

VAL_RE = re.compile(r'step (\d+): val/loss ([\d.]+) val/acc ([\d.]+)')
TRAIN_RE = re.compile(r'step (\d+): train/loss \(avg over last \d+ iters\) = ([\d.]+)')
ITER_RE = re.compile(r'iter (\d+): loss ([\d.]+),')


def parse(path):
    val_steps, val_loss, val_acc = [], [], []
    train_steps, train_loss = [], []
    iters, iter_loss = [], []
    status = 'missing'
    if not os.path.exists(path):
        return dict(status=status)

    with open(path) as f:
        text = f.read()

    for line in text.splitlines():
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
            continue
        m = ITER_RE.search(line)
        if m:
            iters.append(int(m.group(1)))
            iter_loss.append(float(m.group(2)))

    if 'Killed' in text or 'CANCELLED' in text or 'Force Terminated' in text:
        status = 'killed' if val_steps else 'revoked'
    elif val_steps and val_steps[-1] >= 19000:
        status = 'complete'
    elif val_steps:
        status = 'partial'
    elif iters:
        status = 'running'
    else:
        status = 'empty'

    return dict(
        status=status,
        val_steps=np.array(val_steps),
        val_loss=np.array(val_loss),
        val_acc=np.array(val_acc),
        train_steps=np.array(train_steps),
        train_loss=np.array(train_loss),
        iters=np.array(iters),
        iter_loss=np.array(iter_loss),
    )


def queries_for(S):
    # mq4 uses Q=S/4; length-sweeps (lc3/lc4) are single-query.
    return S // 4 if PREFIX == 'mq4' else 1


def log_path(mixer, S):
    Q = queries_for(S)
    if PREFIX == 'mq4':
        name = f'{PREFIX}_{mixer}_ctx{S}_q{Q}.log'
    else:
        name = f'{PREFIX}_{mixer}_s{S}_q{Q}.log'
    return os.path.join(LOGDIR, name)


def main():
    mixers = {
        'gdn': ('GDN', plt.cm.Reds),
        'softmax': ('softmax', plt.cm.Blues),
    }
    shades = {s: 0.42 + 0.5 * i / max(1, len(SEQS) - 1) for i, s in enumerate(SEQS)}
    no_recall = np.log(VOCAB // 2)
    x_label = 'context KV pairs S' if PREFIX == 'mq4' else 'sequence length S'

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_vloss, ax_vacc = axes[0]
    ax_tloss, ax_sum = axes[1]

    finals = {k: {'acc': [], 'loss': [], 'status': []} for k in mixers}

    for key, (label, cmap) in mixers.items():
        for S in SEQS:
            Q = queries_for(S)
            d = parse(log_path(key, S))
            st = d['status']
            tag = f'{label} S={S}' + (f' Q={Q}' if PREFIX == 'mq4' else '')
            if st not in ('missing', 'empty'):
                ls = '-' if st == 'complete' else '--'
                suffix = '' if st == 'complete' else f' ({st})'
                color = cmap(shades[S])

                if len(d['val_steps']):
                    ax_vloss.plot(d['val_steps'], d['val_loss'], color=color, lw=2, ls=ls,
                                  marker='o', ms=3, label=tag + suffix)
                    ax_vacc.plot(d['val_steps'], d['val_acc'], color=color, lw=2, ls=ls,
                                 marker='o', ms=3, label=tag + suffix)
                    finals[key]['acc'].append(d['val_acc'][-1])
                    finals[key]['loss'].append(d['val_loss'][-1])
                else:
                    finals[key]['acc'].append(np.nan)
                    finals[key]['loss'].append(np.nan)

                if len(d['train_steps']):
                    ax_tloss.plot(d['train_steps'], d['train_loss'], color=color, lw=1.8,
                                  ls=ls, label=tag + suffix)
                elif len(d['iters']) >= 2:
                    ax_tloss.plot(d['iters'], d['iter_loss'], color=color, lw=1.0,
                                  alpha=0.5, ls=ls, label=tag + suffix + ' (iter)')
            else:
                finals[key]['acc'].append(np.nan)
                finals[key]['loss'].append(np.nan)
            finals[key]['status'].append(st)

    ax_vloss.axhline(no_recall, color='gray', ls='--', lw=1, alpha=0.7, label=f'no recall ln(V/2)')
    ax_vloss.axhline(0.0, color='gray', ls=':', lw=1, alpha=0.7, label='perfect recall')
    ax_vloss.set_xlabel('iteration')
    ax_vloss.set_ylabel('masked val CE (nats)')
    ax_vloss.set_title('Validation loss (answer-token CE)')
    ax_vloss.legend(fontsize=7, ncol=2)
    ax_vloss.grid(alpha=0.25)

    ax_vacc.axhline(1.0, color='gray', ls=':', lw=1)
    ax_vacc.set_xlabel('iteration')
    ax_vacc.set_ylabel('answer accuracy')
    ax_vacc.set_ylim(-0.03, 1.03)
    ax_vacc.set_title('Validation answer accuracy')
    ax_vacc.legend(fontsize=7, ncol=2)
    ax_vacc.grid(alpha=0.25)

    ax_tloss.set_xlabel('iteration')
    ax_tloss.set_ylabel('train loss (nats)')
    ax_tloss.set_title('Training loss (500-iter rolling avg)')
    ax_tloss.legend(fontsize=7, ncol=2)
    ax_tloss.grid(alpha=0.25)

    width = 0.18
    x = np.arange(len(SEQS))
    for i, (key, (label, cmap)) in enumerate(mixers.items()):
        off = (i - 0.5) * width
        acc = finals[key]['acc']
        ax_sum.bar(x + off, acc, width=width, color=cmap(0.72), alpha=0.9, label=f'{label} acc')
        for xi, a, st in zip(x, acc, finals[key]['status']):
            if not np.isnan(a):
                ax_sum.text(xi + off, a + 0.02, f'{a:.2f}', ha='center', fontsize=7)
            elif st == 'missing':
                ax_sum.text(xi + off, 0.02, 'NA', ha='center', fontsize=7, color='gray')

    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels([str(s) for s in SEQS])
    ax_sum.set_xlabel(x_label)
    ax_sum.set_ylabel('final answer accuracy')
    ax_sum.set_ylim(0, 1.12)
    ax_sum.set_title('Final validation accuracy')
    ax_sum.legend(fontsize=8)
    ax_sum.grid(alpha=0.25, axis='y')

    q_note = 'Q=S/4 (context pairs)' if PREFIX == 'mq4' else 'Q=1'
    model_note = {'lc3': 'd256 4L', 'lc4': 'd256 4L', 'mq4': 'd256 4L', 'lc5': 'd64 2L'}.get(PREFIX, '')
    train_note = {'lc3': '400k train', 'mq4': '400k train',
                  'lc4': '5.12M train', 'lc5': '5.12M train'}.get(PREFIX, '')
    cfg = ', '.join(x for x in (model_note, 'batch=256', train_note, f'vocab={VOCAB}') if x)
    fig.suptitle(
        f'MQAR sweep [{PREFIX}] ({q_note}, masked CE, {cfg}): GDN vs softmax\n'
        f'reds=GDN, blues=softmax; light→dark = smaller→larger S',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(LOGDIR, f'{PREFIX}_len_curves.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)

    print('\nsummary (final val loss / answer acc):')
    for key, (label, _) in mixers.items():
        for S, vl, va, st in zip(SEQS, finals[key]['loss'], finals[key]['acc'], finals[key]['status']):
            if not np.isnan(va):
                print(f'  {label:8s} S={S:<4d}  loss={vl:.4f}  acc={va:.4f}  ({st})')
            else:
                print(f'  {label:8s} S={S:<4d}  —  ({st})')


if __name__ == '__main__':
    main()
