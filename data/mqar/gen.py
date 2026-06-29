r"""Parameterized MQAR generator for the difficulty sweep.

Writes data/mqar_p{P}/{train,val}.bin for P key->value bindings and Q queries
(default Q=P). Example length L = 2*P + 2*Q; train with block_size=L,
block_align=True. Same vocab layout as data/mqar/prepare.py:
    keys   in [0, NUM_KEYS), values in [NUM_KEYS, NUM_KEYS+NUM_VALS).

Knobs via env vars: NUM_PAIRS, NUM_QUERIES, N_TRAIN, N_VAL.
  NUM_PAIRS=1  python data/mqar/gen.py   # easiest: one binding, one query
"""
import os
import numpy as np

NUM_KEYS = 256
NUM_VALS = 256
P = int(os.environ.get('NUM_PAIRS', '8'))
Q = int(os.environ.get('NUM_QUERIES', str(P)))
N_TRAIN = int(os.environ.get('N_TRAIN', '40000'))
N_VAL = int(os.environ.get('N_VAL', '2000'))
SEED = 7

assert P <= NUM_KEYS, "need P distinct keys"
VOCAB = NUM_KEYS + NUM_VALS
L = 2 * P + 2 * Q
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, '..', f'mqar_p{P}')


def build(n_examples, seed):
    r = np.random.default_rng(seed)
    out = np.empty((n_examples, L), dtype=np.uint16)
    for e in range(n_examples):
        keys = r.choice(NUM_KEYS, size=P, replace=False)
        vals = r.integers(0, NUM_VALS, size=P) + NUM_KEYS
        out[e, 0:2 * P:2] = keys
        out[e, 1:2 * P:2] = vals
        qidx = r.integers(0, P, size=Q)
        out[e, 2 * P + 0::2] = keys[qidx]
        out[e, 2 * P + 1::2] = vals[qidx]
    return out.reshape(-1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    build(N_TRAIN, SEED + 1).tofile(os.path.join(OUTDIR, 'train.bin'))
    build(N_VAL, SEED + 2).tofile(os.path.join(OUTDIR, 'val.bin'))
    print(f"[mqar_p{P}] pairs={P} queries={Q} L={L} block_size={L} "
          f"vocab={VOCAB}  answers={Q/L:.0%} of positions  -> {OUTDIR}")


if __name__ == '__main__':
    main()
