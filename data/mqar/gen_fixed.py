r"""Fixed-length MQAR for long-context retrieval (standard Zoology-style setup).

The context is a POOL of K distinct key->value pairs that fills the sequence;
we then query a random subset of Q of them. The queried key's binding sits at a
RANDOM position in the pool, so retrieval distance varies and there is genuine
interference from the other K-1 bindings:

    k1 v1 k2 v2 ............ kK vK   q1 a1 ... qQ aQ
    \_____ pool of K random bindings ____/  \_ Q random queries _/   (total = S)

  K = (S - 2Q)/2 pool pairs   (grows with S -> more distractors + longer distance)
  Q = number of retrievals (the "retrieval" knob: 1, 8, 64)

Longer S => bigger pool + farther bindings => harder for a fixed-state recurrent
model, easy for attention (direct access). Keys are distinct within an example
(unambiguous), values random. Answers are predicted at local positions
[S-2Q + 2t for t in 0..Q-1] -> matches train.py ar_num_queries scoring.

Writes data/mqar_s{S}_q{Q}/{train,val}.bin. Env: SEQ_LEN, NUM_QUERIES.
"""
import os
import numpy as np

NUM_KEYS = 1024          # big enough that the pool keys can be distinct
NUM_VALS = 1024
S = int(os.environ.get('SEQ_LEN', '256'))
Q = int(os.environ.get('NUM_QUERIES', '8'))
N_TRAIN = int(os.environ.get('N_TRAIN', '40000'))
N_VAL = int(os.environ.get('N_VAL', '2000'))
SEED = 11

VOCAB = NUM_KEYS + NUM_VALS
assert (S - 2 * Q) % 2 == 0 and S - 2 * Q > 0, "need S even and S > 2Q"
K = (S - 2 * Q) // 2     # pool size
assert Q <= K, f"need Q<=K; S={S},Q={Q} gives pool K={K}"
assert K <= NUM_KEYS, f"pool K={K} exceeds NUM_KEYS={NUM_KEYS}; raise NUM_KEYS"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, '..', f'mqar_s{S}_q{Q}')


def build(n, seed):
    r = np.random.default_rng(seed)
    out = np.empty((n, S), dtype=np.uint16)
    for e in range(n):
        keys = r.choice(NUM_KEYS, size=K, replace=False)          # distinct pool keys
        vals = r.integers(0, NUM_VALS, size=K) + NUM_KEYS
        out[e, 0:2 * K:2] = keys
        out[e, 1:2 * K:2] = vals
        qidx = r.integers(0, K, size=Q)                           # query random pool entries
        out[e, S - 2 * Q + 0::2] = keys[qidx]
        out[e, S - 2 * Q + 1::2] = vals[qidx]
    return out.reshape(-1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    build(N_TRAIN, SEED + 1).tofile(os.path.join(OUTDIR, 'train.bin'))
    build(N_VAL, SEED + 2).tofile(os.path.join(OUTDIR, 'val.bin'))
    print(f"[mqar_s{S}_q{Q}] S={S} pool_pairs={K} queries={Q} vocab={VOCAB} "
          f"(avg query distance ~{K + Q}) -> {OUTDIR}")


if __name__ == '__main__':
    main()
