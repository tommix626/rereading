r"""fnl2: compact, K/Q-parametrized MQAR (final suite).

Each example (length 2K + 2Q):
    context (2K):   k0 v0 k1 v1 ... k_{K-1} v_{K-1}     (K key-value pairs, packed)
    queries (2Q):   q0  r q1  r ... q_{Q-1}  r          (Q queries; r = random filler)
  - q_j is one of the K keys; the label at the q_j POSITION is that key's value
    (no next-token shift; matches our mqar_masked loss / masked_answer_accuracy).
  - all other positions (context, fillers) are labeled -100 (ignored).
  - keys in [1, V/2), values in [V/2, V).
  - RANDOM_NON_QUERIES=1 (default): filler slots hold RANDOM distractor tokens. This is
    the fnl2 FIX: clean 0-padding (RANDOM_NON_QUERIES=0) is pathological for softmax --
    it plateaus at 1/K (value-copying without matching), and RoPE does NOT rescue the
    compact layout. Random fillers force content-based retrieval -> softmax groks to ~1.0.
    (See reports/mq5_ape_cleanpad_plateau.md.)
  - queries: Q distinct pairs (without replacement) if Q<=K, else with replacement.

Writes data/mqar_{NAME_PREFIX}_k{K}_q{Q}/ with {train,val}_{inputs,labels}.bin + meta.pkl.

Env: K (required), Q (default K), VOCAB_SIZE=8192, N_TRAIN=400000, N_VAL=3000, SEED=1234,
     CHUNK=20000, RANDOM_NON_QUERIES=1, NAME_PREFIX=fnl2
Usage: K=16 python data/mqar/gen_fnl2.py
"""
import os, pickle
import numpy as np

VOCAB = int(os.environ.get("VOCAB_SIZE", "8192"))
N_TRAIN = int(os.environ.get("N_TRAIN", "400000"))
N_VAL = int(os.environ.get("N_VAL", "3000"))
SEED = int(os.environ.get("SEED", "1234"))
CHUNK = int(os.environ.get("CHUNK", "20000"))
# fnl2 fix: fill the query-region filler slots with RANDOM distractor tokens instead of 0.
# Clean 0-padding is pathological for softmax (1/K value-copying plateau); random fillers
# force content-based retrieval (proven in reports/mq5_ape_cleanpad_plateau.md).
RANDOM_NON_QUERIES = os.environ.get("RANDOM_NON_QUERIES", "1") not in ("0", "false", "False")
NAME_PREFIX = os.environ.get("NAME_PREFIX", "fnl2")
IGNORE = -100
HERE = os.path.dirname(os.path.abspath(__file__))


def gen_chunk(n, K, Q, rng):
    key_pool = np.arange(1, VOCAB // 2)
    val_pool = np.arange(VOCAB // 2, VOCAB)
    # K distinct keys/values per row (argpartition = fast unordered top-K of random keys)
    ki = np.argpartition(rng.random((n, key_pool.size)), K, axis=1)[:, :K]
    vi = np.argpartition(rng.random((n, val_pool.size)), K, axis=1)[:, :K]
    keys, vals = key_pool[ki], val_pool[vi]
    ctx = np.zeros((n, 2 * K), dtype=np.int64)
    ctx[:, 0::2] = keys
    ctx[:, 1::2] = vals
    # which pairs are queried
    if Q <= K:
        qsel = np.argpartition(rng.random((n, K)), Q - 1 if Q < K else K - 1, axis=1)[:, :Q]
    else:
        qsel = rng.integers(0, K, size=(n, Q))
    qkeys = np.take_along_axis(keys, qsel, axis=1)
    qvals = np.take_along_axis(vals, qsel, axis=1)
    qreg = np.zeros((n, 2 * Q), dtype=np.int64)
    qreg[:, 0::2] = qkeys            # query keys at even offsets
    if RANDOM_NON_QUERIES:
        qreg[:, 1::2] = rng.integers(1, VOCAB, size=(n, Q))  # random distractor fillers (fnl2 fix)
    inp = np.concatenate([ctx, qreg], axis=1)          # (n, 2K+2Q)
    lab = np.full((n, 2 * K + 2 * Q), IGNORE, dtype=np.int64)
    lab[:, 2 * K::2] = qvals         # value predicted AT the query position
    return inp, lab


def write_split(outdir, split, n, K, Q, seed):
    rng = np.random.default_rng(seed)
    with open(os.path.join(outdir, f"{split}_inputs.bin"), "wb") as fx, \
         open(os.path.join(outdir, f"{split}_labels.bin"), "wb") as fy:
        written = 0
        while written < n:
            m = min(CHUNK, n - written)
            xi, yi = gen_chunk(m, K, Q, rng)
            xi.astype(np.int32).tofile(fx)
            yi.astype(np.int32).tofile(fy)
            written += m
    return written


def main():
    K = int(os.environ["K"]); Q = int(os.environ.get("Q", str(K)))
    seqlen = 2 * K + 2 * Q
    assert VOCAB // 2 - 1 >= K, f"vocab {VOCAB} too small for K={K} distinct keys"
    name = f"mqar_{NAME_PREFIX}_k{K}_q{Q}"
    outdir = os.path.join(HERE, "..", name); os.makedirs(outdir, exist_ok=True)
    write_split(outdir, "train", N_TRAIN, K, Q, SEED)
    write_split(outdir, "val", N_VAL, K, Q, SEED + 777)
    meta = {
        "vocab_size": VOCAB, "block_size": seqlen,
        "num_kv_pairs": K, "num_queries": Q,
        "mqar_masked": True, "ignore_index": IGNORE,
        "format": "fnl1 compact: [k v]*K then [q 0]*Q; label=value at query position",
    }
    with open(os.path.join(outdir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[{name}] K={K} Q={Q} seqlen={seqlen} vocab={VOCAB} train={N_TRAIN} val={N_VAL} -> {outdir}",
          flush=True)


if __name__ == "__main__":
    main()
