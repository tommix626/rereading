r"""Multi-Query Associative Recall (MQAR) -- the standard probe that separates
attention from fixed-state recurrent mixers (Zoology / Based / Mamba papers).

Each example is self-contained and has a FIXED length L = 2*N_PAIRS + 2*N_QUERIES:

    k1 v1 k2 v2 ... kN vN   q1 a1 q2 a2 ... qM aM
    \___ key->value bindings (context) ___/  \___ queries; ai = value bound to qi ___/

Tokens are laid out in one shared vocab with disjoint ranges:
    keys   in [0, NUM_KEYS)
    values in [NUM_KEYS, NUM_KEYS + NUM_VALS)

Under next-token prediction with loss over ALL positions, only the *answer*
positions (ai, which follow a query key) are predictable -- and they are
predictable ONLY by recalling the in-context binding. Keys/values/queries are
random, so they sit at the ~ln(vocab) baseline for every model. The val-loss
gap between models is therefore driven entirely by recall ability:
  - softmax attention: attend to where qi appeared, copy the next token -> ~0 loss
  - GDN: must have stored the binding in its fixed recurrent state -> degrades as
    the number of bindings grows.

The stream is a concatenation of fixed-length examples; train with
`block_size = L` and `block_align = True` so each random window is exactly one
self-contained example.
"""
import os
import numpy as np

# --- task size (tweak to make it harder/easier for the recurrent model) ---
NUM_KEYS = 256          # key vocab
NUM_VALS = 256          # value vocab
N_PAIRS = 64            # key->value bindings per example (state pressure for GDN)
N_QUERIES = 64          # queries per example (= recall signal fraction)
N_TRAIN_EXAMPLES = 40_000
N_VAL_EXAMPLES = 2_000
SEED = 7

VOCAB = NUM_KEYS + NUM_VALS                 # 512
L = 2 * N_PAIRS + 2 * N_QUERIES             # example length = block_size to use
here = os.path.dirname(__file__)


def build(n_examples, seed):
    r = np.random.default_rng(seed)
    out = np.empty((n_examples, L), dtype=np.uint16)
    for e in range(n_examples):
        keys = r.choice(NUM_KEYS, size=N_PAIRS, replace=False)
        vals = r.integers(0, NUM_VALS, size=N_PAIRS) + NUM_KEYS
        ctx = np.empty(2 * N_PAIRS, dtype=np.uint16)
        ctx[0::2] = keys
        ctx[1::2] = vals
        # query a random subset (with replacement) of the bound keys
        qidx = r.integers(0, N_PAIRS, size=N_QUERIES)
        qry = np.empty(2 * N_QUERIES, dtype=np.uint16)
        qry[0::2] = keys[qidx]
        qry[1::2] = vals[qidx]      # the answer to recall
        out[e, :2 * N_PAIRS] = ctx
        out[e, 2 * N_PAIRS:] = qry
    return out.reshape(-1)


def main():
    train = build(N_TRAIN_EXAMPLES, SEED + 1)
    val = build(N_VAL_EXAMPLES, SEED + 2)
    train.tofile(os.path.join(here, 'train.bin'))
    val.tofile(os.path.join(here, 'val.bin'))
    frac = N_QUERIES / L
    print(f"vocab={VOCAB} (keys {NUM_KEYS}, vals {NUM_VALS})  example_len L={L}")
    print(f"N_PAIRS={N_PAIRS} bindings, N_QUERIES={N_QUERIES} -> {frac:.0%} of positions are recall answers")
    print(f"wrote train.bin ({train.size} tok, {N_TRAIN_EXAMPLES} ex) "
          f"val.bin ({val.size} tok, {N_VAL_EXAMPLES} ex)")
    print(f"uniform per-token baseline ln(vocab) = {np.log(VOCAB):.3f} nats; "
          f"answer baseline ln(NUM_VALS) = {np.log(NUM_VALS):.3f} nats")


if __name__ == '__main__':
    main()
