"""Generate a tiny *learnable* synthetic dataset for smoke-testing training.

Writes nanoGPT-native `train.bin` / `val.bin` (flat uint16 token streams) that
the data loader picks up when a config leaves `megatron_train_paths` empty and
sets `dataset='synthetic'`.

The stream is drawn from a fixed random Markov chain over a small vocab, so the
next token is genuinely predictable from the current one. A small model should
drive the loss well below the uniform baseline (ln(vocab)) within a few hundred
steps -- exactly what we want to *see* working.
"""
import os
import numpy as np

VOCAB = 512          # small vocab -> small lm_head, fast
N_TRAIN = 4_000_000  # tokens (~8 MB uint16)
N_VAL = 400_000
SEED = 1234

here = os.path.dirname(__file__)
rng = np.random.default_rng(SEED)

# Fixed Markov transition: each token has a sharply peaked distribution over the
# next token, so there's real structure to learn (but not trivially identity).
# Build a sparse-ish transition by giving each row a few high-probability targets.
logits = rng.normal(0, 1, size=(VOCAB, VOCAB)).astype(np.float64)
# sharpen so each state strongly prefers a handful of successors
logits *= 4.0
trans = np.exp(logits - logits.max(axis=1, keepdims=True))
trans /= trans.sum(axis=1, keepdims=True)
# precompute cumulative distributions for fast sampling
cum = np.cumsum(trans, axis=1)


def gen(n, seed):
    r = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint16)
    s = 0
    draws = r.random(n)
    for t in range(n):
        out[t] = s
        s = int(np.searchsorted(cum[s], draws[t]))
    return out


def main():
    train = gen(N_TRAIN, SEED + 1)
    val = gen(N_VAL, SEED + 2)
    train.tofile(os.path.join(here, 'train.bin'))
    val.tofile(os.path.join(here, 'val.bin'))
    # entropy floor for reference: average -sum p log p over rows
    ent = -(trans * np.log(np.clip(trans, 1e-12, None))).sum(axis=1).mean()
    print(f"wrote train.bin ({train.size} tok) val.bin ({val.size} tok) vocab={VOCAB}")
    print(f"approx per-token entropy floor: {ent:.3f} nats  (uniform baseline ln(V)={np.log(VOCAB):.3f})")


if __name__ == '__main__':
    main()
