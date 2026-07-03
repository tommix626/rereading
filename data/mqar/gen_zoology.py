r"""mq5: Zoology-faithful multi-query associative recall (Algorithm 1 of the paper).

Layout matches HazyResearch/zoology zoology/data/multiquery_ar.py:
    k0 v0 k1 v1 ... k_{D-1} v_{D-1}   [ queries (2nd occurrence of each key) at
                                        power-law-distributed gaps in [2D, N] ]
  - D = num_kv_pairs, ALL D keys queried (second occurrence), value = next-token label.
  - NO separator token.
  - random_non_queries=False by default -> non-query positions are 0 (the paper's
    clean version); set RANDOM_NON_QUERIES=1 for Zoology's default random distractors.
  - keys in [1, V/2), values in [V/2, V). Large vocab (default 8192).

Env:
  KV_PAIRS=D             number of key-value pairs (required)
  SEQ_LEN=N             input sequence length (default 6*D; must be even, >= 4*D)
  VOCAB_SIZE=8192
  RANDOM_NON_QUERIES=0  (0 -> clean 0-pad; 1 -> random distractor tokens)
  POWER_A=0.01          power-law gap parameter (alpha)
  N_TRAIN, N_VAL, SEED, CHUNK_SIZE, DATASET_TAG

Writes data/mqar_zoo_d{D}_n{N}{tag}/ with {train,val}_{inputs,labels}.bin + meta.pkl.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoology import IGNORE_INDEX, build_examples

VOCAB_SIZE = int(os.environ.get('VOCAB_SIZE', '8192'))
N_TRAIN = int(os.environ.get('N_TRAIN', '100000'))
N_VAL = int(os.environ.get('N_VAL', '3000'))
POWER_A = float(os.environ.get('POWER_A', '0.01'))
RANDOM_NON_QUERIES = os.environ.get('RANDOM_NON_QUERIES', '0') not in ('0', 'false', 'False')
SEED = int(os.environ.get('SEED', '11'))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '2000000'))
_TAG = os.environ.get('DATASET_TAG', '')

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_sizing():
    D = int(os.environ['KV_PAIRS'])
    N = int(os.environ.get('SEQ_LEN', str(6 * D)))
    if N % 2:
        N += 1
    assert N >= 4 * D, f"SEQ_LEN={N} must be >= 4*D={4*D} (context 2D + query region >= 2D)"
    name = f'mqar_zoo_d{D}_n{N}{_TAG}'
    return D, N, name


def write_split_chunked(outdir, split, n_total, base_seed, kw):
    os.makedirs(outdir, exist_ok=True)
    xp = os.path.join(outdir, f'{split}_inputs.bin')
    yp = os.path.join(outdir, f'{split}_labels.bin')
    written = ci = 0
    with open(xp, 'wb') as fx, open(yp, 'wb') as fy:
        while written < n_total:
            n = min(CHUNK_SIZE, n_total - written)
            xi, yi = build_examples(n, seed=base_seed + ci, **kw)
            xi.astype(np.int32).tofile(fx)
            yi.astype(np.int32).tofile(fy)
            written += n
            ci += 1
            if n_total > CHUNK_SIZE:
                print(f"  [{split}] {written:,}/{n_total:,} ({ci} chunks)", flush=True)
    return written


def main():
    D, N, basename = resolve_sizing()
    outdir = os.path.join(HERE, '..', basename)
    kw = dict(
        vocab_size=VOCAB_SIZE,
        input_seq_len=N,
        num_kv_pairs=D,
        power_a=POWER_A,
        random_non_queries=RANDOM_NON_QUERIES,
        use_query_sep=False,          # Zoology has no separator
    )
    os.makedirs(outdir, exist_ok=True)
    write_split_chunked(outdir, 'train', N_TRAIN, SEED + 10000, kw)
    write_split_chunked(outdir, 'val', N_VAL, SEED + 2, kw)

    meta = {
        'vocab_size': VOCAB_SIZE,
        'block_size': N,
        'num_kv_pairs': D,
        'num_queries': D,             # all D pairs are queried
        'power_a': POWER_A,
        'random_non_queries': RANDOM_NON_QUERIES,
        'use_query_sep': False,
        'mqar_masked': True,
        'ignore_index': IGNORE_INDEX,
    }
    with open(os.path.join(outdir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f"[{basename}] vocab={VOCAB_SIZE} D={D} seq_len={N} random_non_queries={RANDOM_NON_QUERIES} "
          f"supervised={D}/{N} -> {outdir}", flush=True)


if __name__ == '__main__':
    main()
