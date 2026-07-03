r"""Long-context MQAR: pool of K bindings + Q queries at tail (masked labels).

Zoology-style answer-token-only CE. Writes {train,val}_inputs.bin and labels.

Two sizing modes:
  1) SEQ_LEN + NUM_QUERIES — total sequence length (lc length-sweep, Q=1).
  2) POOL_PAIRS — user's S: number of KV pairs in context before '?' and queries;
     Q = POOL_PAIRS // 4, seq_len = 2*POOL_PAIRS + 1 + 2*Q (padded to even).

Env vars:
  SEQ_LEN=256 | POOL_PAIRS=16
  NUM_QUERIES=1  (ignored when POOL_PAIRS is set; Q = POOL_PAIRS // 4)
  VOCAB_SIZE=2048
  N_TRAIN=40000
  N_VAL=2000
  DATASET_TAG=''  — optional suffix on output dir name

Example:
  SEQ_LEN=64 NUM_QUERIES=1 python data/mqar/gen_fixed.py
  POOL_PAIRS=16 python data/mqar/gen_fixed.py   # S=16 pairs, Q=4, seq_len=42
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoology import IGNORE_INDEX, build_pool_examples, default_sep_token_id, write_split_bins

VOCAB_SIZE = int(os.environ.get('VOCAB_SIZE', '2048'))
N_TRAIN = int(os.environ.get('N_TRAIN', '40000'))
N_VAL = int(os.environ.get('N_VAL', '2000'))
USE_QUERY_SEP = os.environ.get('USE_QUERY_SEP', '1') not in ('0', 'false', 'False')
RANDOM_NON_QUERIES = os.environ.get('RANDOM_NON_QUERIES', '1') not in ('0', 'false', 'False')
SEED = int(os.environ.get('SEED', '11'))
_DATASET_TAG = os.environ.get('DATASET_TAG', '')
# Generate/write in chunks so peak RAM stays bounded for very large N (e.g. lc5).
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '2000000'))

HERE = os.path.dirname(os.path.abspath(__file__))


def seq_len_from_pool_pairs(pool_pairs: int, num_queries: int, *, use_sep: bool = True) -> int:
    """Total tokens: 2*pool_pairs + sep + 2*Q; pad by 1 if odd (zoology requires even)."""
    sep_len = 1 if use_sep else 0
    seq_len = 2 * pool_pairs + sep_len + 2 * num_queries
    if seq_len % 2:
        seq_len += 1
    return seq_len


def resolve_sizing():
    """Return (seq_len, pool_pairs, num_queries, outdir_basename)."""
    if os.environ.get('POOL_PAIRS'):
        pool_pairs = int(os.environ['POOL_PAIRS'])
        num_queries = pool_pairs // 4
        assert num_queries >= 1, f"POOL_PAIRS={pool_pairs} too small for Q=POOL_PAIRS//4"
        seq_len = seq_len_from_pool_pairs(pool_pairs, num_queries, use_sep=USE_QUERY_SEP)
        name = f'mqar_ctx{pool_pairs}_q{num_queries}{_DATASET_TAG}'
        return seq_len, pool_pairs, num_queries, name
    seq_len = int(os.environ.get('SEQ_LEN', '256'))
    num_queries = int(os.environ.get('NUM_QUERIES', '1'))
    sep_len = 1 if USE_QUERY_SEP else 0
    pool_pairs = (seq_len - sep_len - 2 * num_queries) // 2
    name = f'mqar_s{seq_len}_q{num_queries}{_DATASET_TAG}'
    return seq_len, pool_pairs, num_queries, name


def write_split_chunked(outdir, split, n_total, base_seed, kw):
    """Generate a split in CHUNK_SIZE batches (distinct seed per chunk) and stream
    int32 bins to disk, so peak RAM is bounded by one chunk regardless of n_total."""
    os.makedirs(outdir, exist_ok=True)
    x_path = os.path.join(outdir, f'{split}_inputs.bin')
    y_path = os.path.join(outdir, f'{split}_labels.bin')
    written = 0
    ci = 0
    with open(x_path, 'wb') as fx, open(y_path, 'wb') as fy:
        while written < n_total:
            n = min(CHUNK_SIZE, n_total - written)
            xi, yi = build_pool_examples(n, seed=base_seed + ci, **kw)
            xi.astype(np.int32).tofile(fx)
            yi.astype(np.int32).tofile(fy)
            written += n
            ci += 1
            if n_total > CHUNK_SIZE:
                print(f"  [{split}] {written:,}/{n_total:,} ({ci} chunks)", flush=True)
    return written


def main():
    seq_len, pool_pairs, num_queries, basename = resolve_sizing()
    outdir = os.path.join(HERE, '..', basename)

    kw = dict(
        vocab_size=VOCAB_SIZE,
        input_seq_len=seq_len,
        num_queries=num_queries,
        random_non_queries=RANDOM_NON_QUERIES,
        use_query_sep=USE_QUERY_SEP,
    )
    os.makedirs(outdir, exist_ok=True)
    # base seeds offset far apart so train chunk seeds never collide with val's.
    write_split_chunked(outdir, 'train', N_TRAIN, SEED + 10000, kw)
    write_split_chunked(outdir, 'val', N_VAL, SEED + 2, kw)

    meta = {
        'vocab_size': VOCAB_SIZE,
        'block_size': seq_len,
        'pool_pairs': pool_pairs,
        'num_queries': num_queries,
        'use_query_sep': USE_QUERY_SEP,
        'sep_token_id': default_sep_token_id(VOCAB_SIZE) if USE_QUERY_SEP else None,
        'mqar_masked': True,
        'ignore_index': IGNORE_INDEX,
    }
    with open(os.path.join(outdir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    sep_note = f" sep_id={default_sep_token_id(VOCAB_SIZE)}" if USE_QUERY_SEP else ""
    print(
        f"[{basename}] vocab={VOCAB_SIZE} ctx_pairs={pool_pairs} queries={num_queries} "
        f"seq_len={seq_len}{sep_note} supervised={num_queries}/{seq_len} -> {outdir}"
    )


if __name__ == '__main__':
    main()
