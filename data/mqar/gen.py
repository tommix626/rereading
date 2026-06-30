r"""Generate Zoology-style MQAR bins with answer-token-only labels (-100 elsewhere).

Writes data/mqar_p{P}/ or data/mqar_s{S}_p{P}/ with:
  {train,val}_inputs.bin  (int32 token ids)
  {train,val}_labels.bin  (int32, -100 on non-answer positions)
  meta.pkl

Env vars (defaults match Zoology canonical where noted):
  VOCAB_SIZE=8192
  INPUT_SEQ_LEN=256       # block_size for training (= shifted sequence length)
  NUM_PAIRS=64            # num_kv_pairs
  NUM_QUERIES unused (always one query per KV pair)
  N_TRAIN=100000
  N_VAL=2000
  NUM_PASSES=1
  POWER_A=0.01
  RANDOM_NON_QUERIES=1
  USE_QUERY_SEP=1         # place a '?' token (id vocab_size-1) between context and queries
  SEED=0

Examples:
  NUM_PAIRS=8 INPUT_SEQ_LEN=64 VOCAB_SIZE=1024 N_TRAIN=50000 python data/mqar/gen.py
  NUM_PAIRS=64 python data/mqar/gen.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoology import IGNORE_INDEX, build_examples, default_sep_token_id, write_split_bins

VOCAB_SIZE = int(os.environ.get('VOCAB_SIZE', '8192'))
INPUT_SEQ_LEN = int(os.environ.get('INPUT_SEQ_LEN', os.environ.get('SEQ_LEN', '258')))
P = int(os.environ.get('NUM_PAIRS', '64'))
N_TRAIN = int(os.environ.get('N_TRAIN', '100000'))
N_VAL = int(os.environ.get('N_VAL', '2000'))
NUM_PASSES = int(os.environ.get('NUM_PASSES', '1'))
POWER_A = float(os.environ.get('POWER_A', '0.01'))
RANDOM_NON_QUERIES = os.environ.get('RANDOM_NON_QUERIES', '1') not in ('0', 'false', 'False')
USE_QUERY_SEP = os.environ.get('USE_QUERY_SEP', '1') not in ('0', 'false', 'False')
SEED = int(os.environ.get('SEED', '0'))

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, '..', f'mqar_s{INPUT_SEQ_LEN}_p{P}')


def main():
    kw = dict(
        vocab_size=VOCAB_SIZE,
        input_seq_len=INPUT_SEQ_LEN,
        num_kv_pairs=P,
        num_passes=NUM_PASSES,
        power_a=POWER_A,
        random_non_queries=RANDOM_NON_QUERIES,
        use_query_sep=USE_QUERY_SEP,
    )
    train_in, train_lab = build_examples(N_TRAIN, seed=SEED + 1, **kw)
    val_in, val_lab = build_examples(N_VAL, seed=SEED + 2, **kw)

    os.makedirs(OUTDIR, exist_ok=True)
    write_split_bins(OUTDIR, 'train', train_in, train_lab)
    write_split_bins(OUTDIR, 'val', val_in, val_lab)

    meta = {
        'vocab_size': VOCAB_SIZE,
        'block_size': INPUT_SEQ_LEN,
        'num_kv_pairs': P,
        'num_passes': NUM_PASSES,
        'power_a': POWER_A,
        'random_non_queries': RANDOM_NON_QUERIES,
        'use_query_sep': USE_QUERY_SEP,
        'sep_token_id': default_sep_token_id(VOCAB_SIZE) if USE_QUERY_SEP else None,
        'mqar_masked': True,
        'ignore_index': IGNORE_INDEX,
    }
    with open(os.path.join(OUTDIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    ctx = 2 * P * NUM_PASSES
    sep = 1 if USE_QUERY_SEP else 0
    space = (INPUT_SEQ_LEN - ctx - sep) // 2
    sep_note = f" sep_id={default_sep_token_id(VOCAB_SIZE)}" if USE_QUERY_SEP else ""
    print(f"[mqar_s{INPUT_SEQ_LEN}_p{P}] vocab={VOCAB_SIZE} ctx_tokens={ctx}{sep_note} "
          f"query_slots={space} supervised={P}/{INPUT_SEQ_LEN} "
          f"ignore_index={IGNORE_INDEX} -> {OUTDIR}")


if __name__ == '__main__':
    main()
