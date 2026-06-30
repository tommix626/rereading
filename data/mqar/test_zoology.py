"""Invariant tests for Zoology-style MQAR data generation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoology import IGNORE_INDEX, build_examples, build_pool_examples, verify_invariants, verify_pool_invariants


def test_sanity_config():
    inputs, labels = build_examples(
        32,
        vocab_size=1024,
        input_seq_len=64,
        num_kv_pairs=8,
        seed=0,
    )
    assert inputs.shape == (32, 64)
    assert labels.shape == (32, 64)
    verify_invariants(inputs, labels, vocab_size=1024, num_kv_pairs=8)


def test_canonical_config():
    inputs, labels = build_examples(
        16,
        vocab_size=8192,
        input_seq_len=258,  # +2 vs Zoology paper: one '?' sep between context and queries
        num_kv_pairs=64,
        seed=1,
    )
    verify_invariants(inputs, labels, vocab_size=8192, num_kv_pairs=64,
                      context_size=128, sep_token_id=8191, val_hi=8191)
    assert (labels != IGNORE_INDEX).sum() == 16 * 64


def test_difficulty_sweep_pairs():
    for p in (1, 8, 64):
        inputs, labels = build_examples(
            8,
            vocab_size=8192,
            input_seq_len=258 if p == 64 else 256,
            num_kv_pairs=p,
            seed=p,
        )
        verify_invariants(
            inputs, labels, vocab_size=8192, num_kv_pairs=p,
            context_size=2 * p,
            sep_token_id=8191 if p else None,
            val_hi=8191,
        )


def test_no_full_ntp_labels():
    inputs, labels = build_examples(4, vocab_size=512, input_seq_len=64, num_kv_pairs=8, seed=3)
    shifted_wrong = np.concatenate([inputs[:, 1:], inputs[:, :1]], axis=1)
    assert not np.array_equal(labels[labels != IGNORE_INDEX],
                              shifted_wrong[labels != IGNORE_INDEX])


def test_query_separator():
    vocab = 1024
    sep = vocab - 1
    inputs, labels = build_examples(
        16, vocab_size=vocab, input_seq_len=64, num_kv_pairs=8, seed=5,
    )
    context_size = 16
    assert np.all(inputs[:, context_size] == sep)
    verify_invariants(
        inputs, labels, vocab_size=vocab, num_kv_pairs=8,
        context_size=context_size, sep_token_id=sep, val_hi=vocab - 1,
    )


def test_pool_q1():
    for S in (16, 64, 256, 1024):
        inputs, labels = build_pool_examples(
            8, vocab_size=2048, input_seq_len=S, num_queries=1, seed=S,
        )
        K = (S - 1 - 2) // 2
        verify_pool_invariants(
            inputs, labels, vocab_size=2048, num_queries=1,
            context_size=2 * K, sep_token_id=2047, val_hi=2047,
        )


def test_bin_roundtrip():
    import os
    import tempfile

    from zoology import write_split_bins

    inputs, labels = build_examples(4, vocab_size=512, input_seq_len=32, num_kv_pairs=4, seed=9)
    outdir = tempfile.mkdtemp()
    write_split_bins(outdir, 'train', inputs, labels)
    x = np.fromfile(os.path.join(outdir, 'train_inputs.bin'), dtype=np.int32).reshape(4, 32)
    y = np.fromfile(os.path.join(outdir, 'train_labels.bin'), dtype=np.int32).reshape(4, 32)
    assert np.array_equal(x, inputs.astype(np.int32))
    assert np.array_equal(y, labels.astype(np.int32))


if __name__ == '__main__':
    test_sanity_config()
    test_canonical_config()
    test_difficulty_sweep_pairs()
    test_no_full_ntp_labels()
    test_query_separator()
    test_pool_q1()
    test_bin_roundtrip()
    print('all zoology MQAR tests passed')
