"""Zoology-style Multi-Query Associative Recall (MQAR) data generation.

Answer-token-only labels: non-answer positions are -100. Inputs and labels are
already causally shifted (inputs = raw[:, :-1], labels = raw_labels[:, 1:]).

Layout (with query separator enabled, the default):
    k0 v0 ... k_{P-1} v_{P-1}  ?  _ q_k _ v_ans ...
    \_______ context ________/ SEP \___ query suffix ___/
"""
from __future__ import annotations

import numpy as np

IGNORE_INDEX = -100
GAP_PLACEHOLDER = 0  # unfilled suffix slots; replaced with random tokens if enabled


def default_sep_token_id(vocab_size: int) -> int:
    """Reserved '?' token — last vocab id, outside key/value ranges."""
    return vocab_size - 1


def _gap_probs(space: int, power_a: float) -> np.ndarray:
    g = np.arange(space, dtype=np.float64)
    p = power_a * (g + 1.0) ** (power_a - 1.0)
    return p / p.sum()


def build_examples(
    num_examples: int,
    *,
    vocab_size: int,
    input_seq_len: int,
    num_kv_pairs: int,
    num_passes: int = 1,
    power_a: float = 0.01,
    random_non_queries: bool = True,
    use_query_sep: bool = True,
    sep_token_id: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inputs, labels) int64 arrays of shape [num_examples, input_seq_len]."""
    assert input_seq_len % 2 == 0
    assert vocab_size > input_seq_len

    if sep_token_id is None:
        sep_token_id = default_sep_token_id(vocab_size)
    assert 0 <= sep_token_id < vocab_size

    key_lo, key_hi = 1, vocab_size // 2
    val_lo, val_hi = vocab_size // 2, vocab_size - (1 if use_query_sep else 0)
    assert sep_token_id < key_lo or sep_token_id >= key_hi, "sep token must not collide with keys/values"

    context_size = 2 * num_kv_pairs * num_passes
    sep_len = 1 if use_query_sep else 0
    raw_len = input_seq_len + 1

    space = (input_seq_len - context_size - sep_len) // 2
    assert space >= num_kv_pairs, (
        f"need space={space} >= num_kv_pairs={num_kv_pairs} "
        f"(input_seq_len={input_seq_len}, context_size={context_size}, sep={sep_len})"
    )

    query_base = context_size + sep_len
    gap_p = _gap_probs(space, power_a)
    rng = np.random.default_rng(seed)

    raw_examples = np.full((num_examples, raw_len), GAP_PLACEHOLDER, dtype=np.int64)
    raw_labels = np.full((num_examples, raw_len), IGNORE_INDEX, dtype=np.int64)

    for e in range(num_examples):
        keys = rng.choice(np.arange(key_lo, key_hi), size=num_kv_pairs, replace=False)
        vals = rng.choice(np.arange(val_lo, val_hi), size=num_kv_pairs, replace=False)
        ctx = np.empty(context_size, dtype=np.int64)
        for rep in range(num_passes):
            off = rep * 2 * num_kv_pairs
            ctx[off:off + 2 * num_kv_pairs:2] = keys
            ctx[off + 1:off + 2 * num_kv_pairs:2] = vals
        raw_examples[e, :context_size] = ctx
        if use_query_sep:
            raw_examples[e, context_size] = sep_token_id

        gaps = rng.choice(space, size=num_kv_pairs, replace=False, p=gap_p)
        for i in range(num_kv_pairs):
            qpos = query_base + 2 * int(gaps[i])
            raw_examples[e, qpos] = keys[i]
            raw_labels[e, qpos + 1] = vals[i]

    inputs = raw_examples[:, :-1].copy()
    labels = raw_labels[:, 1:].copy()

    if random_non_queries:
        # Fill gap placeholders only; never overwrite the '?' separator.
        filler = rng.integers(0, vocab_size - 1, size=inputs.shape, dtype=np.int64)
        gap_mask = inputs == GAP_PLACEHOLDER
        inputs[gap_mask] = filler[gap_mask]

    verify_invariants(
        inputs, labels,
        vocab_size=vocab_size,
        num_kv_pairs=num_kv_pairs,
        context_size=context_size,
        sep_token_id=sep_token_id if use_query_sep else None,
        val_hi=val_hi,
    )
    return inputs, labels


def build_pool_examples(
    num_examples: int,
    *,
    vocab_size: int,
    input_seq_len: int,
    num_queries: int = 1,
    random_non_queries: bool = True,
    use_query_sep: bool = True,
    sep_token_id: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Long-context MQAR: K-pair pool, optional '?', Q queries at sequence tail.

    K = (input_seq_len - sep - 2*Q) // 2 bindings fill the prefix; one random pool
  entry is queried per query slot. Matches the lc2 length-sweep layout (Q=1 default).
    """
    assert input_seq_len % 2 == 0
    assert vocab_size > input_seq_len
    Q = num_queries

    if sep_token_id is None:
        sep_token_id = default_sep_token_id(vocab_size)
    sep_len = 1 if use_query_sep else 0
    K = (input_seq_len - sep_len - 2 * Q) // 2
    assert K >= Q, f"pool K={K} must be >= num_queries={Q} (seq_len={input_seq_len})"

    key_lo, key_hi = 1, vocab_size // 2
    val_lo, val_hi = vocab_size // 2, vocab_size - (1 if use_query_sep else 0)
    assert K <= key_hi - key_lo

    context_size = 2 * K
    raw_len = input_seq_len + 1
    query_start = input_seq_len - 2 * Q  # shifted index of first query key

    rng = np.random.default_rng(seed)
    raw_examples = np.full((num_examples, raw_len), GAP_PLACEHOLDER, dtype=np.int64)
    raw_labels = np.full((num_examples, raw_len), IGNORE_INDEX, dtype=np.int64)

    for e in range(num_examples):
        keys = rng.choice(np.arange(key_lo, key_hi), size=K, replace=False)
        vals = rng.choice(np.arange(val_lo, val_hi), size=K, replace=False)
        raw_examples[e, 0:context_size:2] = keys
        raw_examples[e, 1:context_size:2] = vals
        if use_query_sep:
            raw_examples[e, context_size] = sep_token_id

        qidx = rng.integers(0, K, size=Q)
        for t in range(Q):
            qpos = query_start + 2 * t
            raw_examples[e, qpos] = keys[qidx[t]]
            raw_labels[e, qpos + 1] = vals[qidx[t]]

    inputs = raw_examples[:, :-1].copy()
    labels = raw_labels[:, 1:].copy()

    if random_non_queries:
        filler = rng.integers(0, vocab_size - 1, size=inputs.shape, dtype=np.int64)
        gap_mask = inputs == GAP_PLACEHOLDER
        inputs[gap_mask] = filler[gap_mask]

    verify_pool_invariants(
        inputs, labels,
        vocab_size=vocab_size,
        num_queries=Q,
        context_size=context_size,
        sep_token_id=sep_token_id if use_query_sep else None,
        val_hi=val_hi,
    )
    return inputs, labels


def verify_pool_invariants(
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    vocab_size: int,
    num_queries: int,
    context_size: int,
    sep_token_id: int | None = None,
    val_hi: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    val_lo = vocab_size // 2
    if val_hi is None:
        val_hi = vocab_size - (1 if sep_token_id is not None else 0)

    supervised_per = (labels != ignore_index).sum(axis=1)
    assert np.all(supervised_per == num_queries)
    sup = labels[labels != ignore_index]
    assert sup.min() >= val_lo and sup.max() < val_hi
    key_hi = vocab_size // 2
    for row_in, row_lab in zip(inputs, labels):
        mask = row_lab != ignore_index
        q_in = row_in[mask]
        assert q_in.min() >= 1 and q_in.max() < key_hi
    if sep_token_id is not None:
        assert np.all(inputs[:, context_size] == sep_token_id)
    assert (labels == ignore_index).mean() > 0.5


def verify_invariants(
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    vocab_size: int,
    num_kv_pairs: int,
    context_size: int | None = None,
    sep_token_id: int | None = None,
    val_hi: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Assert Zoology MQAR label layout."""
    if context_size is None:
        # assumes num_passes=1 when context_size omitted
        context_size = 2 * num_kv_pairs
    val_lo = vocab_size // 2
    if val_hi is None:
        val_hi = vocab_size - (1 if sep_token_id is not None else 0)

    supervised_per = (labels != ignore_index).sum(axis=1)
    assert np.all(supervised_per == num_kv_pairs), (
        f"expected {num_kv_pairs} supervised positions, got {supervised_per}"
    )
    sup = labels[labels != ignore_index]
    assert sup.min() >= val_lo and sup.max() < val_hi
    key_hi = vocab_size // 2
    for row_in, row_lab in zip(inputs, labels):
        mask = row_lab != ignore_index
        q_in = row_in[mask]
        assert q_in.min() >= 1 and q_in.max() < key_hi
    if sep_token_id is not None:
        assert np.all(inputs[:, context_size] == sep_token_id), "missing '?' at context/query boundary"
    frac_ignored = (labels == ignore_index).mean()
    assert frac_ignored > 0.5, f"expected mostly -100 labels, got ignore frac {frac_ignored:.3f}"


def write_split_bins(outdir: str, split: str, inputs: np.ndarray, labels: np.ndarray) -> None:
    import os

    os.makedirs(outdir, exist_ok=True)
    inputs.astype(np.int32).tofile(os.path.join(outdir, f'{split}_inputs.bin'))
    labels.astype(np.int32).tofile(os.path.join(outdir, f'{split}_labels.bin'))
