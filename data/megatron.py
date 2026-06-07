"""Minimal Megatron .bin/.idx loader for nanoGPT-lead.

Reads the .idx header to discover the token dtype, then mmaps the .bin file
as a flat token stream. We drop document-boundary respect — samples can span
documents, matching nanoGPT's native get_batch behavior on np.memmap.

Format reference: ~/dynFLA/lm-engine/lm_engine/data/megatron/indexed_dataset.py
"""

from __future__ import annotations

import struct

import numpy as np


_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_FROM_CODE = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def _read_dtype(idx_path: str) -> type[np.number]:
    with open(idx_path, "rb") as f:
        header = f.read(9)
        assert header == _INDEX_HEADER, f"bad header in {idx_path}: {header!r}"
        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1, f"bad version {version} in {idx_path}"
        code = struct.unpack("<B", f.read(1))[0]
    return _DTYPE_FROM_CODE[code]


def open_megatron(path_prefix: str) -> np.memmap:
    """Open a Megatron .bin/.idx pair as a flat token-stream memmap.

    Args:
        path_prefix: path without extension; expects <prefix>.bin and <prefix>.idx alongside.

    Returns:
        np.memmap over the .bin contents at the dtype specified in the .idx header.
    """
    dtype = _read_dtype(path_prefix + ".idx")
    return np.memmap(path_prefix + ".bin", mode="r", dtype=dtype)


def normalize_weights(weights):
    """Normalize a list of weights to sum to 1.0 (numpy array)."""
    w = np.asarray(weights, dtype=np.float64)
    return w / w.sum()
