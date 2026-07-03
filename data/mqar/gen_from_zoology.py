r"""mq6: export MQAR data by calling the ACTUAL zoology repo generator.

Unlike gen_zoology.py (our reimplementation via build_examples), this imports
`zoology.data.multiquery_ar.multiquery_ar` from the cloned HazyResearch/zoology
repo, so the data is byte-faithful to Zoology Figure 2. Dumps to our .bin format
(mqar_masked): {train,val}_{inputs,labels}.bin (int32) + meta.pkl.

Figure-2 protocol: num_examples=100_000 train / 3_000 test, vocab=8192,
random_non_queries=False, power_a=0.01, seqlen->kv = {64:4,128:8,256:16,512:64}.
Fixed seeds for reproducibility; train/test seeds well separated (no leakage).

Zoology's multiquery_ar already returns position-aligned inputs/labels
(inputs=examples[:,:-1], labels shifted so target sits at the query-key position,
-100 elsewhere) which matches our pre-shifted mqar_masked loss exactly -> direct export.

Usage:
  python data/mqar/gen_from_zoology.py                 # all four Figure-2 cells
  SEQLENS="64 256" python data/mqar/gen_from_zoology.py
"""
import os
import pickle
import sys

import numpy as np

ZOO_REPO = os.environ.get("ZOO_REPO", "/data/cl/u/xwang397/zoology")
sys.path.insert(0, ZOO_REPO)
from zoology.data.multiquery_ar import multiquery_ar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "8192"))
N_TRAIN = int(os.environ.get("N_TRAIN", "100000"))
N_VAL = int(os.environ.get("N_VAL", "3000"))
POWER_A = float(os.environ.get("POWER_A", "0.01"))
RANDOM_NON_QUERIES = os.environ.get("RANDOM_NON_QUERIES", "0") not in ("0", "false", "False")
SEED = int(os.environ.get("SEED", "123"))
IGNORE_INDEX = -100

# Figure 2 axis
SEQLEN_KV = {64: 4, 128: 8, 256: 16, 512: 64}
SEQLENS = [int(s) for s in os.environ.get("SEQLENS", "64 128 256 512").split()]


def dump_split(outdir, split, n, seed, seqlen, kv):
    seg = multiquery_ar(
        vocab_size=VOCAB_SIZE,
        num_examples=n,
        input_seq_len=seqlen,
        seed=seed,
        power_a=POWER_A,
        num_kv_pairs=kv,
        num_passes=1,
        random_non_queries=RANDOM_NON_QUERIES,
        include_slices=True,
    )
    x = seg.inputs.cpu().numpy().astype(np.int32)   # (n, seqlen)
    y = seg.labels.cpu().numpy().astype(np.int32)   # (n, seqlen), -100 masked
    assert x.shape == (n, seqlen) and y.shape == (n, seqlen), (x.shape, y.shape)
    x.tofile(os.path.join(outdir, f"{split}_inputs.bin"))
    y.tofile(os.path.join(outdir, f"{split}_labels.bin"))
    n_answers = int((y != IGNORE_INDEX).sum()) // n
    return n_answers


def main():
    for seqlen in SEQLENS:
        kv = SEQLEN_KV[seqlen]
        name = f"mqar_zoo6_d{kv}_n{seqlen}"
        outdir = os.path.join(HERE, "..", name)
        os.makedirs(outdir, exist_ok=True)
        # train/test seeds far apart to avoid overlap (mirrors zoology's split)
        na = dump_split(outdir, "train", N_TRAIN, SEED, seqlen, kv)
        dump_split(outdir, "val", N_VAL, SEED + 1_000_000, seqlen, kv)
        meta = {
            "vocab_size": VOCAB_SIZE,
            "block_size": seqlen,
            "num_kv_pairs": kv,
            "num_queries": kv,
            "power_a": POWER_A,
            "random_non_queries": RANDOM_NON_QUERIES,
            "mqar_masked": True,
            "ignore_index": IGNORE_INDEX,
            "source": "zoology.data.multiquery_ar (upstream repo)",
        }
        with open(os.path.join(outdir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"[{name}] seqlen={seqlen} kv={kv} vocab={VOCAB_SIZE} "
              f"train={N_TRAIN} val={N_VAL} answers/ex={na} -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
