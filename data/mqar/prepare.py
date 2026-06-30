r"""Multi-Query Associative Recall (MQAR) — Zoology-style.

This repo uses **answer-token-only** masked CE (labels -100 elsewhere), not full
next-token prediction over all positions. See data/mqar/zoology.py and gen.py.

Generate data:
  NUM_PAIRS=64 INPUT_SEQ_LEN=256 VOCAB_SIZE=8192 python data/mqar/gen.py

Train:
  python train.py config/ar_gdn.py
  python train.py config/ar_transformer.py

Invariant tests:
  python data/mqar/test_zoology.py
"""
raise SystemExit(
    "Use data/mqar/gen.py (Zoology MQAR). This legacy prepare.py is retired."
)
