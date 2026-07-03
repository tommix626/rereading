# MQAR softmax sub-1.0 plateau: a learned-APE × clean-padding interaction

**Scope:** `mq5` A/B harness (this repo, `GDN-minimal-BlueVela`), single cell
D=32, lr=1e-3, 1-head softmax. **Date:** 2026-07-02.

---

## TL;DR

A tiny 1-head softmax transformer trained on the *clean* variant of Zoology MQAR
plateaus at **~0.92** answer accuracy (not 1.0), learning in a **grokking staircase**
(0.02 → 0.81 → 0.88 → 0.92). Dissecting the errors by position/token shows the entire
residual error lives in **a few late context binding slots** that mutually alias.

The plateau is caused by an **interaction of two factors**, not either alone:
1. **Learned absolute positional embeddings (APE)**, and
2. the **clean, all-zero-padded query region** (`random_non_queries=False`).

**Fixing either one → ~1.0 accuracy:**

| variant | positional enc. | non-query slots | final val acc |
|---|---|---|---|
| original | learned APE | `0` padding (clean) | **0.921** (plateau, stuck slots) |
| `_rnq` | learned APE | random distractor tokens | **0.9995** |
| `_rope` | RoPE (relative) | `0` padding (clean) | **0.9994** |

So the "clean" simplified MQAR is **pathological** for an APE softmax model — it is
*not* an easier task, it is a trap. Random distractors (Zoology's real default) or
relative positions remove the pathology.

---

## Setup

**Task — Zoology MQAR** (`data/mqar/zoology.py`, `build_examples`), `num_passes=1`,
no separator:
```
context (positions 0..63):  k0 v0 k1 v1 ... k31 v31      (32 KV pairs, fully packed)
query region (64..191):     the 32 keys re-appear at power-law-distributed gaps;
                            the model must predict each key's value as the next token.
```
- vocab 8192; keys ∈ [1, 4096); values ∈ [4096, 8192). D = 32 KV pairs, seq len N = 6D = 192.
- Exactly 32 supervised positions/example (labels `-100` elsewhere). Gaps ~ power-law
  (`power_a=0.01`, small gaps favored).
- **`random_non_queries`** decides what fills the ~96 unused (0-placeholder) slots in the
  query region: `False` → left as `0` (clean); `True` → uniform-random token ids
  (distractors). It never touches the context, the query keys, or the labels — the task
  is identical, only the noise differs.

**Model** (`config/mq5_transformer.py`, `model.py mixer='softmax'`): 2 layers, `n_embd=64`,
**1 attention head**, SwiGLU MLP, RMSNorm, tied embeddings. Original run uses **learned
absolute pos-emb** (`use_rope=False`, `use_learned_pos_emb=True`) — matches Zoology's
attention synthetic baseline.

**Training:** constant lr 1e-3 (no decay/warmup grokking effects), wd 0.1, batch 256,
20 000 iters ≈ single epoch over 5.12M examples, bf16, `compile=False`.

---

## Method

1. **Snapshotting.** Added opt-in full-checkpoint snapshots to `train.py`
   (`--snapshot_checkpoints=True --snapshot_interval=500`) → 41 checkpoints
   (`ckpt_iter{N}.pt`) spanning the trajectory. Re-run is deterministic and reproduces
   the original bumps bit-for-bit.
2. **Per-position error analysis** (`scripts/analyze_mq5_errors.py`). For every snapshot,
   evaluate on the full 3000-example val set and record, for each of the 96 000 supervised
   predictions: correctness, predicted token, **query gap** (distance into query region),
   **binding slot** (which of the 32 context pairs was queried, 0=first…31=last), value
   token, key token. Also classify each error as *another in-context value* (binding
   confusion) / *a key* / *other*.
3. **Visualization** (`scripts/plot_mq5_slot_heatmap.py`, `plot_mq5_slot_compare.py`):
   slot × step accuracy heatmap with the val-acc learning curve aligned beneath it.

> Note: `import model` (via `fla`) crashes on newer/Blackwell GPUs with triton 3.2.0
> (`triton.set_allocator` missing); **pin `--gres=gpu:a6000:1`** for training and analysis.

---

## Findings (original: learned APE + clean padding)

- **Value-token identity is irrelevant.** Accuracy across 32 value-token buckets:
  mean 0.921, **std 0.004**. Failures are not about specific values.
- **Errors are positional — concentrated in specific late context slots.** Final
  per-slot accuracy is ~0.99 for slots 0–27 and 29, but **slots 28, 30, 31 sit at ~0.25**
  (context positions 56/60/62 — the last bindings before the query region). Slot 29
  (position 58) is spared.
- **The staircase is a positional curriculum (front → back).** Each grokking bump = a few
  more late slots switching on:
  - bump 1 (~step 3k, →0.81): all slots except 18,19,28,29,30,31
  - bump 2 (~step 5k, →0.88): slots **18,19** turn on
  - bump 3 (~step 13k, →0.92): slot **29** turns on
  - then it **stalls** — slots 28,30,31 never recover.
- **The error is binding confusion, never garbage.** ~79% of errors emit *another value
  present in the same example*; ~0% emit a key. When it fails on slots {28,30,31} it emits
  one of the *other two stuck slots'* values ~67% of the time, ~20% out-of-context — i.e.,
  those three positions collapse into one aliased cluster under the learned pos-emb.
- **Mild secondary query-side recency:** queries deep in the query region (large gap) are
  slightly worse (gap 0–3 ≈ 0.99 vs far gaps ≈ 0.885), separate from the slot effect.

## Ablations

- **RoPE + clean padding → 0.9994.** All slots reach ~1.0 (worst 0.998, incl. slot 31).
  No striping, no stuck band; slots turn on uniformly in one sweep (~step 5.3k).
- **APE + random fillers (`_rnq`) → 0.9995.** Fastest, single grok (~step 2–2.5k); all
  slots uniform, worst slot 0.998.

## Interpretation

With an all-zero query region, an APE model can lean on absolute-position shortcuts
("this offset is padding, ignore") instead of content-based key matching. Under a single
head with learned pos-emb, those shortcuts fail to *separate* binding positions right at
the context/query boundary, so the last few context slots become mutually indistinguishable
→ the model knows the answer is one of those late values but not which. Adding random
distractors destroys the shortcut and forces genuine retrieval; using relative (RoPE)
positions removes the special boundary. Either way → full recall.

**Takeaway for other agents:** when benchmarking recall on the "clean"
(`random_non_queries=False`) MQAR with absolute positional embeddings, a sub-1.0 plateau
is likely this artifact — **not** a capacity limit. Check per-slot accuracy; if it's the
late context slots aliasing, switch to `random_non_queries=True` or RoPE before concluding
the architecture "can't do MQAR."

---

## Figures

- `slurm/plots/mq5_slot_compare_d32_lr1e-3.png` — 3-way side-by-side (the money shot)
- `slurm/plots/mq5_slot_heatmap_d32_lr1e-3.png` — original APE+clean (slot × step + curve)
- `slurm/plots/mq5_slot_heatmap_d32_lr1e-3_rnq.png` — APE + fillers
- `slurm/plots/mq5_slot_heatmap_d32_lr1e-3_rope.png` — RoPE + clean
- `slurm/plots/mq5_errors_d32_lr1e-3_snap.png` — 6-panel diagnostic (gap/slot/value/error-type/transition)

## Reproduce

```bash
# 0. env: pin a6000 (fla import breaks on newer GPUs). conda env: gdn.
# 1. (clean data already exists: data/mqar_zoo_d32_n192)
#    filler variant:
KV_PAIRS=32 N_TRAIN=5120000 N_VAL=3000 RANDOM_NON_QUERIES=1 DATASET_TAG=_rnq \
  CHUNK_SIZE=500000 python data/mqar/gen_zoology.py     # ~12 min CPU, needs ~32G RAM

# 2. snapshot training runs (each ~10 min on 1×a6000):
SBATCH_EXTRA="--partition=debug --gres=gpu:a6000:1 --mem=24G --cpus-per-task=4" TIME=02:00:00 \
  SLURM=1 bash scripts/debug_mq5_snapshot.sh                                   # APE + clean
DATASET_TAG=_rnq  SBATCH_EXTRA="..." SLURM=1 bash scripts/debug_mq5_snapshot.sh # APE + fillers
RUN_TAG=_rope EXTRA_ARGS="--use_rope=True --use_learned_pos_emb=False" \
  SBATCH_EXTRA="..." SLURM=1 bash scripts/debug_mq5_snapshot.sh               # RoPE + clean

# 3. analysis + heatmap per run (srun on a6000):
python scripts/analyze_mq5_errors.py --out_dir out/mq5_softmax_d32_lr1e-3_snap \
  --dataset mqar_zoo_d32_n192 --tag d32_lr1e-3
python scripts/plot_mq5_slot_heatmap.py \
  --npz out/mq5_softmax_d32_lr1e-3_snap/error_analysis.npz --tag d32_lr1e-3

# 4. combined comparison:
python scripts/plot_mq5_slot_compare.py --tag d32_lr1e-3 --runs \
  "out/mq5_softmax_d32_lr1e-3_snap/error_analysis.npz:APE + clean (0-pad)" \
  "out/mq5_softmax_d32_lr1e-3_rnq_snap/error_analysis.npz:APE + random fillers" \
  "out/mq5_softmax_d32_lr1e-3_rope_snap/error_analysis.npz:RoPE + clean (0-pad)"
```

## Caveats / open questions

- *Which* exact late slots stick (28/30/31 here) is likely **seed-specific**; the robust
  claim is "a few late context slots near the boundary," not those exact indices.
- Single cell (D=32, lr=1e-3, 1 seed). Not yet checked: larger D (64/128), multiple seeds,
  RoPE+fillers (the 4th 2×2 cell), or whether more heads/layers mask the effect.
- The `_rnq` random fillers can coincidentally equal a key/value token; rare, and labels
  still supervise only the true query positions, so it's mild added distraction.
