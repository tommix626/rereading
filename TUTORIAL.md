# TUTORIAL — reading this codebase part by part

A guided walk through the whole repo, in the order that makes it click. Each section
points you at a specific file and line range, explains what that segment does and *why*,
and ends with a one-line "you now understand X". Read with the file open beside this.

The repo is small on purpose: ~5 source files. By the end you'll have read essentially
all the meaningful code. Total time ≈ 60–90 min if you read the cited code carefully.

Suggested order:

1. The big idea (this section)
2. `configurator.py` — how config works (read first; everything else assumes it)
3. `model.py` — the network
4. `data/megatron.py` — where tokens come from
5. `train.py` — how it all runs
6. `config/*.py` — the actual experiment knobs
7. `scripts/submit_lsf.sh` — launching on the cluster

---

## 0. The big idea (read before any code)

This is a **controlled experiment**, not a product. The question being asked is:

> *If I hold the entire language model fixed and swap only the "sequence mixer"
> (the part that lets tokens attend to / mix with each other), how does
> **Gated DeltaNet (GDN, a linear-attention recurrence)** compare to **standard
> softmax attention**?*

To make the comparison fair, everything except the mixer is identical and the two are
**parameter-matched** to <0.5%. So the codebase is structured around one switch
(`mixer = 'gdn' | 'softmax'`) and two near-identical config files.

Mental model of one Transformer block (both variants):

```
x ──► RMSNorm ──► [ MIXER ] ──► + ──► RMSNorm ──► SwiGLU MLP ──► + ──► out
└──────────────────────────────┘     └──────────────────────────────┘
        residual 1                            residual 2
```

Only `[ MIXER ]` changes between the two experiments. Keep that picture in mind.

✅ *You now understand why there are two configs and one model file.*

---

## 1. `configurator.py` — the config mechanism (≈48 lines, read 100%)

Read the **whole file** first. It's weird but you must understand it before `train.py`.

- **Lines 1–15 (docstring):** the honest pitch. This is *not* a Python module. `train.py`
  literally does `exec(open('configurator.py').read())`, so this code runs **inside
  train.py's global namespace** and edits its variables in place.
- **Lines 20–28:** for each CLI arg with no `=`, treat it as a **config file path** and
  `exec` that file too — so `config/train_gdn.py` also runs in train.py's namespace and
  reassigns globals.
- **Lines 29–47:** for each `--key=value` arg, parse the value with `literal_eval`
  (so `--max_iters=100` becomes an int, not a string), assert the **key already exists**
  as a global and that the **type matches**, then overwrite it.

The two consequences that bite people:
- A new tunable must already be a global in `train.py` *before* the exec line, or `--key`
  is rejected (line 47).
- `--lr=1e-4` only works if `lr` is already a float global; the type check (line 42) is strict.

✅ *You now understand how a config file path and `--flags` both mutate `train.py`'s globals.*

---

## 2. `model.py` — the network (≈272 lines)

Read top-to-bottom; here's the map.

### 2a. Module docstring — lines 1–13
States the design contract: both mixers share RMSNorm pre-norm, SwiGLU MLP, tied
embeddings, no biases, depth-scaled output-projection init, and a fused cross-entropy loss.
The only fork is the mixer. The GDN paper link is here (arxiv 2412.06464).

### 2b. RoPE helpers — lines 27–41 (softmax path only)
`_precompute_rope_freqs` builds the cos/sin tables; `_apply_rope` rotates adjacent channel
pairs of `q`/`k`. This is the **positional information for the softmax mixer**. Note: GDN has
*no* RoPE — it gets locality from a short conv instead (see 2e). Skim the math; the point is
just "softmax attention needs explicit positions, GDN doesn't."

### 2c. `CausalSelfAttention` — lines 44–67 (the softmax mixer)
Standard multi-head attention: `c_q/c_k/c_v` projections, apply RoPE to q and k,
`F.scaled_dot_product_attention(..., is_causal=True)`, then `c_proj`. Nothing exotic.
This is the *baseline* arm of the experiment. `head_dim = n_embd // n_head`.

### 2d. `MLP` — lines 70–80 (shared)
SwiGLU. `c_fc` projects to `2 × ffn_intermediate` and is split into `(gate, up)`;
output is `c_proj(silu(gate) * up)`. Same for both experiments.

### 2e. `Block` — lines 83–118 (the switch lives here) ⭐
This is the heart of the repo. In `__init__`:
- `mixer == 'gdn'` → `self.attn = GatedDeltaNet(...)` from `fla`, configured entirely from
  `gdn_*` config fields. Note `conv_bias=False`, `use_short_conv=True`, `conv_size=4` — the
  short conv is GDN's source of local position. `layer_idx` is passed for the recurrence cache.
- `mixer == 'softmax'` → `self.attn = CausalSelfAttention(config)`.
- anything else → raises.

In `forward` (lines 111–118): note GDN returns a **3-tuple** `(out, _, _)` while
`CausalSelfAttention` returns a plain tensor, so there's an `isinstance` check. Then the two
residual adds. Re-read the ASCII diagram in section 0 against these 8 lines.

### 2f. `_default_ffn_intermediate` + `GPTConfig` — lines 121–148
`GPTConfig` is the single source of truth for model shape. Read every field and its default;
the `gdn_*` block (lines 137–145) and the softmax block (lines 147–148) are mutually
exclusive depending on `mixer` (line 135). `ffn_intermediate_size=None` → computed as
8/3·n_embd rounded to a multiple of 64.

### 2g. `GPT.__init__` — lines 151–181
- Builds `wte` embedding, `n_layer` `Block`s, final `ln_f`, and `lm_head`.
- **Tied embeddings** (line 168): `wte.weight = lm_head.weight` — same tensor, counted once.
- **Fused loss** (line 172): `FusedLinearCrossEntropyLoss`. The comment explains *why* —
  materializing the full `[B·T, vocab]` logits would be ~13 GB and OOM. This is important.
- **Depth-scaled init** (lines 175–179): every `*.c_proj.weight` / `*.o_proj.weight`
  (the residual-path *outputs*) is re-initialized with std `0.02/sqrt(2·n_layer)` — the
  GPT-2 trick to keep residual variance stable with depth.

### 2h. `forward` — lines 200–217 ⭐
The training path (`targets is not None`, lines 211–214) calls `self.loss_fn` and returns
`(None, loss)` — **it never produces a logits tensor**. The inference path (lines 215–217)
runs `lm_head` on the **last position only**. Understand why training returns `None` for
logits: it's the fused-loss memory optimization in action.

### 2i. `configure_optimizers` — lines 219–242 ⭐
Splits params into decay / no-decay groups. The rule (lines 224–228): a param goes to
**no-decay** if it carries `_no_weight_decay=True` (GDN sets this on `A_log`, `dt_bias`) **or**
has `dim < 2` (norms, biases). Everything else gets weight decay. Then builds fused AdamW.
Note `eps` is a parameter here (default 1e-8) but the configs pass **1e-10** — keep that in mind.

### 2j. `estimate_mfu` (244–257) and `generate` (259–271)
MFU uses the 6N approximation (drops the attention T² term — fine at this scale). `generate`
is a standard autoregressive sampling loop; not used in training. Skim both.

✅ *You now understand the model end-to-end and exactly where the GDN/softmax fork is.*

---

## 3. `data/megatron.py` — the data source (≈57 lines, read 100%)

Small and worth reading fully.
- **Lines 17–37:** Megatron `.idx` files start with a magic header and a dtype code; `_read_dtype`
  reads just enough of the header to learn the token dtype (e.g. `uint16` vs `int32`).
- **Lines 40–50:** `open_megatron` mmaps the `.bin` as a **flat token stream** at that dtype.
  Crucial simplification (lines 4–6): document boundaries are **ignored** — a training sample can
  span documents, exactly like nanoGPT's original loader.
- **Lines 53–56:** `normalize_weights` for blending multiple data files.

✅ *You now understand that "the dataset" is just one or more big flat arrays of token ids.*

---

## 4. `train.py` — the run (≈472 lines). The big one; read in segments.

### 4a. Docstring + defaults — lines 1–120
Lines 1–17 show the three launch modes (single-GPU, 1-node DDP, multi-node DDP). Lines 34–119
are **all the default globals** — this is the full list of tunables. You don't need to memorize
them, but skim so you know what exists. Note the comments flag the non-obvious ones
(`eps=1e-10`, fixed-set val via `val_seed`, the running-window train-loss metric).

### 4b. The config exec — lines 121–124 ⭐
```python
config_keys = [...globals that are int/float/bool/str...]   # snapshot NAMES
exec(open('configurator.py').read())                         # overrides happen here
config = {k: globals()[k] for k in config_keys}              # snapshot VALUES for logging
```
This is configurator.py (section 1) firing. Everything above line 122 is a *default*;
everything below reads the *resolved* value. This three-line dance is why new knobs must be
declared above line 122.

### 4c. DDP / device setup — lines 126–157
Detects DDP from env vars (`RANK` etc.), sets per-rank device and `seed_offset`, divides
`gradient_accumulation_steps` across ranks (line 140), computes `tokens_per_iter` (line 146).
Sets the autocast `ctx` and dtype. Read the `tokens_per_iter` formula — it's how you reason
about total tokens (`tokens_per_iter × max_iters`).

### 4d. Data loader — lines 159–223 ⭐
- Lines 164–178: resolve the Megatron config — either a list of blended paths+weights or a
  single legacy path; falls back to nanoGPT-native `train.bin/val.bin` if neither set.
- **`get_batch` (180–223):** the key function. Note (lines 181–188) **all randomness routes
  through `data_rng` for train and `_val_rng` for val** — this is what makes batches identical
  across architectures and makes val a fixed set. For each sample it picks a file (by weight),
  picks a random offset, and slices `x = stream[i:i+block]`, `y = stream[i+1:i+1+block]`
  (next-token targets). Then pins + async-copies to GPU.

### 4e. Model + optimizer init — lines 225–308
- Lines 240–289: assemble `model_args`, then either build from scratch or `resume` from
  `out/ckpt.pt` (resume forces the architecture fields to match the checkpoint).
- Line 292: GradScaler (only active for fp16).
- Line 295: `configure_optimizers` (section 2i) — note `eps=eps` is threaded through.
- Lines 301–308: `torch.compile`, then wrap in DDP.

### 4f. Validation — lines 310–328 ⭐
`estimate_loss` (section to internalize): **reseeds `_val_rng` to `val_seed + ddp_rank` every
call** (line 316) so every eval draws the same positions; runs `eval_iters` batches; all-reduces
the per-rank mean with `ReduceOp.AVG`. Combined with `get_batch('val')` this is the "fixed
~20M-token val set" the README describes.

### 4g. LR schedule — lines 330–346
`get_lr(it)`: linear warmup → optional constant hold → cosine down to `lr_decay_factor·lr` by
`lr_decay_iters` → flat floor after. Mirrors lm-engine's scheduler. Trace one value
(e.g. `it = warmup_iters`) by hand to convince yourself.

### 4h. RNG setup — lines 356–364 ⭐
`data_rng` is seeded **here, after model construction** (line 361) — the comment explains this
decouples the data stream from however many RNG draws each architecture's init consumed. `_val_rng`
is created but seeded later (inside `estimate_loss`). This is the linchpin of the fair comparison.

### 4i. The training loop — lines 365–471 ⭐ (read carefully)
- Line 366: prefetch the first batch *before* the loop.
- Lines 376–378: set this iter's LR.
- Lines 382–392: periodic eval + WandB log (gated by `eval_during_training`).
- **Gradient accumulation (401–419):** inner loop over `gradient_accumulation_steps`
  micro-batches. Note (407) DDP grad sync only on the **last** micro-step; (415) loss divided by
  accum steps; (417) **next batch prefetched while the forward result is still on the GPU**;
  (411–414) raw per-micro-batch loss accumulated separately for the metric.
- Lines 421–428: grad clip → optimizer step → zero grad.
- Lines 430–461: timing, the per-iter loss log, MFU, and the **rolling-window train/loss**
  metric (averaged over all iters in the window, flushed every `train_loss_log_interval`).
- Lines 463–468: increment and stop at `max_iters`.

✅ *You now understand a full training step, including grad accumulation, DDP sync timing,
the fixed-set eval, and why data is reproducible across architectures.*

---

## 5. `config/train_gdn.py` and `config/train_transformer.py` — the experiment

Now read both configs side by side (they're ~85 lines each). Because of section 1, every line
here just reassigns a `train.py` global.

- Both: same data blend (lines ~11–22), same `vocab_size=100352`, same shape
  (`n_layer=16, n_embd=768, ffn=2048, block=4096`), same optimizer (`lr=3e-4, eps=1e-10`),
  same LR schedule (`warmup 1500 → cosine over 19000`), same batching
  (`bs=8, grad_accum=2*8`), same eval (`every 500, eval_iters=76`).
- **The only real difference** is the mixer block:
  - GDN (`train_gdn.py` 32–41): `mixer='gdn'`, `gdn_num_heads=6`, `gdn_head_dim=128`,
    `gdn_expand_v=1.0`, `gdn_use_gate=False`. The header comment (lines 1–6) explains that
    `use_gate=False` drops `g_proj` *and* removes fla's 0.75·hidden constraint, letting
    `6×128=768=hidden`.
  - Softmax (`train_transformer.py` 33–35): `mixer='softmax'`, `n_head=6` (→ head_dim 128,
    matching GDN), `rope_theta=10000`.
- Read both header comments (the `# ===` banners) — they document the *param-matching argument*
  (mixer params differ ~0.7%/layer, totals within <0.5%).

✅ *You now understand exactly what is held constant and what varies in the experiment.*

---

## 6. `scripts/submit_lsf.sh` — launching (≈47 lines, read 100%)

- Lines 14–17: `RUN_NAME` (arg 1, default `200M-gdn`) and `CONFIG` (arg 2, default the GDN
  config). Arg 1 becomes the WandB run name.
- Lines 21–23: capture git SHA/branch/dirty for WandB provenance.
- Lines 25–47: the `bsub` heredoc — requests 8 GPUs for 12h, sources the WandB key and the
  prebuilt venv (which **must** have `flash-linear-attention` installed), exports WandB env, and
  finally runs `torchrun --standalone --nproc_per_node=8 train.py <config>`.

✅ *You now understand how a run actually gets onto the cluster.*

---

## 7. Suggested exercises to confirm understanding

1. **Trace a token.** Pick `idx[0,0]`. Follow it: `get_batch` slice → `wte` embed → 16 `Block`s
   → `ln_f` → fused loss against `targets[0,0]`. Where does the GDN-vs-softmax path diverge?
2. **Add a knob.** Where exactly must you declare a new `--my_flag` so configurator accepts it?
   (Answer: as a global in `train.py` above line 122.)
3. **Break param-matching.** If you changed `n_embd` in `train_gdn.py` only, which invariant
   would the repo's whole premise violate, and which file documents that invariant?
4. **Reproduce eval.** Why does `estimate_loss` give identical val positions on every call, and
   what one number do you change to make the val set bigger? (Answer: `eval_iters`.)

When you can answer all four without re-reading, you've understood the codebase.
