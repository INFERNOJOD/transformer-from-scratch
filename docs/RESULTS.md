# Results

This is the publication-style summary of this project's outcomes.
`EXPERIMENTS.md` is the lab notebook (every run, in order, with notes).
This file is the distilled version — what you'd show someone who wants the
headline numbers, not the process.

**Status:** Trained and evaluated — verified end-to-end twice, including
once from a completely fresh clone of the repository. Numbers below are
real, not placeholders.

---

## Final Model

| Property | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Layers (N) | 3 |
| Model dimension (d_model) | 256 |
| Attention heads (h) | 4 |
| FFN dimension (d_ff) | 512 |
| Total parameters | 9,358,576 |
| Dataset | Multi30k (German → English) |
| Training examples | 29,000 sentence pairs (1,014 val / 1,000 test) |
| Epochs | 8 |
| Hardware | Google Colab, Tesla T4 GPU |
| Training time | ~20 minutes (fresh-clone verification run) |
| Unit tests | 39/39 passing |

---

## Headline Numbers

| Metric | This Model | Paper (WMT14, full scale) |
|---|---|---|
| BLEU (greedy) | not separately measured — see `EXPERIMENTS.md` | — |
| BLEU (beam, k=4) | **36.37** (verified twice: 36.30, 36.37) | 27.3 |
| Final validation loss | **1.5295** | — |
| Parameters | 9,358,576 | 65M (base model) |
| Training data | 29K pairs | 4.5M pairs |

**Why this BLEU is higher than the paper's 27.3 — and that's expected, not
a bug:** This project uses the Multi30k German→English dataset (~29k
sentence pairs), whereas the original Transformer paper reports results on
WMT14 English→German (4.5M sentence pairs). Multi30k is substantially
easier, so BLEU scores are not directly comparable. The higher BLEU
reflects dataset difficulty rather than outperforming the original model.
See `LIMITATIONS.md` for the full breakdown.

See `LIMITATIONS.md` for the complete scope comparison — this is a
deliberately scaled-down reproduction, not an attempt to match the paper's
compute budget or dataset.

---

## Training Curve

![Training Loss](training_loss_curve.png)

**Observations:** Loss decreases smoothly and monotonically for both train
and validation across all 8 epochs. Train loss: 6.26 → 1.27. Val loss:
4.57 → 1.53. No overfitting is visible — train and val loss track closely
together throughout, with val loss staying slightly above train loss the
entire time (expected, since the model never sees validation data during
training). The curve visibly flattens from epoch 6 onward (val loss:
1.61 → 1.57 → 1.53), suggesting the model is close to converged at this
scale — more epochs would likely show diminishing returns without either
more data or a larger model. This trajectory was confirmed reproducible
across two independent full training runs, converging to within ~1% of
each other on final validation loss.

---

## Translation Examples

Five translations from the verified fresh-clone run (`evaluate.py`, beam
search k=4), picked to show a range of outcomes — not just the best ones.

### Good — near-perfect
```
EN (model):      A man sleeping on a couch in a green room.
EN (reference):  A man sleeping in a green room on a couch.
```
Same words, different order — semantically identical to the reference.

### Good — accurate paraphrase
```
EN (model):      A bald man wearing a red life jacket sitting in a small boat.
EN (reference):  A balding man wearing a red life jacket is sitting in a small boat.
```
"Bald" vs "balding" — minor difference, otherwise an accurate translation.
Notably, this same sentence was mistranslated in an earlier training run
("a man is catching a red life jacket") — see Failure Analysis for that
run's diagnosis and how the model corrected this on retraining.

### Partially wrong — dropped detail
```
EN (model):      A group of men are loading into a truck.
EN (reference):  A group of men are loading cotton onto a truck
```
Issue: dropped the object "cotton" entirely — the model gets the
structure (men loading something into a truck) but loses a specific noun.

### Partially wrong — restructured but plausible
```
EN (model):      A boy with headphones sitting on his shoulders.
EN (reference):  A boy wearing headphones sits on a woman's shoulders.
```
Issue: "sitting on his shoulders" is a confusing rewrite — the model
dropped "a woman's" and the sentence now reads ambiguously (whose
shoulders?). A real grammatical/coherence error, not just a word swap.

### Wrong — unknown token
```
EN (model):      Two men are building a blue <unk> on a lake floor.
EN (reference):  Two men setting up a blue ice fishing hut on an iced over lake
```
Issue: "ice fishing hut" — a rare compound noun — falls back to `<unk>`.
This same failure mode appeared consistently across multiple training
runs, which makes it a structural limitation (vocabulary coverage) rather
than training variance. See Failure Analysis below.

See the **Failure Analysis** section below for a deeper look at the wrong ones.

---

## Attention Visualization

![Attention Heatmap](assets/attention_sample.png)

_(Representative attention heatmap generated by the visualization pipeline.)_

**What the heads learned:** Different attention heads specialize in
complementary behaviours, including local neighbour attention, broader
sentence-level attention and alignment between related source and target
tokens.

---

## Benchmark Results

Measured on Colab T4, PyTorch 2.11.0+cu128. Run twice independently —
raw data from the first complete run in `benchmark_results.json`.

### Flash Attention vs Standard Attention

| Seq Length | Standard (ms) | Flash (ms) | Speedup | Memory Saved |
|---|---|---|---|---|
| 64 | 0.106 | 0.049 | 2.15× | 0.5 MB |
| 128 | 0.123 | 0.077 | 1.61× | 2.0 MB |
| 256 | 0.264 | 0.185 | 1.43× | 8.0 MB |
| 512 | 0.709 | 0.653 | 1.09× | 32.0 MB |
| 1024 | 2.652 | 2.224 | 1.19× | 128.0 MB |

**Analysis:** Flash Attention wins on speed at every sequence length tested
and the memory savings scale predictably — roughly 4× per doubling of
sequence length, consistent with standard attention's O(seq²) memory vs
Flash's O(seq). The speedup ratio itself isn't perfectly monotonic (2.15×
at seq=64 vs 1.09× at seq=512) — at very short sequences, PyTorch's
overhead for kernel dispatch and the eager-mode standard-attention path
dominate, exaggerating Flash's relative advantage. The picture would
likely look different (and more favorable to Flash) on a larger model and
at sequence lengths in the thousands, where Flash Attention's design
advantages are most pronounced. A second independent benchmark run showed
the same overall shape (2.39× at seq=64, 1.24× at seq=1024).

### KV Cache — Autoregressive Decoding

| Decode Steps | No Cache (ms) | With Cache (ms) | Speedup |
|---|---|---|---|
| 10 | 53.94 | 51.49 | 1.05× |
| 20 | 109.87 | 103.87 | 1.06× |
| 40 | 221.63 | 286.15 | **0.78×** |
| 80 | 439.28 | 433.34 | 1.01× |

**Honest finding — KV cache showed essentially no measurable benefit here,
and was slower at 40 steps in this run.** This is a real result worth
understanding rather than hiding. Two likely causes:

1. **Scale.** This is a 3-layer, d_model=256 model decoding only 10-80
   tokens. KV caching eliminates *redundant* K,V computation for past
   tokens, but the *fixed* per-step overhead (Python loop, kernel launches,
   tensor concatenation in the cache) is roughly constant regardless of
   caching. At this small scale, the fixed overhead dominates the
   savings — the O(n²) → O(n) improvement matters most when n and the
   model are both large enough that recomputation is actually expensive
   relative to everything else happening each step.
2. **GPU scheduling noise.** A second independent benchmark run showed a
   *different* pattern: near-1× at 10/20/80 steps, but a 1.30× speedup
   (cache faster) at 40 steps — the "anomaly" landed at a different point
   each run. This is strong evidence the irregularity is run-to-run timing
   noise rather than a structural problem with the cache implementation.
   `tests.py`'s `TestKVCache` suite independently confirms the cache
   produces numerically correct accumulated K,V tensors, so this is a
   *performance* characteristic at small scale, not a *correctness* issue.

**What this means in practice:** KV caching is a technique that pays off
at production LLM scale (billions of parameters, sequences of hundreds to
thousands of tokens) — exactly where it's used in GPT-4, Claude, etc. At
this project's intentionally small scale, the benefit is genuinely not
visible, and pretending otherwise would be dishonest. Understanding *when*
an optimization technique applies and when it doesn't is itself a useful,
demonstrable piece of engineering judgment.

### Mixed Precision (FP32 vs AMP)

| Batch Size | FP32 (ms) | AMP (ms) | Speedup |
|---|---|---|---|
| 4 | 31.46 | 36.95 | **0.85×** |
| 8 | 31.66 | 37.78 | 0.84× |
| 16 | 32.56 | 38.14 | 0.85× |

**Honest finding — AMP was slower than FP32 in every batch size tested,
confirmed across two independent runs** (second run: 0.61×–0.88×). Mixed
precision's speed advantage comes from GPU Tensor Cores, which need
sufficiently large matrix dimensions to saturate — at d_model=256 with
batch sizes ≤16, the matrices involved are small enough that the
*overhead* of dtype casting (FP32→FP16 and back), gradient unscaling, and
`GradScaler`'s NaN/Inf checks outweighs any Tensor Core speedup. This
matches known guidance that AMP's benefit grows with model and batch
size — it would be expected to look very different on the paper's full
base model (d_model=512, batch up to 256).

### Profiler — Where Time Actually Goes

Measured with `torch.profiler` (`profile_model.py`). Full raw output in
`profiler_summary.txt`. Confirmed consistent across two runs at different
configurations (seq_len=512/batch=8, and seq_len=128/batch=8).

| Operation | % of Total CUDA Time | What it is |
|---|---|---|
| `aten::bmm` | ~16% | Attention's Q·Kᵀ and weights·V matmuls |
| `aten::mm` / `aten::addmm` | ~26% | Linear projections (Q/K/V/output, FFN) |
| `volta_sgemm_*` kernels | ~20% | Underlying GPU matmul kernels |
| `aten::masked_fill` | ~10% | Causal mask application |
| `aten::clone` / `aten::copy_` | ~16% | Multi-head reshape/concat |

**Analysis:** attention computation (`bmm` + `masked_fill`) and the linear
projections (`mm`/`addmm`) together account for roughly half of total GPU
time — exactly where you'd expect compute to concentrate in a Transformer,
and a useful confirmation that Flash Attention (which directly targets the
`bmm` + `masked_fill` + softmax sequence) is optimizing a genuinely
significant fraction of total compute, not a minor operation.

---

## Failure Analysis

Real failures from the evaluation runs above, with actual diagnosis — not
just a BLEU number.

### Failure 1 — `<unk>` on a rare compound noun (consistent across runs)
```
EN (model):      Two men are building a blue <unk> on a lake floor.
EN (reference):  Two men setting up a blue ice fishing hut on an iced over lake
```
**Diagnosis:** The vocabulary was built with `min_freq=2` on a 29,000-sentence
corpus — any token appearing fewer than 2 times collapses to `<unk>`.
"Ice fishing hut" is exactly the kind of rare, multi-word compound concept
that a small-vocabulary, word-level tokenizer struggles with. This failure
appeared in the *same sentence, the same way*, across two independently
trained models — strong evidence this is a structural limitation of
word-level tokenization with a frequency cutoff, not a one-off training
artifact. A subword tokenizer (BPE) would instead fall back to
recognizable pieces ("ice", "fish", "ing", "hut") rather than a single
opaque `<unk>` — the strongest practical argument for the planned BPE
migration (see `LIMITATIONS.md`).

### Failure 2 — same input, different error across two training runs
```
Run A model:     A man is catching a red life jacket sitting in a small boat.
Run B model:     A bald man wearing a red life jacket sitting in a small boat.
Reference:        A balding man wearing a red life jacket is sitting in a small boat.
```
**Diagnosis:** This is a genuinely interesting comparison. Run A's model
substituted "catching" for "wearing" — a real semantic error, likely a
spurious association between "red life jacket" and boat/fishing contexts
common in Multi30k. Run B's model (a separate training run, same config)
got this essentially correct. Since nothing about the architecture or
data changed between runs, this difference comes down to training
stochasticity — different random weight initialization and minibatch
ordering led to different local optima within 8 epochs. This is a useful,
concrete illustration of training variance: a single BLEU number from one
run doesn't fully capture model quality, and the *same* model architecture
can produce meaningfully different outputs on the *same* sentence purely
from random seed differences.

### Failure 3 — dropped object noun
```
EN (model):      A group of men are loading into a truck.
EN (reference):  A group of men are loading cotton onto a truck
```
**Diagnosis:** The model correctly captures sentence structure (subject,
verb, prepositional phrase) but drops the direct object "cotton" entirely.
This looks like a beam-search / decoding issue rather than a vocabulary
gap — "cotton" is a common enough word to be in-vocabulary. More likely,
the model assigned low probability to inserting an object noun at that
position and beam search's length penalty didn't surface a hypothesis
that included it. This is the kind of error more training data or a
larger model would likely reduce, since it reflects insufficient learned
association between "loading" and its typical objects in this domain.

---

## Ablation Summary

| Config | Val Loss | BLEU | Status |
|---|---|---|---|
| Baseline (greedy) | 1.5295 | not separately measured | trained, BLEU not isolated from beam search |
| + Beam Search (k=4) | — | **36.37** | verified twice (36.30, 36.37), full validation set |
| + RoPE | — | — | implemented and unit-tested, not yet run as a retrained ablation |
| + RMSNorm + SwiGLU | — | — | implemented and unit-tested, not yet run as a retrained ablation |
| + Weight Tying | — | — | implemented and unit-tested, not yet isolated as a separate ablation |

Honest gap: the architectural improvements (RoPE, RMSNorm+SwiGLU, weight
tying) are implemented, unit-tested, and individually verified to work
correctly — but have not yet been plugged into `train.py` for a full
retraining ablation run. The numbers above reflect only what has actually
been measured; see `EXPERIMENTS.md` for the open items.

---

## Conclusion

This project reproduced the Transformer architecture from "Attention Is
All You Need" entirely from scratch in PyTorch, trained it end-to-end on
Multi30k, and reached 36.37 BLEU with beam search — verified reproducibly
across two independent training runs and confirmed once more from a
completely fresh repository clone. Beyond the base architecture, six
modern improvements (RoPE, Flash Attention, KV Cache, RMSNorm, SwiGLU,
weight tying) were implemented and unit-tested, with real benchmarking
revealing an honest and useful finding: at this project's intentionally
small scale, KV Cache and Mixed Precision show little to no measurable
speedup, while Flash Attention's advantage is real but modest — a concrete
illustration of *when* these production-LLM optimizations actually start
to matter. The most surprising result wasn't the BLEU score itself, but
how consistently the same failure mode (rare compound nouns collapsing to
`<unk>`) appeared across independently trained models — a clear, specific
signal pointing at vocabulary coverage, not training variance, as the
next thing worth fixing.

---

## What's Next

See `LIMITATIONS.md` for the full roadmap. Short version: BPE tokenizer →
decoder-only mode → Grouped Query Attention, in that order. The BPE
tokenizer is the highest-priority item — Failure Analysis above shows a
concrete, reproducible case (`<unk>` on "ice fishing hut") it would
directly fix.
