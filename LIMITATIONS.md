# Limitations & Future Work

An honest accounting of what this project does and doesn't do. Listing
limitations explicitly is itself a signal of engineering maturity — it
shows the scope was a deliberate choice, not an oversight.

---

## Current Scope

This implementation reproduces the original Transformer architecture for
machine translation and extends it with inference/training optimizations
common in modern LLMs. It does **not** attempt to:

- Match the original paper's training scale (WMT14, 4.5M sentence pairs,
  8×P100 GPUs, 12+ hours) — this repo targets free-tier Colab, which means
  a smaller model (N=3 vs N=6, d_model=256 vs 512) and a smaller dataset
  (Multi30k, ~29k pairs vs WMT14's millions)
- Reach the paper's reported BLEU (27.3 on WMT14 En-De) — see
  `EXPERIMENTS.md` for actual numbers on this smaller setup
- Support languages beyond German→English (Multi30k is bilingual; extending
  to other pairs would need different spaCy tokenizer models and retraining)

### Experimental limitations (why the BLEU gap exists)

The gap between this repo's BLEU and the paper's 27.3 is not a bug — it's
an expected consequence of three deliberate scope reductions, each made to
fit free-tier Colab:

| Factor | This repo | Paper |
|---|---|---|
| Training data | Multi30k, ~29K pairs | WMT14, ~4.5M pairs |
| Model size | N=3, d_model=256 (~5-10M params) | N=6, d_model=512 (65M params) |
| Hardware / time | 1× Colab T4, ~20-30 min | 8× P100, 12 hours |

Each of these alone would lower BLEU; combined, a meaningful gap from 27.3
is expected and not evidence of an implementation bug. Scaling up any one
of these (more data, bigger model, more training time) would be expected
to close part of the gap — see `EXPERIMENTS.md` for what was actually
tested at this smaller scale.

---

## Known Limitations

### The reported model uses spaCy word-level tokenization
All trained results use spaCy's pretrained tokenizers with a word-level
vocabulary (`min_freq=2`), not subwords. A from-scratch byte-level BPE
tokenizer now exists (`src/bpe.py`, opt-in with `python src/train.py --bpe`)
and is unit-tested (exact round trip, fewer tokens after merges, unseen
words never become `<unk>`), but:
- **No model has been trained or evaluated with BPE yet**, so there is no
  BPE BLEU number and no evidence yet that it fixes the `<unk>` failures in
  `RESULTS.md`.
- BLEU for a BPE checkpoint is computed on detokenized text, while the
  spaCy baseline scores space-joined word tokens. The two numbers would not
  be strictly comparable without re-scoring the baseline the same way.
- BPE sequences are longer than word sequences; `max_padding=72` has not
  been re-checked against BPE lengths on Multi30k.
- `visualize.py` does not support BPE checkpoints yet (it exits with a
  message).

### Data pipeline migrated off torchtext
`torchtext` (originally used for dataset loading and vocabulary building,
following the Annotated Transformer reference implementation) was deprecated
by its maintainers in September 2023, with its final release (0.18.0)
compatible only with PyTorch ≤2.3.0. Current Colab runtimes ship PyTorch
2.4+, which torchtext does not support. `data_pipeline.py` replaces the
three torchtext call sites (`datasets.Multi30k`, `build_vocab_from_iterator`,
`to_map_style_dataset`) with HuggingFace `datasets` for loading and a
minimal custom `Vocab` class matching torchtext's original interface — the
rest of the training pipeline (`Batch`, masking, the model itself) is
unchanged. Covered by 7 unit tests in `tests.py`.

### Decoder-only mode exists but has no training pipeline
The translation model follows the original 2017 encoder-decoder design.
A GPT-style decoder-only model (`make_decoder_only_model` in `model.py`)
is implemented and unit-tested (causal masking, shapes, gradient flow,
overfitting a toy sequence), but there is no training script or text
corpus for it. Nothing has been trained with it. Its `generate()` is
greedy-only and recomputes the full prefix each step (no KV cache, no
sampling).

### GQA is a standalone module, not part of the trained model
`src/gqa.py` implements Grouped Query Attention. It is tested to match
standard multi-head attention exactly when `num_kv_heads == h`, and to
shrink the KV cache by `h / num_kv_heads`. However, it is not wired into
`make_model()` or `train.py`. No model has been trained with GQA, and its
memory and speed savings have not been benchmarked (the
`python src/gqa.py` output is a shape calculation, not a measurement).
Converting a trained MHA model with `from_mha(num_kv_heads < h)` would
normally need a short "uptraining" phase, which is not implemented.

### Ablation flags are implemented but not yet trained
`make_model()` / `train.py` can now switch on RoPE, RMSNorm, SwiGLU and
weight tying (defaults reproduce the baseline exactly), and each flag is
tested to build and run. None of these configurations has been trained, so
the ablation table still has only baseline numbers. With `--use-swiglu`,
`d_ff` is not rescaled, so that variant has more parameters than the
baseline.

### No distributed / multi-GPU training
Training is single-GPU (or CPU) only. The original paper's `train_worker`
pattern (in the Annotated Transformer reference) supports
`DistributedDataParallel`, but this repo's `train.py` intentionally omits
that complexity to stay runnable on a single free-tier Colab instance.

### No model quantization
Inference runs in FP32 or FP16 (via AMP) only. INT8/INT4 quantization,
which would further reduce memory and could speed up CPU inference, is not
implemented.

### Beam search is not batched
The current `beam_search.py` implementation processes one sentence at a
time. A production system would batch multiple sentences' beams together
for GPU efficiency.

### Flash Attention's tiled implementation is for education, not production
`flash_attention_tiled()` in `flash_attention.py` is a pure-PyTorch
implementation of the tiling algorithm, useful for understanding *how*
Flash Attention works, but it doesn't achieve the actual speedup of the
real CUDA kernel. Production use should go through
`torch.nn.functional.scaled_dot_product_attention` (the
`flash_attention_pytorch()` function), which dispatches to the real fused
kernel on supported hardware.

---

## Planned Next Steps

### Done (implemented and unit-tested, not yet trained)

- **Ablation flags** — `--use-rope`, `--use-rmsnorm`, `--use-swiglu`,
  `--tie-weights` on `train.py` (`make_model()` keyword arguments).
- **BPE tokenizer from scratch** — `src/bpe.py`, opt-in via `train.py --bpe`.
- **Decoder-only mode** — `make_decoder_only_model()` in `model.py`.
- **Grouped Query Attention** — `src/gqa.py`.
- **`torch.profiler` integration** — `src/profile_model.py` (per-op
  time/memory breakdown; see `RESULTS.md`).

### Next, in rough priority order

1. **Run the ablations** — retrain with each flag (and `--bpe`) under the
   baseline's settings and fill in the ablation table with measured
   numbers, including whether BPE removes the `<unk>` failures.

2. **Train the decoder-only model** — add a small next-token-prediction
   training loop on an unstructured text corpus (e.g. TinyShakespeare),
   using the BPE tokenizer.

3. **Wire GQA into the models** — a `num_kv_heads` option in `make_model()`
   / `make_decoder_only_model()` plus cache-aware decoding, then benchmark
   the cache memory against standard attention.

---

## Why document limitations at all?

Two reasons. First, it's honest — claiming a project does more than it
does is the fastest way to lose credibility in a technical interview.
Second, it's useful: this list *is* the roadmap, and being able to discuss
"here's what I'd build next and why" is a stronger signal than pretending
the project is finished.
