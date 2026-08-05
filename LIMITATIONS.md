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

### Tokenization is not built from scratch
The pipeline currently uses spaCy's pretrained tokenizers for German and
English, not a custom-trained subword tokenizer. The original paper uses
byte-pair encoding (BPE) trained on the translation corpus itself. This is
the most significant remaining gap between this repo and a fully
from-scratch implementation — **planned as the next addition** (see below).

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

### Encoder-decoder, not decoder-only
The architecture follows the original 2017 paper's encoder-decoder design,
which is well-suited to translation (a clear source→target mapping) but is
not the dominant architecture for general-purpose language modeling today.
Modern LLMs (GPT, LLaMA, Mistral) are decoder-only, trained on
next-token-prediction over unstructured text rather than parallel
sentence pairs.

### KV Cache memory grows linearly with sequence length
The current KV cache implementation stores one K,V pair per attention head
per layer per token. For long sequences or large batch sizes, this becomes
the dominant memory cost during inference. Grouped Query Attention (GQA) —
where multiple query heads share a single K,V head — is the standard fix
for this and is not yet implemented here.

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

In rough priority order:

1. **BPE tokenizer from scratch** — train byte-pair merge rules directly on
   the Multi30k corpus, replacing the spaCy dependency. This closes the
   last black-box in the pipeline.

2. **Decoder-only mode** — strip the encoder and cross-attention, train the
   decoder stack alone for next-token prediction on an unstructured text
   corpus (e.g. TinyShakespeare). This is architecturally what GPT/LLaMA
   actually are.

3. **Grouped Query Attention (GQA)** — modify `MultiHeadAttention` so that
   multiple query heads share K,V projections, shrinking the KV cache
   memory footprint at longer context lengths.

4. **`torch.profiler` integration** — replace the wall-clock benchmarking
   in `benchmark.py` with proper PyTorch profiler traces, to get per-layer
   breakdowns of time and memory rather than just end-to-end numbers.

---

## Why document limitations at all?

Two reasons. First, it's honest — claiming a project does more than it
does is the fastest way to lose credibility in a technical interview.
Second, it's useful: this list *is* the roadmap, and being able to discuss
"here's what I'd build next and why" is a stronger signal than pretending
the project is finished.
