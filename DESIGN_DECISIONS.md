# Design Decisions

Every non-obvious choice in this codebase, with the reasoning behind it.
This document is written for two audiences: anyone reading the repo, and
future-me preparing for interviews.

---

## Why Encoder-Decoder (not decoder-only)?

Translation is the task the original paper targets, and it's naturally a
sequence-to-sequence problem: the source sentence (German) is fully known
upfront, so the encoder can attend to it bidirectionally. The decoder then
generates the target (English) autoregressively, attending back to the
encoder's output via cross-attention.

Decoder-only models (GPT-style) are better suited to open-ended generation
where there's no clean "source" / "target" split. A decoder-only model is
now available as a separate option (`make_decoder_only_model`, see below),
but the translation pipeline and all reported results stay encoder-decoder.

---

## Why scale attention by 1/√d_k?

If `q` and `k` have independent components with mean 0 and variance 1, their
dot product `q·k = Σ qᵢkᵢ` has variance `d_k`. As `d_k` grows, raw dot products
grow large in magnitude. Large inputs push softmax into a saturated regime
where most of the probability mass concentrates on one element and gradients
through the other elements vanish.

Dividing by `√d_k` keeps the variance of the scaled scores at approximately 1
regardless of dimension, keeping softmax in a well-behaved region.

---

## Why 8 attention heads (not 1 or 16)?

A single attention head computes one weighted average over the sequence —
averaging inherently discards information about which positions disagreed.
Multiple heads let the model attend to different things simultaneously
(e.g., one head tracks syntactic dependencies, another tracks adjacent
tokens, another tracks rare long-range references). `assets/attention_sample.png`
illustrates this with synthetic data. For real heads from the trained model,
see `attention_maps/`, which is also the source of the README's attention image.
In `decoder_layers_2_cross_attn.png`, all four cross-attention heads learn a
near-diagonal source–target alignment but spread their attention differently
around "soccer" and the final period.

The paper's ablation (Table 3, row A) shows single-head attention is 0.9 BLEU
worse than h=8, but going beyond 8 heads also degrades quality slightly —
each head has fewer dimensions (`d_k = d_model/h`) to represent its slice of
the relationship, so too many heads under-parameterizes each one.

---

## Why residual connections + LayerNorm around every sub-layer?

Without residuals, gradients have to flow through every layer's full
transformation, which becomes unstable in deep networks (vanishing/exploding
gradients). The residual `x + Sublayer(x)` gives gradients a direct path
backward through addition, independent of how deep the network is.

LayerNorm stabilizes the *scale* of activations layer to layer — without it,
the variance of activations can drift as they pass through many layers,
making training sensitive to learning rate and initialization.

---

## Why Adam optimizer with β1=0.9, β2=0.98?

Adam adapts the learning rate per-parameter using running estimates of the
first and second moments of the gradient, which works well for the highly
non-uniform gradient scales typical in Transformers (embedding gradients vs
attention gradients vs FFN gradients can differ by orders of magnitude).

β2=0.98 (vs the more common default of 0.999) means the second-moment
estimate adapts faster — useful because Transformer training has fairly
noisy, fast-changing gradient statistics early in training.

---

## Why the warmup learning rate schedule?

```
lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

Early in training, the model's weights are randomly initialized and
predictions are essentially noise. A large learning rate applied immediately
can cause the first few updates to push the model into a bad region of
parameter space it can't recover from (especially with Adam, which can take
large steps when its moment estimates are still poorly calibrated from only
a few updates).

Warming up linearly for the first `warmup_steps` lets the moment estimates
stabilize before the learning rate reaches its peak. After that, decaying as
`1/√step` is a theoretically motivated schedule that slows learning as the
model approaches convergence.

---

## Why label smoothing (ε=0.1)?

Standard cross-entropy with one-hot targets pushes the model toward 100%
confidence on the correct token, which:
1. Encourages large logit magnitudes (the model has to push the correct
   logit infinitely high to get arbitrarily close to probability 1)
2. Makes the model overconfident on data it hasn't seen, hurting generalization

Label smoothing redistributes a small amount of probability mass (ε=0.1)
from the correct token to all other tokens, capping how confident the model
can become. This costs some perplexity (the model is "less sure" even when
correct) but empirically improves BLEU and calibration.

---

## Why BLEU instead of just validation loss?

Validation loss measures next-token prediction accuracy under teacher
forcing — it tells you how surprised the model is by the *correct* next
token, given the correct previous tokens. It says nothing about translation
quality when the model has to generate the whole sequence itself
(error compounding, fluency, adequacy).

BLEU compares n-gram overlap between the model's own generated output and a
reference translation, which is much closer to what a human means by
"is this a good translation." It's also the metric the original paper
reports (Table 2), so it's the only way to meaningfully compare against it.

---

## Why beam search instead of greedy decoding?

Greedy decoding picks the single highest-probability token at each step.
This is locally optimal but not globally optimal — a token that's slightly
less likely right now might lead to a much better continuation overall.

Beam search keeps the top-k partial hypotheses alive at each step instead of
collapsing to one, exploring more of the search space before committing.
The length penalty (`score = log_prob / length^α`) corrects for beam
search's natural bias toward shorter sequences (each additional token
multiplies in another probability ≤ 1, so unpenalized log-probs always favor
shorter outputs).

---

## Why RoPE instead of sinusoidal positional encoding?

Sinusoidal PE adds a fixed position vector to the token embedding *before*
attention. Position information gets mixed with content information in a
way that's only implicitly recoverable by the attention mechanism.

RoPE instead rotates the Q and K vectors by an angle proportional to their
position, applied *inside* the attention computation. The key mathematical
property: the dot product `q_m · k_n` after rotation depends only on
`(m - n)` — the relative position — not on the absolute positions. This
makes relative position explicit in the attention score itself, and tends
to generalize better to sequence lengths longer than what the model was
trained on.

---

## Why RMSNorm instead of LayerNorm?

LayerNorm computes `(x - mean) / std * γ + β` — it re-centers (subtracts
mean) and rescales (divides by std), with both a learned scale and bias.

RMSNorm computes `x / RMS(x) * γ` — only rescaling, no re-centering, no
bias term. The empirical finding (Zhang & Sennrich, 2019) is that the
re-centering step contributes little to LayerNorm's benefit; the rescaling
is what matters. Dropping mean-subtraction and the bias parameter makes
RMSNorm computationally cheaper (~10-15% faster) with comparable quality —
which is why it's used in LLaMA, Mistral, Gemma, and most current LLMs.

---

## Why SwiGLU instead of a ReLU FFN?

The original FFN is `max(0, xW1+b1)W2+b2` — a fixed nonlinearity (ReLU)
applied uniformly to all features.

SwiGLU instead computes `(Swish(xW1)) ⊙ (xW2) · W3` — a *gating* mechanism
where one linear projection (passed through Swish) controls how much of
another linear projection passes through, per-feature. This gives the
network the ability to learn which features matter for a given input,
rather than applying a fixed nonlinearity everywhere. Empirically this
consistently improves perplexity by a small but consistent margin
(Shazeer, 2020) and is now standard in LLaMA, PaLM, and Gemma.

---

## Why KV Cache?

During autoregressive decoding, generating token `t+1` requires running the
decoder on tokens `[0, ..., t]`. Without caching, the K and V projections
for *every* previous token get recomputed at *every* step — for a sequence
of length `n`, this means `O(n²)` total work across the full generation.

The KV cache stores the K, V tensors for tokens already processed, so at
each new step only the *new* token's K, V need to be computed and appended.
This reduces total work to `O(n)`. The cost is memory: the cache grows
linearly with sequence length, which is why techniques like GQA exist to
shrink the cache further (implemented in `gqa.py`, see below).

---

## Why Flash Attention?

Standard attention computes and stores the full `(seq_len, seq_len)`
attention score matrix, which for long sequences becomes the dominant cost
in both compute and memory I/O — writing and reading this matrix from GPU
HBM (slow memory) is the actual bottleneck, not the matrix multiplication
itself.

Flash Attention restructures the computation to process Q, K, V in tiles
that fit in fast on-chip SRAM, using the online softmax algorithm to
accumulate the mathematically exact result without ever materializing the
full attention matrix. Same output as standard attention, but `O(seq_len)`
memory instead of `O(seq_len²)`, and significantly faster due to reduced
memory traffic. This is what makes long-context models practical and is
implemented in PyTorch's `scaled_dot_product_attention` since 2.0.

---

## Why weight tying?

The target embedding matrix maps token IDs to vectors (`vocab_size × d_model`).
The generator's final linear layer maps vectors back to logits over the
vocabulary (`d_model × vocab_size`) — these are approximate transposes of
each other (a token's embedding should be "close to" what the model outputs
when it wants to predict that token).

Tying them (using the literal same weight matrix for both) eliminates a
full `vocab_size × d_model` parameter block, acting as a strong regularizer
and reducing the parameter count — described in the paper's Section 3.4.

---

## Why mixed precision (AMP) training?

FP32 uses 4 bytes per number; FP16 uses 2. Training in FP16 halves memory
bandwidth requirements and lets modern GPUs use dedicated Tensor Cores that
are 2-8x faster at FP16 matrix multiplication than FP32.

The risk is numerical: some operations (like the softmax sum in attention,
or gradient accumulation) can underflow or become unstable in FP16. AMP
handles this by keeping a master copy of weights in FP32, running the
forward/backward pass in FP16 where safe, and using a dynamically-adjusted
loss scaling factor (`GradScaler`) to prevent small gradients from
underflowing to zero during backpropagation.

---

## Why a custom data pipeline instead of torchtext?

The original implementation followed the data-loading pattern from the
Annotated Transformer reference (torchtext's `Multi30k` dataset loader,
`build_vocab_from_iterator`, `to_map_style_dataset`). `torchtext` was
deprecated by its maintainers in September 2023; its final release
(0.18.0) only supports PyTorch ≤2.3.0, and current Colab runtimes ship
PyTorch 2.4+ without it.

Rather than pin an old PyTorch version (which risks cascading conflicts
with other dependencies and doesn't fix anything long-term), `data_pipeline.py`
replaces exactly the three torchtext call sites with HuggingFace `datasets`
(actively maintained) for loading, and a minimal custom `Vocab` class that
mirrors torchtext's original interface (`vocab(tokens)`, `vocab["<tok>"]`,
`set_default_index`, `get_itos()`) so nothing downstream — masking, batching,
the model itself — needed to change. This was a deliberate scope decision:
fix the deprecated dependency, not rewrite a working training pipeline.

---

## Why ablation flags that default to the baseline?

RoPE, RMSNorm, SwiGLU and weight tying were each implemented and tested in
isolation, but an ablation needs them inside the *same* model and training
loop as the baseline, changing one thing at a time. `make_model()` therefore
takes `use_rope`, `use_rmsnorm`, `use_swiglu` and `tie_weights` (exposed in
`train.py` as `--use-rope` etc. and saved in each checkpoint's config).

Every flag defaults to off, and with all flags off the factory builds the
same modules in the same order as before. The baseline is bit-for-bit
unchanged: same parameter count, same `state_dict` keys, same weights
for a given seed. This keeps the existing checkpoints and reported results
valid, and a unit test pins the baseline parameter count. Checkpoints saved
before the flags existed load as the baseline because missing keys default
to off.

Two details:
- **SwiGLU keeps the given `d_ff`.** SwiGLU has three bias-free matrices vs
  the ReLU FFN's two, so at equal `d_ff` it has more parameters. For a
  parameter-matched comparison, pass `d_ff ≈ 2/3 × d_ff` explicitly; the
  factory doesn't silently rescale it.
- **Only the target embedding is tied.** Source and target use separate
  vocabularies here, so the source embedding has a different shape.

---

## Why RoPE only in self-attention, not cross-attention?

RoPE makes the attention score depend on the *relative* position `m − n`
between a query and a key. In self-attention both come from the same
sequence, so `m − n` is meaningful. In cross-attention, queries are target
positions and keys are source positions, so their difference has no
consistent meaning (German and English word order differ). With
`use_rope=True`, encoder and decoder self-attention use
`RoPEMultiHeadAttention`, cross-attention stays position-free, and the
sinusoidal positional encoding is removed so RoPE is the only source of
position information. A test checks that the model is still order-sensitive.

---

## Why a byte-level BPE tokenizer, and why opt-in?

The word-level vocabulary maps anything seen fewer than twice to `<unk>`,
which is the most consistent failure in `RESULTS.md` ("ice fishing hut").
BPE (Sennrich et al., 2016) builds a subword vocabulary by repeatedly
merging the most frequent adjacent pair of tokens. The *byte-level* variant
(as in GPT-2) starts from the 256 possible bytes rather than characters,
so every string can be encoded and `<unk>` is never produced. A
GPT-2-style pre-tokenizer keeps the leading space attached to each word,
which makes decoding plain concatenation: `decode(encode(s)) == s` for any
`s`.

Implementation choices in `bpe.py`:
- **Fast, exact training.** Pair counts are updated incrementally (only
  words containing the merged pair are touched) and the best pair comes
  from a lazy max-heap, with ties broken by token id so training is
  deterministic. A unit test checks the learned merges against a naive
  recount-everything implementation.
- **Fits the existing pipeline.** Special-token ids match the word-level
  vocab (`<s>`=0, `</s>`=1, `<blank>`=2, `<unk>`=3), and `to_vocab()`
  returns a `data_pipeline.Vocab`, so batching, masking and the model are
  unchanged. Merges that would spell a special token are blocked.
- **Opt-in.** `train.py` still uses spaCy words by default, because all
  reported numbers come from that setup. `--bpe` trains one tokenizer per
  language on the training split only and stores the merges in the
  checkpoint, so `evaluate.py` and `inference.py` tokenize and detokenize
  correctly.

---

## Why does the decoder-only model reuse `EncoderLayer`?

A GPT block is masked self-attention followed by an FFN, each wrapped in a
pre-norm residual. That is exactly `EncoderLayer`; the only difference is
the mask. So `make_decoder_only_model()` stacks `EncoderLayer`s in an
`Encoder` (layers + final norm), adds `Embeddings` + positional encoding
and a `Generator`, and `lm_mask()` supplies the causal (optionally
padded) mask. There is no cross-attention and no memory. Reusing the layers
means the decoder-only model gets the same ablation flags for free, and
with RoPE + RMSNorm + SwiGLU it matches the LLaMA block layout. Tests check
that changing a future token never changes earlier outputs.

---

## Why Grouped Query Attention?

The KV cache stores `2 × batch × n_kv_heads × seq_len × d_k` values per
layer, and with standard MHA `n_kv_heads = h`. GQA (Ainslie et al., 2023)
keeps all `h` query heads but has groups of `h / n_kv` query heads share
one K/V head (`n_kv = 1` is Multi-Query Attention). The cache and the K/V
projections shrink by `h / n_kv`, while quality stays close to MHA because
queries keep their full resolution.

`GroupedQueryAttention` keeps `MultiHeadAttention`'s call signature, caches
only the compact K/V heads, and expands them (`repeat_interleave`) only at
attention time. `from_mha()` converts a trained `MultiHeadAttention`. With
`num_kv_heads == h` it is an exact copy, which is how the tests check
equivalence. With fewer heads it mean-pools each group's K/V projections,
the checkpoint-conversion step from the GQA paper.
