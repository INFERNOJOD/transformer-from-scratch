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
where there's no clean "source" / "target" split — see `LIMITATIONS.md` for
why this is the natural next architecture to implement.

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
tokens, another tracks rare long-range references — see `attention_sample.png`
for an illustration of this with synthetic data).

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
shrink the cache further (see `LIMITATIONS.md`).

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
