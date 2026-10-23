"""
model.py — Transformer Encoder-Decoder

Paper: "Attention Is All You Need" — Vaswani et al., 2017
       https://arxiv.org/abs/1706.03762

Figure 1 — Full Architecture:

                        OUTPUT PROBABILITIES
                              ↑
                           Softmax
                              ↑
                           Linear
                              ↑
                        ┌─────────┐
                        │Add&Norm │  ← (x6)
                        └────┬────┘
                          Feed
                         Forward
                        ┌────┴────┐
                        │Add&Norm │
                        └────┬────┘
                        Multi-Head          ← Cross-attention (queries from decoder,
                         Attention             keys/values from encoder output)
                        ┌────┴────┐
                        │Add&Norm │
                        └────┬────┘
                        Masked Multi-       ← Self-attention with causal mask
                        Head Attention
                              ↑
                       Output Embedding
                        + Pos Encoding
                              ↑
                           OUTPUTS
                        (shifted right)


    ENCODER (left)              DECODER (right)
    ──────────────              ───────────────
    ┌─────────┐                 (described above)
    │Add&Norm │  ← (x6)
    └────┬────┘
      Feed
     Forward
    ┌────┴────┐
    │Add&Norm │
    └────┬────┘
    Multi-Head
     Attention  ← Self-attention
          ↑
   Input Embedding
    + Pos Encoding
          ↑
        INPUTS
"""

import math
import copy
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modern_components import RMSNorm, SwiGLUFeedForward
from optimizations import apply_weight_tying
from rope import RoPEMultiHeadAttention

# Architecture toggles accepted by make_model(); all False = paper baseline.
ABLATION_FLAGS = ("use_rope", "use_rmsnorm", "use_swiglu", "tie_weights")

# A norm factory: called with the feature size, returns a norm module.
NormFactory = Callable[[int], nn.Module]


def ablation_kwargs(config: Dict) -> Dict[str, bool]:
    """
    Pull make_model()'s architecture flags out of a saved config dict.
    Missing keys default to False, so checkpoints saved before these flags
    existed rebuild as the baseline model.
    """
    return {flag: bool(config.get(flag, False)) for flag in ABLATION_FLAGS}


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Causal mask: position i can only attend to positions ≤ i.
    Returns a (1, size, size) boolean tensor.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).bool()
    return ~mask   # True = allowed to attend


# ─────────────────────────────────────────────
# 1. Scaled Dot-Product Attention
# ─────────────────────────────────────────────

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Scaling by sqrt(d_k) prevents vanishingly small gradients when d_k is large
    (dot products grow in magnitude → softmax saturates).

    Args:
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  boolean tensor broadcastable to (batch, heads, seq_q, seq_k)
               True = keep, False = mask out
    Returns:
        output: (batch, heads, seq_q, d_v)
        attn_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    return torch.matmul(attn_weights, value), attn_weights


# ─────────────────────────────────────────────
# 2. Multi-Head Attention
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Uses h=8 heads and d_k = d_v = d_model/h = 64 (for d_model=512).
    Total compute is similar to single-head attention at full dimension.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h = h

        # 4 linear projections: W^Q, W^K, W^V, W^O
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None   # stored for visualization

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        if mask is not None:
            mask = mask.unsqueeze(1)   # broadcast over heads

        # 1) Project Q, K, V → (batch, h, seq, d_k)
        query, key, value = [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]

        # 2) Attention over all heads
        x, self.attn_weights = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) Concat heads: (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # 4) Final linear W^O
        return self.linears[3](x)


# ─────────────────────────────────────────────
# 3. Position-wise Feed-Forward Network
# ─────────────────────────────────────────────

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    Applied identically to each position.
    d_model=512 → d_ff=2048 → d_model=512
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))


# ─────────────────────────────────────────────
# 4. Layer Norm + Residual (Sublayer Connection)
# ─────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Standard layer normalization with learned scale (a) and bias (b)."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    """
    Residual connection + layer norm (pre-norm):
        output = x + Dropout(Sublayer(LayerNorm(x)))

    The paper uses post-norm, LayerNorm(x + Sublayer(x)); the norm is applied
    *before* the sublayer here for training stability, following the
    Annotated Transformer.
    """

    def __init__(self, size: int, dropout: float,
                 norm_cls: NormFactory = LayerNorm):
        super().__init__()
        self.norm = norm_cls(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


# ─────────────────────────────────────────────
# 5. Encoder
# ─────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    One encoder layer = Self-Attention → FFN, each wrapped in Add&Norm.
    """

    def __init__(self, size: int, self_attn: nn.Module,
                 feed_forward: nn.Module, dropout: float,
                 norm_cls: NormFactory = LayerNorm):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, norm_cls), 2)
        self.size = size
        self.norm_cls = norm_cls

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Stack of N encoder layers followed by a final layer norm."""

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = layer.norm_cls(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 6. Decoder
# ─────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    One decoder layer = Masked Self-Attention → Cross-Attention → FFN.
    Each sub-layer wrapped in Add&Norm.
    """

    def __init__(self, size: int, self_attn: nn.Module,
                 cross_attn: nn.Module, feed_forward: nn.Module,
                 dropout: float, norm_cls: NormFactory = LayerNorm):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, norm_cls), 3)
        self.size = size
        self.norm_cls = norm_cls

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # 1) Masked self-attention (causal)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) Cross-attention: Q from decoder, K/V from encoder output
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        # 3) Position-wise FFN
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """Stack of N decoder layers followed by a final layer norm."""

    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = layer.norm_cls(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 7. Embeddings & Positional Encoding
# ─────────────────────────────────────────────

class Embeddings(nn.Module):
    """
    Learned token embeddings scaled by sqrt(d_model).
    Scaling keeps embedding magnitudes comparable to positional encodings.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (fixed, not learned):

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Each dimension corresponds to a sinusoid with a different frequency.
    Allows the model to attend by relative positions: PE(pos+k) is a linear
    function of PE(pos) for any fixed offset k.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                  # (1, max_len, d_model)
        self.register_buffer("pe", pe)        # not a parameter, but saved with model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ─────────────────────────────────────────────
# 8. Generator (final linear + log-softmax)
# ─────────────────────────────────────────────

class Generator(nn.Module):
    """Linear projection from d_model → vocab, then log-softmax."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


# ─────────────────────────────────────────────
# 9. Full Encoder-Decoder Model
# ─────────────────────────────────────────────

class EncoderDecoder(nn.Module):
    """
    Full Transformer encoder-decoder as described in Figure 1 of the paper.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: nn.Sequential, tgt_embed: nn.Sequential,
                 generator: Generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


# ─────────────────────────────────────────────
# 10. Model Factory
# ─────────────────────────────────────────────

def make_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    N: int = 6,          # number of encoder/decoder layers
    d_model: int = 512,  # embedding / model dimension
    d_ff: int = 2048,    # inner FFN dimension
    h: int = 8,          # attention heads
    dropout: float = 0.1,
    use_rope: bool = False,     # RoPE self-attention instead of sinusoidal PE
    use_rmsnorm: bool = False,  # RMSNorm instead of LayerNorm
    use_swiglu: bool = False,   # SwiGLU FFN instead of ReLU FFN
    tie_weights: bool = False,  # share tgt embedding with generator projection
) -> EncoderDecoder:
    """
    Construct a Transformer from hyperparameters.
    Colab-friendly small model (train.py): N=3, d_model=256, d_ff=512, h=4
    Full paper base model:                 N=6, d_model=512, d_ff=2048, h=8

    With every flag False (the default) this builds exactly the baseline
    model used for the reported results. The flags swap in the modules
    from rope.py, modern_components.py and optimizations.py:

      use_rope     Encoder and decoder self-attention use
                   RoPEMultiHeadAttention and the sinusoidal positional
                   encoding is dropped (embeddings keep their dropout).
                   Cross-attention stays position-free: queries and keys
                   come from different sequences, so a relative rotation
                   between them has no meaning.
      use_rmsnorm  Every LayerNorm (sublayers + final encoder/decoder
                   norms) becomes RMSNorm.
      use_swiglu   PositionwiseFeedForward becomes SwiGLUFeedForward with
                   the same d_ff (3 bias-free matrices, so more parameters
                   than the 2-matrix ReLU FFN at equal d_ff).
      tie_weights  Generator projection shares the target embedding matrix
                   (paper §3.4). Source and target vocabularies are
                   separate, so the source embedding is not tied.
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)
    self_attn = (RoPEMultiHeadAttention(h, d_model, dropout, max_seq_len=5000)
                 if use_rope else attn)
    ff = (SwiGLUFeedForward(d_model, d_ff, dropout) if use_swiglu
          else PositionwiseFeedForward(d_model, d_ff, dropout))
    pos_enc = nn.Dropout(dropout) if use_rope else PositionalEncoding(d_model, dropout)
    norm_cls = RMSNorm if use_rmsnorm else LayerNorm

    model = EncoderDecoder(
        encoder=Encoder(
            EncoderLayer(d_model, c(self_attn), c(ff), dropout, norm_cls), N
        ),
        decoder=Decoder(
            DecoderLayer(d_model, c(self_attn), c(attn), c(ff), dropout, norm_cls), N
        ),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab_size), c(pos_enc)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(pos_enc)),
        generator=Generator(d_model, tgt_vocab_size),
    )

    # Xavier uniform init (important for stable training)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if tie_weights:
        apply_weight_tying(model)

    return model
