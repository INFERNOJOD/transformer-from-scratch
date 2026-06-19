"""
rope.py — Rotary Positional Embeddings (RoPE)

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
       Su et al., 2021 — https://arxiv.org/abs/2104.09864

Used in: LLaMA, GPT-NeoX, PaLM, Mistral, Gemma — basically every modern LLM.

WHY RoPE IS BETTER THAN SINUSOIDAL PE
--------------------------------------
Sinusoidal PE (original Transformer):
  - Adds position info to the token embedding BEFORE attention
  - The position info gets mixed in with content info
  - Relative position is implicit, not explicit

RoPE:
  - Encodes position by ROTATING the Q and K vectors before dot product
  - The dot product Q·K naturally becomes a function of (position_i - position_j)
  - Relative position is EXPLICIT in the attention score
  - Works better for long sequences and generalizes beyond training length

HOW IT WORKS (the key insight)
--------------------------------
In 2D, rotating a vector by angle θ means:
    [x, y] → [x·cos(θ) - y·sin(θ), x·sin(θ) + y·cos(θ)]

RoPE applies this in pairs of dimensions. For position m:
    q_rotated[2i]   = q[2i]·cos(m·θᵢ) - q[2i+1]·sin(m·θᵢ)
    q_rotated[2i+1] = q[2i]·sin(m·θᵢ) + q[2i+1]·cos(m·θᵢ)

where θᵢ = 1 / 10000^(2i/d_k)  (same frequencies as sinusoidal PE)

The magic: when you compute q_m · k_n (dot product in attention),
the result depends only on (m - n) — the RELATIVE position.
"""

import torch
import torch.nn as nn
import math


def precompute_rope_freqs(d_k: int, max_seq_len: int, base: float = 10000.0,
                          device=None) -> tuple:
    """
    Precompute cos and sin tables for RoPE.

    Returns:
        cos_cached: (max_seq_len, d_k)
        sin_cached: (max_seq_len, d_k)
    """
    # θᵢ = 1 / 10000^(2i/d_k)  for i = 0, 1, ..., d_k/2 - 1
    half = d_k // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))

    # positions: (max_seq_len,)
    positions = torch.arange(max_seq_len, device=device).float()

    # outer product → (max_seq_len, d_k/2)
    freqs = torch.outer(positions, theta)

    # duplicate each freq for both dims of each pair → (max_seq_len, d_k)
    freqs = torch.cat([freqs, freqs], dim=-1)

    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate each pair (x[2i], x[2i+1]) by 90°:
        [-x[d_k/2:], x[:d_k/2]]
    This is the efficient way to implement the rotation matrix multiplication.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.

    Args:
        x:   (batch, heads, seq_len, d_k)
        cos: (seq_len, d_k)
        sin: (seq_len, d_k)
    Returns:
        rotated x with same shape
    """
    seq_len = x.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)   # (1, 1, seq, d_k)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Positional Embeddings.

    Drop-in replacement for the original MultiHeadAttention.
    Position info is baked into Q and K via rotation — V is unchanged.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        import copy
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

        # Precompute RoPE tables (not parameters — saved as buffers)
        cos, sin = precompute_rope_freqs(self.d_k, max_seq_len)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        B = query.size(0)

        # Project → (B, heads, seq, d_k)
        def reshape(t, proj):
            return proj(t).view(B, -1, self.h, self.d_k).transpose(1, 2)

        q = reshape(query, self.q_proj)
        k = reshape(key, self.k_proj)
        v = reshape(value, self.v_proj)

        # Apply RoPE to Q and K only
        q = apply_rope(q, self.cos_cached, self.sin_cached)
        k = apply_rope(k, self.cos_cached, self.sin_cached)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        self.attn_weights = torch.softmax(scores, dim=-1)
        attn = self.dropout(self.attn_weights)
        out = torch.matmul(attn, v)

        # Concat heads → (B, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out_proj(out)
