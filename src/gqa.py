"""
gqa.py — Grouped Query Attention (GQA)

Paper: "GQA: Training Generalized Multi-Query Transformer Models from
        Multi-Head Checkpoints" — Ainslie et al., 2023
        https://arxiv.org/abs/2305.13245
Related: "Fast Transformer Decoding: One Write-Head is All You Need"
         (Multi-Query Attention) — Shazeer, 2019

Used in: LLaMA 2 70B, LLaMA 3, Mistral, Gemma.

WHY GQA
--------
During autoregressive decoding the KV cache holds, per layer,
    2 (K and V) × batch × n_kv_heads × seq_len × d_k
values. In standard multi-head attention n_kv_heads = n_heads, and for long
contexts this cache dominates memory and memory bandwidth (see kv_cache.py).

GQA keeps all h query heads but lets groups of h / n_kv query heads share
one K/V head:

    n_kv_heads = h   → standard Multi-Head Attention (MHA)
    1 < n_kv_heads < h → Grouped Query Attention
    n_kv_heads = 1   → Multi-Query Attention (MQA)

The KV cache (and the K/V projection weights) shrink by a factor of
h / n_kv_heads, while quality stays close to MHA because queries keep
their full set of heads.

HOW IT WORKS
-------------
    Q = x W_Q  → (B, h,    T, d_k)
    K = x W_K  → (B, n_kv, T, d_k)     ← only n_kv heads are computed/cached
    V = x W_V  → (B, n_kv, T, d_k)
    K, V are repeated h / n_kv times along the head axis, then ordinary
    scaled dot-product attention runs over all h heads.
Query head i uses K/V head i // (h / n_kv).
"""

from typing import Optional

import torch
import torch.nn as nn

from kv_cache import KVCache
from model import MultiHeadAttention, scaled_dot_product_attention


class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with `num_kv_heads` shared key/value heads.

    Same call signature as MultiHeadAttention (query, key, value, mask), so
    it can replace self- or cross-attention in any layer. Optionally takes a
    KVCache for incremental decoding; the cache stores only the n_kv heads.
    """

    def __init__(self, h: int, d_model: int, num_kv_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        assert h % num_kv_heads == 0, "h must be divisible by num_kv_heads"
        self.h = h
        self.num_kv_heads = num_kv_heads
        self.group_size = h // num_kv_heads
        self.d_k = d_model // h

        self.q_proj = nn.Linear(d_model, h * self.d_k)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.out_proj = nn.Linear(h * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None   # stored for visualization

    def _split(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """(B, T, n_heads·d_k) → (B, n_heads, T, d_k)"""
        B = x.size(0)
        return x.view(B, -1, n_heads, self.d_k).transpose(1, 2)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_kv, T, d_k) → (B, h, T, d_k); query head i reads kv head i // group."""
        if self.group_size == 1:
            return x
        return x.repeat_interleave(self.group_size, dim=1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        B = query.size(0)

        q = self._split(self.q_proj(query), self.h)              # (B, h,    Tq, d_k)
        k = self._split(self.k_proj(key), self.num_kv_heads)     # (B, n_kv, Tk, d_k)
        v = self._split(self.v_proj(value), self.num_kv_heads)

        if cache is not None:
            # Cache the compact n_kv heads — this is where GQA saves memory
            k, v = cache.update(layer_id if layer_id is not None else 0, k, v)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)   # broadcast over heads

        x, self.attn_weights = scaled_dot_product_attention(
            q, self._expand_kv(k), self._expand_kv(v),
            mask=mask, dropout=self.dropout,
        )
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out_proj(x)

    def kv_cache_numel(self, batch_size: int, seq_len: int) -> int:
        """Number of values this layer caches for (batch_size, seq_len): K + V."""
        return 2 * batch_size * self.num_kv_heads * seq_len * self.d_k

    @classmethod
    def from_mha(cls, mha: MultiHeadAttention,
                 num_kv_heads: Optional[int] = None) -> "GroupedQueryAttention":
        """
        Convert a trained MultiHeadAttention into GQA.

        With num_kv_heads == mha.h (the default) the weights are copied and
        the result computes exactly the same function. With fewer KV heads,
        each group's K/V projection heads are mean-pooled — the
        checkpoint-conversion step from the GQA paper (normally followed by
        a short "uptraining" phase).
        """
        h, d_k = mha.h, mha.d_k
        num_kv_heads = num_kv_heads or h
        d_model = h * d_k
        gqa = cls(h, d_model, num_kv_heads, dropout=mha.dropout.p)
        gqa = gqa.to(mha.linears[0].weight.device, mha.linears[0].weight.dtype)
        group = h // num_kv_heads

        def pool(linear: nn.Linear):
            # weight rows are grouped per head: (h·d_k, d_model) → (n_kv·d_k, d_model)
            w = linear.weight.view(num_kv_heads, group, d_k, d_model).mean(1)
            b = linear.bias.view(num_kv_heads, group, d_k).mean(1)
            return w.reshape(-1, d_model), b.reshape(-1)

        q_lin, k_lin, v_lin, o_lin = mha.linears
        with torch.no_grad():
            gqa.q_proj.weight.copy_(q_lin.weight)
            gqa.q_proj.bias.copy_(q_lin.bias)
            for dst, src in ((gqa.k_proj, k_lin), (gqa.v_proj, v_lin)):
                w, b = pool(src)
                dst.weight.copy_(w)
                dst.bias.copy_(b)
            gqa.out_proj.weight.copy_(o_lin.weight)
            gqa.out_proj.bias.copy_(o_lin.bias)
        return gqa


def kv_cache_reduction(h: int, num_kv_heads: int) -> float:
    """How many times smaller the KV cache is than standard MHA's."""
    return h / num_kv_heads


if __name__ == "__main__":
    # Cache size for a LLaMA-2-7B-like layer shape (h=32, d_k=128), batch 1,
    # 4096 tokens, fp16 — a pure shape calculation, not a measurement.
    h, d_k, seq = 32, 128, 4096
    print(f"KV cache per layer (batch=1, seq={seq}, fp16):")
    for n_kv in (32, 8, 4, 1):
        attn = GroupedQueryAttention(h, h * d_k, n_kv, dropout=0.0)
        mb = attn.kv_cache_numel(1, seq) * 2 / 2**20
        name = "MHA" if n_kv == h else ("MQA" if n_kv == 1 else "GQA")
        print(f"  {name:>3} n_kv={n_kv:>2}: {mb:7.1f} MB  "
              f"({kv_cache_reduction(h, n_kv):.0f}x reduction vs MHA)")
