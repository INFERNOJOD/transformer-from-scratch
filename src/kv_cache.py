"""
kv_cache.py — Key-Value Cache for Fast Autoregressive Inference

WHY KV CACHE EXISTS
--------------------
During decoding, at step t we generate token t+1.
To do so, we run the decoder on the full sequence [t_0, t_1, ..., t_t].

Without cache: EVERY STEP recomputes K and V for ALL previous tokens.
  Step 1: compute K,V for [t_0]              → 1 token
  Step 2: compute K,V for [t_0, t_1]         → 2 tokens
  Step 3: compute K,V for [t_0, t_1, t_2]    → 3 tokens
  Total work: O(n²) — quadratic in sequence length.

With KV cache: compute K,V for ONE new token per step, reuse the rest.
  Step 1: compute K,V for [t_0], store in cache
  Step 2: compute K,V for [t_1] only, concat with cached → same result
  Step 3: compute K,V for [t_2] only, concat with cached → same result
  Total work: O(n) — linear. Massive speedup for long sequences.

This is why KV caching is standard in production LLM inference.

MEMORY TRADEOFF
----------------
Cached K,V tensors take memory proportional to:
  batch_size × n_layers × n_heads × seq_len × d_k × 2 (K and V)

For long contexts this dominates GPU memory — hence techniques like
grouped-query attention (GQA) and multi-query attention (MQA)
that reduce the number of K,V heads vs Q heads.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class KVCache:
    """
    Stores accumulated K and V tensors for each decoder layer and head.

    Two kinds of entries:
      - self-attention K,V (update): grow by the new token(s) every step.
      - cross-attention K,V (static): projected from the encoder output once
        on the first step and reused unchanged afterwards.

    Usage:
        cache = KVCache()
        # First call — empty cache
        k_full, v_full = cache.update(layer_id=0, new_k=k, new_v=v)
        # Second call — appends to cache
        k_full, v_full = cache.update(layer_id=0, new_k=k2, new_v=v2)
        # k_full is now [k, k2] concatenated along seq dimension

    To decode with an EncoderDecoder, pass the cache to model.decode() and
    feed only the new token each step (see greedy_decode_cached).
    """

    def __init__(self):
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._static: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def update(self, layer_id: int,
               new_k: torch.Tensor,
               new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new_k, new_v to the cache for this layer.
        Returns the full accumulated K, V tensors.

        new_k, new_v: (batch, heads, 1, d_k)  — just the new token
        returns:      (batch, heads, t, d_k)   — all tokens so far
        """
        if layer_id not in self._cache:
            self._cache[layer_id] = (new_k, new_v)
        else:
            cached_k, cached_v = self._cache[layer_id]
            full_k = torch.cat([cached_k, new_k], dim=2)  # concat on seq dim
            full_v = torch.cat([cached_v, new_v], dim=2)
            self._cache[layer_id] = (full_k, full_v)

        return self._cache[layer_id]

    def static(self, layer_id: int,
               compute: Callable[[], Tuple[torch.Tensor, torch.Tensor]]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention K, V for this layer: call compute() on the first
        request, then return the stored tensors on every later step.
        """
        if layer_id not in self._static:
            self._static[layer_id] = compute()
        return self._static[layer_id]

    def seq_len(self, layer_id: int = 0) -> int:
        """Number of tokens cached in this layer's self-attention (0 if empty)."""
        if layer_id not in self._cache:
            return 0
        return self._cache[layer_id][0].size(2)

    def clear(self) -> None:
        self._cache.clear()
        self._static.clear()

    def numel(self) -> int:
        """Total number of cached self-attention K and V values across all layers."""
        return sum(k.numel() + v.numel() for k, v in self._cache.values())

    def __len__(self) -> int:
        """Number of layers with cached self-attention K, V."""
        return len(self._cache)


class CachedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention that supports KV caching during inference.

    During training (cache=None): behaves exactly like the original.
    During inference  (cache provided): only computes K,V for the new token,
    retrieves the rest from cache.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

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

        def proj_reshape(x, proj):
            return proj(x).view(B, -1, self.h, self.d_k).transpose(1, 2)

        q = proj_reshape(query, self.q_proj)   # (B, h, q_len, d_k)

        if cache is not None and layer_id is not None:
            # Inference with cache:
            # Only project K,V for the new query token
            new_k = proj_reshape(key, self.k_proj)    # (B, h, 1, d_k)
            new_v = proj_reshape(value, self.v_proj)
            # Get full K,V (past + current)
            k, v = cache.update(layer_id, new_k, new_v)
        else:
            # Training: project all tokens normally
            k = proj_reshape(key, self.k_proj)
            v = proj_reshape(value, self.v_proj)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        self.attn_weights = torch.softmax(scores, dim=-1)
        attn = self.dropout(self.attn_weights)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out_proj(out)


def greedy_decode_cached(model: nn.Module, src: torch.Tensor,
                         src_mask: torch.Tensor, max_len: int,
                         start_symbol: int, end_symbol: int,
                         device: torch.device) -> torch.Tensor:
    """
    Greedy decode with KV cache. Produces the same tokens as the uncached
    greedy_decode in inference.py: (len,) ids starting with start_symbol.

    Each step feeds only the newest token to model.decode() together with
    the cache. Every decoder layer's self-attention projects K,V for that
    one token, appends them to the cache and attends over all cached
    positions (the new token is last, so no causal mask is needed).
    Cross-attention K,V are projected from the encoder output on the first
    step and reused from the cache afterwards. The positional encoding is
    offset by the number of cached tokens.

    Speedup over the uncached version grows with output length.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)

        # One cache holds K,V for every decoder layer
        cache = KVCache()
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # Feed only the most recent token; past K,V come from the cache
            out = model.decode(memory, src_mask, ys[:, -1:], None, cache=cache)

            logits = model.generator(out[:, -1])
            next_token = logits.argmax(dim=-1).unsqueeze(0)
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == end_symbol:
                break

    return ys.squeeze(0)
