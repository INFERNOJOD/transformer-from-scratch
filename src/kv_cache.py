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

This is why KV caching is critical for production LLM inference.
GPT, Claude, Gemini all use this.

MEMORY TRADEOFF
----------------
Cached K,V tensors take memory proportional to:
  batch_size × n_layers × n_heads × seq_len × d_k × 2 (K and V)

For long contexts this dominates GPU memory — hence techniques like
grouped-query attention (GQA) and multi-query attention (MQA)
that reduce the number of K,V heads vs Q heads.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict
import copy


class KVCache:
    """
    Stores accumulated K and V tensors for each decoder layer and head.

    Usage:
        cache = KVCache()
        # First call — empty cache
        k_full, v_full = cache.update(layer_id=0, new_k=k, new_v=v)
        # Second call — appends to cache
        k_full, v_full = cache.update(layer_id=0, new_k=k2, new_v=v2)
        # k_full is now [k, k2] concatenated along seq dimension
    """

    def __init__(self):
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

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

    def clear(self):
        self._cache.clear()

    def __len__(self):
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


def greedy_decode_cached(model, src: torch.Tensor, src_mask: torch.Tensor,
                         max_len: int, start_symbol: int, end_symbol: int,
                         device) -> torch.Tensor:
    """
    Greedy decode with KV cache.

    Each step only runs the new token through the decoder and appends to cache,
    instead of re-running the full sequence from scratch.

    For a fair comparison, you'd time this vs the uncached version on long sequences.
    Speedup is most dramatic for long outputs (50+ tokens).
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)

        # One KV cache per decoder layer
        cache = KVCache()
        generated = [start_symbol]

        for step in range(max_len - 1):
            # Feed only the most recent token (with cache, that's all we need)
            tgt = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)

            # No causal mask needed — cache handles ordering implicitly
            # (we only ever pass the new token as query)
            out = model.decode(memory, src_mask, tgt, tgt_mask=None)

            logits = model.generator(out[:, -1])
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)

            if next_token == end_symbol:
                break

    return torch.tensor(generated, dtype=torch.long)
