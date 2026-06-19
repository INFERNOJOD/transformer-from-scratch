"""
flash_attention.py — Flash Attention

Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with
        IO-Awareness" — Dao et al., 2022
        https://arxiv.org/abs/2205.14135

WHY STANDARD ATTENTION IS SLOW
---------------------------------
Standard attention computes:
    scores = Q @ K.T / sqrt(d_k)          → (seq, seq) matrix  ← PROBLEM
    weights = softmax(scores)              → (seq, seq) matrix
    output  = weights @ V

The (seq, seq) matrix is written to and read from GPU HBM (slow memory).
For seq_len=2048, d_model=512: that's 2048² × 4 bytes = 16MB per layer.
With 6 layers, 2 passes (fwd + bwd): hundreds of MB of slow memory traffic.

HOW FLASH ATTENTION FIXES IT
------------------------------
Key insight: rewrite attention to never materialise the full (seq, seq) matrix.

Instead, process Q, K, V in TILES that fit in SRAM (fast memory).
Use the online softmax trick to accumulate the correct result tile by tile.

Result:
  - Same mathematical output as standard attention (exact, not approximate)
  - Memory: O(seq) instead of O(seq²)
  - Speed: 2-4× faster on A100 GPUs
  - Enables much longer context windows

This is why GPT-4, Claude, LLaMA 2+ all use Flash Attention.

IMPLEMENTATION APPROACH
-------------------------
A true from-scratch Flash Attention requires CUDA kernels.
We implement three levels here:

1. FlashAttentionV1   — Pure PyTorch tiled implementation (educational)
2. FlashAttentionV2   — Uses torch.nn.functional.scaled_dot_product_attention
                        (which calls CUDA Flash Attention if available)
3. FlashMultiHeadAttn — Drop-in replacement for MultiHeadAttention

In production, always use level 2 (PyTorch's built-in).
Level 1 is here so you can explain the algorithm in interviews.
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# Level 1: Tiled attention (educational)
# Shows the core idea without CUDA kernels
# ─────────────────────────────────────────────

def flash_attention_tiled(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Tiled attention: processes Q, K, V in blocks to reduce peak memory.

    This is a simplified version of the Flash Attention algorithm.
    It avoids storing the full (seq, seq) attention matrix by computing
    softmax incrementally using the online softmax trick.

    Online softmax trick:
        Instead of computing softmax over all scores at once,
        we maintain running max (m) and running sum (l) as we process blocks:
            m_new = max(m_old, max(new_scores))
            l_new = exp(m_old - m_new) * l_old + sum(exp(new_scores - m_new))
        This is mathematically identical to standard softmax.

    Args:
        Q, K, V: (batch, heads, seq_len, d_k)
        block_size: tile size (should fit in L2 cache / SRAM)
    Returns:
        output: (batch, heads, seq_len, d_k)
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Output accumulator and softmax statistics — one entry per query position
    O = torch.zeros_like(Q)                                                   # (B, H, N, d)
    L = torch.zeros(B, H, N, device=Q.device, dtype=Q.dtype)                 # normalizer
    M = torch.full((B, H, N), float("-inf"), device=Q.device, dtype=Q.dtype) # running row max

    # Outer loop: tile over query positions
    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        Qi = Q[:, :, i_start:i_end, :]          # (B, H, q_blk, d)
        q_blk = i_end - i_start

        # Accumulators for this query block
        Oi = torch.zeros(B, H, q_blk, d, device=Q.device, dtype=Q.dtype)
        Li = torch.zeros(B, H, q_blk, device=Q.device, dtype=Q.dtype)
        Mi = torch.full((B, H, q_blk), float("-inf"), device=Q.device, dtype=Q.dtype)

        # Inner loop: tile over key/value positions
        for j_start in range(0, N, block_size):
            j_end = min(j_start + block_size, N)
            Kj = K[:, :, j_start:j_end, :]      # (B, H, k_blk, d)
            Vj = V[:, :, j_start:j_end, :]

            # Scores for this (query_block × key_block) tile
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale  # (B,H,q_blk,k_blk)

            if mask is not None:
                mask_block = mask[:, :, i_start:i_end, j_start:j_end]
                Sij = Sij.masked_fill(mask_block == 0, float("-inf"))

            # Online softmax: update running max
            Mij_new = torch.maximum(Mi, Sij.amax(dim=-1))   # new row max

            # Rescale previous accumulator with correction factor
            rescale = torch.exp(Mi - Mij_new)                # (B, H, q_blk)
            Pij = torch.exp(Sij - Mij_new.unsqueeze(-1))     # unnormalised weights

            Oi = Oi * rescale.unsqueeze(-1) + torch.matmul(Pij, Vj)
            Li = Li * rescale + Pij.sum(dim=-1)
            Mi = Mij_new

        # Write normalised result back for this query block
        O[:, :, i_start:i_end, :] = Oi / Li.unsqueeze(-1)

    return O


# ─────────────────────────────────────────────
# Level 2: PyTorch built-in (production use)
# Uses CUDA Flash Attention kernel when available
# ─────────────────────────────────────────────

def flash_attention_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Flash Attention via torch.nn.functional.scaled_dot_product_attention.

    PyTorch >= 2.0 automatically dispatches to:
      - Flash Attention CUDA kernel (if on GPU + correct dtype)
      - Memory-efficient attention (fallback)
      - Math attention (CPU fallback)

    This is the one-line way to get Flash Attention in production.
    Same outputs as standard attention, just faster and more memory-efficient.
    """
    # PyTorch expects attn_mask: True = attend, or additive float mask
    attn_mask = None
    if mask is not None:
        # Convert bool mask to additive: 0 where attend, -inf where mask
        attn_mask = torch.zeros_like(mask, dtype=Q.dtype)
        attn_mask = attn_mask.masked_fill(mask == 0, float("-inf"))

    return torch.nn.functional.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=attn_mask,
        dropout_p=dropout_p if Q.requires_grad else 0.0,
        scale=1.0 / math.sqrt(Q.size(-1)),
    )


# ─────────────────────────────────────────────
# Level 3: Drop-in MultiHeadAttention with Flash Attention
# ─────────────────────────────────────────────

class FlashMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using Flash Attention backend.
    Drop-in replacement for the original MultiHeadAttention.

    On GPU with PyTorch >= 2.0: uses CUDA Flash Attention automatically.
    On CPU: falls back to standard (tiled) attention.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1,
                 use_tiled: bool = False):
        """
        Args:
            use_tiled: if True, uses the educational tiled implementation.
                       if False (default), uses PyTorch's optimised backend.
        """
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.dropout_p = dropout
        self.use_tiled = use_tiled

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B = query.size(0)

        def reshape(x, proj):
            return proj(x).view(B, -1, self.h, self.d_k).transpose(1, 2)

        Q = reshape(query, self.q_proj)
        K = reshape(key,   self.k_proj)
        V = reshape(value, self.v_proj)

        if self.use_tiled:
            out = flash_attention_tiled(Q, K, V, mask=mask)
        else:
            out = flash_attention_pytorch(
                Q, K, V, mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out_proj(out)


def benchmark_attention(seq_len: int = 512, d_model: int = 256,
                        h: int = 4, batch: int = 8, device: str = "cpu"):
    """
    Compare standard attention vs Flash Attention on speed and memory.
    Run this to generate benchmark numbers for your README.
    """
    import time

    d_k = d_model // h
    Q = torch.randn(batch, h, seq_len, d_k, device=device)
    K = torch.randn(batch, h, seq_len, d_k, device=device)
    V = torch.randn(batch, h, seq_len, d_k, device=device)

    def time_fn(fn, n=20):
        # warmup
        for _ in range(3):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000  # ms

    from model import scaled_dot_product_attention as standard_attn

    t_standard = time_fn(lambda: standard_attn(Q, K, V))
    t_tiled    = time_fn(lambda: flash_attention_tiled(Q, K, V))
    t_flash    = time_fn(lambda: flash_attention_pytorch(Q, K, V))

    print(f"\nAttention Benchmark (seq={seq_len}, d_model={d_model}, batch={batch})")
    print(f"  Standard attention : {t_standard:.2f} ms")
    print(f"  Tiled (educational): {t_tiled:.2f} ms")
    print(f"  Flash (PyTorch)    : {t_flash:.2f} ms")
    print(f"  Speedup (Flash/Std): {t_standard/t_flash:.2f}x")

    return {"standard": t_standard, "tiled": t_tiled, "flash": t_flash}


if __name__ == "__main__":
    benchmark_attention()
