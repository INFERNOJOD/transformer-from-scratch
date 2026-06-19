"""
modern_components.py — RMSNorm + SwiGLU FFN

These two components appear in every modern LLM:
LLaMA, LLaMA 2, Mistral, Gemma, PaLM, Qwen, etc.

They replace the original Transformer's LayerNorm and ReLU FFN.


IMPROVEMENT 1: RMSNorm
------------------------
Paper: "Root Mean Square Layer Normalization" — Zhang & Sennrich, 2019
       https://arxiv.org/abs/1910.07467

Original LayerNorm:
    y = (x - mean) / sqrt(var + ε) * γ + β
    → Computes both mean and variance
    → Has both scale (γ) and bias (β) parameters

RMSNorm:
    y = x / sqrt(mean(x²) + ε) * γ
    → Only computes RMS (root mean square), no mean subtraction
    → Only scale parameter γ, no bias β

WHY RMSNorm:
  - ~10% faster than LayerNorm (skip mean computation + bias)
  - Comparable or better training stability in practice
  - Used in: LLaMA 1/2/3, Mistral, Gemma, PaLM 2, Qwen
  - The bias term in LayerNorm is largely redundant (the FFN already has biases)

Interview question: "What's the difference between LayerNorm and RMSNorm?"
Answer: RMSNorm drops the mean subtraction and bias term, keeping only
RMS scaling. It's faster and works just as well empirically.


IMPROVEMENT 2: SwiGLU FFN
---------------------------
Paper: "GLU Variants Improve Transformer" — Noam Shazeer, 2020
       https://arxiv.org/abs/2002.05202

Original FFN:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂   ← ReLU activation

SwiGLU FFN:
    FFN(x) = (xW₁ ⊙ Swish(xW₂))W₃
    where Swish(x) = x · σ(x)  (sigmoid-weighted linear unit)
    and ⊙ is element-wise multiplication (gating)

WHY SwiGLU:
  - GLU = Gated Linear Unit: one branch gates (multiplies) the other
  - The gate learns WHICH parts of the representation to pass through
  - Swish is smoother than ReLU (differentiable everywhere, slightly negative for x<0)
  - Consistently outperforms ReLU/GELU FFN by ~+1 perplexity point
  - Used in: LLaMA 1/2/3, PaLM, Gemma, Mistral

Note: SwiGLU uses 3 weight matrices instead of 2, so d_ff is usually
set to 2/3 × original d_ff to keep parameter count equal.

Interview question: "What's the difference between the original FFN and SwiGLU?"
Answer: SwiGLU replaces the ReLU with a gating mechanism — one linear
projection gates another element-wise, allowing the network to selectively
suppress or pass through features. It uses Swish as the activation, which
is smoother than ReLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²) + ε)

    Faster than LayerNorm: no mean subtraction, no bias term.
    Drop-in replacement for LayerNorm in any Transformer.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))   # γ — learned scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS over the last dimension (feature dimension)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale

    def extra_repr(self) -> str:
        return f"d_model={self.scale.shape[0]}, eps={self.eps}"


# ─────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────

def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation: x · σ(x)
    Smooth, non-monotonic, slightly negative for x < 0.
    Also known as SiLU (Sigmoid Linear Unit) — same function.
    """
    return x * torch.sigmoid(x)
    # Equivalently: return F.silu(x)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU(x) = (xW₁ ⊙ Swish(xW₂)) W₃

    Three linear projections:
      W₁: d_model → d_ff   (gate projection)
      W₂: d_model → d_ff   (up projection)
      W₃: d_ff   → d_model (down projection)

    Parameter count vs original FFN (2 matrices, d_ff=2048):
      Original:  2 × 512 × 2048 = 2,097,152
      SwiGLU:    3 × 512 × 1365 = 2,097,920  (d_ff ≈ 2/3 × 2048 to match)

    The standard convention is d_ff = int(d_model * 8/3) for SwiGLU,
    but we accept any d_ff for flexibility.
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            # 8/3 × d_model keeps param count ≈ original FFN with 4× expansion
            d_ff = int(d_model * 8 / 3)
            # Round to nearest multiple of 64 for efficiency
            d_ff = (d_ff + 63) // 64 * 64

        self.d_ff = d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)   # W₁
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)   # W₂
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)   # W₃
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: Swish(xW₁) ⊙ xW₂
        gate = swish(self.gate_proj(x))   # Swish-gated branch
        up   = self.up_proj(x)            # Linear branch
        return self.down_proj(self.dropout(gate * up))

    def extra_repr(self) -> str:
        return f"d_ff={self.d_ff}"


# ─────────────────────────────────────────────
# LLaMA-style Transformer Block
# Combines RMSNorm + SwiGLU + RoPE attention
# ─────────────────────────────────────────────

class LLaMAStyleEncoderLayer(nn.Module):
    """
    Encoder layer using modern LLaMA-style components:
      - RMSNorm instead of LayerNorm
      - SwiGLU FFN instead of ReLU FFN
      - Pre-normalization (norm before sublayer, not after)

    This is architecturally similar to a LLaMA transformer block.

    Pre-norm vs post-norm:
      Original paper used post-norm: LayerNorm(x + Sublayer(x))
      LLaMA uses pre-norm:  x + Sublayer(RMSNorm(x))
      Pre-norm is more stable for deep networks and large learning rates.
    """

    def __init__(self, d_model: int, attn_module: nn.Module,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.attn = attn_module
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm self-attention with residual
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, h, h, mask))

        # Pre-norm FFN with residual
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))

        return x
