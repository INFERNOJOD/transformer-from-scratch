"""
optimizations.py — Mixed Precision Training + Weight Tying

IMPROVEMENT 1: WEIGHT TYING
------------------------------
The paper (Section 3.4) says:
  "We share the same weight matrix between the two embedding layers
   and the pre-softmax linear transformation."

The original model.py doesn't implement this. This module adds it.

WHY IT WORKS:
  - The embedding matrix maps token_id → vector  (vocab_size × d_model)
  - The generator's linear layer maps vector → token_id  (d_model × vocab_size)
  - These are approximate transposes of each other
  - Sharing weights reduces parameters significantly:
      Without tying: 2 × vocab_size × d_model  extra params
      With tying:    0 extra params (same matrix reused)
  - Also acts as regularization — the model can't learn contradictory
    representations in embedding vs output projection

IMPLEMENTATION: one line of code, huge conceptual insight.
    model.generator.proj.weight = model.tgt_embed[0].lut.weight


IMPROVEMENT 2: MIXED PRECISION (AMP)
--------------------------------------
Standard training uses FP32 (32-bit floats) for everything.
Mixed precision uses FP16 (16-bit) for most ops, FP32 only where needed.

WHY:
  - FP16 tensors use 2× less memory → larger batch sizes
  - FP16 matrix multiply is 2-8× faster on modern GPUs (Tensor Cores)
  - PyTorch's GradScaler handles the instability of FP16 gradients

HOW IT WORKS:
  - Forward pass and loss computed in FP16
  - Loss scaled up by a large factor (e.g. 2^16) before backward
  - Gradients scaled back down before optimizer step
  - Scaling factor adjusted automatically if gradients overflow (NaN/Inf)

In practice: ~1.5-2× training speedup, same final model quality.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


# ─────────────────────────────────────────────
# Weight Tying
# ─────────────────────────────────────────────

def apply_weight_tying(model) -> None:
    """
    Tie the target embedding weights to the generator projection weights.

    This is the improvement described in Section 3.4 of the paper.
    Call this immediately after make_model().

    Args:
        model: EncoderDecoder instance

    The generator.proj.weight has shape (vocab_size, d_model).
    The tgt_embed[0].lut.weight has shape (vocab_size, d_model).
    They're the same shape — we make them the exact same tensor object.
    """
    # tgt_embed is nn.Sequential(Embeddings(...), PositionalEncoding(...))
    # tgt_embed[0] is the Embeddings module
    # tgt_embed[0].lut is the nn.Embedding
    model.generator.proj.weight = model.tgt_embed[0].lut.weight

    n_tied = model.generator.proj.weight.numel()
    print(f"Weight tying applied — {n_tied:,} parameters shared "
          f"(src_embed, tgt_embed, and generator projection)")


def count_parameters(model) -> int:
    """Count unique trainable parameters (accounts for weight tying)."""
    # Using a set of ids to avoid double-counting shared tensors
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen and p.requires_grad:
            seen.add(id(p))
            total += p.numel()
    return total


# ─────────────────────────────────────────────
# Mixed Precision Training Loop
# ─────────────────────────────────────────────

class AMPTrainer:
    """
    Wraps a training step with Automatic Mixed Precision (AMP).

    Usage:
        trainer = AMPTrainer(model, optimizer, scheduler)
        loss = trainer.step(batch, criterion)
    """

    def __init__(self, model, optimizer, scheduler,
                 accum_iter: int = 4, clip_grad: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accum_iter = accum_iter
        self.clip_grad = clip_grad

        # GradScaler: automatically manages the loss scaling factor
        # Only active if CUDA is available (no-op on CPU)
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self._step_count = 0

    def step(self, batch, criterion) -> float:
        """
        Run one training step with AMP.

        Returns:
            loss value (float)
        """
        # autocast: runs forward pass in FP16 where safe, FP32 elsewhere
        with autocast(enabled=torch.cuda.is_available()):
            out = self.model(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            logits = self.model.generator(out)
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            tgt_flat = batch.tgt_y.contiguous().view(-1)
            loss = criterion(logits_flat, tgt_flat) / batch.n_tokens

        # Scaled backward pass — prevents FP16 underflow
        self.scaler.scale(loss).backward()

        self._step_count += 1
        if self._step_count % self.accum_iter == 0:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad
            )
            # Update weights (scaler skips step if gradients are Inf/NaN)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.item() * batch.n_tokens.item()

    @property
    def scale(self) -> float:
        """Current gradient scale factor (useful for monitoring)."""
        return self.scaler.get_scale()
