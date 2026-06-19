"""
config.py — Centralized Configuration System

Replaces scattered magic numbers with named, documented configs.
Supports multiple presets (base, small, large) and easy overrides.

WHY A CONFIG SYSTEM
---------------------
Bad:  make_model(100, 100, 6, 512, 2048, 8, 0.1)   ← what are these?
Good: make_model(**ModelConfig.base().model_kwargs())

Benefits:
  - All hyperparameters in one place
  - Easy to reproduce experiments (save config with checkpoint)
  - Easy to run ablations (change one param, re-run)
  - Looks like a real engineering project
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os


@dataclass
class ModelConfig:
    """
    Transformer model architecture hyperparameters.

    Three presets:
      - small:  N=3, d_model=256  → Colab free tier (trains in ~20 min)
      - base:   N=6, d_model=512  → Paper's base model (needs GPU)
      - large:  N=6, d_model=1024 → Paper's big model (needs multi-GPU)
    """
    N: int = 3                  # number of encoder/decoder layers
    d_model: int = 256          # model / embedding dimension
    d_ff: int = 512             # FFN inner dimension (usually 4 × d_model)
    h: int = 4                  # number of attention heads
    dropout: float = 0.1        # dropout rate

    # Modern component toggles
    use_rope: bool = False           # RoPE instead of sinusoidal PE
    use_rmsnorm: bool = False        # RMSNorm instead of LayerNorm
    use_swiglu: bool = False         # SwiGLU FFN instead of ReLU FFN
    use_flash_attention: bool = False  # Flash Attention backend
    tie_weights: bool = True         # tie embedding ↔ generator weights

    @classmethod
    def small(cls) -> "ModelConfig":
        """Colab-friendly. Trains in ~20 min on T4."""
        return cls(N=3, d_model=256, d_ff=512, h=4)

    @classmethod
    def base(cls) -> "ModelConfig":
        """Paper base model. Needs decent GPU (~12 hrs on 8×P100 originally)."""
        return cls(N=6, d_model=512, d_ff=2048, h=8)

    @classmethod
    def base_modern(cls) -> "ModelConfig":
        """Base model with all modern improvements (LLaMA-style)."""
        return cls(N=6, d_model=512, d_ff=2048, h=8,
                   use_rope=True, use_rmsnorm=True,
                   use_swiglu=True, use_flash_attention=True,
                   tie_weights=True)

    @classmethod
    def large(cls) -> "ModelConfig":
        """Paper big model."""
        return cls(N=6, d_model=1024, d_ff=4096, h=16, dropout=0.3)

    def model_kwargs(self) -> dict:
        """Returns kwargs for make_model() — excludes toggle flags."""
        return dict(N=self.N, d_model=self.d_model,
                    d_ff=self.d_ff, h=self.h, dropout=self.dropout)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    dataset: str = "multi30k"       # "multi30k" or "wmt14"
    language_pair: tuple = ("de", "en")
    max_padding: int = 72

    # Training
    batch_size: int = 128
    num_epochs: int = 8
    accum_iter: int = 4             # gradient accumulation steps
    label_smoothing: float = 0.1

    # Optimizer (from paper Section 5.3)
    base_lr: float = 1.0            # used with warmup schedule
    warmup_steps: int = 400         # 4000 in paper; smaller for small model
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    clip_grad: float = 1.0

    # Mixed precision
    use_amp: bool = True            # automatic mixed precision

    # Evaluation
    eval_bleu_every: int = 2        # evaluate BLEU every N epochs
    beam_size: int = 4
    length_penalty: float = 0.6

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1             # save checkpoint every N epochs

    @classmethod
    def colab(cls) -> "TrainConfig":
        """Settings tuned for Colab free tier."""
        return cls(batch_size=64, num_epochs=8,
                   warmup_steps=400, use_amp=True)

    @classmethod
    def full(cls) -> "TrainConfig":
        """Settings closer to original paper."""
        return cls(batch_size=256, num_epochs=20,
                   warmup_steps=4000, use_amp=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["language_pair"] = list(d["language_pair"])
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = json.load(f)
        data["language_pair"] = tuple(data["language_pair"])
        return cls(**data)


@dataclass
class ExperimentConfig:
    """
    Full experiment config = model + training.
    Save this alongside every checkpoint for full reproducibility.
    """
    model: ModelConfig = field(default_factory=ModelConfig.small)
    train: TrainConfig = field(default_factory=TrainConfig.colab)
    name: str = "experiment"
    notes: str = ""

    def save(self, path: str):
        d = {
            "name": self.name,
            "notes": self.notes,
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"Experiment config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        train_d = d["train"]
        train_d["language_pair"] = tuple(train_d["language_pair"])
        return cls(
            model=ModelConfig(**d["model"]),
            train=TrainConfig(**train_d),
            name=d.get("name", ""),
            notes=d.get("notes", ""),
        )

    def summary(self) -> str:
        m, t = self.model, self.train
        lines = [
            f"Experiment: {self.name}",
            f"  Model:  N={m.N}, d_model={m.d_model}, h={m.h}, d_ff={m.d_ff}",
            f"  Extras: rope={m.use_rope}, rmsnorm={m.use_rmsnorm}, "
            f"swiglu={m.use_swiglu}, flash={m.use_flash_attention}",
            f"  Train:  epochs={t.num_epochs}, batch={t.batch_size}, "
            f"warmup={t.warmup_steps}, amp={t.use_amp}",
        ]
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Preset configs for ablation experiments
# ─────────────────────────────────────────────

ABLATION_CONFIGS = {
    "baseline": ExperimentConfig(
        model=ModelConfig.small(),
        name="baseline",
        notes="Original sinusoidal PE, LayerNorm, ReLU FFN",
    ),
    "with_rope": ExperimentConfig(
        model=ModelConfig(N=3, d_model=256, d_ff=512, h=4, use_rope=True),
        name="with_rope",
        notes="RoPE instead of sinusoidal PE",
    ),
    "with_rmsnorm": ExperimentConfig(
        model=ModelConfig(N=3, d_model=256, d_ff=512, h=4, use_rmsnorm=True),
        name="with_rmsnorm",
        notes="RMSNorm instead of LayerNorm",
    ),
    "with_swiglu": ExperimentConfig(
        model=ModelConfig(N=3, d_model=256, d_ff=512, h=4, use_swiglu=True),
        name="with_swiglu",
        notes="SwiGLU FFN instead of ReLU FFN",
    ),
    "llama_style": ExperimentConfig(
        model=ModelConfig.base_modern(),
        name="llama_style",
        notes="All modern components: RoPE + RMSNorm + SwiGLU + Flash Attention",
    ),
}


if __name__ == "__main__":
    # Print all configs
    for name, exp in ABLATION_CONFIGS.items():
        print(exp.summary())
        print()
