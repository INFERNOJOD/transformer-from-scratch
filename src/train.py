"""
train.py — Train the Transformer on Multi30k (De→En) translation.

Colab free tier friendly:
  - Small model (N=3, d_model=256)
  - 8 epochs, batch_size=128
  - Saves checkpoints every epoch
  - Plots training loss curve at the end

Run:
    python train.py
or in Colab:
    %run train.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import spacy
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.functional import pad

from model import make_model, subsequent_mask
from data_pipeline import load_multi30k, build_vocab_from_iterator


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CONFIG = {
    # Small model that fits in Colab free T4 / CPU
    "N": 3,
    "d_model": 256,
    "d_ff": 512,
    "h": 4,
    "dropout": 0.1,

    "batch_size": 128,
    "num_epochs": 8,
    "warmup_steps": 400,
    "base_lr": 1.0,
    "max_padding": 72,
    "accum_iter": 4,        # gradient accumulation steps (simulates larger batch)
    "label_smoothing": 0.1,
    "checkpoint_dir": "checkpoints",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# Tokenisation & Vocabulary
# ─────────────────────────────────────────────

def load_tokenizers():
    """Load spaCy models (downloads if needed)."""
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")
    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def build_vocab(spacy_de, spacy_en):
    """Build source (De) and target (En) vocabularies from Multi30k."""
    if os.path.exists("vocab.pt"):
        print("Loading cached vocab...")
        return torch.load("vocab.pt")

    def tok_de(text): return tokenize(text, spacy_de)
    def tok_en(text): return tokenize(text, spacy_en)

    SPECIALS = ["<s>", "</s>", "<blank>", "<unk>"]

    # Load once, reuse for both vocabs (avoids downloading/parsing twice)
    train, val, test = load_multi30k()
    all_pairs = train + val + test

    print("Building German vocabulary...")
    vocab_src = build_vocab_from_iterator(
        (tok_de(pair[0]) for pair in all_pairs),
        min_freq=2, specials=SPECIALS
    )
    vocab_src.set_default_index(vocab_src["<unk>"])

    print("Building English vocabulary...")
    vocab_tgt = build_vocab_from_iterator(
        (tok_en(pair[1]) for pair in all_pairs),
        min_freq=2, specials=SPECIALS
    )
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    torch.save((vocab_src, vocab_tgt), "vocab.pt")
    print(f"Vocab sizes — DE: {len(vocab_src)}, EN: {len(vocab_tgt)}")
    return vocab_src, vocab_tgt


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def collate_fn(batch, src_pipeline, tgt_pipeline, src_vocab, tgt_vocab,
               device, max_padding=72, pad_id=2):
    BOS = torch.tensor([0], device=device)
    EOS = torch.tensor([1], device=device)
    src_list, tgt_list = [], []

    for src_text, tgt_text in batch:
        src_tok = torch.tensor(src_vocab(src_pipeline(src_text)),
                               dtype=torch.long, device=device)
        tgt_tok = torch.tensor(tgt_vocab(tgt_pipeline(tgt_text)),
                               dtype=torch.long, device=device)

        src_seq = torch.cat([BOS, src_tok, EOS])
        tgt_seq = torch.cat([BOS, tgt_tok, EOS])

        src_list.append(pad(src_seq, (0, max_padding - len(src_seq)), value=pad_id))
        tgt_list.append(pad(tgt_seq, (0, max_padding - len(tgt_seq)), value=pad_id))

    return torch.stack(src_list), torch.stack(tgt_list)


def create_dataloaders(spacy_de, spacy_en, vocab_src, vocab_tgt, config, device):
    tok_de = lambda t: tokenize(t, spacy_de)
    tok_en = lambda t: tokenize(t, spacy_en)
    PAD_ID = vocab_src["<blank>"]

    fn = lambda b: collate_fn(b, tok_de, tok_en, vocab_src, vocab_tgt,
                              device, config["max_padding"], PAD_ID)

    # load_multi30k() returns plain Python lists of (de, en) tuples —
    # already "map-style" (indexable, has __len__), so no wrapper needed
    # (this replaces torchtext's to_map_style_dataset() call)
    train_ds, val_ds, _ = load_multi30k()

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=fn)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Batch Helper
# ─────────────────────────────────────────────

class Batch:
    """Holds a training batch with pre-computed masks."""

    def __init__(self, src, tgt, pad_idx=2):
        self.src = src
        self.src_mask = (src != pad_idx).unsqueeze(-2)   # (B, 1, src_len)

        # Teacher forcing: feed tgt[:-1], predict tgt[1:]
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]

        # Causal + padding mask for target
        tgt_mask = (self.tgt != pad_idx).unsqueeze(-2)
        self.tgt_mask = tgt_mask & subsequent_mask(self.tgt.size(-1)).to(src.device)
        self.n_tokens = (self.tgt_y != pad_idx).data.sum()


# ─────────────────────────────────────────────
# Label Smoothing Loss
# ─────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    KL-divergence loss with label smoothing (ε_ls = 0.1 in the paper).
    Distributes smoothing_mass / (V-2) probability to all non-pad, non-target tokens.
    This prevents the model from becoming over-confident and improves BLEU.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, vocab_size) — log-probs from Generator
        target: (N,)            — true token ids
        """
        dist = logits.clone().detach()
        dist.fill_(self.smoothing / (self.vocab_size - 2))
        dist.scatter_(1, target.unsqueeze(1), self.confidence)
        dist[:, self.pad_idx] = 0
        mask = (target == self.pad_idx).nonzero(as_tuple=False)
        if mask.numel() > 0:
            dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(logits, dist.detach())


# ─────────────────────────────────────────────
# Learning Rate Schedule (from paper eq. 3)
# ─────────────────────────────────────────────

def lr_lambda(step: int, d_model: int, warmup: int) -> float:
    """
    lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

    Linearly increases for `warmup` steps, then decays proportionally to
    the inverse square root of the step count.
    """
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def run_epoch(data_loader, model, criterion, optimizer, scheduler,
              pad_idx, device, mode="train", accum_iter=1):
    model.train() if mode == "train" else model.eval()

    total_loss = 0.0
    total_tokens = 0
    tokens_since_log = 0
    t0 = time.time()

    optimizer.zero_grad()

    for i, (src, tgt) in enumerate(data_loader):
        src, tgt = src.to(device), tgt.to(device)
        batch = Batch(src, tgt, pad_idx)

        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        # out: (B, tgt_len-1, d_model)

        # Flatten for loss
        logits = model.generator(out)                        # (B*(T-1), vocab)
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        tgt_flat = batch.tgt_y.contiguous().view(-1)

        loss = criterion(logits_flat, tgt_flat) / batch.n_tokens

        if mode == "train":
            loss.backward()
            if (i + 1) % accum_iter == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        n_tok = batch.n_tokens.item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        tokens_since_log += n_tok

        if mode == "train" and i % 50 == 0 and i > 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {i:4d} | loss {total_loss/total_tokens:.3f} "
                  f"| tok/s {tokens_since_log/elapsed:6.0f} | lr {lr:.2e}")
            tokens_since_log = 0
            t0 = time.time()

    return total_loss / total_tokens


# ─────────────────────────────────────────────
# Main Training Script
# ─────────────────────────────────────────────

def train():
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # Data
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = build_vocab(spacy_de, spacy_en)
    PAD_IDX = vocab_src["<blank>"]

    train_loader, val_loader = create_dataloaders(
        spacy_de, spacy_en, vocab_src, vocab_tgt, CONFIG, DEVICE
    )

    # Model
    model = make_model(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        N=CONFIG["N"],
        d_model=CONFIG["d_model"],
        d_ff=CONFIG["d_ff"],
        h=CONFIG["h"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = LabelSmoothingLoss(
        vocab_size=len(vocab_tgt),
        pad_idx=PAD_IDX,
        smoothing=CONFIG["label_smoothing"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["base_lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    d_model = CONFIG["d_model"]
    warmup = CONFIG["warmup_steps"]
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, d_model, warmup)
    )

    # Training loop
    train_losses, val_losses = [], []

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'='*50}")

        train_loss = run_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            PAD_IDX, DEVICE, mode="train", accum_iter=CONFIG["accum_iter"]
        )
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss = run_epoch(
                val_loader, model, criterion, optimizer, scheduler,
                PAD_IDX, DEVICE, mode="eval"
            )
        val_losses.append(val_loss)

        print(f"  → Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Checkpoint
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch+1:02d}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_src": vocab_src,
            "vocab_tgt": vocab_tgt,
            "config": CONFIG,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # Save final model
    torch.save(model.state_dict(), "transformer_final.pt")
    print("\nTraining complete. Final model saved to transformer_final.pt")

    # ── Plot training curve (reproduces Figure style from paper) ──
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, "o-", label="Train Loss", linewidth=2)
    plt.plot(range(1, len(val_losses)+1), val_losses, "s--", label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Loss (per token)", fontsize=13)
    plt.title("Transformer Training Loss — Multi30k De→En", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("training_loss_curve.png", dpi=150)
    plt.show()
    print("Loss curve saved to training_loss_curve.png")

    return model, vocab_src, vocab_tgt


if __name__ == "__main__":
    train()
