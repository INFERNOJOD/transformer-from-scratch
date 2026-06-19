"""
visualize.py — Attention Visualization

Generates heatmaps showing what each attention head attends to.
These are the kinds of plots you see in the original paper (Figures 3-5)
and in blog posts explaining Transformers.

Great for your README — a visual that shows the model actually working.

HOW TO USE
-----------
After training:
    python visualize.py --checkpoint checkpoints/epoch_08.pt
                        --sentence "Ein Hund läuft durch den Park."

This will generate:
    attention_maps/encoder_layer0_head0.png  (and all other layers/heads)
    attention_maps/summary.png               (all heads in a grid)

WHAT YOU'RE LOOKING AT
-----------------------
Each cell (i, j) in the heatmap = how much token i attends to token j.

In a well-trained model you'd expect to see:
  - Diagonal patterns (each token attends to itself / nearby tokens)
  - Syntactic patterns (verbs attending to their subjects/objects)
  - Some heads specializing in long-range dependencies
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works in Colab and headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Optional


# ─────────────────────────────────────────────
# Core plotting function
# ─────────────────────────────────────────────

def plot_attention_head(
    attn_weights: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str],
    title: str = "",
    ax: plt.Axes = None,
    cmap: str = "Blues",
) -> plt.Axes:
    """
    Plot a single attention head as a heatmap.

    Args:
        attn_weights: (tgt_len, src_len) attention probabilities
        src_tokens:   tokens on x-axis (keys)
        tgt_tokens:   tokens on y-axis (queries)
        title:        plot title
        ax:           existing axes (creates new figure if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.5),
                                        max(4, len(tgt_tokens) * 0.4)))

    weights = attn_weights.detach().cpu().float().numpy()

    im = ax.imshow(weights, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tgt_tokens, fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def plot_all_heads(
    attn_weights: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str],
    layer_name: str = "Layer",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot all attention heads for one layer in a grid.

    Args:
        attn_weights: (n_heads, tgt_len, src_len)
        src_tokens:   source tokens
        tgt_tokens:   target tokens
        layer_name:   title prefix
        save_path:    if provided, saves figure to this path
    """
    n_heads = attn_weights.size(0)
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).flatten() if n_heads > 1 else [axes]

    for h in range(n_heads):
        plot_attention_head(
            attn_weights[h],
            src_tokens, tgt_tokens,
            title=f"{layer_name} — Head {h+1}",
            ax=axes[h],
        )

    # Hide unused axes
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f"Attention Weights: {layer_name}", fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Hook-based attention extraction
# ─────────────────────────────────────────────

class AttentionExtractor:
    """
    Registers forward hooks on all MultiHeadAttention modules to capture
    attention weights after each forward pass.

    Usage:
        extractor = AttentionExtractor(model)
        output = model(...)            # triggers hooks
        weights = extractor.weights    # {layer_name: (batch, heads, tgt, src)}
        extractor.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module):
        self.weights = {}
        self._hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model: torch.nn.Module):
        from model import MultiHeadAttention
        # Also handle RoPE and Flash variants
        try:
            from rope import RoPEMultiHeadAttention
            attn_classes = (MultiHeadAttention, RoPEMultiHeadAttention)
        except ImportError:
            attn_classes = (MultiHeadAttention,)

        for name, module in model.named_modules():
            if isinstance(module, attn_classes):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if hasattr(module, "attn_weights") and module.attn_weights is not None:
                # Store: (batch, heads, tgt_len, src_len)
                self.weights[name] = module.attn_weights.detach().cpu()
        return hook

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear(self):
        self.weights.clear()


# ─────────────────────────────────────────────
# Full visualization pipeline
# ─────────────────────────────────────────────

def visualize_translation(
    sentence: str,
    model,
    vocab_src,
    vocab_tgt,
    spacy_de,
    device,
    save_dir: str = "attention_maps",
    beam_size: int = 1,
):
    """
    Translate a sentence and visualize all attention heads.

    Generates:
        {save_dir}/encoder_layer_X_head_grid.png   for each encoder layer
        {save_dir}/decoder_self_layer_X.png         for each decoder layer
        {save_dir}/decoder_cross_layer_X.png        for each decoder layer
        {save_dir}/summary.png                      4-panel overview
    """
    import spacy as _spacy
    from model import subsequent_mask

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Tokenize
    src_tokens_raw = [tok.text for tok in spacy_de.tokenizer(sentence)]
    src_tokens = ["<s>"] + src_tokens_raw + ["</s>"]
    BOS = vocab_src["<s>"]
    EOS = vocab_src["</s>"]
    PAD = vocab_src["<blank>"]

    src_ids = [BOS] + vocab_src(src_tokens_raw) + [EOS]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = (src != PAD).unsqueeze(-2)

    # Set up extractor
    extractor = AttentionExtractor(model)

    with torch.no_grad():
        # Greedy decode to get translation
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), BOS, dtype=torch.long, device=device)
        for _ in range(min(50, len(src_ids) + 10)):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = model.generator(out[:, -1]).argmax(dim=-1)
            ys = torch.cat([ys, next_tok.unsqueeze(0)], dim=1)
            if next_tok.item() == EOS:
                break

    tgt_itos = vocab_tgt.get_itos()
    tgt_tokens = []
    for tid in ys.squeeze(0).tolist():
        t = tgt_itos[tid]
        tgt_tokens.append(t)
        if t == "</s>":
            break

    translation = " ".join(t for t in tgt_tokens if t not in ("<s>", "</s>", "<blank>"))
    print(f"\nSource:      {sentence}")
    print(f"Translation: {translation}")
    print(f"Saving attention maps to: {save_dir}/")

    # Plot captured attention weights
    for layer_name, weights in extractor.weights.items():
        # weights: (1, heads, tgt_len, src_len)
        w = weights[0]   # remove batch dim → (heads, tgt, src)

        # Determine token labels based on layer type
        if "encoder" in layer_name and "self_attn" in layer_name:
            q_toks = src_tokens[:w.size(1)]
            k_toks = src_tokens[:w.size(2)]
            label = f"Encoder Self-Attn"
        elif "decoder" in layer_name and "self_attn" in layer_name:
            q_toks = tgt_tokens[:w.size(1)]
            k_toks = tgt_tokens[:w.size(2)]
            label = f"Decoder Self-Attn"
        else:
            q_toks = tgt_tokens[:w.size(1)]
            k_toks = src_tokens[:w.size(2)]
            label = f"Decoder Cross-Attn"

        safe_name = layer_name.replace(".", "_")
        save_path = os.path.join(save_dir, f"{safe_name}.png")
        plot_all_heads(w, k_toks, q_toks, layer_name=f"{label} ({layer_name})",
                       save_path=save_path)
        plt.close("all")

    extractor.remove_hooks()
    print(f"Done. {len(extractor.weights)} attention layers saved.")
    return translation


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/epoch_08.pt")
    parser.add_argument("--sentence", default="Ein Hund läuft durch den Park.")
    parser.add_argument("--save-dir", default="attention_maps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torch, spacy
    from model import make_model

    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab_src = ckpt["vocab_src"]
    vocab_tgt = ckpt["vocab_tgt"]
    cfg = ckpt["config"]

    model = make_model(len(vocab_src), len(vocab_tgt),
                       N=cfg["N"], d_model=cfg["d_model"],
                       d_ff=cfg["d_ff"], h=cfg["h"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    visualize_translation(args.sentence, model, vocab_src, vocab_tgt,
                          spacy_de, device, save_dir=args.save_dir)
