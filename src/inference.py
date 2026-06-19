"""
inference.py — Translate German → English using a trained Transformer.

Usage:
    python inference.py
    python inference.py --checkpoint checkpoints/epoch_08.pt --sentence "Ein Hund läuft im Park."
"""

import argparse
import os
import torch
import spacy

from model import make_model, subsequent_mask


# ─────────────────────────────────────────────
# Greedy Decoding
# ─────────────────────────────────────────────

def greedy_decode(model, src, src_mask, max_len: int, start_symbol: int, device):
    """
    Decode one sentence with greedy (argmax) decoding.
    At each step we pick the token with highest probability and feed it back.

    For better translations, replace this with beam search (see README).
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            logits = model.generator(out[:, -1])           # last position
            next_token = logits.argmax(dim=-1).unsqueeze(0)
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == 1:   # </s> token
                break

    return ys.squeeze(0)


# ─────────────────────────────────────────────
# Translate a single sentence
# ─────────────────────────────────────────────

def translate(sentence: str, model, vocab_src, vocab_tgt, spacy_de, device,
              max_len: int = 72):
    """Tokenize → encode → greedy decode → detokenize."""
    model.eval()

    # Tokenize + encode
    tokens = [tok.text for tok in spacy_de.tokenizer(sentence)]
    BOS, EOS = vocab_src["<s>"], vocab_src["</s>"]
    src_ids = [BOS] + vocab_src(tokens) + [EOS]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = (src != vocab_src["<blank>"]).unsqueeze(-2)

    # Decode
    out_ids = greedy_decode(model, src, src_mask, max_len,
                            start_symbol=vocab_tgt["<s>"], device=device)

    # Convert ids → tokens, strip special tokens
    tgt_itos = vocab_tgt.get_itos()
    tokens_out = [tgt_itos[i] for i in out_ids.tolist()]
    result = []
    for t in tokens_out:
        if t in ("<s>", "<blank>"):
            continue
        if t == "</s>":
            break
        result.append(t)
    return " ".join(result)


# ─────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab_src = ckpt["vocab_src"]
    vocab_tgt = ckpt["vocab_tgt"]
    cfg = ckpt["config"]

    model = make_model(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        N=cfg["N"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        h=cfg["h"],
        dropout=0.0,   # no dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} "
          f"(val loss: {ckpt['val_loss']:.4f})")
    return model, vocab_src, vocab_tgt


# ─────────────────────────────────────────────
# Demo sentences
# ─────────────────────────────────────────────

DEMO_SENTENCES = [
    "Ein Hund läuft durch den Park.",
    "Zwei Männer spielen Fußball.",
    "Ein kleines Kind sitzt auf einer Schaukel.",
    "Eine Frau liest ein Buch im Café.",
    "Der Zug fährt durch den Tunnel.",
]


def main():
    parser = argparse.ArgumentParser(description="Translate De→En with Transformer")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_08.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--sentence", default=None,
                        help="German sentence to translate (runs demo if omitted)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        # Fall back to final model
        alt = "transformer_final.pt"
        if os.path.exists(alt):
            args.checkpoint = alt
        else:
            print("No checkpoint found. Train first: python train.py")
            return

    model, vocab_src, vocab_tgt = load_model(args.checkpoint, device)

    # Load tokenizer
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        import os as _os
        _os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    if args.sentence:
        translation = translate(args.sentence, model, vocab_src, vocab_tgt,
                                spacy_de, device)
        print(f"\nDE: {args.sentence}")
        print(f"EN: {translation}")
    else:
        print("\n── Demo Translations ──\n")
        for sent in DEMO_SENTENCES:
            translation = translate(sent, model, vocab_src, vocab_tgt,
                                    spacy_de, device)
            print(f"DE: {sent}")
            print(f"EN: {translation}\n")


if __name__ == "__main__":
    main()
