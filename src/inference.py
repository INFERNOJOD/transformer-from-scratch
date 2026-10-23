"""
inference.py — Translate German → English using a trained Transformer.

Usage (from the repository root):
    python src/inference.py                                    # demo sentences
    python src/inference.py --sentence "Ein Hund läuft im Park."
    python src/inference.py --sentence "Ein Hund läuft im Park." --beam 4
"""

import argparse
import os
from typing import Callable, Dict, List, Optional, Tuple

import spacy
import torch

from beam_search import translate_beam
from bpe import tokenizers_from_checkpoint
from data_pipeline import Vocab
from model import EncoderDecoder, ablation_kwargs, make_model, subsequent_mask


# ─────────────────────────────────────────────
# Greedy Decoding
# ─────────────────────────────────────────────

def greedy_decode(model: EncoderDecoder, src: torch.Tensor,
                  src_mask: torch.Tensor, max_len: int, start_symbol: int,
                  device: torch.device) -> torch.Tensor:
    """
    Decode one sentence with greedy (argmax) decoding.
    At each step we pick the token with highest probability and feed it back.

    For better translations use beam search (--beam N, see beam_search.py).
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

def translate(sentence: str, model: EncoderDecoder, vocab_src: Vocab,
              vocab_tgt: Vocab, spacy_de: Optional["spacy.language.Language"],
              device: torch.device, max_len: int = 72,
              tokenize: Optional[Callable[[str], List[str]]] = None,
              detokenize: Optional[Callable[[List[str]], str]] = None) -> str:
    """
    Tokenize → encode → greedy decode → detokenize.
    tokenize / detokenize default to spaCy words joined by spaces; pass a
    BPETokenizer's tokenize / detokenize for BPE checkpoints.
    """
    model.eval()

    # Tokenize + encode
    if tokenize is None:
        tokens = [tok.text for tok in spacy_de.tokenizer(sentence)]
    else:
        tokens = tokenize(sentence)
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
    return (detokenize or " ".join)(result)


# ─────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load a train.py checkpoint (contains Vocab objects, so not weights_only)."""
    print(f"Loading checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def model_from_checkpoint(ckpt: Dict, device: torch.device
                          ) -> Tuple[EncoderDecoder, Vocab, Vocab]:
    """Rebuild the model described by a checkpoint and load its weights."""
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
        **ablation_kwargs(cfg),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} "
          f"(val loss: {ckpt['val_loss']:.4f})")
    return model, vocab_src, vocab_tgt


def load_model(checkpoint_path: str,
               device: torch.device) -> Tuple[EncoderDecoder, Vocab, Vocab]:
    """Rebuild the model from a train.py checkpoint and load its weights."""
    return model_from_checkpoint(load_checkpoint(checkpoint_path, device), device)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate De→En with Transformer")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_08.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--sentence", default=None,
                        help="German sentence to translate (runs demo if omitted)")
    parser.add_argument("--beam", type=int, default=0,
                        help="Beam size for beam search (0 = greedy decoding)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        # Fall back to final model
        alt = "transformer_final.pt"
        if os.path.exists(alt):
            args.checkpoint = alt
        else:
            print("No checkpoint found. Train first: python src/train.py")
            return

    ckpt = load_checkpoint(args.checkpoint, device)
    model, vocab_src, vocab_tgt = model_from_checkpoint(ckpt, device)

    # Load tokenizer: BPE merges stored in the checkpoint, else spaCy
    bpe = tokenizers_from_checkpoint(ckpt)
    spacy_de, tokenize, detokenize = None, None, None
    if bpe is not None:
        tokenize, detokenize = bpe[0].tokenize, bpe[1].detokenize
    else:
        try:
            spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            os.system("python -m spacy download de_core_news_sm")
            spacy_de = spacy.load("de_core_news_sm")

    def run(sentence: str) -> str:
        if args.beam > 0:
            return translate_beam(sentence, model, vocab_src, vocab_tgt,
                                  spacy_de, device, beam_size=args.beam,
                                  tokenize=tokenize, detokenize=detokenize)
        return translate(sentence, model, vocab_src, vocab_tgt, spacy_de, device,
                         tokenize=tokenize, detokenize=detokenize)

    if args.sentence:
        translation = run(args.sentence)
        print(f"\nDE: {args.sentence}")
        print(f"EN: {translation}")
    else:
        print("\n── Demo Translations ──\n")
        for sent in DEMO_SENTENCES:
            translation = run(sent)
            print(f"DE: {sent}")
            print(f"EN: {translation}\n")


if __name__ == "__main__":
    main()
