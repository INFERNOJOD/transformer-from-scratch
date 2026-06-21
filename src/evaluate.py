"""
evaluate.py — BLEU Score Evaluation

WHY BLEU, NOT LOSS
--------------------
Validation loss measures how surprised the model is at the next token.
BLEU (Bilingual Evaluation Understudy) measures translation quality
the way a human would: do the n-grams in the prediction match the reference?

The original paper reports BLEU scores (Table 2), not loss values.
A model that reports only "val_loss: 2.8" has no way to compare against
the paper's results. A model that reports "BLEU: 24.5" does.

HOW BLEU WORKS
---------------
BLEU-4 is the geometric mean of 1-gram through 4-gram precision scores,
multiplied by a brevity penalty (to penalize overly short translations).

Example:
  Reference: "The dog is running in the park"
  Prediction: "The dog runs in park"

  1-gram precision: 4/5  ("The", "dog", "in", "park" match; "runs" doesn't perfectly)
  2-gram precision: 2/4  ("The dog", "in park" match)
  3-gram precision: 0/3
  4-gram precision: 0/2

  BLEU = BP × exp(0.25 × (log(4/5) + log(2/4) + log(0) + log(0)))
       ≈ 0  (any zero n-gram kills the score — log(0) = -inf)

In practice on the full test set, the zeros smooth out.
The paper's base model achieves 27.3 BLEU on WMT14 En-De.
This Multi30k implementation typically achieves ~25-30 BLEU.

sacrebleu is the standard library — it handles tokenization and normalization
consistently so different papers can be fairly compared.
"""

import torch
import sacrebleu
from typing import List
from tqdm import tqdm


def evaluate_bleu(
    model,
    data_loader,
    vocab_src,
    vocab_tgt,
    device,
    beam_size: int = 4,
    max_len: int = 72,
    use_beam: bool = True,
    n_examples: int = None,
) -> dict:
    """
    Compute BLEU score on a dataset split.

    Args:
        model:        trained EncoderDecoder
        data_loader:  validation DataLoader (batch_size=1 recommended)
        vocab_src:    source vocabulary
        vocab_tgt:    target vocabulary
        device:       torch device
        beam_size:    beam width for beam search decoding
        max_len:      max output tokens
        use_beam:     if False, uses greedy decode (faster, worse BLEU)
        n_examples:   if set, only evaluate first n examples (for quick check)

    Returns:
        dict with 'bleu', 'predictions', 'references'
    """
    from model import subsequent_mask

    if use_beam:
        from beam_search import beam_search as _beam_search

    model.eval()
    PAD = vocab_src["<blank>"]
    BOS_TGT = vocab_tgt["<s>"]
    EOS_TGT = vocab_tgt["</s>"]
    tgt_itos = vocab_tgt.get_itos()

    predictions: List[str] = []
    references: List[str] = []

    with torch.no_grad():
        for i, (src_batch, tgt_batch) in enumerate(tqdm(data_loader,
                                                         desc="Evaluating BLEU")):
            if n_examples and i >= n_examples:
                break

            src = src_batch.to(device)
            tgt = tgt_batch.to(device)
            src_mask = (src != PAD).unsqueeze(-2)

            # Decode prediction
            if use_beam:
                pred_ids = _beam_search(
                    model, src, src_mask,
                    max_len=max_len,
                    start_symbol=BOS_TGT,
                    end_symbol=EOS_TGT,
                    pad_symbol=vocab_tgt["<blank>"],
                    beam_size=beam_size,
                    device=device,
                )
            else:
                # Greedy
                memory = model.encode(src, src_mask)
                ys = torch.full((1, 1), BOS_TGT, dtype=torch.long, device=device)
                for _ in range(max_len - 1):
                    tgt_mask = subsequent_mask(ys.size(1)).to(device)
                    out = model.decode(memory, src_mask, ys, tgt_mask)
                    next_tok = model.generator(out[:, -1]).argmax(dim=-1)
                    ys = torch.cat([ys, next_tok.unsqueeze(0)], dim=1)
                    if next_tok.item() == EOS_TGT:
                        break
                pred_ids = ys.squeeze(0).tolist()[1:]  # strip BOS

            # Detokenise prediction
            pred_tokens = []
            for tid in pred_ids:
                t = tgt_itos[tid]
                if t == "</s>":
                    break
                if t not in ("<s>", "<blank>"):
                    pred_tokens.append(t)
            predictions.append(" ".join(pred_tokens))

            # Detokenise reference (ground truth)
            ref_tokens = []
            for tid in tgt[0].tolist():
                t = tgt_itos[tid]
                if t == "</s>":
                    break
                if t not in ("<s>", "<blank>"):
                    ref_tokens.append(t)
            references.append(" ".join(ref_tokens))

    # Compute corpus-level BLEU using sacrebleu
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    result = {
        "bleu": bleu.score,
        "bleu_str": bleu.format(),
        "predictions": predictions,
        "references": references,
    }

    print(f"\nBLEU Score: {bleu.score:.2f}")
    print(f"Details: {bleu.format()}")

    # Print a few examples
    print("\n── Sample Translations ──")
    for pred, ref in zip(predictions[:5], references[:5]):
        print(f"  REF: {ref}")
        print(f"  PRD: {pred}")
        print()

    return result


def quick_bleu_check(model, val_loader, vocab_src, vocab_tgt, device,
                     n: int = 200) -> float:
    """
    Fast BLEU estimate on n examples — useful during training to track progress
    without waiting for a full eval pass.
    """
    result = evaluate_bleu(
        model, val_loader, vocab_src, vocab_tgt, device,
        use_beam=False,   # greedy for speed
        n_examples=n,
    )
    return result["bleu"]


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    import argparse
    import torch
    from model import make_model

    parser = argparse.ArgumentParser(description="Evaluate BLEU for a trained checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_08.pt",
                        help="Path to a checkpoint saved by train.py "
                             "(must contain model_state, vocab_src, vocab_tgt, config)")
    parser.add_argument("--beam", type=int, default=4,
                        help="Beam size for beam search decoding (0 = greedy)")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Limit evaluation to first N examples (omit for full val set)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")

    # weights_only=False: checkpoint contains data_pipeline.Vocab objects,
    # not just tensors — see train.py / inference.py for the same fix.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab_src = ckpt["vocab_src"]
    vocab_tgt = ckpt["vocab_tgt"]
    cfg = ckpt["config"]

    model = make_model(
        src_vocab_size=len(vocab_src), tgt_vocab_size=len(vocab_tgt),
        N=cfg["N"], d_model=cfg["d_model"], d_ff=cfg["d_ff"],
        h=cfg["h"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded model from epoch {ckpt['epoch']} (val loss: {ckpt['val_loss']:.4f})\n")

    # Build the validation dataloader the same way train.py does
    from train import create_dataloaders, load_tokenizers
    spacy_de, spacy_en = load_tokenizers()
    _, val_loader = create_dataloaders(
        spacy_de, spacy_en, vocab_src, vocab_tgt, cfg, device
    )

    use_beam = args.beam > 0
    label = f"beam search (k={args.beam})" if use_beam else "greedy decoding"
    print(f"Evaluating with {label}"
         + (f", first {args.n_examples} examples" if args.n_examples else ", full validation set")
         + "...\n")

    result = evaluate_bleu(
        model, val_loader, vocab_src, vocab_tgt, device,
        beam_size=args.beam, use_beam=use_beam,
        n_examples=args.n_examples,
    )

    print(f"\n>>> COPY THIS INTO RESULTS.md / EXPERIMENTS.md <<<")
    print(f"BLEU ({label}): {result['bleu']:.2f}")


if __name__ == "__main__":
    main()
