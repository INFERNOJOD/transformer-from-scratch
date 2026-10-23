"""
beam_search.py — Beam Search Decoding

WHY BEAM SEARCH > GREEDY
--------------------------
Greedy decode: at each step, pick the single highest-probability token.
Problem: a locally bad choice early on can ruin the whole translation.

Example (greedy might produce):
  "The dog is run in park"   ← picked "run" greedily at step 3

Beam search keeps the top-k COMPLETE SEQUENCES alive at every step.
With beam_size=4, it explores 4 parallel hypotheses and returns the best.

  Step 1: ["The", "A", "One", "That"]           ← 4 beams
  Step 2: ["The dog", "The cat", "A dog", ...]  ← expand each
  Step 3: prune back to top-4 by cumulative log-prob
  ...
  Final: pick highest score (with length penalty)

LENGTH PENALTY
--------------
Without it, beam search prefers short translations (fewer terms to multiply).
Hypotheses are ranked by the GNMT length penalty (Wu et al., 2016), with
α=0.6 as in the paper:

    score = log_prob / ((5 + length) / 6)^α

Why length^α and not just length? Log-probs are negative, so dividing by
the full length would over-correct and favour long outputs; α in [0, 1]
tunes how strongly length is normalised.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from data_pipeline import Vocab


@dataclass
class BeamHypothesis:
    """One hypothesis (partial translation) in the beam."""
    token_ids: List[int]
    log_prob: float = 0.0

    def score(self, alpha: float = 0.6) -> float:
        """Length-normalized log-probability (higher = better)."""
        length_penalty = ((5 + len(self.token_ids)) / 6) ** alpha
        return self.log_prob / length_penalty


def beam_search(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    pad_symbol: int,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    device: Optional[torch.device] = None,
    return_hypotheses: bool = False,
) -> Union[List[int], List[BeamHypothesis]]:
    """
    Beam search decoder.

    Args:
        model:          trained EncoderDecoder
        src:            (1, src_len) source token ids
        src_mask:       (1, 1, src_len) padding mask
        max_len:        maximum output length
        start_symbol:   BOS token id
        end_symbol:     EOS token id
        pad_symbol:     PAD token id
        beam_size:      number of beams (4 in paper)
        length_penalty: α in length penalty formula (0.6 in paper)
        device:         torch device
        return_hypotheses: return every final hypothesis instead of the best

    Returns:
        best token id sequence (excluding BOS, including EOS), or with
        return_hypotheses=True the final hypotheses (completed ones, or the
        surviving beams if none completed), best first, token_ids with BOS
    """
    if device is None:
        device = src.device

    model.eval()
    with torch.no_grad():
        # Encode source once — reuse memory for all beams
        memory = model.encode(src, src_mask)                       # (1, S, d)
        memory = memory.expand(beam_size, -1, -1)                  # (B, S, d)
        src_mask = src_mask.expand(beam_size, -1, -1)              # (B, 1, S)

        # Initialise with a single BOS beam. Starting with beam_size copies
        # of the same BOS beam would expand identical prefixes and fill the
        # beam with duplicate hypotheses; the first step's top-k expansion
        # of this one beam already yields beam_size distinct hypotheses.
        beams: List[BeamHypothesis] = [
            BeamHypothesis(token_ids=[start_symbol], log_prob=0.0)
        ]
        completed: List[BeamHypothesis] = []

        for step in range(max_len):
            if not beams:
                break

            # Build decoder input from all current beams
            B = len(beams)
            tgt = torch.tensor(
                [h.token_ids for h in beams], dtype=torch.long, device=device
            )  # (B, current_len)

            # Causal mask
            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device), diagonal=1
            ).bool()
            tgt_mask = (~tgt_mask).unsqueeze(0)   # (1, tgt_len, tgt_len)

            # Decode one step
            mem_b = memory[:B]
            src_mask_b = src_mask[:B]
            out = model.decode(mem_b, src_mask_b, tgt, tgt_mask)
            logits = model.generator(out[:, -1, :])   # (B, vocab)

            # top-(beam_size) candidates per beam
            log_probs = logits                         # already log_softmax
            top_log_probs, top_ids = log_probs.topk(beam_size, dim=-1)

            # Expand beams
            candidates: List[BeamHypothesis] = []
            for b_idx, beam in enumerate(beams):
                for k in range(beam_size):
                    new_id = top_ids[b_idx, k].item()
                    new_lp = beam.log_prob + top_log_probs[b_idx, k].item()
                    new_hyp = BeamHypothesis(
                        token_ids=beam.token_ids + [new_id],
                        log_prob=new_lp,
                    )
                    if new_id == end_symbol:
                        completed.append(new_hyp)
                    else:
                        candidates.append(new_hyp)

            # Prune to top beam_size by length-normalized score
            candidates.sort(key=lambda h: h.score(length_penalty), reverse=True)
            beams = candidates[:beam_size]

        # If nothing completed, take the best partial hypothesis
        if not completed:
            completed = beams

        if return_hypotheses:
            return sorted(completed, key=lambda h: h.score(length_penalty),
                          reverse=True)
        best = max(completed, key=lambda h: h.score(length_penalty))
        return best.token_ids[1:]   # strip BOS


def translate_beam(sentence: str, model: nn.Module, vocab_src: Vocab,
                   vocab_tgt: Vocab, spacy_de: Optional["spacy.language.Language"],
                   device: torch.device, beam_size: int = 4,
                   max_len: int = 72,
                   tokenize: Optional[Callable[[str], List[str]]] = None,
                   detokenize: Optional[Callable[[List[str]], str]] = None) -> str:
    """
    Convenience wrapper: string → beam search → string.
    tokenize / detokenize default to spaCy words joined by spaces.
    """
    model.eval()
    if tokenize is None:
        tokens = [tok.text for tok in spacy_de.tokenizer(sentence)]
    else:
        tokens = tokenize(sentence)
    BOS, EOS = vocab_src["<s>"], vocab_src["</s>"]
    PAD = vocab_src["<blank>"]

    src_ids = [BOS] + vocab_src(tokens) + [EOS]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = (src != PAD).unsqueeze(-2)

    out_ids = beam_search(
        model, src, src_mask,
        max_len=max_len,
        start_symbol=vocab_tgt["<s>"],
        end_symbol=vocab_tgt["</s>"],
        pad_symbol=vocab_tgt["<blank>"],
        beam_size=beam_size,
        device=device,
    )

    tgt_itos = vocab_tgt.get_itos()
    tokens_out = []
    for i in out_ids:
        t = tgt_itos[i]
        if t == "</s>":
            break
        if t not in ("<s>", "<blank>"):
            tokens_out.append(t)

    return (detokenize or " ".join)(tokens_out)
