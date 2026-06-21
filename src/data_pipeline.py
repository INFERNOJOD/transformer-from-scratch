"""
data_pipeline.py — Custom Multi30k Data Pipeline (torchtext-free)

WHY THIS FILE EXISTS
----------------------
torchtext's last release (0.18.0) only supports PyTorch up to 2.3.0.
Development stopped in September 2023, and current Colab runtimes ship
PyTorch 2.4+ with torchtext removed entirely. Rather than fight version
pinning, this module replaces the three torchtext call sites with plain
Python + HuggingFace `datasets` (a maintained, stable library) — while
keeping the exact same output format (vocab objects, dataloaders) that
train.py already expects. No other file changes.

WHAT GETS REPLACED
--------------------
    torchtext.datasets.Multi30k          → datasets.load_dataset(...)
    torchtext.vocab.build_vocab_from_iterator → Vocab (this file)
    torchtext.data.functional.to_map_style_dataset → plain list (already
                                                        map-style by default)

The `Vocab` class below intentionally mimics torchtext's Vocab interface
(`vocab(tokens)` callable, `vocab["<token>"]` indexing, `.set_default_index`,
`.get_itos()`, `len(vocab)`) so collate_fn / create_dataloaders in train.py
don't need to change at all.
"""

import os
import torch
from collections import Counter
from typing import List, Callable, Iterable


# ─────────────────────────────────────────────
# Vocab — drop-in replacement for torchtext.vocab.Vocab
# ─────────────────────────────────────────────

class Vocab:
    """
    Minimal vocabulary class matching the subset of torchtext.vocab.Vocab's
    interface actually used in train.py:

        vocab(["hello", "world"])  -> [id1, id2]      (callable, list in)
        vocab["<unk>"]             -> id               (indexing, str in)
        vocab.set_default_index(i)                      (for OOV tokens)
        vocab.get_itos()           -> ["<s>", ...]      (id -> string list)
        len(vocab)                                       (vocab size)
    """

    def __init__(self, stoi: dict, itos: List[str]):
        self._stoi = stoi
        self._itos = itos
        self._default_index = None

    def set_default_index(self, index: int):
        self._default_index = index

    def __getitem__(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token '{token}' not in vocab and no default index set")

    def __call__(self, tokens: List[str]) -> List[int]:
        """Convert a list of string tokens to a list of integer ids."""
        return [self[t] for t in tokens]

    def __len__(self) -> int:
        return len(self._itos)

    def get_itos(self) -> List[str]:
        return self._itos

    def get_stoi(self) -> dict:
        return self._stoi


def build_vocab_from_iterator(
    token_iterator: Iterable[List[str]],
    min_freq: int = 1,
    specials: List[str] = None,
) -> Vocab:
    """
    Build a Vocab from an iterator of tokenized sentences.

    Drop-in replacement for torchtext.vocab.build_vocab_from_iterator.
    Same signature, same behavior: counts token frequencies, keeps tokens
    appearing >= min_freq times, prepends `specials` at fixed low indices.
    """
    specials = specials or []
    counter = Counter()
    for tokens in token_iterator:
        counter.update(tokens)

    # Special tokens first, at fixed indices (matches torchtext convention)
    itos = list(specials)
    seen = set(specials)

    # Then everything else meeting min_freq, sorted by frequency (desc)
    # then alphabetically for determinism (matches torchtext's tie-breaking)
    for token, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq and token not in seen:
            itos.append(token)
            seen.add(token)

    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(stoi, itos)


# ─────────────────────────────────────────────
# Dataset loading — replaces torchtext.datasets.Multi30k
# ─────────────────────────────────────────────

def load_multi30k():
    """
    Load Multi30k German-English translation pairs via HuggingFace `datasets`.

    Returns:
        train, val, test — each a list of (german_text, english_text) tuples,
        matching the (src, tgt) tuple format torchtext.datasets.Multi30k
        used to yield. Already "map-style" (plain Python list, indexable,
        has __len__) so no to_map_style_dataset() wrapper is needed.

    Source: https://huggingface.co/datasets/bentrevett/multi30k
    Same 29K/1K/1K train/val/test split as the original torchtext loader.
    """
    from datasets import load_dataset

    print("Loading Multi30k from HuggingFace datasets...")
    ds = load_dataset("bentrevett/multi30k")

    def to_pairs(split) -> List[tuple]:
        # bentrevett/multi30k rows are {"en": ..., "de": ...}
        # torchtext's Multi30k(language_pair=("de","en")) yields (de, en)
        return [(row["de"], row["en"]) for row in split]

    train = to_pairs(ds["train"])
    val = to_pairs(ds["validation"])
    test = to_pairs(ds["test"])

    print(f"Loaded — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


# ─────────────────────────────────────────────
# Self-test — run this file directly to sanity-check the pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing data_pipeline.py in isolation...\n")

    train, val, test = load_multi30k()
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    print(f"\nSample pair: {train[0]}")

    # Build a tiny vocab from a few sentences to verify the Vocab class
    sample_tokens = [s[1].split() for s in train[:200]]  # English side
    vocab = build_vocab_from_iterator(
        sample_tokens, min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    print(f"\nVocab size (200 sentences, min_freq=2): {len(vocab)}")
    print(f"vocab['<s>'] = {vocab['<s>']}")
    print(f"vocab(['a', 'dog', 'qwertyzzz']) = {vocab(['a', 'dog', 'qwertyzzz'])}")
    print(f"get_itos()[:6] = {vocab.get_itos()[:6]}")

    print("\n✅ data_pipeline.py self-test passed")
