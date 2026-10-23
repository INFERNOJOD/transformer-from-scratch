"""
bpe.py — Byte-Pair Encoding Tokenizer (from scratch)

Papers: "Neural Machine Translation of Rare Words with Subword Units"
        Sennrich et al., 2016 — https://arxiv.org/abs/1508.07909
        Byte-level variant as in GPT-2 (Radford et al., 2019)

WHY BPE
--------
The word-level vocabulary in train.py drops every token seen fewer than
min_freq=2 times to <unk> — RESULTS.md shows "ice fishing hut" collapsing
to a single opaque <unk>. BPE instead builds words out of learned subword
pieces, so a rare word is split into known fragments rather than lost.

HOW IT WORKS
-------------
1. Start from the 256 possible bytes as the base vocabulary. Because every
   string is a byte sequence, ANY input can be encoded: <unk> is never needed.
2. Pre-tokenize text into words (GPT-2 style — a leading space stays
   attached to the word, so decoding is plain concatenation).
3. Repeatedly find the most frequent adjacent pair of tokens across the
   corpus and merge it into a new token, until the target vocab size.
4. Encoding applies the learned merges in the order they were learned.

Token ids: special tokens first (<s>=0, </s>=1, <blank>=2, <unk>=3, matching
the word-level vocab in train.py), then the 256 byte tokens, then merges.

Usage:
    tok = BPETokenizer.train(sentences, vocab_size=8000)
    ids = tok.encode("Ein Hund läuft durch den Park.")
    tok.decode(ids)            # → "Ein Hund läuft durch den Park."
    tok.save("bpe_de.json"); tok = BPETokenizer.load("bpe_de.json")

    # Plug into the existing train.py pipeline (Vocab + token-string lists):
    vocab = tok.to_vocab()
    vocab(tok.tokenize("Ein Hund"))   # same ids as tok.encode("Ein Hund")
"""

import heapq
import json
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from data_pipeline import Vocab


SPECIALS: Tuple[str, ...] = ("<s>", "</s>", "<blank>", "<unk>")

# GPT-2-style pre-tokenizer: words and punctuation runs keep one leading
# space; other whitespace is kept as its own piece. Every character matches
# one alternative, so "".join(PRETOKENIZE.findall(text)) == text.
PRETOKENIZE = re.compile(r" ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+")

Pair = Tuple[int, int]


def _bytes_to_unicode() -> Dict[int, str]:
    """
    Map each byte to a printable unicode character (GPT-2's trick), so that
    byte-level tokens have readable, whitespace-free string forms: " Hund"
    becomes "ĠHund". Printable ASCII/Latin-1 bytes map to themselves.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTE = {c: b for b, c in BYTE_TO_UNICODE.items()}


def _merge_pair(ids: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right)."""
    out: List[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """
    Byte-level BPE tokenizer: train on text, encode, decode, save/load.

    The whole learned state is the ordered list of merges — the vocabulary
    (byte sequence of every token) is derived from it.
    """

    def __init__(self, merges: Optional[Sequence[Pair]] = None,
                 specials: Sequence[str] = SPECIALS):
        self.specials: List[str] = list(specials)
        self.merges: List[Pair] = []
        self._byte_offset = len(self.specials)
        self._token_bytes: List[bytes] = [s.encode("utf-8") for s in self.specials]
        self._token_bytes += [bytes([b]) for b in range(256)]
        self._ranks: Dict[Pair, int] = {}
        self._cache: Dict[str, List[int]] = {}
        self._token_to_id: Optional[Dict[str, int]] = None
        for a, b in merges or []:
            self._add_merge((int(a), int(b)))

    # ── vocabulary ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._token_bytes)

    @property
    def vocab_size(self) -> int:
        return len(self)

    @property
    def unk_id(self) -> int:
        return self.specials.index("<unk>")

    def _add_merge(self, pair: Pair) -> int:
        a, b = pair
        n_base = self._byte_offset + 256
        new_id = len(self._token_bytes)
        for t in pair:
            if not self._byte_offset <= t < new_id:
                raise ValueError(f"merge {pair} references unknown or special token {t}")
        self.merges.append(pair)
        self._ranks[pair] = new_id - n_base
        self._token_bytes.append(self._token_bytes[a] + self._token_bytes[b])
        self._cache.clear()
        self._token_to_id = None
        return new_id

    def id_to_token(self, idx: int) -> str:
        """Readable string form of a token (specials verbatim, bytes mapped)."""
        if idx < self._byte_offset:
            return self.specials[idx]
        return "".join(BYTE_TO_UNICODE[b] for b in self._token_bytes[idx])

    def token_to_id(self, token: str) -> int:
        if self._token_to_id is None:
            self._token_to_id = {self.id_to_token(i): i for i in range(len(self))}
        return self._token_to_id[token]

    def to_vocab(self) -> Vocab:
        """
        A data_pipeline.Vocab over this tokenizer's token strings, so the
        existing train.py collate/eval code can consume BPE tokens unchanged.
        """
        itos = [self.id_to_token(i) for i in range(len(self))]
        stoi = {tok: i for i, tok in enumerate(itos)}
        assert len(stoi) == len(itos), "token strings must be unique"
        vocab = Vocab(stoi, itos)
        vocab.set_default_index(self.unk_id)
        return vocab

    # ── training ────────────────────────────────────────────────

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int,
              min_frequency: int = 2,
              specials: Sequence[str] = SPECIALS) -> "BPETokenizer":
        """
        Learn merges from `texts` until the vocabulary reaches `vocab_size`
        or no adjacent pair occurs at least `min_frequency` times.

        Pair counts are updated incrementally (only words containing the
        merged pair are touched) and the best pair is kept in a lazy max-heap,
        so training is fast enough for the full Multi30k corpus.
        Ties are broken by the smaller token ids, making training deterministic.
        """
        tok = cls(specials=specials)
        n_merges = vocab_size - len(tok)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {len(tok)} "
                             f"({len(tok.specials)} specials + 256 bytes)")

        word_freqs: Counter = Counter()
        for text in texts:
            word_freqs.update(PRETOKENIZE.findall(text))

        words = [[tok._byte_offset + b for b in w.encode("utf-8")] for w in word_freqs]
        freqs = list(word_freqs.values())

        pair_counts: Dict[Pair, int] = defaultdict(int)
        where: Dict[Pair, set] = defaultdict(set)   # pair → indices of words containing it
        for i, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(i)

        heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
        heapq.heapify(heap)
        special_bytes = {s.encode("utf-8") for s in tok.specials}
        blocked: set = set()

        while len(tok.merges) < n_merges and heap:
            neg_count, a, b = heapq.heappop(heap)
            pair = (a, b)
            count = -neg_count
            if pair in blocked or pair_counts.get(pair, 0) != count:
                continue   # stale heap entry
            if count < min_frequency:
                break
            if tok._token_bytes[a] + tok._token_bytes[b] in special_bytes:
                blocked.add(pair)   # would collide with a special token's string
                continue

            new_id = tok._add_merge(pair)
            touched: set = set()
            for i in where.pop(pair, ()):
                old = words[i]
                new = _merge_pair(old, pair, new_id)
                if len(new) == len(old):
                    continue
                f = freqs[i]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                    touched.add(p)
                words[i] = new
            for p in touched:
                c = pair_counts[p]
                if c > 0:
                    heapq.heappush(heap, (-c, p[0], p[1]))
                else:
                    del pair_counts[p]

        return tok

    # ── encoding / decoding ─────────────────────────────────────

    def _encode_word(self, word: str) -> List[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [self._byte_offset + b for b in word.encode("utf-8")]
        n_base = self._byte_offset + 256
        while len(ids) >= 2:
            # Apply the earliest-learned merge present in the word
            best = min(zip(ids, ids[1:]),
                       key=lambda p: self._ranks.get(p, float("inf")))
            rank = self._ranks.get(best)
            if rank is None:
                break
            ids = _merge_pair(ids, best, n_base + rank)
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        """Text → token ids. Never emits <unk>: every byte has a token."""
        ids: List[int] = []
        for word in PRETOKENIZE.findall(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Token ids → text. Exact inverse of encode() for any input string."""
        out = bytearray()
        for i in ids:
            if i < self._byte_offset and skip_special_tokens:
                continue
            out += self._token_bytes[i]
        return out.decode("utf-8", errors="replace")

    def tokenize(self, text: str) -> List[str]:
        """Text → token strings (the form stored in to_vocab())."""
        return [self.id_to_token(i) for i in self.encode(text)]

    def detokenize(self, tokens: Iterable[str]) -> str:
        """Token strings → text; special tokens are dropped."""
        return self.decode(self.token_to_id(t) for t in tokens)

    # ── persistence ─────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {"type": "byte_bpe", "version": 1,
                "specials": self.specials,
                "merges": [list(p) for p in self.merges]}

    @classmethod
    def from_dict(cls, data: Dict) -> "BPETokenizer":
        if data.get("type") != "byte_bpe":
            raise ValueError("not a byte-level BPE tokenizer file")
        return cls(merges=[tuple(p) for p in data["merges"]],
                   specials=data["specials"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def tokenizers_from_checkpoint(
        ckpt: Dict) -> Optional[Tuple[BPETokenizer, BPETokenizer]]:
    """(src, tgt) tokenizers stored by `train.py --bpe`; None for spaCy checkpoints."""
    if ckpt.get("bpe_src") is None:
        return None
    return (BPETokenizer.from_dict(ckpt["bpe_src"]),
            BPETokenizer.from_dict(ckpt["bpe_tgt"]))
