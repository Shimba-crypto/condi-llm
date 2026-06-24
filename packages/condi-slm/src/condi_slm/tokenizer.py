"""Condi tokenizer — a compact BPE tokenizer shared with condi-llm."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Byte-level token IDs (mirrors GPT-2's byte->unicode mapping)
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE | re.IGNORECASE,
)


def _byte_encoder() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class CondiTokenizer:
    """A minimal BPE tokenizer compatible with the condi 128k vocabulary."""

    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self._enc = _byte_encoder()
        self._dec = {v: k for k, v in self._enc.items()}
        self.pad_id = vocab.get("<|pad|>", 0)
        self.bos_id = vocab.get("<|bos|>", 1)
        self.eos_id = vocab.get("<|eos|>", 2)

    @classmethod
    def from_file(cls, path: str | Path) -> "CondiTokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab=data["vocab"], merges=[tuple(m) for m in data["merges"]])

    @classmethod
    def tiny(cls) -> "CondiTokenizer":
        """A built-in byte-level fallback used when no vocab file is present."""
        vocab = {chr(i): i for i in range(256)}
        vocab.update({"<|pad|>": 256, "<|bos|>": 257, "<|eos|>": 258})
        return cls(vocab=vocab, merges=[])

    def _bpe(self, tokens: List[str]) -> List[str]:
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            merged = best[0] + best[1]
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids: List[int] = []
        if add_special:
            ids.append(self.bos_id)
        for chunk in PAT.findall(text):
            chunk_bytes = chunk.encode("utf-8")
            symbols = [self._enc[b] for b in chunk_bytes]
            for tok in self._bpe(symbols):
                ids.append(self.vocab.get(tok, self.vocab.get("<|unk|>", self.pad_id)))
        return ids

    def decode(self, ids: List[int]) -> str:
        text = "".join(self.inv_vocab.get(i, "") for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id))
        byte_array = bytes(self._dec.get(ord(c), ord("?")) for c in text)
        return byte_array.decode("utf-8", errors="replace")
