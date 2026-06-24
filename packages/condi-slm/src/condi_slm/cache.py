"""Paged key-value cache for efficient autoregressive generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class PagedKVCache:
    """Block-paged KV cache that avoids fragmentation on long contexts.

    Memory is allocated in fixed-size blocks; sequences reference blocks
    rather than owning contiguous rows, so prefix caching and batch padding
    are nearly free.
    """

    n_layers: int
    n_heads: int
    head_dim: int
    block_size: int = 16
    max_blocks: int = 1024
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)

    def __post_init__(self) -> None:
        # Free list of block indices.
        self._free: List[int] = list(range(self.max_blocks))
        # Per-layer storage: (block, n_heads, block_size, head_dim)
        shape = (self.max_blocks, self.n_heads, self.block_size, self.head_dim)
        self.k = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.n_layers)]
        self.v = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.n_layers)]
        # sequence_id -> list of block indices
        self._blocks: dict[int, List[int]] = {}
        # sequence_id -> current position within the last block
        self._pos: dict[int, int] = {}

    def allocate(self, seq_id: int) -> None:
        self._blocks[seq_id] = []
        self._pos[seq_id] = 0

    def append(self, seq_id: int, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one new token's K/V (shape: n_heads, head_dim) for a sequence."""
        if seq_id not in self._blocks:
            self.allocate(seq_id)
        if self._pos[seq_id] == 0 or self._pos[seq_id] >= self.block_size:
            if not self._free:
                raise RuntimeError("PagedKVCache exhausted: no free blocks")
            self._blocks[seq_id].append(self._free.pop())
            self._pos[seq_id] = 0
        block_idx = self._blocks[seq_id][-1]
        slot = self._pos[seq_id]
        self.k[layer][block_idx, :, slot, :] = k.to(self.dtype).to(self.device)
        self.v[layer][block_idx, :, slot, :] = v.to(self.dtype).to(self.device)
        self._pos[seq_id] += 1

    def get(self, seq_id: int, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (K, V) of shape (seq_len, n_heads, head_dim) for the sequence."""
        blocks = self._blocks.get(seq_id, [])
        if not blocks:
            return (
                torch.zeros(0, self.n_heads, self.head_dim, device=self.device),
                torch.zeros(0, self.n_heads, self.head_dim, device=self.device),
            )
        k = self.k[layer][blocks]  # (n_blocks, n_heads, block_size, head_dim)
        v = self.v[layer][blocks]
        k = k.reshape(-1, self.n_heads, self.head_dim)[: self.length(seq_id)]
        v = v.reshape(-1, self.n_heads, self.head_dim)[: self.length(seq_id)]
        return k, v

    def length(self, seq_id: int) -> int:
        return (len(self._blocks.get(seq_id, [])) - 1) * self.block_size + self._pos.get(seq_id, 0)

    def free(self, seq_id: int) -> None:
        self._free.extend(self._blocks.pop(seq_id, []))
        self._pos.pop(seq_id, None)
