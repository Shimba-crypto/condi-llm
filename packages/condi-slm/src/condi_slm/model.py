"""A compact decoder-only transformer for the condi-mini family."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SLMConfig:
    vocab_size: int = 32000
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _rope(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, D) ; freqs: (T, D/2)
    T = x.shape[1]
    cos = freqs[:T].cos()[None, :, None, :]
    sin = freqs[:T].sin()[None, :, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rot.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg
        self.n_kv = cfg.n_kv_heads
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.freqs = self._build_freqs(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)

    @staticmethod
    def _build_freqs(T: int, D: int, theta: float) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
        t = torch.arange(T).float()
        return torch.outer(t, freqs)

    def forward(self, x: torch.Tensor, cache=None, seq_id: int = 0, layer: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q = self.wq(x).view(B, T, H, Dh)
        k = self.wk(x).view(B, T, self.n_kv, Dh)
        v = self.wv(x).view(B, T, self.n_kv, Dh)

        q = _rope(self.freqs.to(x.device), q)
        k = _rope(self.freqs.to(x.device), k)

        # Expand GQA: repeat KV heads to match Q heads.
        rep = H // self.n_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

        # Use cache if provided (single-token decode path).
        if cache is not None and T == 1:
            cache.append(seq_id, layer, k[0, 0], v[0, 0])
            kk, vv = cache.get(seq_id, layer)  # (S, H, Dh)
            kk = kk.transpose(0, 1)[None]  # (1, H, S, Dh)
            vv = vv.transpose(0, 1)[None]
            scores = (q.transpose(1, 2) @ kk.transpose(-2, -1)) / math.sqrt(Dh)
            mask = None
        else:
            kk = k.transpose(1, 2)  # (B, H, T, Dh)
            vv = v.transpose(1, 2)
            scores = (q.transpose(1, 2) @ kk.transpose(-2, -1)) / math.sqrt(Dh)
            mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
            scores = scores + mask

        attn = F.softmax(scores, dim=-1) @ vv  # (B, H, T, Dh)
        attn = attn.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.wo(attn)


class FeedForward(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cache=None, seq_id=0, layer=0):
        x = x + self.attn(self.norm1(x), cache=cache, seq_id=seq_id, layer=layer)
        x = x + self.ffn(self.norm2(x))
        return x


class SLMModel(nn.Module):
    """The raw transformer — weights + forward only. The `SLM` wrapper adds I/O."""

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, ids: torch.Tensor, cache=None, seq_id: int = 0) -> torch.Tensor:
        x = self.tok_emb(ids)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, seq_id=seq_id, layer=i)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
