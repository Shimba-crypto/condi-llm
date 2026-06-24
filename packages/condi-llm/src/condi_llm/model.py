"""AutoModel + large-scale transformer with FSDP-friendly shardable layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMConfig:
    vocab_size: int = 128000
    n_layers: int = 80
    n_heads: int = 64
    n_kv_heads: int = 8
    d_model: int = 8192
    d_ff: int = 28672
    max_seq_len: int = 32768
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


_PRESETS = {
    "condi-7b": LLMConfig(vocab_size=128000, n_layers=32, n_heads=32, n_kv_heads=8, d_model=4096, d_ff=14336, max_seq_len=8192),
    "condi-70b": LLMConfig(vocab_size=128000, n_layers=80, n_heads=64, n_kv_heads=8, d_model=8192, d_ff=28672, max_seq_len=32768),
    "condi-405b": LLMConfig(vocab_size=128000, n_layers=126, n_heads=128, n_kv_heads=8, d_model=16384, d_ff=53248, max_seq_len=32768),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        Dh = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * Dh, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * Dh, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * Dh, bias=False)
        self.wo = nn.Linear(cfg.n_heads * Dh, cfg.d_model, bias=False)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, Dh, 2).float() / Dh))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        freqs = torch.outer(torch.arange(T, device=x.device, dtype=self.inv_freq.dtype), self.inv_freq)
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q = self._rope(self.wq(x).view(B, T, H, Dh))
        k = self._rope(self.wk(x).view(B, T, self.cfg.n_kv_heads, Dh))
        v = self.wv(x).view(B, T, self.cfg.n_kv_heads, Dh)
        rep = H // self.cfg.n_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(attn.transpose(1, 2).contiguous().view(B, T, H * Dh))


class FeedForward(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LLMModel(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AutoModel:
    """Factory that loads pretrained configs and checkpoints."""

    @staticmethod
    def from_pretrained(name_or_path: str, device: Optional[str] = None) -> LLMModel:
        path = Path(name_or_path)
        if path.is_dir():
            cfg = LLMConfig(**json.loads((path / "config.json").read_text()))
            model = LLMModel(cfg)
            state = torch.load(path / "model.safetensors", map_location="cpu")
            model.load_state_dict(state, strict=False)
        else:
            cfg = _PRESETS.get(name_or_path, _PRESETS["condi-7b"])
            model = LLMModel(cfg)
        if device:
            model.to(device)
        return model

    @staticmethod
    def available_models() -> list[str]:
        return list(_PRESETS.keys())
