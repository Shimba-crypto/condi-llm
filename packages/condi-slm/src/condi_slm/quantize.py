"""Symmetric per-channel int4/int8 quantization for SLM weight compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # <-- Fixed import here

@dataclass
class QuantizationConfig:
    bits: Literal[4, 8] = 4
    group_size: int = 128
    device: str = "cpu"


def _quantize_tensor(w: torch.Tensor, bits: int, group_size: int):
    """Quantize a 2D weight tensor along the output dimension."""
    if w.dim() == 1:
        return w, None
    out_f, in_f = w.shape
    g = min(group_size, in_f)
    w = w.reshape(out_f, in_f // g, g)
    mx = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = mx / (2 ** (bits - 1) - 1)
    q = torch.round(w / scale).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return q, scale


def _dequantize_tensor(q: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if scale is None:
        return q
    return (q * scale).reshape(q.shape[0], -1)


class _QuantizedLinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: QuantizationConfig):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        w = base.weight.detach()
        self.qw, self.scale = _quantize_tensor(w, cfg.bits, cfg.group_size)
        # Store quantized weights as float for portability; pack to int in production.
        self.register_buffer("qw", self.qw.float())
        if self.scale is not None:
            self.register_buffer("scale", self.scale.float())
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _dequantize_tensor(self.qw, getattr(self, "scale", None))
        return F.linear(x, w, self.bias)


def quantize(model: nn.Module, config: QuantizationConfig | None = None) -> nn.Module:
    """Replace every nn.Linear with a quantized equivalent. Returns a new model in-place."""
    cfg = config or QuantizationConfig()
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, _QuantizedLinear(module, cfg))
        else:
            quantize(module, cfg)
    return model
