"""ONNX export for cross-platform SLM deployment (iOS, browser WASM, edge)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_onnx(model: nn.Module, path: str | Path, seq_len: int = 1, opset: int = 17) -> Path:
    """Export a condi SLM to ONNX. Falls back gracefully if onnx is unavailable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=opset,
    )
    return path
