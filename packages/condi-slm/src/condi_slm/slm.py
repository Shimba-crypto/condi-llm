"""The SLM public API: load, generate, save, export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn as nn

from .cache import PagedKVCache
from .model import SLMConfig, SLMModel
from .quantize import QuantizationConfig, quantize
from .tokenizer import CondiTokenizer


# Registry of named preset configs (param count in name for convenience).
_PRESETS = {
    "condi-mini-1.2b": SLMConfig(vocab_size=32000, n_layers=24, n_heads=24, n_kv_heads=8, d_model=2048, d_ff=5632, max_seq_len=4096),
    "condi-tiny-350m": SLMConfig(vocab_size=32000, n_layers=18, n_heads=12, n_kv_heads=4, d_model=1024, d_ff=2816, max_seq_len=2048),
    "condi-nano-120m": SLMConfig(vocab_size=32000, n_layers=12, n_heads=8, n_kv_heads=4, d_model=512, d_ff=1408, max_seq_len=1024),
}


class SLM:
    """A loaded small language model ready for generation."""

    def __init__(self, model: SLMModel, tokenizer: CondiTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.eval()

    # ---- construction ---------------------------------------------------

    @classmethod
    def empty(cls, params: str = "1.2b", device: str = "cpu") -> "SLM":
        """Create an untrained model with the given parameter budget."""
        name = f"condi-mini-{params}" if not params.startswith("condi") else params
        cfg = _PRESETS.get(name) or _PRESETS["condi-nano-120m"]
        return cls(SLMModel(cfg), CondiTokenizer.tiny(), device=device)

    @classmethod
    def load(
        cls,
        name_or_path: str,
        quantization: Optional[str] = None,
        device: str = "cpu",
        kv_cache: str = "paged",
    ) -> "SLM":
        """Load a named preset or a checkpoint directory."""
        path = Path(name_or_path)
        if path.is_dir():
            cfg = SLMConfig(**json.loads((path / "config.json").read_text()))
            model = SLMModel(cfg)
            state = torch.load(path / "model.safetensors", map_location="cpu")
            model.load_state_dict(state, strict=False)
            tok = CondiTokenizer.from_file(path / "tokenizer.json")
        else:
            cfg = _PRESETS.get(name_or_path, _PRESETS["condi-nano-120m"])
            model = SLMModel(cfg)
            tok = CondiTokenizer.tiny()
        if quantization in ("int4", "int8"):
            model = quantize(model, QuantizationConfig(bits=4 if quantization == "int4" else 8))
        return cls(model, tok, device=device)

    # ---- persistence ----------------------------------------------------

    def save(self, name: str) -> Path:
        out = Path(name)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(self.model.cfg.__dict__, indent=2))
        torch.save(self.model.state_dict(), out / "model.safetensors")
        return out

    def export(self, path: str, fmt: str = "onnx") -> Path:
        if fmt != "onnx":
            raise ValueError(f"Unsupported export format: {fmt}")
        from .export import export_onnx
        return export_onnx(self.model, path)

    # ---- inference ------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        ids = self.tokenizer.encode(prompt)
        cache = PagedKVCache(
            n_layers=self.model.cfg.n_layers,
            n_heads=self.model.cfg.n_heads,
            head_dim=self.model.cfg.head_dim,
            device=self.device,
        )
        cache.allocate(seq_id=0)
        out_ids: List[int] = []
        for tok in self._stream(ids, max_tokens, temperature, top_k, top_p, cache, stop):
            out_ids.append(tok)
        return self.tokenizer.decode(out_ids)

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> Iterator[str]:
        """Yield decoded tokens one at a time for streaming UIs."""
        ids = self.tokenizer.encode(prompt)
        cache = PagedKVCache(
            n_layers=self.model.cfg.n_layers,
            n_heads=self.model.cfg.n_heads,
            head_dim=self.model.cfg.head_dim,
            device=self.device,
        )
        cache.allocate(seq_id=0)
        for tok in self._stream(ids, max_tokens, temperature, top_k, 1.0, cache, None):
            yield self.tokenizer.decode([tok])

    def _stream(self, ids, max_tokens, temperature, top_k, top_p, cache, stop) -> Iterator[int]:
        eos = self.tokenizer.eos_id
        # Prefill
        x = torch.tensor([ids], device=self.device, dtype=torch.long)
        logits = self.model(x, cache=cache, seq_id=0)[0, -1]
        for _ in range(max_tokens):
            tok = self._sample(logits, temperature, top_k, top_p)
            yield tok
            if tok == eos:
                break
            x = torch.tensor([[tok]], device=self.device, dtype=torch.long)
            logits = self.model(x, cache=cache, seq_id=0)[0, -1]

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
        if temperature <= 0:
            return int(logits.argmax().item())
        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[-1], torch.full_like(logits, float("-inf")), logits)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def batch_generate(self, prompts: List[str], max_tokens: int = 128) -> List[str]:
        """Generate for a batch of prompts (left-padded)."""
        enc = [self.tokenizer.encode(p) for p in prompts]
        max_len = max(len(e) for e in enc)
        pad = self.tokenizer.pad_id
        padded = [e + [pad] * (max_len - len(e)) for e in enc]
        x = torch.tensor(padded, device=self.device, dtype=torch.long)
        results = []
        for _ in range(max_tokens):
            logits = self.model(x)[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_tok], dim=1)
        for row in x:
            results.append(self.tokenizer.decode(row.tolist()))
        return results

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
