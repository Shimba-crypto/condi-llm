"""Shared condi tokenizer (re-exports the condi-slm implementation)."""

from __future__ import annotations

try:
    from condi_slm.tokenizer import CondiTokenizer  # type: ignore
except Exception:  # pragma: no cover - condi-slm optional at runtime
    CondiTokenizer = None  # type: ignore

__all__ = ["CondiTokenizer"]
