"""Trainer with gradient accumulation, mixed precision, and FSDP support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import trace
from .strategy import Strategy


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100_000
    grad_accum: int = 32
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 2000
    strategy: Strategy = field(default_factory=Strategy)


def _lr_schedule(step: int, max_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


class Trainer:
    """Orchstrates a training loop with sharding, accumulation, and tracing."""

    def __init__(self, model: nn.Module, config: Optional[TrainingConfig] = None, strategy: Optional[Strategy] = None, grad_accum: Optional[int] = None):
        if strategy is not None:
            config = config or TrainingConfig()
            config.strategy = strategy
        if grad_accum is not None:
            config = config or TrainingConfig()
            config.grad_accum = grad_accum
        self.config = config or TrainingConfig()
        self.raw_model = model
        self.model = self.config.strategy.apply(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.step = 0

    @classmethod
    def load(cls, name: str, **kwargs) -> "Trainer":
        from .model import AutoModel
        model = AutoModel.from_pretrained(name)
        return cls(model, **kwargs)

    def fit(self, dataset, epochs: int = 1, loss_fn: Optional[Callable] = None) -> dict:
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        loss_fn = loss_fn or self._ce_loss
        history: List[float] = []
        total_steps = epochs * len(loader) // max(1, self.config.grad_accum)
        for epoch in range(epochs):
            self.model.train()
            accum = 0
            self.optim.zero_grad(set_to_none=True)
            for batch in loader:
                ids = batch["input_ids"].to(self.device) if isinstance(batch, dict) else batch.to(self.device)
                with trace.span("train.forward"):
                    logits = self.model(ids)
                    loss = loss_fn(logits, ids) / self.config.grad_accum
                with trace.span("train.backward"):
                    loss.backward()
                accum += 1
                if accum >= self.config.grad_accum:
                    self._step()
                    history.append(loss.item() * self.config.grad_accum)
                    if self.step >= self.config.max_steps:
                        break
            if self.step >= self.config.max_steps:
                break
        return {"steps": self.step, "final_loss": history[-1] if history else None, "history": history}

    def _step(self) -> None:
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        lr = self.config.lr * _lr_schedule(self.step, self.config.max_steps, self.config.warmup_steps)
        for pg in self.optim.param_groups:
            pg["lr"] = lr
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        self.step += 1
        if self.step % self.config.log_every == 0:
            trace.log({"step": self.step, "lr": lr})

    @staticmethod
    def _ce_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # Shift for next-token prediction.
        return nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
