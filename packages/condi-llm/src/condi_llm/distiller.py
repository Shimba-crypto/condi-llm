"""Knowledge distillation: train a small student from a large teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import trace


@dataclass
class Distiller:
    teacher: nn.Module
    student: nn.Module
    temperature: float = 2.0
    loss: str = "forward_kl"
    alpha: float = 0.5  # weight on distillation loss vs. supervised CE

    def __post_init__(self) -> None:
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def fit(self, dataset, epochs: int = 4, lr: float = 3e-4) -> dict:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        optim = torch.optim.AdamW(self.student.parameters(), lr=lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher.to(device)
        self.student.to(device)
        history = []
        for epoch in range(epochs):
            self.student.train()
            for batch in loader:
                ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
                with torch.no_grad():
                    t_logits = self.teacher(ids)
                s_logits = self.student(ids)
                loss = self._combine(s_logits, t_logits, ids)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                with trace.span("distill.step"):
                    pass
            history.append(loss.item())
        return {"epochs": epochs, "final_loss": history[-1] if history else None, "history": history}

    def _combine(self, s_logits: torch.Tensor, t_logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        # Distillation loss (KL between softened distributions).
        kl = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        ce = F.cross_entropy(
            s_logits[:, :-1, :].reshape(-1, s_logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        return self.alpha * kl + (1 - self.alpha) * ce
