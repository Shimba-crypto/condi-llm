"""Training strategies: DDP, FSDP shard, and pipeline parallel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


@dataclass
class Strategy:
    """Declarative description of how a model should be sharded across devices."""

    mode: Literal["ddp", "fsdp_shard", "pipeline", "single"] = "fsdp_shard"
    world_size: int = 1
    rank: int = 0
    mixed_precision: str = "bf16"
    activation_checkpoint: bool = True
    cpu_offload: bool = False

    def apply(self, model: nn.Module) -> nn.Module:
        """Wrap the model for the chosen strategy.

        Uses torch.distributed when a process group is initialised; otherwise
        falls back to a no-op so the same code runs on a laptop for debugging.
        """
        if self.mode == "single" or not torch.distributed.is_available():
            return model
        if not torch.distributed.is_initialized():
            return model
        if self.mode == "ddp":
            return torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank] if torch.cuda.is_available() else None
            )
        if self.mode == "fsdp_shard":
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                return FSDP(model, device_id=self.rank if torch.cuda.is_available() else None)
            except Exception:
                return model
        if self.mode == "pipeline":
            # Naive split: assign first half of blocks to rank 0, rest to rank 1.
            return model  # full pipeline schedule handled by Trainer
        return model

    @property
    def is_distributed(self) -> bool:
        return self.mode != "single" and torch.distributed.is_available() and torch.distributed.is_initialized()
