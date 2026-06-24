"""condi-llm: distributed training and serving for large language models."""

from . import trace
from .model import AutoModel
from .trainer import Trainer, TrainingConfig
from .distiller import Distiller
from .strategy import Strategy

__version__ = "0.9.0"

__all__ = [
    "AutoModel",
    "Trainer",
    "TrainingConfig",
    "Distiller",
    "Strategy",
    "trace",
    "__version__",
]
