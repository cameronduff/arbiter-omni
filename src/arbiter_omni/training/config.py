"""
Training configuration parameters for ArbiterOmni.
Supports GPU batch scaling, mixed precision (FP16/AMP), and gradient accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Hyperparameters and hardware configuration for ArbiterOmni training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 5
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    device: Optional[str] = None
    save_path: Optional[str] = None
    log_interval: int = 5

    # GPU scaling & Mixed Precision
    fp16: bool = True
    accumulate_grad_batches: int = 1
    pin_memory: bool = True
    num_workers: int = 0
