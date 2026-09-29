"""
Training configuration parameters for ArbiterOmni.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for ArbiterOmni training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 5
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    device: Optional[str] = None
    save_path: Optional[str] = None
    log_interval: int = 5
