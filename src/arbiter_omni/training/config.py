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

    # Hard-Negative Contrastive Margin Loss
    contrastive_lambda: float = 0.0
    margin_gamma: float = 0.5

    # Modality Dropout Regularization [AO-14]
    modality_dropout_prob: float = 0.0  # Fraction of present sensory modalities dropped during training (e.g. 0.15)

    # Global Hard-Negative Memory Bank [AO-23]
    use_memory_bank: bool = False
    memory_bank_capacity: int = 50000
    memory_bank_k_foils: int = 10
    memory_bank_min_sim: float = 0.25
    memory_bank_max_sim: float = 0.98
    global_contrastive_lambda: float = 1.0
    global_margin_gamma: float = 0.5

