"""
ArbiterOmni Modular Training Pipeline.
Trains only the lightweight fusion network and decision head while keeping
multimodal encoders frozen.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arbiter_omni.data.dataset import (
    MultimodalDecisionDataset,
    collate_multimodal_decision,
)
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class ArbiterOmniTrainer:
    """
    Lightweight Trainer for ArbiterOmni.
    
    Verifies encoder parameter freezing, executes dynamic-batch cross-entropy training,
    and tracks calibration metrics (entropy, temperature, accuracy).
    """

    def __init__(
        self,
        model: ArbiterOmniModel,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()

        # Resolve compute device (CPU / CUDA / ROCm)
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Enforce that encoder parameters are frozen
        self.model.encoder.freeze()
        trainable = self.model.trainable_parameters()
        if len(trainable) == 0:
            raise ValueError("No trainable parameters found in fusion or decision head!")

        frozen_count = sum(p.numel() for p in self.model.encoder.parameters())
        trainable_count = sum(p.numel() for p in trainable)
        logger.info(
            f"Initialized ArbiterOmniTrainer on {self.device}. "
            f"Trainable params: {trainable_count:,} | Frozen encoder params: {frozen_count:,}"
        )

        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        self.model.encoder.eval()  # Keep encoder in eval mode (dropout/layernorm frozen)

        total_loss = 0.0
        correct = 0
        total_samples = 0
        total_entropy = 0.0

        for batch_idx, batch in enumerate(dataloader):
            targets = batch["targets"]
            if targets is None:
                continue

            targets = targets.to(self.device)
            self.optimizer.zero_grad()

            logits, probs, entropy, _ = self.model(
                questions=batch["questions"],
                candidates=batch["candidates"],
                texts=batch["texts"],
                images=batch["images"],
                videos=batch["videos"],
                audios=batch["audios"],
            )

            loss = self.criterion(logits, targets)
            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            # Metrics
            preds = torch.argmax(probs, dim=-1)
            correct += int((preds == targets).sum().item())
            total_samples += len(targets)
            total_loss += float(loss.item()) * len(targets)
            total_entropy += float(entropy.sum().item())

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        avg_entropy = total_entropy / max(1, total_samples)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "entropy": avg_entropy,
            "temperature": self.model.decision_head.temperature,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance on validation data."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0
        total_entropy = 0.0

        with torch.no_grad():
            for batch in dataloader:
                targets = batch["targets"]
                if targets is None:
                    continue

                targets = targets.to(self.device)

                logits, probs, entropy, _ = self.model(
                    questions=batch["questions"],
                    candidates=batch["candidates"],
                    texts=batch["texts"],
                    images=batch["images"],
                    videos=batch["videos"],
                    audios=batch["audios"],
                )

                loss = self.criterion(logits, targets)
                preds = torch.argmax(probs, dim=-1)

                correct += int((preds == targets).sum().item())
                total_samples += len(targets)
                total_loss += float(loss.item()) * len(targets)
                total_entropy += float(entropy.sum().item())

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        avg_entropy = total_entropy / max(1, total_samples)

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_entropy": avg_entropy,
        }

    def fit(
        self,
        train_dataset: MultimodalDecisionDataset,
        val_dataset: Optional[MultimodalDecisionDataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Executes end-to-end training loop for config.num_epochs.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_multimodal_decision,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_multimodal_decision,
            )

        history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "entropy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            history["loss"].append(train_metrics["loss"])
            history["accuracy"].append(train_metrics["accuracy"])
            history["entropy"].append(train_metrics["entropy"])

            val_str = ""
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_accuracy"].append(val_metrics["val_accuracy"])
                val_str = f" | Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']*100:.1f}%"

            logger.info(
                f"Epoch {epoch:02d}/{self.config.num_epochs:02d} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']*100:.1f}% | "
                f"Entropy: {train_metrics['entropy']:.3f} | "
                f"Temp: {train_metrics['temperature']:.2f}{val_str}"
            )

        if self.config.save_path:
            self.save_checkpoint(self.config.save_path)

        return history

    def save_checkpoint(self, path: str):
        """Saves only the trainable weights (fusion + decision head) and config."""
        state = {
            "fusion": self.model.fusion.state_dict(),
            "decision_head": self.model.decision_head.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(state, path)
        logger.info(f"Saved trainable checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Loads trained weights into fusion and decision head."""
        state = torch.load(path, map_location=self.device)
        self.model.fusion.load_state_dict(state["fusion"])
        self.model.decision_head.load_state_dict(state["decision_head"])
        logger.info(f"Loaded checkpoint from {path}")
