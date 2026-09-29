"""
ArbiterOmni Modular Training Pipeline.
Trains only the lightweight fusion network and decision head while keeping
multimodal encoders frozen.
Supports GPU batch scaling, mixed precision (FP16/AMP), and throughput telemetry.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arbiter_omni.data.dataset import (
    MultimodalDecisionDataset,
    collate_multimodal_decision,
)
from arbiter_omni.data.cached import (
    CachedMultimodalDataset,
    collate_cached_multimodal_decision,
)
from arbiter_omni.device import resolve_device, get_device_telemetry
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class ArbiterOmniTrainer:
    """
    Lightweight Trainer for ArbiterOmni.
    
    Verifies encoder parameter freezing, executes dynamic-batch cross-entropy training,
    tracks calibration metrics (entropy, temperature, accuracy), and accelerates
    execution via PyTorch AMP mixed precision and gradient accumulation.
    """

    def __init__(
        self,
        model: ArbiterOmniModel,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()

        # Resolve compute device (CPU / CUDA / ROCm / DirectML)
        self.device = resolve_device(self.config.device)
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

        # Setup AMP Mixed Precision
        use_amp = self.config.fp16 and (self.device.type in ("cuda", "cpu"))
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        self.amp_enabled = use_amp
        
        # GradScaler only applies to CUDA
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.config.fp16 and self.device.type == "cuda")
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch with mixed precision and gradient accumulation."""
        self.model.train()
        self.model.encoder.eval()  # Keep encoder in eval mode (dropout/layernorm frozen)

        total_loss = 0.0
        correct = 0
        total_samples = 0
        total_entropy = 0.0
        accum_steps = max(1, self.config.accumulate_grad_batches)

        start_time = time.perf_counter()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            targets = batch["targets"]
            if targets is None:
                continue

            targets = targets.to(self.device)

            with torch.amp.autocast(
                device_type=self.amp_device_type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                if batch.get("is_cached", False):
                    logits, probs, entropy, _ = self.model.forward_cached(
                        question_embed=batch["question_embed"].to(self.device),
                        modality_embeds={k: v.to(self.device) for k, v in batch["modality_embeds"].items()},
                        presence_mask={k: v.to(self.device) for k, v in batch["presence_mask"].items()},
                        candidate_embeds=batch["candidate_embeds"].to(self.device),
                        candidate_mask=batch["candidate_mask"].to(self.device),
                    )
                else:
                    logits, probs, entropy, _ = self.model(
                        questions=batch["questions"],
                        candidates=batch["candidates"],
                        texts=batch["texts"],
                        images=batch["images"],
                        videos=batch["videos"],
                        audios=batch["audios"],
                    )
                raw_loss = self.criterion(logits, targets)
                loss = raw_loss / accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Metrics
            preds = torch.argmax(probs, dim=-1)
            correct += int((preds == targets).sum().item())
            total_samples += len(targets)
            total_loss += float(raw_loss.item()) * len(targets)
            total_entropy += float(entropy.sum().item())

        elapsed = max(1e-5, time.perf_counter() - start_time)
        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        avg_entropy = total_entropy / max(1, total_samples)
        samples_per_sec = total_samples / elapsed

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "entropy": avg_entropy,
            "temperature": self.model.decision_head.temperature,
            "samples_per_sec": samples_per_sec,
        }

        if self.device.type == "cuda" and torch.cuda.is_available():
            metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024**2)

        return metrics

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

                with torch.amp.autocast(
                    device_type=self.amp_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    if batch.get("is_cached", False):
                        logits, probs, entropy, _ = self.model.forward_cached(
                            question_embed=batch["question_embed"].to(self.device),
                            modality_embeds={k: v.to(self.device) for k, v in batch["modality_embeds"].items()},
                            presence_mask={k: v.to(self.device) for k, v in batch["presence_mask"].items()},
                            candidate_embeds=batch["candidate_embeds"].to(self.device),
                            candidate_mask=batch["candidate_mask"].to(self.device),
                        )
                    else:
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
        # Select collate_fn based on whether dataset contains pre-cached embeddings
        is_train_cached = isinstance(train_dataset, CachedMultimodalDataset) or (
            len(train_dataset) > 0 and hasattr(train_dataset[0], "question_embed")
        )
        train_collate = collate_cached_multimodal_decision if is_train_cached else collate_multimodal_decision

        pin_mem = self.config.pin_memory and (self.device.type == "cuda")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=train_collate,
            pin_memory=pin_mem,
            num_workers=self.config.num_workers,
        )

        val_loader = None
        if val_dataset is not None:
            is_val_cached = isinstance(val_dataset, CachedMultimodalDataset) or (
                len(val_dataset) > 0 and hasattr(val_dataset[0], "question_embed")
            )
            val_collate = collate_cached_multimodal_decision if is_val_cached else collate_multimodal_decision
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=val_collate,
                pin_memory=pin_mem,
                num_workers=self.config.num_workers,
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

            speed_str = f"{train_metrics.get('samples_per_sec', 0.0):.1f} samples/s"
            logger.info(
                f"Epoch {epoch:02d}/{self.config.num_epochs:02d} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']*100:.1f}% | "
                f"Entropy: {train_metrics['entropy']:.3f} | "
                f"Speed: {speed_str}{val_str}"
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
