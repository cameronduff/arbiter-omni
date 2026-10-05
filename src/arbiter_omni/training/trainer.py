"""
ArbiterOmni Modular Training Pipeline.
Trains only the lightweight fusion network and decision head while keeping
multimodal encoders frozen.
Supports GPU batch scaling, mixed precision (FP16/AMP), and throughput telemetry.
"""

from __future__ import annotations

import logging
import os
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
from arbiter_omni.model.decision_head import contrastive_margin_loss
from arbiter_omni.training.config import TrainingConfig
from arbiter_omni.training.telemetry import memory_summary


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
        if hasattr(self.config, "modality_dropout_prob") and self.config.modality_dropout_prob > 0.0:
            self.model.modality_dropout_prob = self.config.modality_dropout_prob

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

        # Global Hard-Negative Memory Bank [AO-23]
        if getattr(self.config, "use_memory_bank", False):
            from arbiter_omni.data.memory_bank import PersistentMemoryBank
            self.memory_bank = PersistentMemoryBank(
                capacity=self.config.memory_bank_capacity,
                candidate_dim=self.model.encoder.text_dim,
                context_dim=getattr(self.model.fusion, "fusion_dim", None),
                device="cpu",
            )
            logger.info(
                f"Initialized PersistentMemoryBank [AO-23] (capacity: {self.config.memory_bank_capacity:,}, "
                f"dim: {self.model.encoder.text_dim}, shared RAM: {self.memory_bank.memory_usage_mb:.2f} MB)"
            )
        else:
            self.memory_bank = None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch with mixed precision and gradient accumulation."""
        self.model.train()
        self.model.encoder.eval()  # Keep encoder in eval mode (dropout/layernorm frozen)

        total_loss = 0.0
        total_margin_loss = 0.0
        correct = 0
        total_samples = 0
        total_entropy = 0.0
        accum_steps = max(1, self.config.accumulate_grad_batches)
        num_batches = len(dataloader)
        last_grad_norm = 0.0

        start_time = time.perf_counter()
        self.optimizer.zero_grad()

        if getattr(self.config, "async_prefetch", True) and len(dataloader) > 1:
            from arbiter_omni.training.prefetcher import AsyncDMADataPrefetcher
            data_iter = AsyncDMADataPrefetcher(
                dataloader=dataloader,
                device=self.device,
                queue_size=getattr(self.config, "prefetch_queue_size", 2),
            )
        else:
            data_iter = dataloader

        for batch_idx, batch in enumerate(data_iter):
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
                    cand_mask = batch["candidate_mask"].to(self.device)
                    batch_patches = batch.get("image_patches", None)
                    if batch_patches is not None and isinstance(batch_patches, torch.Tensor):
                        batch_patches = batch_patches.to(self.device)
                    logits, probs, entropy, fused_context = self.model.forward_cached(
                        question_embed=batch["question_embed"].to(self.device),
                        modality_embeds={k: v.to(self.device) for k, v in batch["modality_embeds"].items()},
                        presence_mask={k: v.to(self.device) for k, v in batch["presence_mask"].items()},
                        candidate_embeds=batch["candidate_embeds"].to(self.device),
                        candidate_mask=cand_mask,
                        image_patches=batch_patches,
                    )
                else:
                    cand_mask = batch.get("candidate_mask", None)
                    if cand_mask is not None and isinstance(cand_mask, torch.Tensor):
                        cand_mask = cand_mask.to(self.device)
                    logits, probs, entropy, fused_context = self.model(
                        questions=batch["questions"],
                        candidates=batch["candidates"],
                        texts=batch["texts"],
                        images=batch["images"],
                        videos=batch["videos"],
                        audios=batch["audios"],
                    )

                # Global Hard-Negative Memory Bank Contrast [AO-23]
                global_foil_logits = None
                global_foil_mask = None
                if self.memory_bank is not None:
                    batch_cands = None
                    if batch.get("is_cached", False):
                        batch_cands = batch["candidate_embeds"]
                    elif "candidate_embeds" in batch and isinstance(batch["candidate_embeds"], torch.Tensor):
                        batch_cands = batch["candidate_embeds"]
                    elif hasattr(self.model, "encode_candidates"):
                        batch_cands, cand_m = self.model.encode_candidates(batch["candidates"])
                        if cand_mask is None:
                            cand_mask = cand_m

                    if batch_cands is not None and batch_cands.numel() > 0:
                        B_curr = batch_cands.shape[0]
                        pos_indices = targets.clamp(0, batch_cands.shape[1] - 1)
                        pos_embeds = batch_cands[torch.arange(B_curr, device=batch_cands.device), pos_indices]

                        if len(self.memory_bank) > 0:
                            foil_embeds, _, foil_mask = self.memory_bank.query_hard_foils(
                                query_embed=pos_embeds,
                                k=self.config.memory_bank_k_foils,
                                min_sim=self.config.memory_bank_min_sim,
                                max_sim=self.config.memory_bank_max_sim,
                            )
                            if foil_mask.any():
                                global_foil_logits = self.model.decision_head.score_foils(
                                    context_embed=fused_context,
                                    foil_embeds=foil_embeds.to(self.device),
                                )
                                global_foil_mask = foil_mask.to(self.device)

                        # Enqueue in-batch candidate representations into resident memory bank
                        self.memory_bank.enqueue(
                            candidate_embeds=batch_cands,
                            candidate_mask=cand_mask if cand_mask is not None else None,
                            context_embeds=fused_context,
                        )

                ce_loss = self.criterion(logits, targets)
                if self.config.contrastive_lambda > 0.0:
                    margin_loss = contrastive_margin_loss(
                        logits=logits,
                        targets=targets,
                        candidate_mask=cand_mask,
                        margin=self.config.margin_gamma,
                        global_foil_logits=global_foil_logits,
                        global_foil_mask=global_foil_mask,
                        global_margin=self.config.global_margin_gamma,
                        global_lambda=self.config.global_contrastive_lambda,
                    )
                    raw_loss = ce_loss + self.config.contrastive_lambda * margin_loss
                else:
                    margin_loss = torch.tensor(0.0, device=self.device)
                    raw_loss = ce_loss

                # Sparse MoE Load-Balancing Aux Loss [AO-27, AO-28]
                if hasattr(self.model, "moe_aux_loss"):
                    aux = self.model.moe_aux_loss
                    if isinstance(aux, torch.Tensor) and aux.item() > 0:
                        raw_loss = raw_loss + getattr(self.config, "moe_aux_lambda", 0.01) * aux.to(self.device)

                loss = raw_loss / accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(), self.config.grad_clip
                    )
                    last_grad_norm = float(grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Metrics
            preds = torch.argmax(probs, dim=-1)
            correct += int((preds == targets).sum().item())
            total_samples += len(targets)
            total_loss += float(raw_loss.item()) * len(targets)
            total_margin_loss += float(margin_loss.item()) * len(targets)
            total_entropy += float(entropy.sum().item())

            # Verbose per-batch progress telemetry
            log_every = int(getattr(self.config, "log_interval", 0) or 0)
            if log_every > 0 and ((batch_idx + 1) % log_every == 0 or (batch_idx + 1) == num_batches):
                done = batch_idx + 1
                elapsed_now = max(1e-5, time.perf_counter() - start_time)
                sec_per_batch = elapsed_now / done
                eta_min = sec_per_batch * (num_batches - done) / 60.0
                aux_val = 0.0
                aux_t = getattr(self.model, "moe_aux_loss", None)
                if isinstance(aux_t, torch.Tensor):
                    aux_val = float(aux_t.item())
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  [train {done:>4d}/{num_batches}] "
                    f"loss={total_loss / max(1, total_samples):.4f} "
                    f"batch_loss={float(raw_loss.item()):.4f} "
                    f"ce={float(ce_loss.item()):.4f} "
                    f"margin={float(margin_loss.item()):.4f} "
                    f"moe_aux={aux_val:.4f} "
                    f"acc={100.0 * correct / max(1, total_samples):.1f}% "
                    f"grad_norm={last_grad_norm:.3f} "
                    f"lr={current_lr:.2e} "
                    f"{total_samples / elapsed_now:.2f} samples/s "
                    f"epoch_eta={eta_min:.1f}min "
                    f"{memory_summary()}"
                )
        elapsed = max(1e-5, time.perf_counter() - start_time)
        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        avg_entropy = total_entropy / max(1, total_samples)
        samples_per_sec = total_samples / elapsed

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "entropy": avg_entropy,
            "margin_loss": total_margin_loss / max(1, total_samples),
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
        total_margin_loss = 0.0
        correct = 0
        total_samples = 0
        total_entropy = 0.0

        if getattr(self.config, "async_prefetch", True) and len(dataloader) > 1:
            from arbiter_omni.training.prefetcher import AsyncDMADataPrefetcher
            val_iter = AsyncDMADataPrefetcher(
                dataloader=dataloader,
                device=self.device,
                queue_size=getattr(self.config, "prefetch_queue_size", 2),
            )
        else:
            val_iter = dataloader

        with torch.no_grad():
            for batch in val_iter:
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
                        cand_mask = batch["candidate_mask"].to(self.device)
                        batch_patches = batch.get("image_patches", None)
                        if batch_patches is not None and isinstance(batch_patches, torch.Tensor):
                            batch_patches = batch_patches.to(self.device)
                        logits, probs, entropy, fused_context = self.model.forward_cached(
                            question_embed=batch["question_embed"].to(self.device),
                            modality_embeds={k: v.to(self.device) for k, v in batch["modality_embeds"].items()},
                            presence_mask={k: v.to(self.device) for k, v in batch["presence_mask"].items()},
                            candidate_embeds=batch["candidate_embeds"].to(self.device),
                            candidate_mask=cand_mask,
                            image_patches=batch_patches,
                        )
                    else:
                        cand_mask = batch.get("candidate_mask", None)
                        if cand_mask is not None and isinstance(cand_mask, torch.Tensor):
                            cand_mask = cand_mask.to(self.device)
                        logits, probs, entropy, fused_context = self.model(
                            questions=batch["questions"],
                            candidates=batch["candidates"],
                            texts=batch["texts"],
                            images=batch["images"],
                            videos=batch["videos"],
                            audios=batch["audios"],
                        )

                    # Global Hard-Negative Memory Bank Contrast [AO-23]
                    global_foil_logits = None
                    global_foil_mask = None
                    if self.memory_bank is not None and len(self.memory_bank) > 0:
                        batch_cands = None
                        if batch.get("is_cached", False):
                            batch_cands = batch["candidate_embeds"]
                        elif "candidate_embeds" in batch and isinstance(batch["candidate_embeds"], torch.Tensor):
                            batch_cands = batch["candidate_embeds"]
                        elif hasattr(self.model, "encode_candidates"):
                            batch_cands, cand_m = self.model.encode_candidates(batch["candidates"])
                            if cand_mask is None:
                                cand_mask = cand_m

                        if batch_cands is not None and batch_cands.numel() > 0:
                            B_curr = batch_cands.shape[0]
                            pos_indices = targets.clamp(0, batch_cands.shape[1] - 1)
                            pos_embeds = batch_cands[torch.arange(B_curr, device=batch_cands.device), pos_indices]
                            foil_embeds, _, foil_mask = self.memory_bank.query_hard_foils(
                                query_embed=pos_embeds,
                                k=self.config.memory_bank_k_foils,
                                min_sim=self.config.memory_bank_min_sim,
                                max_sim=self.config.memory_bank_max_sim,
                            )
                            if foil_mask.any():
                                global_foil_logits = self.model.decision_head.score_foils(
                                    context_embed=fused_context,
                                    foil_embeds=foil_embeds.to(self.device),
                                )
                                global_foil_mask = foil_mask.to(self.device)

                    ce_loss = self.criterion(logits, targets)
                    if self.config.contrastive_lambda > 0.0:
                        margin_loss = contrastive_margin_loss(
                            logits=logits,
                            targets=targets,
                            candidate_mask=cand_mask,
                            margin=self.config.margin_gamma,
                            global_foil_logits=global_foil_logits,
                            global_foil_mask=global_foil_mask,
                            global_margin=self.config.global_margin_gamma,
                            global_lambda=self.config.global_contrastive_lambda,
                        )
                        loss = ce_loss + self.config.contrastive_lambda * margin_loss
                    else:
                        margin_loss = torch.tensor(0.0, device=self.device)
                        loss = ce_loss

                    # Sparse MoE Load-Balancing Aux Loss [AO-27, AO-28]
                    if hasattr(self.model, "moe_aux_loss"):
                        aux = self.model.moe_aux_loss
                        if isinstance(aux, torch.Tensor) and aux.item() > 0:
                            loss = loss + getattr(self.config, "moe_aux_lambda", 0.01) * aux.to(self.device)

                preds = torch.argmax(probs, dim=-1)
                correct += int((preds == targets).sum().item())
                total_samples += len(targets)
                total_loss += float(loss.item()) * len(targets)
                total_margin_loss += float(margin_loss.item()) * len(targets)
                total_entropy += float(entropy.sum().item())

        avg_loss = total_loss / max(1, total_samples)
        accuracy = correct / max(1, total_samples)
        avg_entropy = total_entropy / max(1, total_samples)

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_entropy": avg_entropy,
            "val_margin_loss": total_margin_loss / max(1, total_samples),
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
            "margin_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_entropy": [],
            "val_margin_loss": [],
        }

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            history["loss"].append(train_metrics["loss"])
            history["accuracy"].append(train_metrics["accuracy"])
            history["entropy"].append(train_metrics["entropy"])
            history["margin_loss"].append(train_metrics.get("margin_loss", 0.0))

            val_str = ""
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_accuracy"].append(val_metrics["val_accuracy"])
                history["val_entropy"].append(val_metrics.get("val_entropy", 0.0))
                history["val_margin_loss"].append(val_metrics.get("val_margin_loss", 0.0))
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
        fusion_mod = self.model.fusion
        state = {
            "fusion": fusion_mod.state_dict(),
            "decision_head": self.model.decision_head.state_dict(),
            "config": self.config.__dict__,
            "model_config": {
                "hidden_dim": getattr(fusion_mod, "hidden_dim", 256),
                "num_heads": getattr(fusion_mod, "num_heads", 4),
                "num_layers": getattr(fusion_mod, "num_layers", 2),
                "scoring_dim": getattr(self.model.decision_head, "scoring_dim", 256),
                "model_name": getattr(self.model.encoder, "model_name", "ViT-B-32"),
                "enable_spatial_cross_attention": getattr(fusion_mod, "enable_spatial_cross_attention", False),
                "max_spatial_patches": getattr(fusion_mod, "max_spatial_patches", 196),
                "use_moe": getattr(fusion_mod, "use_moe", False),
                "moe_num_layers": getattr(getattr(fusion_mod, "moe_transformer", None), "num_moe_layers", 4),
                "moe_num_experts": getattr(getattr(fusion_mod, "moe_transformer", None), "num_experts", 4),
                "moe_top_k": getattr(getattr(fusion_mod, "moe_transformer", None), "top_k", 2),
                "use_shared_expert": getattr(fusion_mod, "use_shared_expert", False),
                "enable_speculative_early_exit": getattr(self.model, "enable_speculative_early_exit", False),
            },
        }
        if getattr(self.model, "speculative_head", None) is not None:
            state["speculative_head"] = self.model.speculative_head.state_dict()

        # torch.save() on DirectML tensors is killed by the OS (SIGKILL) at around 1 GB of
        # state, so copy each tensor to CPU first, and write atomically so a failed save can
        # never leave a truncated checkpoint behind.
        def _to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().to("cpu")
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            return obj

        state = _to_cpu(state)
        tmp_path = f"{path}.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"Saved trainable checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Loads trained weights into fusion and decision head."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        self.model.fusion.load_state_dict(state["fusion"])
        self.model.decision_head.load_state_dict(state["decision_head"])
        logger.info(f"Loaded checkpoint from {path}")
