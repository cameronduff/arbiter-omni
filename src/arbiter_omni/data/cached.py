"""
ArbiterOmni Pre-Cached Multimodal Dataset.
Pre-extracts invariant latent representations from frozen multimodal encoders
(OpenCLIP, CLAP, Temporal Video Attention) once, eliminating redundant vision/audio
forward passes during training and accelerating training loops by >1,000x.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence
import torch
from torch.utils.data import DataLoader, Dataset

from arbiter_omni.data.dataset import MultimodalDecisionDataset, collate_multimodal_decision
from arbiter_omni.types import ModalityType, MultimodalSample

logger = logging.getLogger(__name__)


class CachedSample:
    """Container holding pre-computed frozen encoder embeddings for a single sample."""

    def __init__(
        self,
        question_embed: torch.Tensor,
        modality_embeds: Dict[str, torch.Tensor],
        presence_mask: Dict[str, bool],
        candidate_embeds: torch.Tensor,
        target_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.question_embed = question_embed.detach().cpu()
        self.modality_embeds = {k: v.detach().cpu() for k, v in modality_embeds.items()}
        self.presence_mask = presence_mask
        self.candidate_embeds = candidate_embeds.detach().cpu()
        self.target_idx = target_idx
        self.metadata = metadata or {}


class CachedMultimodalDataset(Dataset):
    """
    Dataset containing pre-extracted frozen multimodal representations.
    Allows training fusion networks and decision heads at thousands of steps/sec.
    """

    def __init__(self, samples: List[CachedSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CachedSample:
        return self.samples[idx]

    @classmethod
    def from_dataset(
        cls,
        dataset: MultimodalDecisionDataset,
        model: Any,  # ArbiterOmniModel
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> CachedMultimodalDataset:
        """
        Pre-computes and caches frozen embeddings for all samples in a MultimodalDecisionDataset.
        """
        dev = device or model.device
        model.eval()
        model.encoder.eval()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_multimodal_decision,
        )

        cached_samples: List[CachedSample] = []
        total = len(dataset)
        if verbose:
            logger.info(f"Pre-caching {total} samples through frozen encoders on {dev}...")

        with torch.no_grad():
            sample_offset = 0
            for batch in loader:
                questions = batch["questions"]
                candidates = batch["candidates"]
                texts = batch["texts"]
                images = batch["images"]
                videos = batch["videos"]
                audios = batch["audios"]
                targets = batch["targets"]
                meta = batch.get("metadata", [{} for _ in range(len(questions))])

                # 1. Encode questions and multimodal inputs
                q_embeds, mod_embeds, pres_masks = model.encode_inputs(
                    questions=questions,
                    texts=texts,
                    images=images,
                    videos=videos,
                    audios=audios,
                )

                # 2. Encode candidates per sample
                # Each sample can have varying candidate lengths K
                b_size = len(questions)
                for i in range(b_size):
                    cands = candidates[i]
                    c_embed = model.encoder.encode_text(cands).detach().cpu()
                    targ = int(targets[i].item()) if targets is not None else None

                    s_q = q_embeds[i].detach().cpu()
                    s_mods = {m: mod_embeds[m][i].detach().cpu() for m in mod_embeds}
                    s_mask = {m: bool(pres_masks[m][i].item()) for m in pres_masks}

                    cached_samples.append(
                        CachedSample(
                            question_embed=s_q,
                            modality_embeds=s_mods,
                            presence_mask=s_mask,
                            candidate_embeds=c_embed,
                            target_idx=targ,
                            metadata=meta[i] if i < len(meta) else {},
                        )
                    )

                sample_offset += b_size
                if verbose and (sample_offset % 100 == 0 or sample_offset == total):
                    logger.info(f"  Cached {sample_offset}/{total} samples ({sample_offset/total*100:.1f}%)")

        if verbose:
            logger.info(f"✅ Pre-caching complete! {len(cached_samples)} samples in memory.")
        return cls(cached_samples)

    def save(self, path: str) -> None:
        """Serializes pre-cached dataset tensors to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.samples, path)
        logger.info(f"Saved {len(self.samples)} cached samples to {path}")

    @classmethod
    def load(cls, path: str) -> CachedMultimodalDataset:
        """Loads pre-cached dataset tensors from disk."""
        samples = torch.load(path, weights_only=False)
        logger.info(f"Loaded {len(samples)} cached samples from {path}")
        return cls(samples)


def collate_cached_multimodal_decision(batch: List[CachedSample]) -> Dict[str, Any]:
    """
    Collator for batched cached samples.
    Pads candidate embeddings and constructs batch presence tensors.
    """
    B = len(batch)
    text_dim = batch[0].question_embed.shape[-1]

    # 1. Question embeddings [B, text_dim]
    q_embeds = torch.stack([s.question_embed for s in batch], dim=0)

    # 2. Modality embeddings {mod: [B, mod_dim]} & presence masks {mod: [B]}
    modality_keys = list(batch[0].modality_embeds.keys())
    mod_embeds: Dict[str, torch.Tensor] = {}
    presence_mask: Dict[str, torch.Tensor] = {}

    for k in modality_keys:
        mod_embeds[k] = torch.stack([s.modality_embeds[k] for s in batch], dim=0)
        presence_mask[k] = torch.tensor([s.presence_mask[k] for s in batch], dtype=torch.bool)

    # 3. Dynamic candidate padding [B, max_K, text_dim] & candidate mask [B, max_K]
    max_K = max(s.candidate_embeds.shape[0] for s in batch)
    cnd_embeds = torch.zeros((B, max_K, text_dim), dtype=torch.float32)
    cnd_mask = torch.zeros((B, max_K), dtype=torch.bool)

    for i, s in enumerate(batch):
        k = s.candidate_embeds.shape[0]
        cnd_embeds[i, :k, :] = s.candidate_embeds
        cnd_mask[i, :k] = True

    # 4. Targets
    has_targets = all(s.target_idx is not None for s in batch)
    targets = (
        torch.tensor([s.target_idx for s in batch], dtype=torch.long)
        if has_targets
        else None
    )

    return {
        "is_cached": True,
        "question_embed": q_embeds,
        "modality_embeds": mod_embeds,
        "presence_mask": presence_mask,
        "candidate_embeds": cnd_embeds,
        "candidate_mask": cnd_mask,
        "targets": targets,
    }
