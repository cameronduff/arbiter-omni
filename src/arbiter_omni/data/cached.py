"""
ArbiterOmni Pre-Cached Multimodal Dataset.
Pre-extracts invariant latent representations from frozen multimodal encoders
(OpenCLIP, CLAP, Temporal Video Attention) once, eliminating redundant vision/audio
forward passes during training and accelerating training loops by >1,000x.

AO-25: INT8 Dynamic Quantization Support
Spatial visual patch embeddings and candidate embeddings can be stored in INT8
format, reducing RAM footprint by ~60% (from float32 @ 4 bytes/element to INT8
@ 1 byte/element), enabling larger effective batch sizes within system RAM limits.
Uses symmetric per-tensor dynamic quantization (scale derived from abs-max).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from arbiter_omni.data.dataset import MultimodalDecisionDataset, collate_multimodal_decision
from arbiter_omni.types import ModalityType, MultimodalSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# INT8 Dynamic Quantization Utilities [AO-25]
# ---------------------------------------------------------------------------

def int8_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantizes a float32 tensor to INT8 with per-tensor scaling.

    Uses the absolute maximum value to derive a symmetric scale factor, then
    clamps and rounds to the INT8 range [-127, 127] (leaving -128 unused to
    keep the mapping symmetric). Reduces memory by ~75% vs float32.

    Args:
        tensor: Input float32 tensor of any shape.

    Returns:
        q_tensor: Quantized INT8 tensor (same shape, dtype=torch.int8).
        scale:    Scalar float32 scale factor (q_tensor * scale ≈ tensor).
    """
    abs_max = tensor.abs().max().clamp(min=1e-8)
    scale = abs_max / 127.0
    q_tensor = (tensor / scale).clamp(-127, 127).round().to(torch.int8)
    return q_tensor, scale.to(torch.float32)


def int8_dequantize(q_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantizes an INT8 tensor back to float32 using the stored scale factor.

    Args:
        q_tensor: INT8 quantized tensor of any shape.
        scale:    Scalar float32 scale factor returned by :func:`int8_quantize`.

    Returns:
        Reconstructed float32 tensor (same shape as q_tensor).
    """
    return q_tensor.to(torch.float32) * scale


class CachedSample:
    """Container holding pre-computed frozen encoder embeddings for a single sample.

    AO-25: When ``use_int8=True``, candidate embeddings and spatial image patches are
    stored in INT8 format (symmetric per-tensor quantization) reducing RAM usage by
    ~75% vs float32.  Access via :attr:`candidate_embeds` and :attr:`image_patches`
    properties — they transparently dequantize to float32 on the fly.
    """

    def __init__(
        self,
        question_embed: torch.Tensor,
        modality_embeds: Dict[str, torch.Tensor],
        presence_mask: Dict[str, bool],
        candidate_embeds: torch.Tensor,
        target_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_patches: Optional[torch.Tensor] = None,
        use_int8: bool = False,
    ):
        self.question_embed = question_embed.detach().cpu()
        self.modality_embeds = {k: v.detach().cpu() for k, v in modality_embeds.items()}
        self.presence_mask = presence_mask
        self.target_idx = target_idx
        self.metadata = metadata or {}
        self._use_int8 = use_int8

        # ---- Candidate embeddings (optionally INT8) ----
        cands_cpu = candidate_embeds.detach().cpu()
        if use_int8 and cands_cpu.numel() > 0:
            self._candidate_embeds_q, self._candidate_embeds_scale = int8_quantize(cands_cpu)
            self._candidate_embeds: Optional[torch.Tensor] = None
        else:
            self._candidate_embeds = cands_cpu
            self._candidate_embeds_q = None
            self._candidate_embeds_scale = None

        # ---- Spatial image patches (optionally INT8) ----
        if image_patches is not None:
            patches_cpu = image_patches.detach().cpu()
            if use_int8 and patches_cpu.numel() > 0:
                self._image_patches_q, self._image_patches_scale = int8_quantize(patches_cpu)
                self._image_patches: Optional[torch.Tensor] = None
            else:
                self._image_patches = patches_cpu
                self._image_patches_q = None
                self._image_patches_scale = None
        else:
            self._image_patches = None
            self._image_patches_q = None
            self._image_patches_scale = None

    @property
    def candidate_embeds(self) -> torch.Tensor:
        """Returns candidate embeddings as float32, dequantizing INT8 if needed."""
        if self._candidate_embeds_q is not None:
            return int8_dequantize(self._candidate_embeds_q, self._candidate_embeds_scale)
        return self._candidate_embeds

    @candidate_embeds.setter
    def candidate_embeds(self, value: torch.Tensor) -> None:
        """Allows direct float32 assignment (bypasses quantization for legacy compat)."""
        self._candidate_embeds = value
        self._candidate_embeds_q = None
        self._candidate_embeds_scale = None

    @property
    def image_patches(self) -> Optional[torch.Tensor]:
        """Returns spatial image patches as float32, dequantizing INT8 if needed."""
        if self._image_patches_q is not None:
            return int8_dequantize(self._image_patches_q, self._image_patches_scale)
        return self._image_patches

    @image_patches.setter
    def image_patches(self, value: Optional[torch.Tensor]) -> None:
        """Allows direct float32 assignment (bypasses quantization for legacy compat)."""
        self._image_patches = value
        self._image_patches_q = None
        self._image_patches_scale = None

    @property
    def memory_bytes(self) -> int:
        """Returns approximate resident RAM bytes for this sample's tensors."""
        total = self.question_embed.nelement() * self.question_embed.element_size()
        for v in self.modality_embeds.values():
            total += v.nelement() * v.element_size()
        if self._candidate_embeds_q is not None:
            total += self._candidate_embeds_q.nelement() * self._candidate_embeds_q.element_size()
        elif self._candidate_embeds is not None:
            total += self._candidate_embeds.nelement() * self._candidate_embeds.element_size()
        if self._image_patches_q is not None:
            total += self._image_patches_q.nelement() * self._image_patches_q.element_size()
        elif self._image_patches is not None:
            total += self._image_patches.nelement() * self._image_patches.element_size()
        return total



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
        use_int8: bool = False,
    ) -> CachedMultimodalDataset:
        """Pre-computes and caches frozen embeddings for all samples in a MultimodalDecisionDataset.

        Args:
            dataset: Source multimodal decision dataset.
            model: ArbiterOmniModel instance (frozen encoders).
            batch_size: Encoding batch size.
            device: Inference device (defaults to model.device).
            verbose: Log progress to logger.
            use_int8: [AO-25] Store candidate embeddings and image patches in INT8
                format to reduce RAM footprint by ~75% vs float32.
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
                q_embeds, mod_embeds, pres_masks, patch_embeds = model.encode_inputs(
                    questions=questions,
                    texts=texts,
                    images=images,
                    videos=videos,
                    audios=audios,
                )

                # 2. Encode candidates per sample in a single batched pass
                b_size = len(questions)
                flat_cands: List[str] = []
                sample_cand_ranges: List[Tuple[int, int]] = []
                for cands in candidates:
                    start_idx = len(flat_cands)
                    flat_cands.extend(cands)
                    end_idx = len(flat_cands)
                    sample_cand_ranges.append((start_idx, end_idx))

                if flat_cands:
                    all_c_embeds = model.encoder.encode_text(flat_cands).detach().cpu()
                else:
                    all_c_embeds = torch.empty((0, model.encoder.text_dim))

                for i in range(b_size):
                    start_idx, end_idx = sample_cand_ranges[i]
                    c_embed = all_c_embeds[start_idx:end_idx]
                    targ = int(targets[i].item()) if targets is not None else None

                    s_q = q_embeds[i].detach().cpu()
                    s_mods = {m: mod_embeds[m][i].detach().cpu() for m in mod_embeds}
                    s_mask = {m: bool(pres_masks[m][i].item()) for m in pres_masks}
                    s_patches = patch_embeds[i].detach().cpu() if patch_embeds is not None else None

                    cached_samples.append(
                        CachedSample(
                            question_embed=s_q,
                            modality_embeds=s_mods,
                            presence_mask=s_mask,
                            candidate_embeds=c_embed,
                            target_idx=targ,
                            metadata=meta[i] if i < len(meta) else {},
                            image_patches=s_patches,
                            use_int8=use_int8,
                        )
                    )

                sample_offset += b_size
                if verbose and ((sample_offset - b_size) // 200 != sample_offset // 200 or sample_offset >= total):
                    logger.info(f"  Cached {min(sample_offset, total)}/{total} samples ({min(100.0, sample_offset/total*100):.1f}%)")

        if verbose:
            logger.info(f"✅ Pre-caching complete! {len(cached_samples)} samples in memory.")
        return cls(cached_samples)

    @property
    def total_memory_mb(self) -> float:
        """Returns approximate total RAM occupied by all cached sample tensors in MB."""
        return sum(s.memory_bytes for s in self.samples) / (1024.0 * 1024.0)

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

    # 5. Spatial visual patches [B, P, img_dim] if cached
    has_patches = any(hasattr(s, "image_patches") and s.image_patches is not None for s in batch)
    if has_patches:
        valid_sample = next(s for s in batch if hasattr(s, "image_patches") and s.image_patches is not None)
        P, img_dim = valid_sample.image_patches.shape[-2], valid_sample.image_patches.shape[-1]
        batch_patches = torch.zeros((B, P, img_dim), dtype=torch.float32)
        for i, s in enumerate(batch):
            if hasattr(s, "image_patches") and s.image_patches is not None:
                batch_patches[i] = s.image_patches
    else:
        batch_patches = None

    return {
        "is_cached": True,
        "question_embed": q_embeds,
        "modality_embeds": mod_embeds,
        "presence_mask": presence_mask,
        "candidate_embeds": cnd_embeds,
        "candidate_mask": cnd_mask,
        "targets": targets,
        "image_patches": batch_patches,
    }

