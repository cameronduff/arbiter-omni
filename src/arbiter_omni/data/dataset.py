"""
Multimodal Decision PyTorch Dataset and Custom Collate Protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from arbiter_omni.types import MultimodalSample


class MultimodalDecisionDataset(Dataset):
    """
    PyTorch Dataset containing MultimodalSample instances.
    Supports samples with variable candidate sets and missing modalities.
    """

    def __init__(self, samples: List[MultimodalSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        return self.samples[idx]


def collate_multimodal_decision(batch: List[MultimodalSample]) -> Dict[str, Any]:
    """
    Custom collator for dynamic multimodal decision samples.
    Pads candidate lists and structures modality lists for batch processing.
    """
    questions = [s.question for s in batch]
    candidates = [s.candidates for s in batch]
    texts = [s.text for s in batch]
    images = [s.image for s in batch]
    videos = [s.video for s in batch]
    audios = [s.audio for s in batch]

    targets = []
    has_targets = True
    for s in batch:
        if s.target_idx is not None:
            targets.append(s.target_idx)
        else:
            has_targets = False

    collated = {
        "questions": questions,
        "candidates": candidates,
        "texts": texts,
        "images": images,
        "videos": videos,
        "audios": audios,
        "metadata": [s.metadata for s in batch],
    }

    if has_targets:
        collated["targets"] = torch.tensor(targets, dtype=torch.long)
    else:
        collated["targets"] = None

    return collated
