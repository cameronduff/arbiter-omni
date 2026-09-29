"""
Unit tests for Dataset and Synthetic Data Generator.
"""

import pytest
from torch.utils.data import DataLoader

from arbiter_omni.data.dataset import (
    MultimodalDecisionDataset,
    collate_multimodal_decision,
)
from arbiter_omni.data.synthetic import generate_synthetic_dataset


def test_synthetic_generation():
    samples = generate_synthetic_dataset(num_samples=25, missing_modality_prob=0.4, seed=123)
    assert len(samples) == 25

    # Check that at least one modality is present for every sample
    for s in samples:
        assert len(s.present_modalities()) >= 1
        assert len(s.candidates) >= 2
        assert s.target_idx is not None


def test_dataset_loader_collate():
    samples = generate_synthetic_dataset(num_samples=10, seed=1)
    ds = MultimodalDecisionDataset(samples)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_multimodal_decision)

    batch = next(iter(loader))
    assert len(batch["questions"]) == 4
    assert len(batch["candidates"]) == 4
    assert batch["targets"].shape == (4,)
