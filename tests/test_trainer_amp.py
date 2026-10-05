"""
Unit tests for ArbiterOmniTrainer GPU batch scaling, mixed precision (AMP), and gradient accumulation.
"""

import pytest
import torch

from arbiter_omni import (
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    TrainingConfig,
    generate_synthetic_dataset,
    collate_multimodal_decision,
)


def test_trainer_amp_and_grad_accum():
    samples = generate_synthetic_dataset(num_samples=32, seed=42)
    train_ds = MultimodalDecisionDataset(samples[:24])
    val_ds = MultimodalDecisionDataset(samples[24:])

    encoder = MockMultimodalEncoder(embed_dim=64)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=64, scoring_dim=64)

    config = TrainingConfig(
        batch_size=8,
        num_epochs=2,
        learning_rate=1e-3,
        fp16=True,
        accumulate_grad_batches=2,
    )

    trainer = ArbiterOmniTrainer(model=model, config=config)
    history = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    assert len(history["loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert history["loss"][-1] >= 0.0


def test_trainer_batch_scaling():
    # Test batch size 64 scaling
    samples = generate_synthetic_dataset(num_samples=128, seed=123)
    train_ds = MultimodalDecisionDataset(samples)

    encoder = MockMultimodalEncoder(embed_dim=64)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=64, scoring_dim=64)

    config = TrainingConfig(
        batch_size=64,
        num_epochs=1,
        learning_rate=2e-3,
        fp16=True,
        accumulate_grad_batches=1,
    )

    trainer = ArbiterOmniTrainer(model=model, config=config)
    metrics = trainer.train_epoch(
        torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.batch_size,
            collate_fn=collate_multimodal_decision,
        )
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "samples_per_sec" in metrics
    assert metrics["samples_per_sec"] > 0
