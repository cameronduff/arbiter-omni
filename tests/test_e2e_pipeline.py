"""
Unit tests for end-to-end training and inference pipeline.
"""

import tempfile
import os
import pytest
import torch

from arbiter_omni import (
    ArbiterOmniEngine,
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    TrainingConfig,
    generate_synthetic_dataset,
)


def test_e2e_training_and_inference():
    # 1. Dataset
    samples = generate_synthetic_dataset(num_samples=20, seed=42)
    train_ds = MultimodalDecisionDataset(samples[:16])
    val_ds = MultimodalDecisionDataset(samples[16:])

    # 2. Model
    encoder = MockMultimodalEncoder(embed_dim=64)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=64, scoring_dim=64)

    # 3. Trainer
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.pt")
        config = TrainingConfig(
            batch_size=8,
            num_epochs=2,
            learning_rate=1e-3,
            save_path=save_path,
        )
        trainer = ArbiterOmniTrainer(model=model, config=config)
        history = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

        assert len(history["loss"]) == 2
        assert os.path.exists(save_path)

        # 4. Engine & Checkpoint Loading
        engine = ArbiterOmniEngine(model=model)
        engine.load_weights(save_path)

        # 5. Inference with missing modalities
        res = engine.decide(
            question="What is the next action?",
            candidates=["Action 1", "Action 2", "Action 3"],
            text="Sensor report: nominal",
        )
        assert res.winner in ["Action 1", "Action 2", "Action 3"]
        assert 0.0 <= res.confidence <= 1.0
        assert len(res.probabilities) == 3
        assert res.entropy >= 0.0
