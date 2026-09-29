"""
Unit tests for ArbiterOmni checkpoint training and from_pretrained loading.
"""

import os
import tempfile
import pytest
import torch
from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.data.dataset import MultimodalDecisionDataset
from arbiter_omni.data.synthetic import generate_synthetic_dataset
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.training.config import TrainingConfig
from arbiter_omni.training.trainer import ArbiterOmniTrainer


def test_checkpoint_save_and_from_pretrained():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")

        # 1. Train lightweight model on synthetic samples
        encoder = MockMultimodalEncoder()
        model = ArbiterOmniModel(encoder=encoder)
        samples = generate_synthetic_dataset(num_samples=16)
        dataset = MultimodalDecisionDataset(samples)

        config = TrainingConfig(
            num_epochs=1,
            batch_size=8,
            save_path=ckpt_path,
        )
        trainer = ArbiterOmniTrainer(model=model, config=config)
        trainer.fit(dataset)

        assert os.path.exists(ckpt_path)

        # 2. Inspect checkpoint structure
        ckpt_data = torch.load(ckpt_path, weights_only=False)
        assert "fusion" in ckpt_data
        assert "decision_head" in ckpt_data
        assert "config" in ckpt_data

        # 3. Load via ArbiterOmniEngine.from_pretrained
        engine = ArbiterOmniEngine.from_pretrained(
            checkpoint_name_or_path=ckpt_path,
            encoder_type="mock",
        )
        assert engine is not None

        # 4. Perform dynamic decision inference
        result = engine.decide(
            question="What is the triage action?",
            candidates=["Action Alpha", "Action Beta", "Action Gamma"],
            text="High thermal sensor reading",
        )
        assert result.winner in ["Action Alpha", "Action Beta", "Action Gamma"]
        assert len(result.probabilities) == 3
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-4


def test_from_pretrained_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        ArbiterOmniEngine.from_pretrained(
            checkpoint_name_or_path="nonexistent_checkpoint_12345.pt",
            encoder_type="mock",
        )
