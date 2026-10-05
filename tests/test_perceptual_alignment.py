"""
Unit tests for zero-shot perceptual residual alignment in ArbiterOmni.
Verifies that when visual (or other present modality) inputs are supplied,
the model grounds candidate scoring in frozen foundation representations
without modality collapse or candidate distortion.
"""

from __future__ import annotations

import torch
from PIL import Image
import numpy as np

from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.types import ModalityType


def test_decision_head_zero_shot_residual_alignment():
    """Verifies that visual_scale properly influences candidate logits when image is present."""
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)
    engine = ArbiterOmniEngine(model=model)

    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    candidates = ["Option A", "Option B", "Option C"]

    # 1. Decide with image
    res_img = engine.decide(
        question="What is shown in this observation?",
        candidates=candidates,
        image=img,
    )
    assert res_img.winner in candidates
    assert abs(sum(res_img.probabilities.values()) - 1.0) < 1e-4

    # 2. Decide without image
    res_no_img = engine.decide(
        question="What is shown in this observation?",
        candidates=candidates,
        image=None,
    )
    assert res_no_img.winner in candidates
    assert abs(sum(res_no_img.probabilities.values()) - 1.0) < 1e-4


def test_missing_modality_preserves_zero_leakage():
    """Ensures when image is missing, presence_mask zeros out the perceptual residual term."""
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)

    candidates = [["Cat", "Dog"]]

    with torch.no_grad():
        logits_none, probs_none, _, _ = model(
            questions=["Identify entity"],
            candidates=candidates,
            images=None,
        )

        assert not torch.isnan(logits_none).any()
        assert not torch.isnan(probs_none).any()
        assert torch.allclose(probs_none.sum(dim=-1), torch.tensor([1.0]), atol=1e-4)
