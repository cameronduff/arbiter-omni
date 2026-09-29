"""
Unit tests for ArbiterOmni type schemas.
"""

from PIL import Image
import numpy as np
import pytest

from arbiter_omni.types import DecisionResult, ModalityType, MultimodalSample


def test_multimodal_sample_modalities():
    # Only text
    s1 = MultimodalSample(
        question="Q?",
        candidates=["A", "B"],
        text="Status log",
    )
    assert s1.present_modalities() == [ModalityType.TEXT]

    # Image + Audio
    img = Image.new("RGB", (32, 32))
    audio = np.zeros(1600)
    s2 = MultimodalSample(
        question="Q?",
        candidates=["A", "B"],
        image=img,
        audio=audio,
    )
    assert set(s2.present_modalities()) == {ModalityType.IMAGE, ModalityType.AUDIO}


def test_decision_result_structure():
    res = DecisionResult(
        question="What to do?",
        winner="B",
        winner_index=1,
        confidence=0.85,
        probabilities={"A": 0.15, "B": 0.85},
        entropy=0.42,
        active_modalities=["image", "audio"],
        boolean_noul={"true_probability": 0.85, "calibrated_certainty": 0.70},
    )
    assert res.winner == "B"
    assert res.confidence == 0.85
    assert res.probabilities["A"] == 0.15
