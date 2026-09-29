"""
Unit tests for ScienceQA dataset adapter and dynamic candidate handling.
"""

from PIL import Image
import pytest

from arbiter_omni.data.scienceqa import (
    ScienceQAAdapter,
    create_mock_scienceqa_samples,
    load_scienceqa_dataset,
)
from arbiter_omni.types import MultimodalSample


def test_scienceqa_adapter_record_conversion():
    dummy_record = {
        "question": "Which state of matter has a fixed volume but no fixed shape?",
        "choices": ["Solid", "Liquid", "Gas", "Plasma"],
        "answer": 1,
        "hint": "Water in a graduated cylinder is an example.",
        "image": Image.new("RGB", (64, 64), color="blue"),
        "subject": "natural science",
        "topic": "physics",
    }

    sample = ScienceQAAdapter.record_to_sample(dummy_record)
    assert isinstance(sample, MultimodalSample)
    assert sample.question == dummy_record["question"]
    assert sample.candidates == ["Solid", "Liquid", "Gas", "Plasma"]
    assert sample.target_idx == 1
    assert sample.image is not None
    assert sample.text == dummy_record["hint"]
    assert sample.metadata["subject"] == "natural science"


def test_mock_scienceqa_samples_generation():
    samples = create_mock_scienceqa_samples(num_samples=12, seed=99)
    assert len(samples) == 12
    for s in samples:
        assert isinstance(s, MultimodalSample)
        assert len(s.candidates) >= 2
        assert 0 <= s.target_idx < len(s.candidates)
        assert s.image is not None
        assert s.text is not None


def test_load_scienceqa_fallback():
    # If network/HF is unavailable, load_scienceqa_dataset falls back to mock samples gracefully
    samples = load_scienceqa_dataset(max_samples=5, use_mock_fallback=True)
    assert len(samples) <= 5
    assert all(isinstance(s, MultimodalSample) for s in samples)
