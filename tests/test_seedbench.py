"""
Unit tests for SEED-Bench and SEED-Bench-2 dataset adapter.
"""

from PIL import Image
import pytest

from arbiter_omni.data.seedbench import (
    SEEDBenchAdapter,
    create_mock_seedbench_samples,
    load_seedbench_dataset,
)
from arbiter_omni.types import MultimodalSample


def test_seedbench_record_to_sample_image():
    record = {
        "question": "What is on the kitchen table?",
        "choice_a": "A bowl of apples",
        "choice_b": "A ceramic vase",
        "choice_c": "A hardcover textbook",
        "choice_d": "A pair of sunglasses",
        "answer": "B",
        "data_type": "image",
        "question_id": "1001",
    }
    img = Image.new("RGB", (64, 64), color="blue")
    sample = SEEDBenchAdapter.record_to_sample(record, image=img)

    assert isinstance(sample, MultimodalSample)
    assert sample.question == record["question"]
    assert sample.candidates == [
        "A bowl of apples",
        "A ceramic vase",
        "A hardcover textbook",
        "A pair of sunglasses",
    ]
    assert sample.target_idx == 1  # "B" -> 1
    assert sample.image is not None
    assert sample.video is None
    assert sample.metadata["data_type"] == "image"


def test_seedbench_record_to_sample_video():
    record = {
        "question": "Which action happens first in the video clip?",
        "choice_a": "The actor opens the microwave door",
        "choice_b": "The actor pours milk into the mug",
        "choice_c": "The actor stirs the spoon",
        "choice_d": "The actor sits down at the counter",
        "answer": "A",
        "data_type": "video",
        "question_id": "2002",
    }
    frames = [Image.new("RGB", (64, 64), color="red") for _ in range(4)]
    sample = SEEDBenchAdapter.record_to_sample(record, video_frames=frames)

    assert isinstance(sample, MultimodalSample)
    assert sample.target_idx == 0  # "A" -> 0
    assert sample.video is not None
    assert len(sample.video) == 4
    assert sample.image is None
    assert sample.metadata["data_type"] == "video"


def test_create_mock_seedbench_samples():
    samples = create_mock_seedbench_samples(num_samples=16, seed=42)
    assert len(samples) == 16

    has_video = False
    has_image = False
    for s in samples:
        assert len(s.candidates) == 4
        assert 0 <= s.target_idx < 4
        if s.video is not None:
            has_video = True
        if s.image is not None:
            has_image = True

    assert has_video, "Expected some video samples in mock SEED-Bench"
    assert has_image, "Expected some image samples in mock SEED-Bench"


def test_load_seedbench_fallback(monkeypatch):
    import urllib.request
    def mock_urlopen(*args, **kwargs):
        raise ConnectionError("Simulated offline condition")
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    samples = load_seedbench_dataset(max_samples=5, use_mock_fallback=True)
    assert len(samples) == 5
    assert all(isinstance(s, MultimodalSample) for s in samples)
