"""
Unit tests for AI2D and GQA dataset adapters.
Tests mock sample generation, record conversion, candidate construction,
and load function fallback behaviour — all without network access.
"""

from __future__ import annotations

import random
import pytest
from PIL import Image

from arbiter_omni.data.ai2d import (
    AI2DAdapter,
    _make_mock_ai2d_samples,
    load_ai2d_dataset,
)
from arbiter_omni.data.gqa import (
    GQAAdapter,
    _make_mock_gqa_samples,
    _build_candidates,
    _guess_foil_pool,
    load_gqa_dataset,
)
from arbiter_omni.types import MultimodalSample


# ---------------------------------------------------------------------------
# AI2D Adapter Tests
# ---------------------------------------------------------------------------

class TestAI2DAdapter:
    def test_mock_samples_count(self):
        samples = _make_mock_ai2d_samples(num_samples=12)
        assert len(samples) == 12

    def test_mock_sample_structure(self):
        samples = _make_mock_ai2d_samples(num_samples=4)
        for s in samples:
            assert isinstance(s, MultimodalSample)
            assert len(s.question) > 0
            assert len(s.candidates) >= 2
            assert s.target_idx is not None
            assert 0 <= s.target_idx < len(s.candidates)
            assert isinstance(s.image, Image.Image)

    def test_adapter_record_to_sample_standard(self):
        adapter = AI2DAdapter()
        img = Image.new("RGB", (64, 64))
        record = {
            "question": "What does the arrow represent?",
            "options": ["A. Energy flow", "B. Water cycle", "C. Gravity", "D. None of the above"],
            "answer": "B",
            "image": img,
        }
        sample = adapter.record_to_sample(record)
        assert sample.question == "What does the arrow represent?"
        assert len(sample.candidates) == 4
        # Strip "A. " prefix → "Energy flow" should be in candidates
        assert "Energy flow" in sample.candidates
        assert sample.target_idx == 1  # "B" → index 1
        assert sample.image is img

    def test_adapter_record_numeric_answer(self):
        adapter = AI2DAdapter()
        record = {
            "question": "What is shown?",
            "options": ["apple", "banana", "orange"],
            "answer": 1,
            "image": None,
        }
        sample = adapter.record_to_sample(record)
        assert sample.target_idx == 1
        assert sample.candidates[sample.target_idx] == "banana"

    def test_adapter_record_strips_letter_prefix(self):
        adapter = AI2DAdapter()
        record = {
            "question": "Q?",
            "options": ["A) Alpha", "B) Beta", "C) Gamma", "D) Delta"],
            "answer": "C",
            "image": None,
        }
        sample = adapter.record_to_sample(record)
        assert "Alpha" in sample.candidates
        assert "Gamma" in sample.candidates
        assert sample.target_idx == 2

    def test_load_ai2d_uses_mock_fallback(self):
        # Force mock fallback by using an invalid dataset name via monkeypatching
        samples = load_ai2d_dataset(
            max_samples=8,
            use_mock_fallback=True,
            # Trigger network failure by using a non-existent split
            split="__nonexistent_split__",
        )
        assert len(samples) >= 1
        for s in samples:
            assert isinstance(s, MultimodalSample)


# ---------------------------------------------------------------------------
# GQA Adapter Tests
# ---------------------------------------------------------------------------

class TestGQAAdapter:
    def test_guess_foil_pool_bool(self):
        pool = _guess_foil_pool("yes")
        assert "no" in pool

    def test_guess_foil_pool_color(self):
        pool = _guess_foil_pool("red")
        assert "blue" in pool

    def test_guess_foil_pool_count(self):
        pool = _guess_foil_pool("3")
        assert "1" in pool

    def test_build_candidates_has_correct_answer(self):
        candidates, target_idx = _build_candidates(
            "yes", foil_pool=["yes", "no"], num_distractors=1, rng=random.Random(0)
        )
        assert candidates[target_idx] == "yes"
        assert len(candidates) == 2

    def test_build_candidates_no_duplicate_answer(self):
        for seed in range(10):
            candidates, target_idx = _build_candidates(
                "red", num_distractors=3, rng=random.Random(seed)
            )
            assert candidates.count("red") == 1
            assert candidates[target_idx] == "red"

    def test_mock_samples_count(self):
        samples = _make_mock_gqa_samples(num_samples=15)
        assert len(samples) == 15

    def test_mock_sample_structure(self):
        samples = _make_mock_gqa_samples(num_samples=5)
        for s in samples:
            assert isinstance(s, MultimodalSample)
            assert len(s.candidates) >= 2
            assert s.target_idx is not None
            assert 0 <= s.target_idx < len(s.candidates)
            assert s.candidates[s.target_idx] in s.candidates

    def test_adapter_record_to_sample(self):
        adapter = GQAAdapter(seed=0)
        img = Image.new("RGB", (32, 32))
        record = {"question": "What color is the ball?", "answer": "red", "image": img, "question_id": "q1"}
        sample = adapter.record_to_sample(record)
        assert sample is not None
        assert "red" in sample.candidates
        assert sample.candidates[sample.target_idx] == "red"
        assert sample.image is img

    def test_adapter_record_multiple_choice_answer(self):
        adapter = GQAAdapter(seed=0)
        record = {
            "question": "Where is he?",
            "multiple_choice_answer": "inside",
            "image": None,
            "question_id": "vqa_1",
        }
        sample = adapter.record_to_sample(record)
        assert sample is not None
        assert "inside" in sample.candidates
        assert sample.candidates[sample.target_idx] == "inside"

    def test_adapter_skips_empty_answer(self):
        adapter = GQAAdapter(seed=0)
        record = {"question": "Empty?", "answer": "", "image": None, "question_id": "q_empty"}
        sample = adapter.record_to_sample(record)
        assert sample is None

    def test_load_gqa_uses_mock_fallback(self):
        samples = load_gqa_dataset(
            max_samples=6,
            use_mock_fallback=True,
            split="__nonexistent__",
        )
        assert len(samples) >= 1
        for s in samples:
            assert isinstance(s, MultimodalSample)
