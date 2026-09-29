"""
Unit tests for the open-domain interactive_demo module.
Tests the core arbitration function, the examples gallery structure,
and Gradio Blocks construction — all without network access.
"""

import pytest
from PIL import Image
from examples.interactive_demo import (
    EXAMPLES,
    _default_image,
    arbitrate_decision,
    build_app,
)


def test_arbitrate_decision_basic():
    winner, probs, entropy, noul, score = arbitrate_decision(
        question="What animal is in this image?",
        candidates_text="Cat\nDog\nHorse",
    )
    assert "Winning Decision" in winner or winner.startswith("##")
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    assert "nats" in entropy
    assert isinstance(noul, str)
    assert isinstance(score, str)


def test_arbitrate_variable_candidates():
    # 2 candidates
    _, p2, _, _, _ = arbitrate_decision(
        question="True or false?",
        candidates_text="True\nFalse",
    )
    assert len(p2) == 2

    # 6 candidates
    cands = "\n".join([f"Option {i}" for i in range(6)])
    _, p6, _, _, _ = arbitrate_decision(
        question="Pick one:",
        candidates_text=cands,
    )
    assert len(p6) == 6


def test_arbitrate_with_image():
    img = Image.new("RGB", (64, 64), color=(180, 100, 50))
    _, probs, _, _, _ = arbitrate_decision(
        question="What colour is this?",
        candidates_text="Red\nBlue\nGreen\nOrange",
        image_input=img,
    )
    assert len(probs) == 4


def test_arbitrate_missing_all_modalities():
    _, probs, _, _, _ = arbitrate_decision(
        question="Pick one",
        candidates_text="A\nB",
        text_context=None,
        image_input=None,
        audio_input=None,
    )
    assert len(probs) == 2


def test_arbitrate_empty_candidates_fallback():
    # Empty candidates_text should fall back to ["Option A", "Option B"]
    _, probs, _, _, _ = arbitrate_decision(
        question="Anything?",
        candidates_text="   \n   ",
    )
    assert len(probs) == 2


def test_default_image_is_pil():
    img = _default_image()
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_examples_gallery_structure():
    assert len(EXAMPLES) >= 4
    for ex in EXAMPLES:
        # Each example: [question, candidates, text_context, image, audio, optional temperature]
        assert len(ex) in (5, 6)
        question, candidates, text_ctx, image, audio = ex[:5]
        assert isinstance(question, str) and len(question) > 0
        assert "\n" in candidates, "Candidates must contain multiple options"
        assert isinstance(image, Image.Image) or image is None


def test_build_gradio_app():
    demo = build_app()
    assert demo is not None
