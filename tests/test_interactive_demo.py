"""
Unit tests for the refactored open-domain interactive_demo module.
Tests predict_arbitration (with Image, Video, and Audio), arbitrate_decision backward compatibility,
the examples gallery structure, and Gradio Blocks construction.
"""

import os
from PIL import Image
import pytest

from examples.interactive_demo import (
    EXAMPLES,
    _default_image,
    arbitrate_decision,
    build_app,
    parse_candidates,
    predict_arbitration,
)


def test_parse_candidates():
    # Comma-separated
    assert parse_candidates("Cat, Dog, Fox") == ["Cat", "Dog", "Fox"]
    # Newline-separated
    assert parse_candidates("Cat\nDog\nFox") == ["Cat", "Dog", "Fox"]
    # Empty / whitespace fallback
    assert parse_candidates("   \n   ") == ["Choice A", "Choice B"]


def test_predict_arbitration_basic():
    probs, conf, entropy, score = predict_arbitration(
        question="What animal is shown in this image?",
        candidates_raw="Cat, Dog, Fox",
    )
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert 0.0 <= conf <= 1.0
    assert entropy >= 0.0
    assert isinstance(score, float)


def test_predict_arbitration_image_modality():
    cat_img = "examples/assets/kitten.jpg"
    probs, conf, entropy, score = predict_arbitration(
        question="What animal is this?",
        candidates_raw="Cat\nDog\nRabbit",
        image=cat_img if os.path.exists(cat_img) else None,
        context="A fluffy orange pet.",
        temperature=0.7,
    )
    assert len(probs) == 3
    assert "Cat" in probs


def test_predict_arbitration_video_modality():
    vid_file = "examples/assets/sample_action.mp4"
    probs, conf, entropy, score = predict_arbitration(
        question="What movement is shown in this video clip?",
        candidates_raw="Horizontal motion, Vertical drop, Circular rotation",
        video=vid_file if os.path.exists(vid_file) else None,
        temperature=0.7,
    )
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_arbitrate_decision_legacy_compatibility():
    winner, probs, entropy, noul, score = arbitrate_decision(
        question="What animal is in this image?",
        candidates_text="Cat\nDog\nHorse",
    )
    assert "Winning Decision" in winner or winner.startswith("##")
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert "nats" in entropy
    assert isinstance(noul, str)
    assert isinstance(score, str)


def test_default_image_is_pil():
    img = _default_image()
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_examples_gallery_structure():
    assert len(EXAMPLES) >= 4
    for ex in EXAMPLES:
        # Schema: [question, candidates, image, video, audio, context, temperature]
        assert len(ex) == 7
        question, candidates, image, video, audio, context, temp = ex
        assert isinstance(question, str) and len(question) > 0
        assert isinstance(candidates, str) and len(candidates) > 0
        assert ("," in candidates or "\n" in candidates), "Candidates must contain multiple options"
        assert image is None or isinstance(image, (str, Image.Image))
        assert video is None or isinstance(video, str)
        assert audio is None or isinstance(audio, str)
        assert isinstance(context, str)
        assert isinstance(temp, (int, float))


def test_build_gradio_app():
    demo = build_app()
    assert demo is not None
