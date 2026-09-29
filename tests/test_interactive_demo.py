"""
Unit tests for interactive_demo module, general presets, and Gradio Blocks construction.
"""

import pytest
from PIL import Image
from examples.interactive_demo import (
    PRESETS,
    arbitrate_decision,
    build_app,
    get_preset_payload,
    load_preset,
)


def test_arbitrate_decision_programmatic():
    winner, probs, entropy, noul, score = arbitrate_decision(
        question="Which physical state is observed?",
        candidates_text="Solid phase\nLiquid phase\nGaseous phase",
        text_context="Thermal sensor high",
    )

    assert "Winning Decision" in winner
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    assert "nats" in entropy
    assert "Certainty" in noul
    assert "Score" in score


def test_load_all_general_presets():
    for name in PRESETS.keys():
        q, cands, text, img, audio = load_preset(name)
        assert len(q) > 0
        assert len(cands) > 0
        assert isinstance(img, Image.Image)
        assert isinstance(audio, tuple)
        assert audio[0] == 16000


def test_get_preset_payload_structure():
    first_name = list(PRESETS.keys())[0]
    payload = get_preset_payload(first_name)
    assert len(payload) == 5
    q, cands, text, img, aud = payload
    assert isinstance(q, str) and len(q) > 0
    assert "\n" in cands  # Multiple dynamic candidate choices
    assert isinstance(img, Image.Image)


def test_arbitrate_variable_candidates():
    # 2 candidates
    w2, p2, _, _, _ = arbitrate_decision(
        question="True or false?",
        candidates_text="True assertion\nFalse assertion",
    )
    assert len(p2) == 2

    # 6 dynamic candidates
    cands_6 = "\n".join([f"Option {chr(65+i)}" for i in range(6)])
    w6, p6, _, _, _ = arbitrate_decision(
        question="Select best candidate:",
        candidates_text=cands_6,
    )
    assert len(p6) == 6


def test_arbitrate_missing_modalities():
    # Text only
    w_text, p_text, _, _, _ = arbitrate_decision(
        question="Text-only question",
        candidates_text="Choice 1\nChoice 2",
        text_context="Context note",
        image_input=None,
        audio_input=None,
    )
    assert len(p_text) == 2

    # Image only
    img = Image.new("RGB", (64, 64), color="blue")
    w_img, p_img, _, _, _ = arbitrate_decision(
        question="Vision-only question",
        candidates_text="Choice 1\nChoice 2",
        text_context=None,
        image_input=img,
        audio_input=None,
    )
    assert len(p_img) == 2


def test_build_gradio_app():
    demo = build_app()
    assert demo is not None
