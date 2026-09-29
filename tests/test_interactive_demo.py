"""
Unit tests for interactive_demo module and Gradio Blocks construction.
"""

import pytest
from examples.interactive_demo import arbitrate_decision, build_app, load_preset, PRESETS


def test_arbitrate_decision_programmatic():
    winner, probs, entropy, noul, score = arbitrate_decision(
        question="What is the hazard response?",
        candidates_text="Emergency halt\nProceed slowly\nReroute path",
        text_context="Thermal sensor high",
    )

    assert "Winning Decision" in winner
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    assert "nats" in entropy
    assert "Certainty" in noul
    assert "Score" in score


def test_load_presets():
    for name in PRESETS.keys():
        q, cands, text, img, audio = load_preset(name)
        assert len(q) > 0
        assert len(cands) > 0
        assert img is not None
        assert audio is not None


def test_build_gradio_app():
    demo = build_app()
    assert demo is not None
