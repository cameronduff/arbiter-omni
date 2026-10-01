"""
Tests for Tier-0 Speculative Draft Head, Early-Exit Gate, and End-to-End Early Exit [AO-29].
"""

import time
import pytest
import torch
from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.model.speculative import (
    SpeculativeDraftDecision,
    SpeculativeDraftHead,
    SpeculativeGate,
)
from arbiter_omni.types import DecisionResult, ModalityType, MultimodalSample


def test_speculative_draft_head_parameters_lean():
    head = SpeculativeDraftHead(input_dim=768, candidate_dim=768, draft_dim=128)
    total_params = sum(p.numel() for p in head.parameters())
    # 768*128 + 768*128 + 2 scalar params = 196,610 parameters (<250k)
    assert total_params < 250_000
    assert total_params > 50_000


def test_speculative_draft_head_forward():
    head = SpeculativeDraftHead(input_dim=768, candidate_dim=768, draft_dim=128)
    B, K = 4, 3
    input_embed = torch.randn(B, 768)
    cand_embeds = torch.randn(B, K, 768)

    logits, probs, entropy = head(input_embed, cand_embeds)
    assert logits.shape == (B, K)
    assert probs.shape == (B, K)
    assert entropy.shape == (B,)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (entropy >= 0.0).all()


def test_speculative_draft_head_residual_visual():
    head = SpeculativeDraftHead(input_dim=768, candidate_dim=768, draft_dim=128)
    B, K = 2, 3
    input_embed = torch.randn(B, 768)
    cand_embeds = torch.randn(B, K, 768)
    cand_embeds[0, 1] = torch.tensor([1.0] * 768)  # candidate 1 matches visual
    vis_embed = torch.tensor([[1.0] * 768, [0.0] * 768])
    vis_present = torch.tensor([True, False])

    logits, probs, entropy = head(
        input_embed,
        cand_embeds,
        residual_visual_embed=vis_embed,
        visual_present=vis_present,
    )
    # In sample 0, candidate 1 has high visual similarity
    assert probs[0, 1] > probs[0, 0]
    assert probs[0, 1] > probs[0, 2]


def test_speculative_gate_early_exit_logic():
    gate = SpeculativeGate(
        margin_threshold=0.40,
        entropy_threshold=0.50,
        min_confidence=0.65,
        enabled=True,
    )

    # Sample 0: Clear winner (probs: 0.85, 0.10, 0.05 -> margin 0.75, low entropy) -> SHOULD EXIT
    # Sample 1: Ambiguous (probs: 0.45, 0.40, 0.15 -> margin 0.05, high entropy) -> SHOULD NOT EXIT
    probs = torch.tensor([
        [0.85, 0.10, 0.05],
        [0.45, 0.40, 0.15],
    ])
    entropy = torch.tensor([0.25, 0.98])

    can_exit, margins, top1 = gate.should_exit(probs, entropy)
    assert can_exit[0].item() is True
    assert can_exit[1].item() is False
    assert margins[0].item() == pytest.approx(0.75, abs=1e-3)
    assert margins[1].item() == pytest.approx(0.05, abs=1e-3)


def test_speculative_gate_disabled():
    gate = SpeculativeGate(enabled=False)
    probs = torch.tensor([[0.99, 0.01]])
    entropy = torch.tensor([0.05])
    can_exit, _, _ = gate.should_exit(probs, entropy)
    assert can_exit[0].item() is False


def test_model_forward_speculative_integration():
    engine = ArbiterOmniEngine.create(
        encoder_type="mock",
        hidden_dim=64,
        scoring_dim=64,
        embed_dim=64,
        enable_speculative_early_exit=True,
        speculative_margin_threshold=0.30,
        speculative_entropy_threshold=0.80,
        speculative_min_confidence=0.50,
    )
    assert engine.model.speculative_head is not None
    assert engine.model.speculative_gate is not None

    q_embed = torch.randn(1, 64)
    c_embeds = torch.randn(1, 3, 64)
    logits, probs, ent, can_exit, margins, top1 = engine.model.forward_speculative(
        question_embed=q_embed,
        candidate_embeds=c_embeds,
    )
    assert logits.shape == (1, 3)
    assert probs.shape == (1, 3)
    assert isinstance(can_exit.item(), bool)


def test_engine_decide_speculative_early_exit():
    engine = ArbiterOmniEngine.create(
        encoder_type="mock",
        hidden_dim=64,
        scoring_dim=64,
        embed_dim=64,
        enable_speculative_early_exit=True,
        # Force low threshold so it triggers early exit on random mock weights
        speculative_margin_threshold=0.0,
        speculative_entropy_threshold=2.00,
        speculative_min_confidence=0.01,
    )

    res = engine.decide(
        question="What animal is this?",
        candidates=["cat", "dog", "car"],
    )
    assert res.speculative_early_exit is True
    assert res.draft_telemetry is not None
    assert res.draft_telemetry["early_exit_taken"] is True
    assert res.draft_telemetry["draft_latency_ms"] >= 0.0
    assert res.winner in ["cat", "dog", "car"]
    assert len(res.probabilities) == 3


def test_engine_decide_speculative_fallback_to_deep_fusion():
    engine = ArbiterOmniEngine.create(
        encoder_type="mock",
        hidden_dim=64,
        scoring_dim=64,
        embed_dim=64,
        enable_speculative_early_exit=True,
        # High impossible threshold so it NEVER early exits
        speculative_margin_threshold=0.9999,
        speculative_entropy_threshold=0.0001,
        speculative_min_confidence=0.9999,
    )

    res = engine.decide(
        question="What animal is this?",
        candidates=["cat", "dog", "car"],
    )
    assert res.speculative_early_exit is False
    assert res.draft_telemetry is not None
    assert res.draft_telemetry["early_exit_taken"] is False
