"""
Unit tests for Test-Time Compute (TTC) Deliberation Tournament & Foil Harvesting [AO-31].
"""

import pytest
import torch
from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.calibration.deliberator import (
    TestTimeDeliberationSummary,
    TestTimeDeliberator,
    TournamentBracket,
)
from arbiter_omni.data.memory_bank import PersistentMemoryBank
from arbiter_omni.model.decision_head import DynamicDecisionHead


def test_test_time_deliberator_multi_pass_stochastic():
    """Verify stochastic perturbation across T passes calculates variance and consistency."""
    deliberator = TestTimeDeliberator(
        num_passes=5,
        router_noise_std=0.08,
        stability_threshold=0.60,
    )
    head = DynamicDecisionHead(context_dim=64, candidate_dim=64, scoring_dim=64)
    ctx = torch.randn(1, 64)
    cands = ["advance", "halt", "divert"]
    cand_embeds = torch.randn(1, 3, 64)

    summary = deliberator.deliberate(
        context_embed=ctx,
        candidate_embeds=cand_embeds,
        candidates=cands,
        decision_head=head,
    )

    assert isinstance(summary, TestTimeDeliberationSummary)
    assert summary.deliberation_passes == 5
    assert 0.0 <= summary.pass_winner_consistency <= 1.0
    assert summary.epistemic_variance >= 0.0
    assert 0.0 <= summary.stability_index <= 1.0
    assert len(summary.calibrated_probabilities) == 3


def test_test_time_deliberator_with_memory_bank_foil_tournament():
    """Verify tournament harvesting pits candidates against mined memory foils."""
    bank = PersistentMemoryBank(capacity=100, candidate_dim=64)
    # Populate memory bank with some foils
    foil_vectors = torch.randn(20, 64)
    bank.enqueue(candidate_embeds=foil_vectors, texts=[f"AdversarialFoil_{i}" for i in range(20)])

    deliberator = TestTimeDeliberator(
        memory_bank=bank,
        num_passes=3,
        foil_k=4,
        min_foil_margin=0.10,
    )
    head = DynamicDecisionHead(context_dim=64, candidate_dim=64, scoring_dim=64)
    ctx = torch.randn(1, 64)
    cands = ["proceed_forward", "emergency_stop"]
    cand_embeds = torch.randn(1, 2, 64)

    summary = deliberator.deliberate(
        context_embed=ctx,
        candidate_embeds=cand_embeds,
        candidates=cands,
        decision_head=head,
    )

    assert summary.tournament_bracket is not None
    assert isinstance(summary.tournament_bracket, TournamentBracket)
    assert len(summary.tournament_bracket.user_candidates) == 2
    assert len(summary.tournament_bracket.mined_foils) > 0
    assert summary.tournament_bracket.tournament_winner != ""


def test_engine_integration_test_time_deliberation():
    """End-to-end test of engine.decide with test_time_deliberate=True."""
    engine = ArbiterOmniEngine.create(
        encoder_type="mock",
        hidden_dim=64,
        scoring_dim=64,
        embed_dim=64,
    )
    bank = PersistentMemoryBank(capacity=50, candidate_dim=64)
    bank.enqueue(candidate_embeds=torch.randn(10, 64), texts=[f"Foil_{i}" for i in range(10)])
    engine.attach_memory_bank(bank)

    res = engine.decide(
        question="Which navigation route is safe?",
        candidates=["route_alpha", "route_beta", "route_gamma"],
        test_time_deliberate=True,
    )

    assert res.deliberation_passes == 3
    assert res.tournament_bracket is not None
    assert "tournament_winner" in res.tournament_bracket
    assert "foil_margin" in res.tournament_bracket
    assert res.winner in ["route_alpha", "route_beta", "route_gamma"]
