"""
Unit tests for Dynamic Decision Head.
"""

import pytest
import torch

from arbiter_omni.model.decision_head import DynamicDecisionHead


def test_dynamic_candidates_scoring():
    head = DynamicDecisionHead(context_dim=32, candidate_dim=64, scoring_dim=32)
    B = 4
    K = 5
    context = torch.randn(B, 32)
    candidates = torch.randn(B, K, 64)

    logits, probs, entropy = head(context, candidates)

    assert logits.shape == (B, K)
    assert probs.shape == (B, K)
    assert entropy.shape == (B,)

    # Probabilities must sum to 1.0
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5)
    # Entropy should be non-negative
    assert (entropy >= 0.0).all()


def test_masked_variable_candidates():
    head = DynamicDecisionHead(context_dim=32, candidate_dim=64, scoring_dim=32)
    B = 2
    max_K = 4
    context = torch.randn(B, 32)
    candidates = torch.randn(B, max_K, 64)

    # Sample 0 has 2 valid candidates, Sample 1 has 3 valid candidates
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
    ], dtype=torch.bool)

    logits, probs, entropy = head(context, candidates, candidate_mask=mask)

    # Masked positions should have 0 probability
    assert probs[0, 2] < 1e-6
    assert probs[0, 3] < 1e-6
    assert probs[1, 3] < 1e-6

    # Active positions sum to 1.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_boolean_and_score_primitives():
    head = DynamicDecisionHead(context_dim=32, candidate_dim=64, scoring_dim=32)
    context = torch.randn(3, 32)
    bool_certs = head.predict_boolean_noul(context)
    scores = head.predict_score(context)

    assert bool_certs.shape == (3,)
    assert (bool_certs >= 0.0).all() and (bool_certs <= 1.0).all()
    assert scores.shape == (3,)
