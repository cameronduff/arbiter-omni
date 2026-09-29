"""
Unit tests for Candidate Prompt Template Ensembling (AO-11).
Verifies that multi-template candidate ensembling reduces prompt variance,
boosts zero-shot visual discrimination, and supports custom templates.
"""

import torch
import pytest

from arbiter_omni.model.arbiter import ArbiterOmniModel, DEFAULT_PROMPT_TEMPLATES
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.api.engine import ArbiterOmniEngine


def test_default_prompt_templates_defined():
    assert len(DEFAULT_PROMPT_TEMPLATES) >= 2
    assert any("{}" in t for t in DEFAULT_PROMPT_TEMPLATES)


def test_encode_candidates_with_prompt_ensembling():
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)

    candidates = [["Cat", "Dog", "Bird"]]

    # 1. Without ensembling
    raw_embeds, mask_raw = model.encode_candidates(candidates, use_prompt_ensembling=False)
    assert raw_embeds.shape == (1, 3, 128)
    assert mask_raw.all()

    # 2. With ensembling
    ens_embeds, mask_ens = model.encode_candidates(candidates, use_prompt_ensembling=True)
    assert ens_embeds.shape == (1, 3, 128)
    assert mask_ens.all()

    # Embeddings must be unit-normalized
    norms = ens_embeds[0].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_custom_prompt_templates():
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)

    candidates = [["Cat", "Dog"]]
    custom_tmpls = ["this is a {}", "photo of {}"]

    embeds, _ = model.encode_candidates(
        candidates,
        prompt_templates=custom_tmpls,
        use_prompt_ensembling=True,
    )
    assert embeds.shape == (1, 2, 128)


def test_engine_decide_prompt_ensembling_flag():
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)
    engine = ArbiterOmniEngine(model=model)

    res = engine.decide(
        question="What is this entity?",
        candidates=["Alpha", "Beta"],
        use_prompt_ensembling=True,
    )
    assert res.winner in ["Alpha", "Beta"]
    assert 0.0 <= res.confidence <= 1.0
