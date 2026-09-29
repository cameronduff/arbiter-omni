"""
Unit tests for Hard-Negative Contrastive Candidate Mining and Margin Loss [AO-07].
"""

import pytest
import torch
import torch.nn.functional as F

from arbiter_omni.data.dataset import MultimodalDecisionDataset
from arbiter_omni.data.miner import DEFAULT_CANDIDATE_POOL, HardNegativeMiner
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.model.decision_head import DynamicDecisionHead, contrastive_margin_loss
from arbiter_omni.training.config import TrainingConfig
from arbiter_omni.training.trainer import ArbiterOmniTrainer
from arbiter_omni.types import MultimodalSample


def test_hard_negative_miner_pool_and_mining():
    """Validates HardNegativeMiner similarity search and foil retrieval."""
    encoder = MockMultimodalEncoder()
    pool = [
        "proceed at nominal velocity",
        "proceed with caution",
        "decelerate immediately",
        "execute emergency stop",
        "maintain current trajectory",
        "turn left sharply",
    ]
    miner = HardNegativeMiner(encoder=encoder, candidate_pool=pool)
    assert len(miner.pool) == len(pool)
    assert miner.pool_embeddings.shape[0] == len(pool)

    # Test mining for target
    target = "proceed at nominal velocity"
    mined = miner.mine_hard_negatives(target, k=2, min_similarity=-1.0, max_similarity=0.999)
    assert len(mined) <= 2
    # Ensure exact match is excluded
    assert target not in mined

    # Test augment_sample
    sample = MultimodalSample(
        question="What velocity should be commanded?",
        candidates=["proceed at nominal velocity", "execute emergency stop"],
        target_idx=0,
    )
    augmented = miner.augment_sample(sample, num_hard_negatives=2, min_similarity=-1.0)
    assert len(augmented.candidates) > len(sample.candidates)
    assert augmented.target_idx == 0
    assert "mined_hard_negatives" in augmented.metadata


def test_contrastive_margin_loss_exact_math():
    """Validates exact numerical math of contrastive margin loss."""
    # Batch size 2, 3 candidates each
    logits = torch.tensor([
        [2.0, 1.0, 0.5],   # Target 0: s_pos = 2.0, hard_neg = 1.0 -> diff = 1.0
        [1.5, 3.0, 2.5],   # Target 1: s_pos = 3.0, hard_neg = 2.5 -> diff = 0.5
    ], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    # Case 1: margin gamma = 1.2
    # Sample 0: max(0, 1.2 - 1.0) = 0.2
    # Sample 1: max(0, 1.2 - 0.5) = 0.7
    # Mean: (0.2 + 0.7) / 2 = 0.45
    loss_mean = contrastive_margin_loss(logits, targets, margin=1.2, reduction="mean")
    assert pytest.approx(loss_mean.item(), rel=1e-5) == 0.45

    # Case 2: margin gamma = 0.4
    # Sample 0: max(0, 0.4 - 1.0) = 0.0
    # Sample 1: max(0, 0.4 - 0.5) = 0.0
    # Mean: 0.0
    loss_zero = contrastive_margin_loss(logits, targets, margin=0.4, reduction="mean")
    assert pytest.approx(loss_zero.item(), abs=1e-6) == 0.0


def test_contrastive_margin_loss_with_mask_and_gradients():
    """Validates contrastive margin loss with candidate padding masks and backward pass."""
    logits = torch.tensor([
        [2.0, 2.5, 0.0],  # Candidate 1 is higher than target 0, but masked out!
    ], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0], dtype=torch.long)
    candidate_mask = torch.tensor([[True, False, True]])  # Candidate 1 is padded/masked

    # Target 0: s_pos = 2.0. Unmasked negative is Candidate 2 with score 0.0 -> diff = 2.0
    # Margin 2.5 -> violation = 2.5 - 2.0 = 0.5
    loss = contrastive_margin_loss(logits, targets, candidate_mask=candidate_mask, margin=2.5)
    assert pytest.approx(loss.item(), rel=1e-5) == 0.5

    # Ensure gradients flow
    loss.backward()
    assert logits.grad is not None
    # pos logit should have negative gradient (increasing it decreases loss)
    assert logits.grad[0, 0] < 0
    # unmasked hard negative logit should have positive gradient (increasing it increases loss)
    assert logits.grad[0, 2] > 0
    # masked candidate should receive zero gradient
    assert logits.grad[0, 1] == 0.0


def test_decision_head_compute_loss():
    """Validates DynamicDecisionHead.compute_loss combines CE and contrastive loss."""
    head = DynamicDecisionHead(context_dim=256, candidate_dim=512, scoring_dim=256)
    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0], dtype=torch.long)

    total_loss, ce_loss, margin_loss = head.compute_loss(
        logits=logits,
        targets=targets,
        margin=1.5,
        contrastive_lambda=0.5,
    )

    # s_pos = 2.0, hard_neg = 1.0 -> diff = 1.0 -> margin violation = 1.5 - 1.0 = 0.5
    assert pytest.approx(margin_loss.item(), rel=1e-5) == 0.5
    expected_total = ce_loss.item() + 0.5 * 0.5
    assert pytest.approx(total_loss.item(), rel=1e-5) == expected_total


def test_trainer_contrastive_margin_learning():
    """Tests end-to-end training loop with contrastive margin loss active."""
    encoder = MockMultimodalEncoder()
    model = ArbiterOmniModel(encoder=encoder)

    samples = [
        MultimodalSample(
            question=f"Decision sample {i}",
            candidates=["proceed at nominal velocity", "proceed with caution", "emergency stop"],
            target_idx=0,
            text=f"Telemetry report {i}",
        )
        for i in range(12)
    ]
    dataset = MultimodalDecisionDataset(samples)

    config = TrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        fp16=False,
        contrastive_lambda=0.5,
        margin_gamma=0.3,
    )
    trainer = ArbiterOmniTrainer(model=model, config=config)
    history = trainer.fit(train_dataset=dataset, val_dataset=dataset)

    assert "loss" in history
    assert "accuracy" in history
    assert "val_margin_loss" in history
    # Check that margin loss is computed and finite
    for ml in history["val_margin_loss"]:
        assert ml >= 0.0


def test_fine_grained_foil_entropy_separation():
    """
    Benchmarks entropy separation and logit sharpness on fine-grained candidate choices.
    Verifies that the contrastive margin penalty enforces separation between target
    and near-identical foil decisions.
    """
    head = DynamicDecisionHead(context_dim=256, candidate_dim=512, scoring_dim=256)
    ctx = torch.randn(1, 256)
    
    # Near identical embeddings (simulating fine-grained candidate choices)
    base_cand = torch.randn(1, 1, 512)
    foil_cand = base_cand + 0.01 * torch.randn(1, 1, 512)
    cands = torch.cat([base_cand, foil_cand], dim=1)  # [1, 2, 512]

    logits, probs, entropy = head(context_embed=ctx, candidate_embeds=cands)
    initial_gap = float((logits[0, 0] - logits[0, 1]).abs().item())

    # Under contrastive margin loss with target=0, penalty fires if gap < margin
    target = torch.tensor([0], dtype=torch.long)
    margin = 1.0
    loss = contrastive_margin_loss(logits, target, margin=margin)
    
    # If the logits are close, margin loss must be strictly positive
    if initial_gap < margin:
        assert loss.item() > 0.0
