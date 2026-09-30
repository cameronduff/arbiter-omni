"""
Unit tests for PersistentMemoryBank and Global Hard-Negative Contrastive Learning [AO-23].
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from arbiter_omni.data.dataset import MultimodalDecisionDataset
from arbiter_omni.data.memory_bank import PersistentMemoryBank
from arbiter_omni.data.miner import DEFAULT_CANDIDATE_POOL, HardNegativeMiner
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.model.decision_head import DynamicDecisionHead, contrastive_margin_loss
from arbiter_omni.training.config import TrainingConfig
from arbiter_omni.training.trainer import ArbiterOmniTrainer
from arbiter_omni.types import MultimodalSample


def test_memory_bank_initialization_and_capacity():
    """Validates memory bank pre-allocation, memory footprint, and sizing."""
    capacity = 1000
    dim = 768
    bank = PersistentMemoryBank(capacity=capacity, candidate_dim=dim, device="cpu")

    assert len(bank) == 0
    assert not bank.is_full
    assert bank.candidate_bank.shape == (capacity, dim)
    # Memory size for 1000 float32 x 768 is ~2.93 MB
    assert 2.0 < bank.memory_usage_mb < 3.5

    # AO-25: 100k full-scale fp32 memory verification (~293 MB)
    full_bank = PersistentMemoryBank(capacity=100000, candidate_dim=768, device="cpu")
    # 100000 * 768 * 4 bytes = 307,200,000 bytes ≈ 293.0 MB
    assert 285.0 < full_bank.memory_usage_mb < 305.0

    # AO-25: 100k fp16 memory verification (~147 MB — half of fp32)
    fp16_bank = PersistentMemoryBank(capacity=100000, candidate_dim=768, device="cpu", store_fp16=True)
    assert fp16_bank.dtype == torch.float16
    assert fp16_bank.candidate_bank.dtype == torch.float16
    assert 140.0 < fp16_bank.memory_usage_mb < 160.0


def test_memory_bank_cyclic_fifo_enqueue():
    """Validates cyclic FIFO queue insertion and wrap-around semantics."""
    capacity = 10
    dim = 16
    bank = PersistentMemoryBank(capacity=capacity, candidate_dim=dim)

    # 1. Enqueue 4 items
    v1 = torch.randn(4, dim)
    enqueued = bank.enqueue(v1, texts=["a", "b", "c", "d"])
    assert enqueued == 4
    assert len(bank) == 4
    assert bank.ptr == 4
    assert not bank.is_full
    # Verify L2 normalization
    norms = torch.norm(bank.candidate_bank[:4], p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    # 2. Enqueue 6 more items -> reaches exact capacity
    v2 = torch.randn(6, dim)
    enqueued2 = bank.enqueue(v2)
    assert enqueued2 == 6
    assert len(bank) == 10
    assert bank.is_full
    assert bank.ptr == 0

    # 3. Overflow cyclic wrap-around: enqueue 3 new items
    v3 = torch.randn(3, dim)
    enqueued3 = bank.enqueue(v3, texts=["x", "y", "z"])
    assert enqueued3 == 3
    assert len(bank) == 10  # Stays at capacity
    assert bank.ptr == 3
    assert bank.texts[0] == "x"
    assert bank.texts[1] == "y"
    assert bank.texts[2] == "z"


def test_memory_bank_3d_batched_masked_enqueue():
    """Validates enqueuing [B, K, D] candidate representations with boolean masks."""
    capacity = 20
    dim = 32
    bank = PersistentMemoryBank(capacity=capacity, candidate_dim=dim)

    # Batch of 2 samples, 4 candidates each
    cands = torch.randn(2, 4, dim)
    mask = torch.tensor([
        [True, True, False, False],
        [True, False, False, False],
    ], dtype=torch.bool)

    enqueued = bank.enqueue(cands, candidate_mask=mask)
    # Exactly 3 valid candidates should be enqueued
    assert enqueued == 3
    assert len(bank) == 3


def test_memory_bank_query_hard_foils():
    """Validates top-k hard foil semantic retrieval and threshold filtering."""
    capacity = 100
    dim = 64
    bank = PersistentMemoryBank(capacity=capacity, candidate_dim=dim)

    # Base query vector
    query = F.normalize(torch.randn(1, dim), p=2, dim=-1)

    # Construct an orthogonal unit vector
    perp = torch.randn(1, dim)
    perp = F.normalize(perp - (perp @ query.T) * query, p=2, dim=-1)

    # Candidate 1: Exact cosine similarity 0.70 -> comfortably within [0.3, 0.98]
    c1 = 0.7 * query + ((1.0 - 0.7**2) ** 0.5) * perp
    # Candidate 2: Identical similarity (1.0) -> exceeds max_sim 0.98 -> excluded
    c2 = query.clone()
    # Candidate 3: Orthogonal similarity (0.0) -> below min_sim 0.3 -> excluded
    c3 = perp.clone()

    bank.enqueue(torch.cat([c1, c2, c3], dim=0))
    assert len(bank) == 3

    # Query top-5 hard foils
    foils, sims, mask = bank.query_hard_foils(query, k=5, min_sim=0.3, max_sim=0.98)
    assert foils.shape == (1, 5, dim)
    assert sims.shape == (1, 5)
    assert mask.shape == (1, 5)

    # Only c1 should qualify
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is False
    assert pytest.approx(sims[0, 0].item(), abs=1e-4) == 0.70


def test_memory_bank_serialization():
    """Validates saving and loading memory bank state to/from disk."""
    capacity = 50
    dim = 32
    bank = PersistentMemoryBank(capacity=capacity, candidate_dim=dim)
    bank.enqueue(torch.randn(15, dim), texts=[f"item_{i}" for i in range(15)])
    assert len(bank) == 15

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "memory_bank.pt"
        bank.save(save_path)
        assert save_path.is_file()

        restored = PersistentMemoryBank(capacity=capacity, candidate_dim=dim)
        restored.load(save_path)
        assert len(restored) == 15
        assert restored.texts[0] == "item_0"
        assert torch.allclose(restored.candidate_bank[:15], bank.candidate_bank[:15])


def test_decision_head_score_foils_and_global_contrastive_margin():
    """Validates DynamicDecisionHead foil scoring and contrastive margin penalty."""
    context_dim = 64
    candidate_dim = 128
    scoring_dim = 64
    head = DynamicDecisionHead(context_dim=context_dim, candidate_dim=candidate_dim, scoring_dim=scoring_dim)

    B = 2
    ctx = torch.randn(B, context_dim)
    cands = torch.randn(B, 3, candidate_dim)
    targets = torch.tensor([0, 1], dtype=torch.long)

    # Score local candidates
    logits, probs, entropy = head(context_embed=ctx, candidate_embeds=cands)

    # Generate external global foils
    foils = torch.randn(B, 4, candidate_dim)
    foil_logits = head.score_foils(context_embed=ctx, foil_embeds=foils)
    assert foil_logits.shape == (B, 4)

    foil_mask = torch.ones((B, 4), dtype=torch.bool)

    # Contrastive margin loss with global foils
    loss = contrastive_margin_loss(
        logits=logits,
        targets=targets,
        margin=1.0,
        global_foil_logits=foil_logits,
        global_foil_mask=foil_mask,
        global_margin=1.0,
        global_lambda=0.5,
    )
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Check backpropagation gradients flow to both local logits and foil logits
    logits_leaf = logits.clone().detach().requires_grad_(True)
    foil_logits_leaf = foil_logits.clone().detach().requires_grad_(True)
    loss_leaf = contrastive_margin_loss(
        logits=logits_leaf,
        targets=targets,
        margin=1.0,
        global_foil_logits=foil_logits_leaf,
        global_foil_mask=foil_mask,
        global_margin=1.0,
        global_lambda=0.5,
    )
    loss_leaf.backward()
    assert logits_leaf.grad is not None
    assert foil_logits_leaf.grad is not None


def test_trainer_end_to_end_with_memory_bank():
    """Validates complete training epoch with active PersistentMemoryBank."""
    encoder = MockMultimodalEncoder()
    model = ArbiterOmniModel(encoder=encoder)

    samples = [
        MultimodalSample(
            question=f"Navigation decision {i}",
            candidates=[
                "maintain nominal speed",
                "decelerate immediately",
                "emergency full stop",
            ],
            target_idx=0,
            text=f"Sensory report {i}",
        )
        for i in range(16)
    ]
    dataset = MultimodalDecisionDataset(samples)

    config = TrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        fp16=False,
        contrastive_lambda=0.3,
        margin_gamma=0.5,
        use_memory_bank=True,
        memory_bank_capacity=200,
        memory_bank_k_foils=4,
        memory_bank_min_sim=0.1,
    )
    trainer = ArbiterOmniTrainer(model=model, config=config)
    assert trainer.memory_bank is not None
    assert len(trainer.memory_bank) == 0

    history = trainer.fit(train_dataset=dataset, val_dataset=dataset)
    assert "loss" in history
    assert "accuracy" in history
    assert len(trainer.memory_bank) > 0
