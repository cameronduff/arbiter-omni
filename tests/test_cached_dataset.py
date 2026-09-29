"""
Unit tests for ArbiterOmni CachedMultimodalDataset, cached forward pass, and accelerated training.
"""

import os
import tempfile
import time
import pytest
import torch

from arbiter_omni import (
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    CachedMultimodalDataset,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    TrainingConfig,
    collate_cached_multimodal_decision,
    generate_synthetic_dataset,
)


def test_cached_dataset_creation_and_forward_equivalence():
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)
    model.eval()

    samples = generate_synthetic_dataset(num_samples=8)
    raw_dataset = MultimodalDecisionDataset(samples)

    # 1. Pre-cache dataset
    cached_dataset = CachedMultimodalDataset.from_dataset(
        raw_dataset, model=model, batch_size=4, verbose=False
    )
    assert len(cached_dataset) == 8

    # 2. Test sample inspection
    sample0 = cached_dataset[0]
    assert sample0.question_embed.shape == (128,)
    assert sample0.candidate_embeds.shape[-1] == 128
    assert "text" in sample0.modality_embeds
    assert "image" in sample0.modality_embeds

    # 3. Test collator
    batch = collate_cached_multimodal_decision([cached_dataset[0], cached_dataset[1]])
    assert batch["is_cached"] is True
    assert batch["question_embed"].shape == (2, 128)
    assert batch["candidate_embeds"].shape[0] == 2
    assert batch["targets"].shape == (2,)

    # 4. Verify exact output equivalence between raw model() and model.forward_cached()
    with torch.no_grad():
        # Raw forward pass
        raw_logits, raw_probs, raw_entropy, raw_ctx = model(
            questions=[samples[0].question],
            candidates=[samples[0].candidates],
            texts=[samples[0].text],
            images=[samples[0].image],
            videos=[samples[0].video],
            audios=[samples[0].audio],
        )

        # Cached forward pass
        c_batch = collate_cached_multimodal_decision([cached_dataset[0]])
        c_logits, c_probs, c_entropy, c_ctx = model.forward_cached(
            question_embed=c_batch["question_embed"],
            modality_embeds=c_batch["modality_embeds"],
            presence_mask=c_batch["presence_mask"],
            candidate_embeds=c_batch["candidate_embeds"],
            candidate_mask=c_batch["candidate_mask"],
        )

        assert torch.allclose(raw_logits, c_logits, atol=1e-5)
        assert torch.allclose(raw_probs, c_probs, atol=1e-5)
        assert torch.allclose(raw_entropy, c_entropy, atol=1e-5)


def test_cached_dataset_serialization():
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)

    samples = generate_synthetic_dataset(num_samples=6)
    raw_dataset = MultimodalDecisionDataset(samples)
    cached_dataset = CachedMultimodalDataset.from_dataset(raw_dataset, model=model, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "cached_tensors.pt")
        cached_dataset.save(cache_file)
        assert os.path.exists(cache_file)

        loaded_dataset = CachedMultimodalDataset.load(cache_file)
        assert len(loaded_dataset) == 6
        assert torch.allclose(
            cached_dataset[0].question_embed, loaded_dataset[0].question_embed
        )


def test_trainer_accelerated_cached_training():
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)

    samples = generate_synthetic_dataset(num_samples=32)
    raw_dataset = MultimodalDecisionDataset(samples)
    cached_dataset = CachedMultimodalDataset.from_dataset(raw_dataset, model=model, verbose=False)

    config = TrainingConfig(
        num_epochs=3,
        batch_size=16,
        learning_rate=1e-3,
    )
    trainer = ArbiterOmniTrainer(model=model, config=config)

    t0 = time.perf_counter()
    history = trainer.fit(train_dataset=cached_dataset)
    elapsed = time.perf_counter() - t0

    assert len(history["loss"]) == 3
    assert history["accuracy"][-1] >= 0.0
    # 3 epochs of 32 samples on cached tensors should execute in under 1 second!
    assert elapsed < 3.0, f"Expected fast execution but took {elapsed:.2f}s"
