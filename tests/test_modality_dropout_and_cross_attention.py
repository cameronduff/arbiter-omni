"""
Unit tests for Modality Dropout and Question-Conditioned Cross-Attention [AO-14].
"""

import pytest
import torch
import torch.nn as nn
from PIL import Image

from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.fusion.transformer import TransformerMultimodalFusion
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.training.config import TrainingConfig
from arbiter_omni.training.trainer import ArbiterOmniTrainer
from arbiter_omni.types import ModalityType, MultimodalSample


def test_modality_dropout_eval_vs_train():
    """Validates that modality dropout is strictly deactivated in eval mode and active during training."""
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128, modality_dropout_prob=0.5)

    presence_mask = {
        ModalityType.TEXT: torch.ones(100, dtype=torch.bool),
        ModalityType.IMAGE: torch.ones(100, dtype=torch.bool),
        ModalityType.VIDEO: torch.zeros(100, dtype=torch.bool),
    }

    # 1. In eval mode: presence mask must remain 100% intact
    model.eval()
    eval_mask = model.apply_modality_dropout(presence_mask)
    assert torch.equal(eval_mask[ModalityType.TEXT], presence_mask[ModalityType.TEXT])
    assert torch.equal(eval_mask[ModalityType.IMAGE], presence_mask[ModalityType.IMAGE])
    assert torch.equal(eval_mask[ModalityType.VIDEO], presence_mask[ModalityType.VIDEO])

    # 2. In train mode: present modalities must experience dropout (~50% kept)
    model.train()
    train_mask = model.apply_modality_dropout(presence_mask)
    text_kept = train_mask[ModalityType.TEXT].float().mean().item()
    image_kept = train_mask[ModalityType.IMAGE].float().mean().item()
    # Statistical tolerance for 100 samples with p=0.5
    assert 0.25 <= text_kept <= 0.75
    assert 0.25 <= image_kept <= 0.75
    # Video was absent to begin with, must remain strictly False
    assert (train_mask[ModalityType.VIDEO] == False).all().item()


def test_question_conditioned_query_token():
    """Validates that question-conditioned query tokens properly integrate question embeddings."""
    modality_dims = {
        "question": 64,
        ModalityType.TEXT.value: 64,
        ModalityType.IMAGE.value: 64,
        ModalityType.VIDEO.value: 64,
        ModalityType.AUDIO.value: 64,
    }
    fusion_static = TransformerMultimodalFusion(
        modality_dims=modality_dims,
        hidden_dim=32,
        condition_query_on_question=False,
    )
    fusion_conditioned = TransformerMultimodalFusion(
        modality_dims=modality_dims,
        hidden_dim=32,
        condition_query_on_question=True,
    )

    B = 2
    q1 = torch.randn(B, 64)
    q2 = torch.randn(B, 64)
    mod_embeds = {mod: torch.randn(B, 64) for mod in [ModalityType.TEXT, ModalityType.IMAGE]}
    presence = {mod: torch.ones(B, dtype=torch.bool) for mod in mod_embeds}

    fused_cond_1 = fusion_conditioned(q1, mod_embeds, presence)
    fused_cond_2 = fusion_conditioned(q2, mod_embeds, presence)

    assert fused_cond_1.shape == (B, 32)
    assert not torch.isnan(fused_cond_1).any()
    # Different questions produce distinct conditioned decision representations
    assert not torch.allclose(fused_cond_1, fused_cond_2, atol=1e-3)


def test_spatial_cross_attention_with_and_without_images():
    """Validates multi-head cross-attention over visual spatial patches, including zero-leakage missing handling."""
    modality_dims = {
        "question": 64,
        ModalityType.TEXT.value: 64,
        ModalityType.IMAGE.value: 64,
        ModalityType.VIDEO.value: 64,
        ModalityType.AUDIO.value: 64,
    }
    fusion = TransformerMultimodalFusion(
        modality_dims=modality_dims,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        enable_spatial_cross_attention=True,
    )

    B = 2
    P = 49
    q_embed = torch.randn(B, 64)
    mod_embeds = {
        ModalityType.TEXT: torch.randn(B, 64),
        ModalityType.IMAGE: torch.randn(B, 64),
    }
    patches = torch.randn(B, P, 64)

    # Case 1: Both samples have image present
    presence_both = {
        ModalityType.TEXT: torch.tensor([True, True]),
        ModalityType.IMAGE: torch.tensor([True, True]),
    }
    out_both = fusion(q_embed, mod_embeds, presence_both, image_patches=patches)
    assert out_both.shape == (B, 32)
    assert not torch.isnan(out_both).any()

    # Case 2: One sample has image, one has image MISSING
    presence_partial = {
        ModalityType.TEXT: torch.tensor([True, True]),
        ModalityType.IMAGE: torch.tensor([True, False]),
    }
    out_partial = fusion(q_embed, mod_embeds, presence_partial, image_patches=patches)
    assert out_partial.shape == (B, 32)
    assert not torch.isnan(out_partial).any()

    # Case 3: ALL samples have image MISSING (must safely bypass cross attention without NaN)
    presence_none = {
        ModalityType.TEXT: torch.tensor([True, True]),
        ModalityType.IMAGE: torch.tensor([False, False]),
    }
    out_none = fusion(q_embed, mod_embeds, presence_none, image_patches=patches)
    assert out_none.shape == (B, 32)
    assert not torch.isnan(out_none).any()


def test_end_to_end_model_and_trainer_modality_dropout():
    """Validates end-to-end forward pass and trainer configuration with modality dropout."""
    encoder = MockMultimodalEncoder(embed_dim=128)
    config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=1,
        modality_dropout_prob=0.25,
        fp16=False,
    )
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)
    trainer = ArbiterOmniTrainer(model=model, config=config)

    # Verify trainer initialized model's modality_dropout_prob
    assert model.modality_dropout_prob == 0.25

    # Run forward pass during training
    model.train()
    images = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),
        Image.new("RGB", (224, 224), color=(0, 255, 0)),
    ]
    logits, probs, entropy, fused = model(
        questions=["Is this a red image?", "Is this a green image?"],
        candidates=[["yes", "no"], ["yes", "no"]],
        images=images,
    )
    assert probs.shape == (2, 2)
    assert not torch.isnan(probs).any()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
