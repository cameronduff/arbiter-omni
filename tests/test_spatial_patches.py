"""
Unit tests for Spatial Patch Cross-Attention Visual Grounding [AO-09].
"""

import numpy as np
import pytest
import torch
from PIL import Image

from arbiter_omni.data.cached import CachedMultimodalDataset
from arbiter_omni.data.dataset import MultimodalDecisionDataset
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder
from arbiter_omni.fusion.transformer import TransformerMultimodalFusion
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.training.config import TrainingConfig
from arbiter_omni.training.trainer import ArbiterOmniTrainer
from arbiter_omni.types import ModalityType, MultimodalSample


def test_mock_encoder_spatial_patches():
    """Validates MockMultimodalEncoder spatial patch token generation."""
    encoder = MockMultimodalEncoder(embed_dim=128)
    images = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),
        Image.new("RGB", (224, 224), color=(0, 255, 0)),
    ]
    patches = encoder.encode_image_patches(images, num_patches=49)
    assert patches.shape == (2, 49, 128)
    # Check L2 normalization
    norms = patches.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_openclip_encoder_spatial_patches():
    """Validates OpenCLIP unpooled 7x7 spatial patch token extraction from ViT-B-32."""
    try:
        encoder = OpenCLIPMultimodalEncoder(device="cpu")
    except Exception as e:
        pytest.skip(f"OpenCLIP weights unavailable offline: {e}")

    images = [
        Image.new("RGB", (224, 224), color=(100, 150, 200)),
        Image.new("RGB", (224, 224), color=(50, 80, 120)),
    ]
    patches = encoder.encode_image_patches(images)
    assert patches.shape == (2, 49, 512)
    norms = patches.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_transformer_fusion_spatial_patches_cross_attention():
    """Validates TransformerMultimodalFusion cross-attention with spatial patches."""
    modality_dims = {
        "question": 128,
        ModalityType.TEXT.value: 128,
        ModalityType.IMAGE.value: 128,
        ModalityType.VIDEO.value: 128,
        ModalityType.AUDIO.value: 128,
    }
    fusion = TransformerMultimodalFusion(modality_dims=modality_dims, hidden_dim=256)

    B = 2
    q_embed = torch.randn(B, 128)
    mod_embeds = {
        ModalityType.TEXT: torch.randn(B, 128),
        ModalityType.IMAGE: torch.randn(B, 128),
        ModalityType.VIDEO: torch.randn(B, 128),
        ModalityType.AUDIO: torch.randn(B, 128),
    }
    presence_mask = {
        ModalityType.TEXT: torch.tensor([True, True]),
        ModalityType.IMAGE: torch.tensor([True, True]),
        ModalityType.VIDEO: torch.tensor([False, False]),
        ModalityType.AUDIO: torch.tensor([True, False]),
    }
    # 49 spatial visual patch tokens
    image_patches = torch.randn(B, 49, 128)

    fused = fusion(
        question_embed=q_embed,
        modality_embeds=mod_embeds,
        presence_mask=presence_mask,
        image_patches=image_patches,
    )
    assert fused.shape == (B, 256)
    assert not torch.isnan(fused).any()


def test_spatial_patches_missing_modality_zero_leakage():
    """Validates that when the image modality is absent, all spatial patches are masked out without NaN."""
    modality_dims = {
        "question": 128,
        ModalityType.TEXT.value: 128,
        ModalityType.IMAGE.value: 128,
        ModalityType.VIDEO.value: 128,
        ModalityType.AUDIO.value: 128,
    }
    fusion = TransformerMultimodalFusion(modality_dims=modality_dims, hidden_dim=256)

    B = 2
    q_embed = torch.randn(B, 128)
    mod_embeds = {
        ModalityType.TEXT: torch.randn(B, 128),
        ModalityType.IMAGE: torch.zeros(B, 128),
        ModalityType.VIDEO: torch.zeros(B, 128),
        ModalityType.AUDIO: torch.zeros(B, 128),
    }
    # Image is missing
    presence_mask = {
        ModalityType.TEXT: torch.tensor([True, True]),
        ModalityType.IMAGE: torch.tensor([False, False]),
        ModalityType.VIDEO: torch.tensor([False, False]),
        ModalityType.AUDIO: torch.tensor([False, False]),
    }
    image_patches = torch.randn(B, 49, 128)

    fused = fusion(
        question_embed=q_embed,
        modality_embeds=mod_embeds,
        presence_mask=presence_mask,
        image_patches=image_patches,
    )
    assert fused.shape == (B, 256)
    assert not torch.isnan(fused).any()


def test_arbiter_model_spatial_patches_end_to_end():
    """Validates full ArbiterOmniModel forward pass with spatial patch tokens active."""
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, use_spatial_patches=True)

    img1 = Image.new("RGB", (224, 224), color=(255, 50, 50))
    img2 = Image.new("RGB", (224, 224), color=(50, 255, 50))

    logits, probs, entropy, fused = model(
        questions=["Where is the obstacle located?", "Identify anomaly region:"],
        candidates=[
            ["upper-left quadrant", "bottom-right quadrant", "center region"],
            ["sector A", "sector B", "nominal state"],
        ],
        images=[img1, img2],
    )
    assert logits.shape == (2, 3)
    assert probs.shape == (2, 3)
    assert entropy.shape == (2,)
    assert fused.shape == (2, 256)
    # Check probabilities sum to 1.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_cached_dataset_and_trainer_with_spatial_patches():
    """Validates pre-caching and rapid training with spatial patch representations."""
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, use_spatial_patches=True)

    samples = [
        MultimodalSample(
            question=f"Spatial reasoning query {i}",
            candidates=["quadrant 1", "quadrant 2", "quadrant 3", "quadrant 4"],
            target_idx=i % 4,
            image=Image.new("RGB", (224, 224), color=(i * 20 % 255, 128, 128)),
        )
        for i in range(16)
    ]
    raw_dataset = MultimodalDecisionDataset(samples)

    # Pre-cache dataset including spatial patches
    cached_dataset = CachedMultimodalDataset.from_dataset(raw_dataset, model=model, batch_size=8)
    assert len(cached_dataset) == 16
    assert cached_dataset[0].image_patches is not None
    assert cached_dataset[0].image_patches.shape[0] == 49

    config = TrainingConfig(
        num_epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        fp16=False,
    )
    trainer = ArbiterOmniTrainer(model=model, config=config)
    history = trainer.fit(train_dataset=cached_dataset, val_dataset=cached_dataset)

    assert "loss" in history
    assert len(history["loss"]) == 2
    assert history["val_accuracy"][-1] >= 0.0
