"""
Unit tests for Multimodal Fusion modules and missing modality handling.
"""

import pytest
import torch

from arbiter_omni.fusion.gated import GatedMultimodalFusion
from arbiter_omni.fusion.transformer import TransformerMultimodalFusion
from arbiter_omni.types import ModalityType


@pytest.fixture
def modality_dims():
    return {
        "question": 64,
        ModalityType.TEXT.value: 64,
        ModalityType.IMAGE.value: 64,
        ModalityType.VIDEO.value: 64,
        ModalityType.AUDIO.value: 64,
    }


def test_transformer_fusion_full_and_missing(modality_dims):
    fusion = TransformerMultimodalFusion(modality_dims=modality_dims, hidden_dim=32, num_heads=2, num_layers=1)
    B = 3

    q_embed = torch.randn(B, 64)
    mod_embeds = {
        ModalityType.TEXT: torch.randn(B, 64),
        ModalityType.IMAGE: torch.randn(B, 64),
        ModalityType.VIDEO: torch.randn(B, 64),
        ModalityType.AUDIO: torch.randn(B, 64),
    }

    # Case 1: All modalities present
    presence_full = {mod: torch.ones(B, dtype=torch.bool) for mod in mod_embeds}
    fused_full = fusion(q_embed, mod_embeds, presence_full)
    assert fused_full.shape == (B, 32)
    assert not torch.isnan(fused_full).any()

    # Case 2: Only audio present, text/image/video missing
    presence_audio_only = {
        ModalityType.TEXT: torch.zeros(B, dtype=torch.bool),
        ModalityType.IMAGE: torch.zeros(B, dtype=torch.bool),
        ModalityType.VIDEO: torch.zeros(B, dtype=torch.bool),
        ModalityType.AUDIO: torch.ones(B, dtype=torch.bool),
    }
    fused_audio = fusion(q_embed, mod_embeds, presence_audio_only)
    assert fused_audio.shape == (B, 32)
    assert not torch.isnan(fused_audio).any()


def test_gated_fusion(modality_dims):
    fusion = GatedMultimodalFusion(modality_dims=modality_dims, hidden_dim=32)
    B = 2
    q_embed = torch.randn(B, 64)
    mod_embeds = {
        ModalityType.TEXT: torch.randn(B, 64),
        ModalityType.IMAGE: torch.randn(B, 64),
        ModalityType.VIDEO: torch.randn(B, 64),
        ModalityType.AUDIO: torch.randn(B, 64),
    }
    presence = {mod: torch.ones(B, dtype=torch.bool) for mod in mod_embeds}
    fused = fusion(q_embed, mod_embeds, presence)
    assert fused.shape == (B, 32)
    assert not torch.isnan(fused).any()
