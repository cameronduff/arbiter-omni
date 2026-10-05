"""
Unit tests for CLAPAudioEncoder and spectral fallback.
"""

import numpy as np
import pytest
import torch

from arbiter_omni.encoders.clap import CLAPAudioEncoder


def test_clap_audio_encoder_fallback_shapes():
    encoder = CLAPAudioEncoder(output_dim=128, load_pretrained=False)
    
    # 2 audio waveforms (16 kHz, 1 second = 16000 samples)
    audio1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    audio2 = np.random.randn(8000).astype(np.float32)
    
    embeds = encoder.encode_audio([audio1, audio2])
    assert embeds.shape == (2, 128)
    
    norms = embeds.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_clap_audio_encoder_torch_inputs():
    encoder = CLAPAudioEncoder(output_dim=64, load_pretrained=False)
    tensor_audio = torch.randn(12000)
    
    embeds = encoder.encode_audio([tensor_audio])
    assert embeds.shape == (1, 64)
    assert torch.isclose(embeds.norm(), torch.tensor(1.0), atol=1e-4)


def test_clap_freeze():
    encoder = CLAPAudioEncoder(output_dim=64, load_pretrained=False)
    encoder.freeze()
    
    for p in encoder.parameters():
        assert not p.requires_grad
