"""
Unit tests for Multimodal Encoders.
"""

from PIL import Image
import numpy as np
import pytest
import torch

from arbiter_omni.encoders.mock import MockMultimodalEncoder


def test_mock_encoder_shapes_and_normalization():
    dim = 64
    encoder = MockMultimodalEncoder(embed_dim=dim)
    assert encoder.text_dim == dim
    assert encoder.image_dim == dim
    assert encoder.video_dim == dim
    assert encoder.audio_dim == dim

    # Text
    texts = ["halt", "proceed forward"]
    text_emb = encoder.encode_text(texts)
    assert text_emb.shape == (2, dim)
    assert torch.allclose(text_emb.norm(dim=-1), torch.ones(2), atol=1e-4)

    # Image
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img_emb = encoder.encode_image([img])
    assert img_emb.shape == (1, dim)
    assert torch.allclose(img_emb.norm(dim=-1), torch.ones(1), atol=1e-4)

    # Video
    vid_emb = encoder.encode_video([[img, img]])
    assert vid_emb.shape == (1, dim)
    assert torch.allclose(vid_emb.norm(dim=-1), torch.ones(1), atol=1e-4)

    # Audio
    audio = np.random.randn(1600).astype(np.float32)
    aud_emb = encoder.encode_audio([audio])
    assert aud_emb.shape == (1, dim)
    assert torch.allclose(aud_emb.norm(dim=-1), torch.ones(1), atol=1e-4)


def test_encoder_freezing():
    encoder = MockMultimodalEncoder(embed_dim=64)
    encoder.freeze()
    for p in encoder.parameters():
        assert not p.requires_grad


def test_openclip_siglip_resolution():
    from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder
    encoder = OpenCLIPMultimodalEncoder(model_name="ViT-B-16-SigLIP", device="cpu")
    assert encoder.pretrained == "webli"
    assert encoder.text_dim == 768
    assert encoder.image_dim == 768

    # Test text encoding
    t_emb = encoder.encode_text(["a car", "a dog"])
    assert t_emb.shape == (2, 768)
    assert torch.isclose(t_emb.norm(dim=-1), torch.ones(2), atol=1e-4).all()

    # Test patch extraction
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    patches = encoder.encode_image_patches([img])
    assert patches.shape == (1, 196, 768)
