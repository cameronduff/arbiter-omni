"""
Unit tests for SpatioTemporalVideoAttention.
Verifies multi-frame sequence processing, normalization, and chronological sensitivity.
"""

import pytest
import torch

from arbiter_omni.encoders.temporal import SpatioTemporalVideoAttention


def test_spatio_temporal_video_attention_forward():
    attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=8, num_heads=4, num_layers=1)
    
    # Batched input: 2 videos, 6 frames each, 64-dim features
    frame_feats = torch.randn(2, 6, 64)
    out = attn(frame_feats)
    
    assert out.shape == (2, 64)
    # Verify unit normalization
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_spatio_temporal_video_attention_unbatched():
    attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=8, num_heads=4, num_layers=1)
    
    # Unbatched input: 4 frames, 64-dim
    frame_feats = torch.randn(4, 64)
    out = attn(frame_feats)
    
    assert out.shape == (64,)
    assert torch.isclose(out.norm(), torch.tensor(1.0), atol=1e-4)


def test_spatio_temporal_video_attention_exceeds_max_frames():
    attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=8, num_heads=4, num_layers=1)
    
    # Input has 20 frames, exceeding max_frames of 8
    frame_feats = torch.randn(2, 20, 64)
    out = attn(frame_feats)
    
    assert out.shape == (2, 64)


def test_temporal_sensitivity_order_dependence():
    attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=8, num_heads=4, num_layers=1)
    attn.eval()
    
    # Generate distinct sequential frames
    frames = torch.randn(6, 64)
    reversed_frames = frames.flip(dims=[0])
    
    with torch.no_grad():
        out_forward = attn(frames)
        out_reversed = attn(reversed_frames)
        
    dot_sim = torch.dot(out_forward, out_reversed).item()
    # Unlike commutative mean-pooling where dot_sim is exactly 1.0000,
    # temporal attention with positional embeddings is strictly order-sensitive
    assert dot_sim < 0.9999, f"Temporal attention failed to distinguish reversed video sequence: {dot_sim}"
    assert not torch.allclose(out_forward, out_reversed, atol=1e-4)
