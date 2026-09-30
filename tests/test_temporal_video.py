"""
Unit tests for SpatioTemporalVideoAttention.
Verifies multi-frame sequence processing, normalization, and chronological sensitivity.
"""

import pytest
import torch

from arbiter_omni.encoders.temporal import (
    SpatioTemporalVideoAttention,
    compute_patch_centroid_flow,
)


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


def test_video_file_decoding(tmp_path):
    import imageio.v3 as iio
    import numpy as np
    from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder

    vid_path = str(tmp_path / "test_clip.mp4")
    frames = [np.full((32, 32, 3), fill_value=i * 50, dtype=np.uint8) for i in range(4)]
    iio.imwrite(vid_path, np.stack(frames), fps=4)

    encoder = OpenCLIPMultimodalEncoder(device="cpu")
    emb = encoder.encode_video([vid_path])
    assert emb.shape == (1, encoder.video_dim)
    assert torch.isclose(emb.norm(), torch.tensor(1.0), atol=1e-4)


def test_motion_delta_velocity_projection():
    attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=8, num_heads=4, num_layers=1)
    attn.eval()

    # Create synthetic linear motion: frame_t = base + t * velocity
    base = torch.randn(1, 64)
    vel_right = torch.ones(1, 64) * 0.5
    vel_left = -torch.ones(1, 64) * 0.5

    frames_right = torch.cat([base + i * vel_right for i in range(5)], dim=0) # [5, 64]
    frames_left = torch.cat([base + i * vel_left for i in range(5)], dim=0)   # [5, 64]

    with torch.no_grad():
        out_right = attn(frames_right)
        out_left = attn(frames_left)

    # Opposite motion vectors should yield distinct embeddings
    assert not torch.allclose(out_right, out_left, atol=1e-3)
    sim = torch.dot(out_right, out_left).item()
    assert sim < 0.998, f"Motion delta projection failed to separate opposing velocities: sim={sim:.4f}"


def test_patch_centroid_flow():
    from arbiter_omni.encoders.temporal import compute_patch_centroid_flow

    grid_size = 14
    T = 4
    # Case A: Object translating horizontally right: patch (7, 2) -> (7, 5) -> (7, 8) -> (7, 11)
    patches_right = torch.zeros(1, T, grid_size * grid_size, 32)
    for t in range(T):
        x = 2 + t * 3
        y = 7
        idx = y * grid_size + x
        patches_right[0, t, idx, :] = 10.0  # high energy at centroid

    vel_right = compute_patch_centroid_flow(patches_right, grid_size=grid_size)
    assert vel_right.shape == (1, T, 2)

    # dx should be strictly positive for t >= 1
    assert (vel_right[0, 1:, 0] > 0.05).all(), f"Expected positive dx: {vel_right[0, :, 0]}"
    # dy should be approximately 0
    assert torch.allclose(vel_right[0, 1:, 1], torch.zeros(T - 1), atol=1e-3)

    # Case B: Object dropping vertically: patch (2, 7) -> (5, 7) -> (8, 7) -> (11, 7)
    patches_drop = torch.zeros(1, T, grid_size * grid_size, 32)
    for t in range(T):
        y = 2 + t * 3
        x = 7
        idx = y * grid_size + x
        patches_drop[0, t, idx, :] = 10.0

    vel_drop = compute_patch_centroid_flow(patches_drop, grid_size=grid_size)
    assert (vel_drop[0, 1:, 1] > 0.05).all(), f"Expected positive dy: {vel_drop[0, :, 1]}"
    assert torch.allclose(vel_drop[0, 1:, 0], torch.zeros(T - 1), atol=1e-3)


def test_spatio_temporal_attention_with_patches():
    attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=8, num_heads=4, num_layers=1)
    attn.eval()

    T = 4
    grid_size = 14
    frames = torch.randn(1, T, 64)
    patches = torch.randn(1, T, grid_size * grid_size, 64)

    with torch.no_grad():
        out = attn(frames, patch_features=patches)

    assert out.shape == (1, 64)
    assert torch.isclose(out.norm(), torch.tensor(1.0), atol=1e-4)


def test_dense_64_frame_video_buffering():
    """Validates that SpatioTemporalVideoAttention handles up to 64 dense temporal frames seamlessly."""
    attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=64, num_heads=4, num_layers=1)
    attn.eval()

    T = 64
    grid_size = 14
    frames = torch.randn(2, T, 64)
    patches = torch.randn(2, T, grid_size * grid_size, 64)

    with torch.no_grad():
        out = attn(frames, patch_features=patches)

    assert out.shape == (2, 64)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_multi_stride_centroid_flow():
    """Validates multi-stride centroid velocity flow over long temporal horizons (T >= 8)."""
    grid_size = 14
    T = 10
    patches = torch.zeros(1, T, grid_size * grid_size, 32)
    for t in range(T):
        x = min(grid_size - 1, 1 + t)
        y = 7
        idx = y * grid_size + x
        patches[0, t, idx, :] = 10.0

    # Flow with long_stride=4
    vel = compute_patch_centroid_flow(patches, grid_size=grid_size, long_stride=4, blend_ratio=0.5)
    assert vel.shape == (1, T, 2)
    # Both short-range and long-range should confirm positive dx
    assert (vel[0, 1:, 0] > 0.01).all()
    # dy should be near 0
    assert torch.allclose(vel[0, :, 1], torch.zeros(T), atol=1e-3)


def test_video_attention_load_from_32_frames_state_dict():
    """Ensures older 32-frame checkpoints load seamlessly into 64-frame architecture."""
    old_attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=32, num_heads=4, num_layers=1)
    old_state = old_attn.state_dict()

    new_attn = SpatioTemporalVideoAttention(embed_dim=64, max_frames=64, num_heads=4, num_layers=1)
    # Loading 32-frame state dict should not raise shape mismatch
    new_attn.load_state_dict(old_state, strict=True)
    assert new_attn.pos_embed.shape == (1, 64, 64)

    # Output remains unit normalized
    test_frames = torch.randn(1, 16, 64)
    out = new_attn(test_frames)
    assert out.shape == (1, 64)
    assert torch.isclose(out.norm(), torch.tensor(1.0), atol=1e-4)


def test_chunked_video_sub_batching(tmp_path):
    """Verifies that OpenCLIPMultimodalEncoder handles long frame sequences via sub-batched chunking."""
    import imageio.v3 as iio
    import numpy as np
    from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder

    vid_path = str(tmp_path / "long_clip.mp4")
    # 32 synthetic frames
    frames = [np.full((32, 32, 3), fill_value=i * 7, dtype=np.uint8) for i in range(32)]
    iio.imwrite(vid_path, np.stack(frames), fps=8)

    encoder = OpenCLIPMultimodalEncoder(device="cpu")
    # Encode with num_frames=32, sub_batch_size=8
    emb = encoder.encode_video([vid_path], num_frames=32, sub_batch_size=8)
    assert emb.shape == (1, encoder.video_dim)
    assert torch.isclose(emb.norm(), torch.tensor(1.0), atol=1e-4)

