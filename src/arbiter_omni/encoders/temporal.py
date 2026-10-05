"""
Spatio-Temporal Video Attention for continuous video clip arbitration.
Aggregates multi-frame image sequences using sinusoidal temporal positional embeddings
and cross-frame self-attention.
"""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sinusoidal_pos_embed(seq_len: int, embed_dim: int) -> torch.Tensor:
    """Builds sinusoidal positional encoding table."""
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(1, seq_len, embed_dim, dtype=torch.float32)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


def compute_patch_centroid_flow(
    patch_features: torch.Tensor,
    grid_size: int = 14,
    long_stride: int = 4,
    blend_ratio: float = 0.4,
) -> torch.Tensor:
    """
    Computes universal (dx, dy) spatial centroid velocity vectors across video frames,
    combining instant single-frame velocity (stride 1) and long-range trajectory flow (stride k).
    
    Args:
        patch_features: [B, T, P, D] or [T, P, D] unpooled patch embeddings (P = grid_size * grid_size).
        grid_size: Spatial grid dimension (default 14 for 196 patches).
        long_stride: Frame stride for long-range trajectory tracking (default 4).
        blend_ratio: Weight for long-range velocity component in [0.0, 1.0].
        
    Returns:
        velocities: [B, T, 2] continuous (dx, dy) velocity vectors per frame, normalized [-1, 1].
    """
    is_unbatched = patch_features.ndim == 3
    if is_unbatched:
        patch_features = patch_features.unsqueeze(0)  # [1, T, P, D]

    B, T, P, _ = patch_features.shape
    device = patch_features.device

    import math
    if grid_size * grid_size != P:
        calc_grid = int(math.isqrt(P))
        if calc_grid * calc_grid == P or grid_size == 14:
            grid_size = max(1, calc_grid)

    # Normalized spatial coordinate grid in [-1, +1]
    y_coords = (
        torch.linspace(-1.0, 1.0, grid_size, device=device)
        .unsqueeze(1)
        .expand(grid_size, grid_size)
        .reshape(-1)
    )  # [P]
    x_coords = (
        torch.linspace(-1.0, 1.0, grid_size, device=device)
        .unsqueeze(0)
        .expand(grid_size, grid_size)
        .reshape(-1)
    )  # [P]

    actual_p = min(P, grid_size * grid_size)
    y_coords = y_coords[:actual_p]
    x_coords = x_coords[:actual_p]
    sub_patches = patch_features[:, :, :actual_p, :]

    # Patch activation energy/salience: L2 norm across feature channels
    raw_norm = sub_patches.norm(dim=-1)  # [B, T, P]
    if raw_norm.std(dim=-1).max() < 1e-3:
        # Unit-normalized patch embeddings (e.g. from OpenCLIP ViT):
        # Use spatial contrast relative to the background mean across patches
        patch_energy = (sub_patches - sub_patches.mean(dim=-2, keepdim=True)).norm(dim=-1) * 10.0
    else:
        patch_energy = raw_norm

    patch_weights = torch.softmax(patch_energy, dim=-1)  # [B, T, P]

    # Center-of-mass coordinates per frame: [B, T]
    cx = (patch_weights * x_coords).sum(dim=-1)
    cy = (patch_weights * y_coords).sum(dim=-1)

    # Frame-to-frame velocity vectors: (dx, dy)
    dx = torch.zeros(B, T, device=device)
    dy = torch.zeros(B, T, device=device)
    if T >= 2:
        dx_short = cx[:, 1:] - cx[:, :-1]
        dy_short = cy[:, 1:] - cy[:, :-1]
        dx[:, 1:] = dx_short
        dy[:, 1:] = dy_short

        if T > long_stride and blend_ratio > 0.0:
            dx_long = (cx[:, long_stride:] - cx[:, :-long_stride]) / float(long_stride)
            dy_long = (cy[:, long_stride:] - cy[:, :-long_stride]) / float(long_stride)
            dx[:, long_stride:] = (1.0 - blend_ratio) * dx[:, long_stride:] + blend_ratio * dx_long
            dy[:, long_stride:] = (1.0 - blend_ratio) * dy[:, long_stride:] + blend_ratio * dy_long

    velocities = torch.stack([dx, dy], dim=-1)  # [B, T, 2]
    if is_unbatched:
        return velocities.squeeze(0)
    return velocities


class SpatioTemporalVideoAttention(nn.Module):
    """
    Spatio-Temporal Video Attention Transformer with Dense Multi-Frame Support (up to 64 frames).
    
    Processes a sequence of T frame feature embeddings [B, T, D] extracted by a visual
    backbone (e.g. SigLIP / ViT-B-16). Enriches each frame with chronological positional
    embeddings and processes cross-frame dependencies via temporal self-attention.
    Outputs a normalized video summary token capturing sequence dynamics, speed,
    multi-stride trajectories, and action direction.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        max_frames: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_frames = max_frames

        # Sinusoidal + learnable chronological frame position embeddings
        pe = _build_sinusoidal_pos_embed(max_frames, embed_dim)
        self.pos_embed = nn.Parameter(pe)

        # Learned summary query token [VIDEO_SUMMARY]
        self.summary_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Inter-frame motion velocity projection: projects ΔF_t = F_{t+1} - F_t
        self.motion_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.motion_proj.bias)
        nn.init.xavier_uniform_(self.motion_proj.weight)

        # Spatial patch centroid velocity projection: projects (dx, dy) -> embed_dim
        self.velocity_proj = nn.Linear(2, embed_dim)
        nn.init.zeros_(self.velocity_proj.bias)
        nn.init.xavier_uniform_(self.velocity_proj.weight)

        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(embed_dim)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Allows loading from checkpoints with different max_frames (e.g. 32 -> 64)."""
        pe_key = prefix + "pos_embed"
        if pe_key in state_dict:
            saved_pe = state_dict[pe_key]
            if saved_pe.shape != self.pos_embed.shape:
                new_pe = self.pos_embed.clone()
                min_len = min(saved_pe.shape[1], new_pe.shape[1])
                new_pe[:, :min_len, :] = saved_pe[:, :min_len, :]
                state_dict[pe_key] = new_pe
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        patch_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregates frame features into a single temporal video representation.
        
        Args:
            frame_features: [B, T, D] or [T, D] tensor of frame embeddings.
            patch_features: Optional [B, T, P, D] or [T, P, D] unpooled patch embeddings for spatial flow.
            
        Returns:
            video_embed: [B, D] or [D] normalized temporal embedding.
        """
        is_unbatched = frame_features.ndim == 2
        if is_unbatched:
            frame_features = frame_features.unsqueeze(0)

        # Normalize frame representations
        frame_features = frame_features / (
            frame_features.norm(dim=-1, keepdim=True) + 1e-8
        )

        B, T, D = frame_features.shape
        device = frame_features.device

        if T > self.max_frames:
            step = max(1, T // self.max_frames)
            frame_features = frame_features[:, ::step, :][:, : self.max_frames, :]
            if patch_features is not None:
                patch_features = patch_features[:, ::step, :, :][:, : self.max_frames, :, :]
            T = frame_features.shape[1]

        # Add temporal positional embeddings
        pos = self.pos_embed[:, :T, :].to(device)
        x = frame_features + pos

        # Inject inter-frame motion delta vectors: ΔF_t = F_{t+1} - F_t
        if T >= 2:
            deltas = frame_features[:, 1:, :] - frame_features[:, :-1, :]  # [B, T-1, D]
            motion_tokens = self.motion_proj(deltas)
            zero_motion = torch.zeros(B, 1, D, device=device)
            motion_features = torch.cat([zero_motion, motion_tokens], dim=1)  # [B, T, D]
            x = x + motion_features

        # Inject spatial patch centroid velocity flow if patches provided
        if patch_features is not None:
            patch_velocities = compute_patch_centroid_flow(patch_features)
            if patch_velocities.ndim == 2:
                patch_velocities = patch_velocities.unsqueeze(0)
            vel_tokens = self.velocity_proj(patch_velocities[:, :T, :].to(device))  # [B, T, D]
            x = x + vel_tokens

        # Prepend [VIDEO_SUMMARY] token
        summary = self.summary_token.expand(B, -1, -1).to(device)
        tokens = torch.cat([summary, x], dim=1)  # [B, 1 + T, D]

        # Process through temporal transformer
        transformed = self.temporal_transformer(tokens)

        # Extract output at summary token position
        out = self.norm(transformed[:, 0, :])
        normalized = out / (out.norm(dim=-1, keepdim=True) + 1e-8)

        if is_unbatched:
            return normalized.squeeze(0)
        return normalized
