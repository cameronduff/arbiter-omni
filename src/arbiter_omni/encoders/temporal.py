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


def compute_patch_centroid_flow(patch_features: torch.Tensor, grid_size: int = 14) -> torch.Tensor:
    """
    Computes universal (dx, dy) spatial centroid velocity vectors across video frames.
    
    Args:
        patch_features: [B, T, P, D] or [T, P, D] unpooled patch embeddings (P = grid_size * grid_size).
        grid_size: Spatial grid dimension (default 14 for 196 patches).
        
    Returns:
        velocities: [B, T, 2] continuous (dx, dy) velocity vectors per frame, normalized [-1, 1].
    """
    is_unbatched = patch_features.ndim == 3
    if is_unbatched:
        patch_features = patch_features.unsqueeze(0)  # [1, T, P, D]

    B, T, P, _ = patch_features.shape
    device = patch_features.device

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
    patch_energy = sub_patches.norm(dim=-1)  # [B, T, P]
    patch_weights = torch.softmax(patch_energy, dim=-1)  # [B, T, P]

    # Center-of-mass coordinates per frame: [B, T]
    cx = (patch_weights * x_coords).sum(dim=-1)
    cy = (patch_weights * y_coords).sum(dim=-1)

    # Frame-to-frame velocity vectors: (dx, dy)
    dx = torch.zeros(B, T, device=device)
    dy = torch.zeros(B, T, device=device)
    if T >= 2:
        dx[:, 1:] = cx[:, 1:] - cx[:, :-1]
        dy[:, 1:] = cy[:, 1:] - cy[:, :-1]

    velocities = torch.stack([dx, dy], dim=-1)  # [B, T, 2]
    if is_unbatched:
        return velocities.squeeze(0)
    return velocities


class SpatioTemporalVideoAttention(nn.Module):
    """
    Spatio-Temporal Video Attention Transformer.
    
    Processes a sequence of T frame feature embeddings [B, T, D] extracted by a visual
    backbone (e.g. ViT-B-32). Enriches each frame with chronological positional
    embeddings and processes cross-frame dependencies via temporal self-attention.
    Outputs a normalized 512-dim video summary token capturing sequence dynamics,
    speed, and action direction.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        max_frames: int = 32,
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
