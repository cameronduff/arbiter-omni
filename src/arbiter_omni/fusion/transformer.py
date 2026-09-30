"""
Transformer-based Multimodal Fusion with Modality-Type Embeddings and Missing-Modality Masking.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from arbiter_omni.fusion.base import BaseMultimodalFusion
from arbiter_omni.types import ModalityType


class TransformerMultimodalFusion(BaseMultimodalFusion):
    """
    Multimodal Transformer Fusion inspired by Perceiver / Cross-Attention.
    
    Each present modality is projected to a shared hidden dimension, enriched with
    a learned modality-type embedding, and attended to by a learned [DECISION_QUERY] token.
    Missing modalities are strictly masked using PyTorch key_padding_mask, preventing any
    leakage or zero-vector skew.

    AO-27: When ``use_moe=True``, the standard dense TransformerEncoder stack is replaced
    with a 4-layer Sparse Mixture-of-Experts (SparseMoEMultimodalFusion) block.
    The load-balancing auxiliary loss is exposed via ``moe_aux_loss`` and should be
    added to the training objective with a small weight (e.g. 0.01).
    """

    MODALITY_ORDER = [
        ModalityType.TEXT,
        ModalityType.IMAGE,
        ModalityType.VIDEO,
        ModalityType.AUDIO,
    ]

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        condition_query_on_question: bool = False,
        enable_spatial_cross_attention: bool = False,
        max_spatial_patches: int = 980,
        use_moe: bool = False,
        moe_num_layers: int = 4,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__(hidden_dim=hidden_dim)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.condition_query_on_question = condition_query_on_question
        self.enable_spatial_cross_attention = enable_spatial_cross_attention
        self.max_spatial_patches = max_spatial_patches
        self.use_moe = use_moe

        # Projections for each input stream to shared hidden dimension
        self.projections = nn.ModuleDict()
        for mod_name, dim in modality_dims.items():
            self.projections[mod_name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Modality type embeddings: [0: DecisionQuery, 1: Question, 2: Text, 3: Image, 4: Video, 5: Audio]
        self.modality_type_embed = nn.Embedding(6, hidden_dim)

        # Learnable [DECISION_QUERY] latent token
        self.decision_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Spatial 2D patch positional embeddings for fine-grained visual grounding
        # Supports multi-scale tiling (1 global overview + 4 quadrant tiles = up to 980 patches)
        self.max_spatial_patches = 980
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 196, hidden_dim) * 0.02)

        # Optional Question-Conditioned Cross-Attention over visual spatial patches
        if enable_spatial_cross_attention:
            self.spatial_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)

        # ---- Transformer stack: dense OR Sparse MoE [AO-27] ----
        if use_moe:
            from arbiter_omni.fusion.moe import SparseMoEMultimodalFusion
            self.moe_transformer = SparseMoEMultimodalFusion(
                hidden_dim=hidden_dim,
                num_moe_layers=moe_num_layers,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                ffn_dim=dim_feedforward,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.transformer = None
            self.output_norm = nn.Identity()  # norm is inside SparseMoEMultimodalFusion
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )
            self.moe_transformer = None
            self.output_norm = nn.LayerNorm(hidden_dim)

    @property
    def moe_aux_loss(self) -> torch.Tensor:
        """Returns the accumulated MoE load-balancing auxiliary loss from the last forward.

        Returns zero when ``use_moe=False``. Add to training loss with a small coefficient
        (recommended: 0.01) to prevent expert routing collapse.
        """
        if self.moe_transformer is not None:
            return self.moe_transformer.accumulated_aux_loss
        return torch.tensor(0.0)

    def forward(
        self,
        question_embed: torch.Tensor,
        modality_embeds: Dict[ModalityType, torch.Tensor],
        presence_mask: Dict[ModalityType, torch.Tensor],
        image_patches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuses modalities into a single context vector.
        
        Args:
            question_embed: [B, D_q]
            modality_embeds: Dict of ModalityType -> [B, D_m]
            presence_mask: Dict of ModalityType -> [B] boolean (True if present)
            image_patches: Optional [B, P, D_img] unpooled visual spatial patch tokens.
            
        Returns:
            fused_context: [B, hidden_dim]
        """
        batch_size = question_embed.shape[0]
        device = question_embed.device

        # Token 1: Question
        q_proj = self.projections["question"](question_embed).unsqueeze(1)
        q_token = q_proj + self.modality_type_embed(torch.tensor(1, device=device))

        # Token 0: Decision Query (optionally conditioned on question representation)
        query_base = self.decision_query.expand(batch_size, -1, -1)
        if self.condition_query_on_question:
            query_base = query_base + q_proj
        query_token = query_base + self.modality_type_embed(
            torch.tensor(0, device=device)
        )

        token_list = [query_token, q_token]
        mask_list = [
            torch.zeros((batch_size, 1), dtype=torch.bool, device=device),  # Query is never masked
            torch.zeros((batch_size, 1), dtype=torch.bool, device=device),  # Question is never masked
        ]

        # Tokens 2..5: Text, Image, Video, Audio
        for idx, mod in enumerate(self.MODALITY_ORDER, start=2):
            mod_embed = modality_embeds.get(mod)
            is_present = presence_mask.get(
                mod, torch.zeros(batch_size, dtype=torch.bool, device=device)
            )

            if mod_embed is None:
                mod_embed = torch.zeros(
                    (batch_size, self.projections[mod.value][0].in_features),
                    device=device,
                )

            proj = self.projections[mod.value](mod_embed).unsqueeze(1)
            token = proj + self.modality_type_embed(torch.tensor(idx, device=device))
            token_list.append(token)

            # PyTorch key_padding_mask: True indicates element should be IGNORED (masked out)
            mask_list.append((~is_present).unsqueeze(1))

        # Spatial Visual Patch Tokens (Fine-Grained Visual Grounding & Dynamic Multi-Scale Tiling)
        if image_patches is not None and image_patches.numel() > 0:
            P = min(image_patches.size(1), self.max_spatial_patches)
            valid_patches = image_patches[:, :P, :]
            # Project patches through image projection layers
            patch_proj = self.projections["image"](valid_patches)  # [B, P, hidden_dim]

            # Multi-scale positional embeddings: repeat coordinate grid across tiles
            repeats = math.ceil(P / 196) if P > 196 else 1
            patch_pos = self.spatial_pos_embed.repeat(1, repeats, 1)[:, :P, :].to(device)
            patch_type = self.modality_type_embed(torch.tensor(3, device=device))
            patch_tokens = patch_proj + patch_pos + patch_type

            img_present = presence_mask.get(
                ModalityType.IMAGE, torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
            # If image missing, mask out all spatial patches
            patch_mask = (~img_present).unsqueeze(1).expand(-1, P)

            # Optional Question-Conditioned Cross-Attention over visual patches
            if self.enable_spatial_cross_attention and hasattr(self, "spatial_cross_attn"):
                if img_present.any():
                    safe_mask = patch_mask.clone()
                    if (~img_present).any():
                        safe_mask[~img_present, 0] = False
                    q_cross, _ = self.spatial_cross_attn(
                        query=q_token,
                        key=patch_tokens,
                        value=patch_tokens,
                        key_padding_mask=safe_mask,
                    )
                    # Zero out for samples where image is missing
                    q_cross = q_cross * img_present.float().unsqueeze(1).unsqueeze(2)
                    q_token = self.cross_norm(q_token + q_cross)
                    # Update q_token in token_list
                    token_list[1] = q_token

            token_list.append(patch_tokens)
            mask_list.append(patch_mask)

        # Shape: [B, Seq_Len, hidden_dim]
        tokens = torch.cat(token_list, dim=1)
        # Shape: [B, Seq_Len]
        key_padding_mask = torch.cat(mask_list, dim=1)

        # ---- Attention processing: dense OR Sparse MoE [AO-27] ----
        if self.use_moe and self.moe_transformer is not None:
            transformed = self.moe_transformer(tokens, src_key_padding_mask=key_padding_mask)
        else:
            transformed = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
            transformed = self.output_norm(transformed)

        # Extract the decision query output token (index 0)
        fused_context = transformed[:, 0, :]
        return fused_context
