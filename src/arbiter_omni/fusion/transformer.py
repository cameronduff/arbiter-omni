"""
Transformer-based Multimodal Fusion with Modality-Type Embeddings and Missing-Modality Masking.
"""

from __future__ import annotations

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
    ):
        super().__init__(hidden_dim=hidden_dim)

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

        # Transformer encoder layers
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
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        question_embed: torch.Tensor,
        modality_embeds: Dict[ModalityType, torch.Tensor],
        presence_mask: Dict[ModalityType, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuses modalities into a single context vector.
        
        Args:
            question_embed: [B, D_q]
            modality_embeds: Dict of ModalityType -> [B, D_m]
            presence_mask: Dict of ModalityType -> [B] boolean (True if present)
            
        Returns:
            fused_context: [B, hidden_dim]
        """
        batch_size = question_embed.shape[0]
        device = question_embed.device

        # Token 0: Decision Query
        query_token = self.decision_query.expand(batch_size, -1, -1) + self.modality_type_embed(
            torch.tensor(0, device=device)
        )

        # Token 1: Question
        q_proj = self.projections["question"](question_embed).unsqueeze(1)
        q_token = q_proj + self.modality_type_embed(torch.tensor(1, device=device))

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

        # Shape: [B, Seq_Len, hidden_dim]
        tokens = torch.cat(token_list, dim=1)
        # Shape: [B, Seq_Len]
        key_padding_mask = torch.cat(mask_list, dim=1)

        # Attention processing
        transformed = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        # Extract the decision query output token (index 0)
        fused_context = self.output_norm(transformed[:, 0, :])
        return fused_context
