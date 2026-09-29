"""
Gated Multimodal Unit (GMU) Fusion.
Provides a lightweight feedforward gating alternative to self-attention.
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from arbiter_omni.fusion.base import BaseMultimodalFusion
from arbiter_omni.types import ModalityType


class GatedMultimodalFusion(BaseMultimodalFusion):
    """
    Gated Multimodal Unit (GMU) Fusion.
    Computes modality-specific gating weights conditioned on question and state.
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
        dropout: float = 0.1,
    ):
        super().__init__(hidden_dim=hidden_dim)

        self.projections = nn.ModuleDict()
        self.gates = nn.ModuleDict()

        for mod_name, dim in modality_dims.items():
            self.projections[mod_name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            # Gate combines modality features + question representation
            self.gates[mod_name] = nn.Sequential(
                nn.Linear(dim + modality_dims.get("question", dim), 1),
                nn.Sigmoid(),
            )

        self.q_proj = nn.Sequential(
            nn.Linear(modality_dims["question"], hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        question_embed: torch.Tensor,
        modality_embeds: Dict[ModalityType, torch.Tensor],
        presence_mask: Dict[ModalityType, torch.Tensor],
        image_patches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = question_embed.shape[0]

        device = question_embed.device

        # Start with projected question representation
        accumulated = self.q_proj(question_embed)
        gate_sum = torch.ones((batch_size, 1), device=device)

        for mod in self.MODALITY_ORDER:
            mod_embed = modality_embeds.get(mod)
            is_present = presence_mask.get(
                mod, torch.zeros(batch_size, dtype=torch.bool, device=device)
            )

            if mod_embed is None:
                continue

            # Compute gate
            gate_input = torch.cat([mod_embed, question_embed], dim=-1)
            raw_gate = self.gates[mod.value](gate_input)  # [B, 1]

            # Zero out gate for missing modalities
            masked_gate = raw_gate * is_present.unsqueeze(1).float()

            proj = self.projections[mod.value](mod_embed)
            accumulated = accumulated + (proj * masked_gate)
            gate_sum = gate_sum + masked_gate

        fused = accumulated / (gate_sum + 1e-8)
        return self.out_proj(fused)
