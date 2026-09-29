"""
Abstract Base Class for Multimodal Fusion Modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from arbiter_omni.types import ModalityType


class BaseMultimodalFusion(nn.Module, ABC):
    """
    Abstract Interface for Fusing Multimodal Embeddings into a Shared Decision Representation.
    
    Guarantees graceful handling of missing modalities via explicit masking.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        question_embed: torch.Tensor,
        modality_embeds: Dict[ModalityType, torch.Tensor],
        presence_mask: Dict[ModalityType, torch.Tensor],
        image_patches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Fuses present modality embeddings conditioned on question representation.
        
        Args:
            question_embed: [B, D_q] embedding of the decision question.
            modality_embeds: Dictionary mapping ModalityType -> [B, D_m] embedding tensor.
            presence_mask: Dictionary mapping ModalityType -> [B] boolean tensor (True if present, False if missing).
            
        Returns:
            fused_context: [B, hidden_dim] unified representation of multimodal state.
        """
        pass
