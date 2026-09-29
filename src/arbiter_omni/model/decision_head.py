"""
Dynamic Decision Scoring Head inspired by Jev System 1 Decision Architecture.
Scores arbitrary runtime candidate decision sets and produces calibrated probability distributions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDecisionHead(nn.Module):
    """
    Dynamic Candidate Decision Head.
    
    Accepts the fused multimodal context vector and a variable number of candidate
    decision embeddings. Projects both into a shared semantic interaction manifold,
    evaluating bilinear and non-linear compatibility scores to yield a calibrated
    softmax probability distribution.
    """

    def __init__(
        self,
        context_dim: int = 256,
        candidate_dim: int = 512,
        scoring_dim: int = 256,
        init_temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.candidate_dim = candidate_dim
        self.scoring_dim = scoring_dim

        # Context and Candidate semantic projection networks
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, scoring_dim),
            nn.LayerNorm(scoring_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scoring_dim, scoring_dim),
        )

        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_dim, scoring_dim),
            nn.LayerNorm(scoring_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(scoring_dim, scoring_dim),
        )

        # Non-linear cross-interaction MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(scoring_dim * 2, scoring_dim),
            nn.GELU(),
            nn.Linear(scoring_dim, 1),
        )

        # Learnable log-temperature for calibrated probabilities
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temperature), dtype=torch.float32))

        # Auxiliary Jev System 1 Primitives: Boolean/Noul Certainty and Continuous Score
        self.boolean_head = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.score_head = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    @property
    def temperature(self) -> float:
        return float(torch.clamp(self.log_temp.exp(), min=0.01, max=100.0).item())

    def forward(
        self,
        context_embed: torch.Tensor,
        candidate_embeds: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates decision logits and probability distributions over dynamic candidates.
        
        Args:
            context_embed: [B, context_dim] Fused multimodal state.
            candidate_embeds: [B, K, candidate_dim] Embeddings of K candidate decisions per sample.
            candidate_mask: Optional [B, K] boolean tensor (True for valid candidates, False for padded).
            
        Returns:
            logits: [B, K] Unnormalized decision logits.
            probs: [B, K] Calibrated softmax probability distribution over candidates.
            entropy: [B] Shannon entropy across candidate probabilities.
        """
        B, K, _ = candidate_embeds.shape

        # Project to scoring manifold
        ctx_proj = self.context_proj(context_embed)  # [B, scoring_dim]
        cnd_proj = self.candidate_proj(candidate_embeds)  # [B, K, scoring_dim]

        # Bilinear dot-product interaction
        ctx_expanded = ctx_proj.unsqueeze(1).expand(-1, K, -1)  # [B, K, scoring_dim]
        dot_scores = (ctx_expanded * cnd_proj).sum(dim=-1) / math.sqrt(self.scoring_dim)  # [B, K]

        # Non-linear interaction refinement
        concat_interact = torch.cat([ctx_expanded, cnd_proj], dim=-1)  # [B, K, 2*scoring_dim]
        mlp_scores = self.interaction_mlp(concat_interact).squeeze(-1)  # [B, K]

        # Combined scaled logits
        raw_logits = dot_scores + mlp_scores
        temp = torch.clamp(self.log_temp.exp(), min=0.01, max=100.0)
        scaled_logits = raw_logits / temp

        # Apply candidate mask if padded
        if candidate_mask is not None:
            scaled_logits = scaled_logits.masked_fill(~candidate_mask, -1e9)

        probs = F.softmax(scaled_logits, dim=-1)

        # Shannon Entropy: -sum(p * log(p))
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        if candidate_mask is not None:
            masked_p_log_p = torch.where(candidate_mask, probs * log_probs, torch.zeros_like(probs))
            entropy = -masked_p_log_p.sum(dim=-1)
        else:
            entropy = -(probs * log_probs).sum(dim=-1)

        return scaled_logits, probs, entropy

    def predict_boolean_noul(self, context_embed: torch.Tensor) -> torch.Tensor:
        """Computes Jev-style binary verification certainty [0.0, 1.0]."""
        return torch.sigmoid(self.boolean_head(context_embed)).squeeze(-1)

    def predict_score(self, context_embed: torch.Tensor) -> torch.Tensor:
        """Computes continuous regression score along an ordinal/continuous scale."""
        return self.score_head(context_embed).squeeze(-1)
