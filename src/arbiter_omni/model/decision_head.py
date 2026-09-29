"""
Dynamic Decision Scoring Head inspired by Jev System 1 Decision Architecture.
Scores arbitrary runtime candidate decision sets and produces calibrated probability distributions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from arbiter_omni.types import ModalityType


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

        # Zero-shot perceptual residual alignment scale
        self.visual_scale = nn.Parameter(torch.tensor(15.0, dtype=torch.float32))

    @property
    def temperature(self) -> float:
        return float(torch.clamp(self.log_temp.exp(), min=0.01, max=100.0).item())

    @temperature.setter
    def temperature(self, val: float) -> None:
        self.set_temperature(val)

    def set_temperature(self, temperature: float) -> None:
        """Sets the calibrated scaling temperature."""
        temp_val = max(1e-3, float(temperature))
        with torch.no_grad():
            self.log_temp.copy_(torch.tensor(math.log(temp_val), dtype=torch.float32, device=self.log_temp.device))

    def forward(
        self,
        context_embed: torch.Tensor,
        candidate_embeds: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        modality_embeds: Optional[Dict[Any, torch.Tensor]] = None,
        presence_mask: Optional[Dict[Any, Any]] = None,
        temperature: Optional[float] = None,
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

        # Direct zero-shot perceptual residual alignment (preserves foundation model zero-shot mapping)
        if modality_embeds is not None and presence_mask is not None:
            # Check Image modality
            img_embed = None
            img_present = None
            for k in [ModalityType.IMAGE, "image", ModalityType.IMAGE.value]:
                if k in modality_embeds:
                    img_embed = modality_embeds[k]
                    break
            for k in [ModalityType.IMAGE, "image", ModalityType.IMAGE.value]:
                if k in presence_mask:
                    img_present = presence_mask[k]
                    break

            if img_embed is not None and img_present is not None and img_embed.shape[-1] == candidate_embeds.shape[-1]:
                # Cosine similarity in foundation encoder space
                vis_sim = (img_embed.unsqueeze(1) * candidate_embeds).sum(dim=-1)  # [B, K]
                if isinstance(img_present, torch.Tensor):
                    pres_float = img_present.float().unsqueeze(1) if img_present.ndim == 1 else img_present.float()
                else:
                    pres_float = torch.tensor(img_present, dtype=torch.float32, device=candidate_embeds.device).unsqueeze(1)
                raw_logits = raw_logits + (vis_sim * self.visual_scale * pres_float)

        if temperature is not None:
            temp = torch.tensor(max(0.01, float(temperature)), device=candidate_embeds.device)
        else:
            temp = torch.clamp(self.log_temp.exp(), min=0.01, max=100.0)
        scaled_logits = raw_logits / temp

        # Apply candidate mask if padded
        if candidate_mask is not None:
            scaled_logits = scaled_logits.masked_fill(~candidate_mask, -10000.0)

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

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        margin: float = 0.5,
        contrastive_lambda: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes total loss combining cross-entropy and contrastive margin loss:
            L_total = L_CE + lambda * max(0, gamma - (s_pos - s_hard_neg))

        Returns:
            total_loss: L_total
            ce_loss: L_CE
            margin_loss: L_margin
        """
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        if contrastive_lambda > 0.0:
            margin_loss = contrastive_margin_loss(
                logits=logits,
                targets=targets,
                candidate_mask=candidate_mask,
                margin=margin,
            )
            total_loss = ce_loss + contrastive_lambda * margin_loss
        else:
            margin_loss = torch.tensor(0.0, device=logits.device)
            total_loss = ce_loss

        return total_loss, ce_loss, margin_loss


def contrastive_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    margin: float = 0.5,
    hard_neg_indices: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes pairwise contrastive margin loss between ground truth and hard negative candidates:
        L_margin = max(0, gamma - (s_pos - s_hard_neg))

    Args:
        logits: [B, K] Decision logits across candidates.
        targets: [B] Ground-truth target indices.
        candidate_mask: Optional [B, K] boolean tensor indicating valid candidates.
        margin: Float margin gamma enforcing separation between pos and hard neg.
        hard_neg_indices: Optional [B] specific index of mined hard negative.
                          If None, dynamically mines the hardest negative in the candidate set:
                          s_hard_neg = max_{j != target, mask[j]} logits[j].
        reduction: "mean", "sum", or "none".

    Returns:
        Loss tensor according to reduction mode.
    """
    B, K = logits.shape
    device = logits.device
    batch_indices = torch.arange(B, device=device)

    # Positive candidate score: s_pos = logits[b, targets[b]]
    s_pos = logits[batch_indices, targets]  # [B]

    if hard_neg_indices is not None:
        s_hard_neg = logits[batch_indices, hard_neg_indices]
    else:
        # Exclude positive target from negative pool
        neg_logits = logits.clone()
        neg_logits[batch_indices, targets] = -1e9
        if candidate_mask is not None:
            neg_logits = neg_logits.masked_fill(~candidate_mask, -1e9)

        # Hardest negative is the maximum among valid negative candidates
        s_hard_neg, _ = neg_logits.max(dim=-1)  # [B]

    # Margin violation: max(0, margin - (s_pos - s_hard_neg))
    margin_diff = margin - (s_pos - s_hard_neg)
    loss = F.relu(margin_diff)

    # If sample has no valid negatives (e.g. only 1 candidate), zero out loss
    valid_neg_mask = s_hard_neg > -1e8
    loss = torch.where(valid_neg_mask, loss, torch.zeros_like(loss))

    if reduction == "mean":
        num_valid = valid_neg_mask.sum().float().clamp(min=1.0)
        return loss.sum() / num_valid
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

