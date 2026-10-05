"""
Test-Time Compute (TTC) Deliberation Tournament & Memory Foil Harvesting [AO-31].

Inspired by Test-Time Scaling literature (Snell et al. 2024, OpenAI o1 principles)
adapted for discrete candidate multimodal decision arbitration.

Rather than evaluating user candidates in isolation with a static single forward pass,
the Test-Time Deliberator scales compute at inference:
1. Dynamic Adversarial Foil Harvesting: Queries the resident 100,000-candidate memory bank
   in host DDR4 RAM to extract the hardest semantic counter-options (foils) for the scene.
2. Multi-Pass Stochastic Deliberation: Perturbs the interaction manifold across T stochastic passes
   to quantify epistemic uncertainty vs aleatoric certainty.
3. Tournament Stress-Testing: Pits user candidates against harvested counter-hypotheses.
   If the chosen candidate dominates all memory-bank foils across perturbation passes,
   certainty is mathematically certified. If an adversarial foil ties or beats it,
   a FOIL_COLLISION escalation is triggered.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from arbiter_omni.data.memory_bank import PersistentMemoryBank


class TournamentBracket(BaseModel):
    """Details of the Test-Time Deliberation Tournament against hardest memory foils."""
    user_candidates: List[str] = Field(..., description="Candidates provided by the caller.")
    user_candidate_logits: Dict[str, float] = Field(..., description="Decision logits for user candidates.")
    mined_foils: List[str] = Field(default_factory=list, description="Adversarial counter-options harvested from memory bank.")
    mined_foil_logits: Dict[str, float] = Field(default_factory=dict, description="Decision logits for mined foils.")
    tournament_winner: str = Field(..., description="Overall winner across user candidates and mined foils.")
    winner_is_user_candidate: bool = Field(..., description="True if a caller candidate won; False if a memory foil won.")
    foil_margin: float = Field(..., description="Logit difference between top user candidate and hardest mined foil.")
    foil_collision: bool = Field(..., description="True if a mined foil beat or came within margin of top user candidate.")


class TestTimeDeliberationSummary(BaseModel):
    """Complete summary of a Test-Time Compute (TTC) deliberation pass."""
    __test__ = False
    certified_stable: bool = Field(..., description="Whether decision satisfies epistemic stability and foil clearance.")
    stability_index: float = Field(..., description="Continuous stability score between 0.0 (fragile) and 1.0 (certified).")
    deliberation_passes: int = Field(..., description="Number of stochastic perturbation passes evaluated.")
    pass_winner_consistency: float = Field(..., description="Fraction of stochastic passes where top candidate won (0.0 to 1.0).")
    epistemic_variance: float = Field(..., description="Variance of probability predictions across stochastic passes.")
    calibrated_probabilities: Dict[str, float] = Field(..., description="Ensembled probabilities over user candidates.")
    tournament_bracket: Optional[TournamentBracket] = Field(default=None, description="Detailed tournament stress test outcome.")
    escalate_reason: Optional[str] = Field(default=None, description="Diagnostic reason for escalation if deliberation failed.")


class TestTimeDeliberator:
    """
    Test-Time Compute Engine for Multimodal Decision Arbitration.
    """
    __test__ = False

    def __init__(
        self,
        memory_bank: Optional[PersistentMemoryBank] = None,
        num_passes: int = 3,
        router_noise_std: float = 0.05,
        foil_k: int = 6,
        min_foil_margin: float = 0.20,
        stability_threshold: float = 0.70,
    ):
        self.memory_bank = memory_bank
        self.num_passes = max(1, int(num_passes))
        self.router_noise_std = float(router_noise_std)
        self.foil_k = max(1, int(foil_k))
        self.min_foil_margin = float(min_foil_margin)
        self.stability_threshold = float(stability_threshold)

    def deliberate(
        self,
        context_embed: torch.Tensor,
        candidate_embeds: torch.Tensor,
        candidates: Sequence[str],
        decision_head: Any,
        candidate_mask: Optional[torch.Tensor] = None,
        modality_embeds: Optional[Dict[Any, torch.Tensor]] = None,
        presence_mask: Optional[Dict[Any, Any]] = None,
        temperature: float = 1.0,
    ) -> TestTimeDeliberationSummary:
        """
        Executes Test-Time Compute (TTC) multi-pass deliberation and memory foil stress testing.

        Args:
            context_embed: [1, context_dim] fused multimodal representation.
            candidate_embeds: [1, K, candidate_dim] candidate embeddings.
            candidates: List of candidate string labels.
            decision_head: DynamicDecisionHead module.
            candidate_mask: Optional [1, K] boolean mask.
            modality_embeds: Optional dict of modality embeddings.
            presence_mask: Optional dict of modality presence flags.
            temperature: Softmax scaling temperature.

        Returns:
            TestTimeDeliberationSummary with stability certification and tournament bracket.
        """
        K = len(candidates)
        device = context_embed.device

        # ---- Step 1: Multi-Pass Stochastic Manifold Deliberation ----
        pass_probs_list = []
        winner_indices = []

        # Pass 0: Baseline unperturbed
        with torch.no_grad():
            base_logits, base_probs, _ = decision_head(
                context_embed=context_embed,
                candidate_embeds=candidate_embeds,
                candidate_mask=candidate_mask,
                modality_embeds=modality_embeds,
                presence_mask=presence_mask,
                temperature=temperature,
            )
        base_p = base_probs[0].cpu().numpy()
        pass_probs_list.append(base_p)
        top1_idx = int(np.argmax(base_p))
        winner_indices.append(top1_idx)

        # Stochastic jitter passes (T - 1)
        for _ in range(self.num_passes - 1):
            noise = torch.randn_like(context_embed) * self.router_noise_std
            perturbed_ctx = context_embed + noise
            with torch.no_grad():
                _, p_jitter, _ = decision_head(
                    context_embed=perturbed_ctx,
                    candidate_embeds=candidate_embeds,
                    candidate_mask=candidate_mask,
                    modality_embeds=modality_embeds,
                    presence_mask=presence_mask,
                    temperature=temperature,
                )
            p_j = p_jitter[0].cpu().numpy()
            pass_probs_list.append(p_j)
            winner_indices.append(int(np.argmax(p_j)))

        all_p = np.stack(pass_probs_list, axis=0)  # [T, K]
        mean_p = all_p.mean(axis=0)  # [K]
        var_p = float(all_p.var(axis=0).sum())

        # Winner consistency across passes
        primary_winner_idx = int(np.argmax(mean_p))
        primary_winner = candidates[primary_winner_idx]
        consistency = float(np.mean([1.0 if idx == primary_winner_idx else 0.0 for idx in winner_indices]))

        calibrated_probs = {cand: float(mean_p[i]) for i, cand in enumerate(candidates)}

        # ---- Step 2: Adversarial Memory Bank Foil Harvesting & Tournament ----
        bracket: Optional[TournamentBracket] = None
        foil_margin = float(base_logits[0, primary_winner_idx].item())
        foil_collision = False
        escalate_reason = None

        if self.memory_bank is not None and len(self.memory_bank) > 0:
            target_vec = candidate_embeds[0, primary_winner_idx]
            hard_foils, foil_sims, foil_mask = self.memory_bank.query_hard_foils(
                target_vec,
                k=self.foil_k,
                min_sim=0.20,
                max_sim=0.98,
            )

            valid_foil_count = int(foil_mask[0].sum().item())
            if valid_foil_count == 0:
                # If no foils strictly met the cosine boundary, harvest top available foils
                valid_foil_count = min(self.foil_k, len(self.memory_bank))

            if valid_foil_count > 0:
                active_foils = hard_foils[0:1, :valid_foil_count]  # [1, M, D]
                with torch.no_grad():
                    foil_logits = decision_head.score_foils(
                        context_embed=context_embed,
                        foil_embeds=active_foils,
                        temperature=temperature,
                    )  # [1, M]

                f_logits_np = foil_logits[0].cpu().numpy()
                max_foil_logit = float(np.max(f_logits_np))
                winner_logit = float(base_logits[0, primary_winner_idx].item())
                foil_margin = winner_logit - max_foil_logit

                foil_names = [f"Adversarial Foil {idx+1}" for idx in range(valid_foil_count)]

                user_logits_dict = {cand: float(base_logits[0, i].item()) for i, cand in enumerate(candidates)}
                mined_logits_dict = {name: float(f_logits_np[i]) for i, name in enumerate(foil_names)}

                if max_foil_logit >= winner_logit:
                    foil_collision = True
                    escalate_reason = "FOIL_COLLISION"
                    tourn_winner = foil_names[int(np.argmax(f_logits_np))]
                    winner_is_user = False
                else:
                    tourn_winner = primary_winner
                    winner_is_user = True
                    if foil_margin < self.min_foil_margin:
                        foil_collision = True
                        escalate_reason = "HIGH_FOIL_AMBIGUITY"

                bracket = TournamentBracket(
                    user_candidates=list(candidates),
                    user_candidate_logits=user_logits_dict,
                    mined_foils=foil_names,
                    mined_foil_logits=mined_logits_dict,
                    tournament_winner=tourn_winner,
                    winner_is_user_candidate=winner_is_user,
                    foil_margin=foil_margin,
                    foil_collision=foil_collision,
                )

        # ---- Step 3: Compute Stability Index S in [0.0, 1.0] ----
        margin_factor = 1.0
        if bracket is not None:
            if foil_collision:
                margin_factor = max(0.0, foil_margin / self.min_foil_margin)
            else:
                margin_factor = min(1.0, max(0.5, foil_margin / (self.min_foil_margin * 2.0)))

        variance_penalty = math.exp(-10.0 * var_p)
        stability_index = float(np.clip(consistency * margin_factor * variance_penalty, 0.0, 1.0))

        certified_stable = (stability_index >= self.stability_threshold) and not foil_collision

        if not certified_stable and escalate_reason is None:
            if consistency < 0.67:
                escalate_reason = "HIGH_TEST_TIME_VARIANCE"
            elif stability_index < self.stability_threshold:
                escalate_reason = "LOW_STABILITY_INDEX"

        return TestTimeDeliberationSummary(
            certified_stable=certified_stable,
            stability_index=stability_index,
            deliberation_passes=self.num_passes,
            pass_winner_consistency=consistency,
            epistemic_variance=var_p,
            calibrated_probabilities=calibrated_probs,
            tournament_bracket=bracket,
            escalate_reason=escalate_reason,
        )
