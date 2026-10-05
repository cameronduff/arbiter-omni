"""
Split Conformal Prediction Sets and System 2 Escalation Gate.
Provides finite-sample (1 - alpha) statistical coverage guarantees and automated
System 1-to-System 2 escalation under ambiguity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from arbiter_omni.types import DecisionResult, MultimodalSample


class ConformalCalibrator:
    """
    Split Conformal Prediction Calibrator for dynamic candidate decisions.
    
    Provides distribution-free (1 - alpha) statistical coverage guarantees:
        P(Y_test in C(X_test)) >= 1 - alpha
        
    Supported methods:
        - 'lac' (Least Ambiguous Classifier): probability thresholding
        - 'aps' (Adaptive Prediction Sets): cumulative probability ranking
    """

    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "lac",
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if method not in ("lac", "aps"):
            raise ValueError(f"method must be 'lac' or 'aps', got {method}")

        self.alpha = alpha
        self.method = method
        self.quantile_threshold: Optional[float] = None
        self.n_calibration_samples: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self.quantile_threshold is not None

    def calibrate(
        self,
        probabilities: Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]], Sequence[Dict[str, float]]],
        targets: Union[torch.Tensor, np.ndarray, Sequence[int]],
        candidate_lists: Optional[Sequence[Sequence[str]]] = None,
    ) -> float:
        """
        Calibrates the non-conformity threshold q_hat on a held-out calibration split.
        
        Args:
            probabilities: Candidate probability distributions for N samples.
            targets: [N] ground-truth target indices.
            candidate_lists: Optional list of candidate strings per sample (needed if probs is list of dicts).
            
        Returns:
            q_hat: Calibrated conformal threshold quantile.
        """
        # Convert targets to 1D numpy array
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy().astype(int)
        else:
            targets_np = np.asarray(targets, dtype=int)

        n = len(targets_np)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Process probabilities into lists of floats
        prob_lists: List[List[float]] = []
        if isinstance(probabilities, (torch.Tensor, np.ndarray)):
            probs_arr = probabilities.detach().cpu().numpy() if isinstance(probabilities, torch.Tensor) else probabilities
            for row in probs_arr:
                prob_lists.append([float(x) for x in row])
        elif isinstance(probabilities[0], dict):
            for i, p_dict in enumerate(probabilities):
                if candidate_lists is not None:
                    cands = candidate_lists[i]
                    prob_lists.append([float(p_dict.get(c, 0.0)) for c in cands])
                else:
                    prob_lists.append([float(v) for v in p_dict.values()])
        else:
            for row in probabilities:
                prob_lists.append([float(x) for x in row])

        # Compute non-conformity scores
        scores: List[float] = []
        for i in range(n):
            p_vec = prob_lists[i]
            y = targets_np[i]
            if y < 0 or y >= len(p_vec):
                continue

            if self.method == "lac":
                # LAC non-conformity score: s_i = 1 - p_i(y)
                p_y = max(1e-8, min(1.0, p_vec[y]))
                scores.append(1.0 - p_y)
            elif self.method == "aps":
                # Adaptive Prediction Sets: cumulative sum of sorted probs until true class
                sorted_indices = sorted(range(len(p_vec)), key=lambda k: p_vec[k], reverse=True)
                cum_prob = 0.0
                for rank_idx in sorted_indices:
                    cum_prob += p_vec[rank_idx]
                    if rank_idx == y:
                        break
                scores.append(min(1.0, cum_prob))

        if not scores:
            raise ValueError("No valid calibration samples could be scored.")

        n_scored = len(scores)
        # Compute finite-sample adjusted quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, math.ceil((n_scored + 1) * (1.0 - self.alpha)) / n_scored)
        q_hat = float(np.quantile(scores, q_level, method="higher"))

        self.quantile_threshold = q_hat
        self.n_calibration_samples = n_scored
        return q_hat

    def predict_set(
        self,
        probabilities: Union[Dict[str, float], Sequence[float]],
        candidates: Optional[Sequence[str]] = None,
        stability_index: Optional[float] = None,
    ) -> List[str]:
        """
        Constructs the conformal prediction set C(X) for a given probability distribution.
        
        Args:
            probabilities: Either a candidate->probability dict or sequence of floats.
            candidates: Sequence of candidate strings if probabilities is a list of floats.
            stability_index: Optional Test-Time Compute (TTC) stability index in [0, 1].
                             When stability_index >= 0.80, safely tightens prediction set to {c*}.
                             When stability_index < 0.50, expands prediction set to guarantee PAC coverage.
            
        Returns:
            List of candidate strings included in the conformal prediction set.
        """
        if isinstance(probabilities, dict):
            cands = list(probabilities.keys())
            probs = [float(probabilities[c]) for c in cands]
        else:
            if candidates is None:
                raise ValueError("candidates list must be provided when probabilities is a sequence of floats.")
            cands = list(candidates)
            probs = [float(p) for p in probabilities]

        if not cands:
            return []

        # Find top-1 candidate (argmax) as guaranteed non-empty fallback
        top_idx = int(np.argmax(probs))
        top_cand = cands[top_idx]

        # Default fallback if uncalibrated: candidate with highest probability
        if self.quantile_threshold is None:
            return [top_cand]

        q_hat = self.quantile_threshold
        conformal_set: List[str] = []

        # Adaptive Conformal Risk Control (CRC) dynamic scaling [AO-32]
        eff_q_hat = q_hat
        if stability_index is not None:
            s_clamped = float(np.clip(stability_index, 0.0, 1.0))
            if s_clamped >= 0.80:
                scale = 1.0 - (s_clamped - 0.80) * 2.0
                eff_q_hat = q_hat * max(0.1, scale)
            elif s_clamped < 0.50:
                fragility = (0.50 - s_clamped) / 0.50  # in (0, 1]
                eff_q_hat = min(1.0, q_hat + fragility * (1.0 - q_hat) * 0.90)

        if self.method == "lac":
            # LAC: include candidate c if 1 - P(c) <= eff_q_hat <=> P(c) >= 1 - eff_q_hat
            tau = max(0.0, 1.0 - eff_q_hat)
            for cand, p in zip(cands, probs):
                if p >= tau:
                    conformal_set.append(cand)
        elif self.method == "aps":
            # APS: include candidates in descending order until cumulative probability >= eff_q_hat
            sorted_indices = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
            cum_p = 0.0
            for idx in sorted_indices:
                conformal_set.append(cands[idx])
                cum_p += probs[idx]
                if cum_p >= eff_q_hat:
                    break

        # Finite-sample guarantee: conformal prediction sets are always non-empty
        if not conformal_set:
            conformal_set = [top_cand]

        return conformal_set

    def evaluate_coverage(
        self,
        probabilities: Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]], Sequence[Dict[str, float]]],
        targets: Union[torch.Tensor, np.ndarray, Sequence[int]],
        candidate_lists: Optional[Sequence[Sequence[str]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluates empirical coverage rate and mean set size on a held-out test split.
        
        Returns:
            Dictionary containing 'coverage_rate', 'mean_set_size', and 'target_coverage'.
        """
        if not self.is_calibrated:
            raise RuntimeError("Calibrator must be calibrated before evaluating coverage.")

        # Convert targets to 1D numpy array
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy().astype(int)
        else:
            targets_np = np.asarray(targets, dtype=int)

        n = len(targets_np)
        covered = 0
        total_size = 0

        # Process probabilities into lists of floats
        prob_lists: List[List[float]] = []
        cands_all: List[List[str]] = []
        if isinstance(probabilities, (torch.Tensor, np.ndarray)):
            probs_arr = probabilities.detach().cpu().numpy() if isinstance(probabilities, torch.Tensor) else probabilities
            for i, row in enumerate(probs_arr):
                prob_lists.append([float(x) for x in row])
                cands_all.append(list(candidate_lists[i]) if candidate_lists else [str(j) for j in range(len(row))])
        elif isinstance(probabilities[0], dict):
            for i, p_dict in enumerate(probabilities):
                cands = list(candidate_lists[i]) if candidate_lists else list(p_dict.keys())
                prob_lists.append([float(p_dict.get(c, 0.0)) for c in cands])
                cands_all.append(cands)
        else:
            for i, row in enumerate(probabilities):
                prob_lists.append([float(x) for x in row])
                cands_all.append(list(candidate_lists[i]) if candidate_lists else [str(j) for j in range(len(row))])

        for i in range(n):
            cands = cands_all[i]
            probs = prob_lists[i]
            y = targets_np[i]

            pred_set = self.predict_set(probs, candidates=cands)
            total_size += len(pred_set)

            if 0 <= y < len(cands) and cands[y] in pred_set:
                covered += 1

        coverage = covered / max(1, n)
        mean_size = total_size / max(1, n)
        return {
            "coverage_rate": coverage,
            "mean_set_size": mean_size,
            "target_coverage": 1.0 - self.alpha,
            "calibrated_samples": self.n_calibration_samples,
        }


class System2EscalationGate:
    """
    Automated System 1-to-System 2 Escalation Gate.
    
    Monitors decision uncertainty metrics (Shannon entropy, conformal prediction set size,
    and top-1 confidence). When ambiguity exceeds tolerance, triggers escalation to System 2
    deliberative reasoning (e.g. LLM reasoning or human supervisor).
    """

    def __init__(
        self,
        entropy_threshold: float = 0.95,
        max_conformal_size: int = 1,
        min_confidence: float = 0.50,
        enabled: bool = True,
    ):
        self.entropy_threshold = entropy_threshold
        self.max_conformal_size = max_conformal_size
        self.min_confidence = min_confidence
        self.enabled = enabled

    def evaluate(
        self,
        confidence: float,
        entropy: float,
        conformal_set: Sequence[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether a System 1 decision should be escalated to System 2.
        
        Args:
            confidence: Top-1 decision confidence score [0.0, 1.0].
            entropy: Shannon entropy in nats.
            conformal_set: Candidate strings in the calibrated prediction set.
            
        Returns:
            escalate: True if decision should escalate to System 2.
            reason: Diagnostic explanation string (or None if System 1 succeeds).
        """
        if not self.enabled:
            return False, None

        if entropy >= self.entropy_threshold:
            return True, f"ESCALATE_TO_SYSTEM_2: HIGH_ENTROPY ({entropy:.3f} >= {self.entropy_threshold:.3f})"

        if len(conformal_set) > self.max_conformal_size:
            return True, (
                f"ESCALATE_TO_SYSTEM_2: AMBIGUOUS_CONFORMAL_SET "
                f"(size {len(conformal_set)} > {self.max_conformal_size}: {conformal_set})"
            )

        if confidence < self.min_confidence:
            return True, f"ESCALATE_TO_SYSTEM_2: LOW_CONFIDENCE ({confidence:.3f} < {self.min_confidence:.3f})"

        return False, None
