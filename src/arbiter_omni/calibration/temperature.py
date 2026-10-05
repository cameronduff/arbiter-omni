"""
Calibrated Temperature Scaling and Expected Calibration Error (ECE) Optimization.
Implements Platt / Temperature Scaling (Guo et al., 2017) to optimize probability calibration
and output sharpness across dynamic multi-candidate decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arbiter_omni.types import DecisionResult, MultimodalSample


@dataclass
class CalibrationSummary:
    """Statistical evaluation metrics before and after temperature calibration."""

    optimal_temperature: float
    initial_ece: float
    calibrated_ece: float
    initial_nll: float
    calibrated_nll: float
    bin_stats: List[Dict[str, Any]] = field(default_factory=list)


def compute_calibration_metrics(
    confidences: Union[np.ndarray, torch.Tensor, Sequence[float]],
    predictions: Union[np.ndarray, torch.Tensor, Sequence[int]],
    targets: Union[np.ndarray, torch.Tensor, Sequence[int]],
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and per-bin reliability diagram statistics.
    """
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.detach().cpu().numpy()
    elif not isinstance(confidences, np.ndarray):
        confidences = np.array(confidences, dtype=np.float32)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions, dtype=np.int64)

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    elif not isinstance(targets, np.ndarray):
        targets = np.array(targets, dtype=np.int64)

    accuracies = (predictions == targets).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats: List[Dict[str, Any]] = []

    ece = 0.0
    mce = 0.0
    total_samples = len(confidences)
    if total_samples == 0:
        return 0.0, 0.0, []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            gap = abs(bin_acc - bin_conf)
            weight = bin_count / total_samples

            ece += weight * gap
            if gap > mce:
                mce = gap

            bin_stats.append({
                "bin_idx": i,
                "range": (float(lower), float(upper)),
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": gap,
            })

    return float(ece), float(mce), bin_stats


class TemperatureCalibrator:
    """
    Learned scalar temperature scaling post-processor for multi-candidate decision heads.
    Optimizes negative log-likelihood on held-out validation samples to minimize ECE
    without altering accuracy or rank-ordering of decisions.
    """

    def __init__(self, init_temperature: float = 1.0):
        self.temperature: float = float(init_temperature)

    def fit(
        self,
        logits_list: Sequence[torch.Tensor],
        targets: Sequence[int],
        lr: float = 0.05,
        max_iter: int = 50,
        n_bins: int = 10,
    ) -> CalibrationSummary:
        """
        Fits optimal scalar temperature T on a list of candidate logit tensors.

        Args:
            logits_list: Sequence of 1D tensors, each containing unnormalized logits for sample i.
            targets: Sequence of integer target candidate indices.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Maximum optimization iterations.
            n_bins: Number of probability bins for ECE calculation.

        Returns:
            CalibrationSummary with before/after calibration metrics and optimal temperature.
        """
        if len(logits_list) != len(targets):
            raise ValueError(f"Length mismatch: {len(logits_list)} logits vs {len(targets)} targets")
        if len(logits_list) == 0:
            raise ValueError("Cannot calibrate on empty dataset")

        device = logits_list[0].device
        log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, self.temperature)), device=device))
        target_tensors = [torch.tensor([t], device=device, dtype=torch.long) for t in targets]

        # Compute initial NLL and ECE
        with torch.no_grad():
            init_nll = 0.0
            init_confs, init_preds = [], []
            for l, y in zip(logits_list, targets):
                p = F.softmax(l / self.temperature, dim=-1)
                pred = int(torch.argmax(p).item())
                conf = float(p[pred].item())
                init_preds.append(pred)
                init_confs.append(conf)
                init_nll += F.cross_entropy((l / self.temperature).unsqueeze(0), torch.tensor([y], device=device)).item()
            init_nll /= len(logits_list)
            init_ece, _, _ = compute_calibration_metrics(init_confs, init_preds, targets, n_bins=n_bins)

        # Optimize scalar temperature via L-BFGS
        optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            T = torch.clamp(log_temp.exp(), min=0.01, max=50.0)
            loss = 0.0
            for l, y_t in zip(logits_list, target_tensors):
                loss = loss + F.cross_entropy((l / T).unsqueeze(0), y_t)
            loss = loss / len(logits_list)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = float(torch.clamp(log_temp.exp(), min=0.01, max=50.0).item())

        # Compute calibrated NLL and ECE
        with torch.no_grad():
            cal_nll = 0.0
            cal_confs, cal_preds = [], []
            for l, y in zip(logits_list, targets):
                p = F.softmax(l / self.temperature, dim=-1)
                pred = int(torch.argmax(p).item())
                conf = float(p[pred].item())
                cal_preds.append(pred)
                cal_confs.append(conf)
                cal_nll += F.cross_entropy((l / self.temperature).unsqueeze(0), torch.tensor([y], device=device)).item()
            cal_nll /= len(logits_list)
            cal_ece, _, bin_stats = compute_calibration_metrics(cal_confs, cal_preds, targets, n_bins=n_bins)

        return CalibrationSummary(
            optimal_temperature=self.temperature,
            initial_ece=init_ece,
            calibrated_ece=cal_ece,
            initial_nll=init_nll,
            calibrated_nll=cal_nll,
            bin_stats=bin_stats,
        )
