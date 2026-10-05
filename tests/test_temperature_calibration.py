"""
Unit tests for Calibrated Temperature Scaling & Output Sharpness [AO-15].
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.calibration.temperature import (
    CalibrationSummary,
    TemperatureCalibrator,
    compute_calibration_metrics,
)
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.model.decision_head import DynamicDecisionHead
from arbiter_omni.types import MultimodalSample


def test_compute_calibration_metrics():
    """Validates ECE and MCE metric calculations."""
    # Case 1: Perfectly calibrated predictions
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    targets = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 9 correct, 1 wrong -> acc in bin 0.9 is 9/9=1.0, gap=0.1
    ece, mce, bin_stats = compute_calibration_metrics(confidences, predictions, targets, n_bins=5)
    assert 0.0 <= ece <= 0.2
    assert len(bin_stats) > 0

    # Case 2: Extreme overconfidence (always predicts with 0.99 confidence but only 50% accurate)
    over_conf = np.full(100, 0.99)
    over_preds = np.zeros(100, dtype=int)
    over_targets = np.array([0]*50 + [1]*50)
    ece_over, mce_over, _ = compute_calibration_metrics(over_conf, over_preds, over_targets, n_bins=10)
    assert ece_over > 0.40  # Gap is |0.50 - 0.99| = 0.49

    # Case 3: Empty inputs
    ece_empty, mce_empty, bins_empty = compute_calibration_metrics([], [], [])
    assert ece_empty == 0.0
    assert mce_empty == 0.0
    assert bins_empty == []


def test_temperature_calibrator_optimization():
    """Validates that TemperatureCalibrator minimizes NLL and adjusts temperature appropriately."""
    calibrator = TemperatureCalibrator(init_temperature=1.0)

    # Synthetic overconfident logits where target is correct only 50% of the time
    torch.manual_seed(42)
    logits_list = []
    targets = []
    for i in range(60):
        # Large logits producing ~99% softmax confidence
        y = 0 if i < 30 else 1
        logits_list.append(torch.tensor([10.0, -10.0]))
        targets.append(y)

    summary = calibrator.fit(logits_list, targets, lr=0.05, max_iter=50)

    assert isinstance(summary, CalibrationSummary)
    # To reduce overconfidence penalty on misclassified samples, temperature must increase (T > 1.0)
    assert summary.optimal_temperature > 1.0
    assert summary.calibrated_nll <= summary.initial_nll + 1e-4


def test_dynamic_decision_head_temperature_setter():
    """Validates DynamicDecisionHead temperature getter, setter, and inference override."""
    head = DynamicDecisionHead(context_dim=64, candidate_dim=64, scoring_dim=64, init_temperature=1.0)
    assert abs(head.temperature - 1.0) < 1e-4

    # Direct setter
    head.set_temperature(0.5)
    assert abs(head.temperature - 0.5) < 1e-4

    # Property setter
    head.temperature = 0.75
    assert abs(head.temperature - 0.75) < 1e-4

    # Test forward pass with dynamic temperature override
    ctx = torch.randn(2, 64)
    cnd = torch.randn(2, 4, 64)

    # Inference with low temperature (sharpened)
    _, probs_sharp, ent_sharp = head(ctx, cnd, temperature=0.2)
    # Inference with high temperature (softened)
    _, probs_soft, ent_soft = head(ctx, cnd, temperature=3.0)

    # Sharpened entropy must be strictly less than softened entropy
    assert (ent_sharp < ent_soft).all().item()
    # Max probability of sharpened distribution must be higher
    assert (probs_sharp.max(dim=-1).values >= probs_soft.max(dim=-1).values).all().item()


def test_engine_temperature_scaling_and_calibration():
    """Validates ArbiterOmniEngine calibrate_temperature and sharpness control."""
    engine = ArbiterOmniEngine.create(encoder_type="mock", hidden_dim=64, scoring_dim=64)

    # 1. Test set_temperature and decide sharpness
    engine.set_temperature(0.3)
    res_sharp = engine.decide(
        question="Which category?",
        candidates=["option A", "option B", "option C"],
    )

    res_soft = engine.decide(
        question="Which category?",
        candidates=["option A", "option B", "option C"],
        temperature=2.5,
    )

    assert res_sharp.entropy < res_soft.entropy
    assert res_sharp.confidence > res_soft.confidence

    # 2. Test calibrate_temperature on validation samples
    val_samples = [
        MultimodalSample(question="Task 1", candidates=["cat", "dog"], target_idx=0),
        MultimodalSample(question="Task 2", candidates=["cat", "dog"], target_idx=1),
        MultimodalSample(question="Task 3", candidates=["apple", "banana", "cherry"], target_idx=2),
        MultimodalSample(question="Task 4", candidates=["red", "blue"], target_idx=0),
    ]

    summary = engine.calibrate_temperature(val_samples, max_iter=20)
    assert summary.optimal_temperature > 0.0
    assert engine.temperature == summary.optimal_temperature
    assert abs(engine.model.decision_head.temperature - summary.optimal_temperature) < 1e-4
