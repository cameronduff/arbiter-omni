"""
Unit tests for Conformal Prediction Sets and System 2 Escalation Gate [AO-08].
"""

import math
import numpy as np
import pytest
import torch

from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.calibration.conformal import ConformalCalibrator, System2EscalationGate
from arbiter_omni.types import DecisionResult, MultimodalSample


def test_conformal_calibrator_lac_math():
    """Validates Least Ambiguous Classifier (LAC) threshold calibration and set prediction."""
    # 5 calibration samples with 3 choices each
    probs = np.array([
        [0.8, 0.1, 0.1],  # target 0 -> p = 0.8 -> score = 0.2
        [0.3, 0.6, 0.1],  # target 1 -> p = 0.6 -> score = 0.4
        [0.1, 0.2, 0.7],  # target 2 -> p = 0.7 -> score = 0.3
        [0.5, 0.4, 0.1],  # target 0 -> p = 0.5 -> score = 0.5
        [0.9, 0.05, 0.05], # target 0 -> p = 0.9 -> score = 0.1
    ])
    targets = np.array([0, 1, 2, 0, 0])
    # Scores: [0.1, 0.2, 0.3, 0.4, 0.5]
    # For n=5, alpha=0.2 (80% coverage): ceil((5 + 1) * 0.8) / 5 = ceil(4.8) / 5 = 5/5 = 1.0
    # quantile 1.0 is max score = 0.5

    calibrator = ConformalCalibrator(alpha=0.2, method="lac")
    q_hat = calibrator.calibrate(probs, targets)
    assert pytest.approx(q_hat, abs=1e-4) == 0.5

    # Prediction test:
    # tau = 1 - q_hat = 1 - 0.5 = 0.5
    # Any candidate with p >= 0.5 is included
    test_prob = {"option_a": 0.6, "option_b": 0.3, "option_c": 0.1}
    c_set = calibrator.predict_set(test_prob)
    assert c_set == ["option_a"]

    # When ambiguous:
    test_ambiguous = {"option_a": 0.50, "option_b": 0.50, "option_c": 0.0}
    c_set_ambiguous = calibrator.predict_set(test_ambiguous)
    assert "option_a" in c_set_ambiguous and "option_b" in c_set_ambiguous


def test_conformal_coverage_guarantee_empirical():
    """Validates distribution-free coverage guarantee on a held-out test split."""
    np.random.seed(42)
    N_calib = 300
    N_test = 200
    K = 4

    # Generate synthetic Dirichlet-distributed probability profiles
    def generate_data(n):
        probs = []
        targets = []
        for _ in range(n):
            true_class = np.random.randint(0, K)
            # Bias probabilities towards true class
            alpha_dir = [0.8] * K
            alpha_dir[true_class] = 3.5
            p = np.random.dirichlet(alpha_dir)
            probs.append(p)
            targets.append(true_class)
        return np.array(probs), np.array(targets)

    calib_probs, calib_targets = generate_data(N_calib)
    test_probs, test_targets = generate_data(N_test)

    # 90% coverage guarantee (alpha = 0.10)
    calibrator = ConformalCalibrator(alpha=0.10, method="lac")
    calibrator.calibrate(calib_probs, calib_targets)

    eval_results = calibrator.evaluate_coverage(test_probs, test_targets)
    assert eval_results["coverage_rate"] >= 0.88  # Within statistical noise of 90%
    assert 1.0 <= eval_results["mean_set_size"] <= K
    assert eval_results["target_coverage"] == 0.90


def test_conformal_aps_method():
    """Validates Adaptive Prediction Sets (APS) cumulative sorting."""
    probs = [
        [0.7, 0.2, 0.1],  # target 0 -> score = 0.7
        [0.4, 0.5, 0.1],  # target 1 -> sorted: [0.5, 0.4, 0.1], rank 0 -> score = 0.5
        [0.1, 0.3, 0.6],  # target 2 -> sorted: [0.6, 0.3, 0.1], rank 0 -> score = 0.6
    ]
    targets = [0, 1, 2]
    calibrator = ConformalCalibrator(alpha=0.1, method="aps")
    q_hat = calibrator.calibrate(probs, targets)
    assert 0.5 <= q_hat <= 1.0

    c_set = calibrator.predict_set([0.8, 0.15, 0.05], candidates=["A", "B", "C"])
    assert "A" in c_set


def test_system2_escalation_gate():
    """Validates automated System 1-to-System 2 escalation conditions."""
    gate = System2EscalationGate(
        entropy_threshold=0.80,
        max_conformal_size=1,
        min_confidence=0.60,
    )

    # Case 1: Confident, sharp System 1 decision -> No escalation
    esc, reason = gate.evaluate(
        confidence=0.92,
        entropy=0.25,
        conformal_set=["proceed at nominal velocity"],
    )
    assert esc is False
    assert reason is None

    # Case 2: High Entropy -> Escalate
    esc, reason = gate.evaluate(
        confidence=0.70,
        entropy=0.85,
        conformal_set=["proceed at nominal velocity"],
    )
    assert esc is True
    assert "HIGH_ENTROPY" in reason

    # Case 3: Ambiguous Conformal Set (multiple candidates required for 1-alpha) -> Escalate
    esc, reason = gate.evaluate(
        confidence=0.65,
        entropy=0.70,
        conformal_set=["proceed at nominal velocity", "proceed with caution"],
    )
    assert esc is True
    assert "AMBIGUOUS_CONFORMAL_SET" in reason

    # Case 4: Low Confidence -> Escalate
    esc, reason = gate.evaluate(
        confidence=0.45,
        entropy=0.75,
        conformal_set=["proceed at nominal velocity"],
    )
    assert esc is True
    assert "LOW_CONFIDENCE" in reason


def test_engine_conformal_and_escalation_integration():
    """Validates ArbiterOmniEngine integration with conformal sets and escalation gate."""
    engine = ArbiterOmniEngine.create(encoder_type="mock")

    # Run uncalibrated decision
    res = engine.decide(
        question="What velocity should be commanded?",
        candidates=["proceed at nominal velocity", "execute emergency stop"],
    )
    assert isinstance(res, DecisionResult)
    assert len(res.conformal_set) >= 1
    assert isinstance(res.escalate_system2, bool)

    # Test calibrate_conformal on engine
    calib_samples = [
        MultimodalSample(
            question=f"Decision sample {i}",
            candidates=["option_a", "option_b"],
            target_idx=0 if i % 2 == 0 else 1,
        )
        for i in range(10)
    ]
    q_hat = engine.calibrate_conformal(calib_samples, alpha=0.10)
    assert 0.0 < q_hat <= 1.0
    assert engine.conformal_calibrator.is_calibrated

    # Verify decision after calibration
    res_calibrated = engine.decide(
        question="Select option:",
        candidates=["option_a", "option_b"],
    )
    assert len(res_calibrated.conformal_set) >= 1
    # Check escalation gate configuration
    engine.configure_escalation_gate(entropy_threshold=0.01)  # Force escalation
    res_forced_esc = engine.decide(
        question="Select option:",
        candidates=["option_a", "option_b"],
    )
    assert res_forced_esc.escalate_system2 is True
    assert "ESCALATE_TO_SYSTEM_2" in res_forced_esc.escalation_reason

def test_adaptive_conformal_risk_control_contraction_on_certified_stability():
    """Verify Adaptive CRC contracts prediction sets to singleton on high test-time stability [AO-32]."""
    probs = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.9, 0.1],
    ])
    targets = np.array([0, 0, 0, 0])
    calibrator = ConformalCalibrator(alpha=0.10, method="lac")
    calibrator.calibrate(probs, targets)

    test_probs = {"cand_1": 0.65, "cand_2": 0.35}
    # Unconditioned calibration might include both or be borderline
    # With certified test-time stability (e.g. 0.95), set tightens to singleton
    c_set_stable = calibrator.predict_set(test_probs, stability_index=0.95)
    assert c_set_stable == ["cand_1"]


def test_adaptive_conformal_risk_control_expansion_on_fragility():
    """Verify Adaptive CRC expands prediction set to cover alternative candidates on low stability [AO-32]."""
    probs = np.array([
        [0.95, 0.05],
        [0.90, 0.10],
        [0.85, 0.15],
        [0.99, 0.01],
    ])
    targets = np.array([0, 0, 0, 0])
    calibrator = ConformalCalibrator(alpha=0.10, method="lac")
    calibrator.calibrate(probs, targets)

    test_probs = {"cand_1": 0.55, "cand_2": 0.45}
    # Under epistemic fragility / foil collision risk (stability_index = 0.15), set expands to guarantee coverage
    c_set_fragile = calibrator.predict_set(test_probs, stability_index=0.15)
    assert "cand_1" in c_set_fragile
    assert "cand_2" in c_set_fragile
