"""
Unit tests for extended real-world benchmark evaluation module (run_extended_eval).
"""

import numpy as np
import pytest
from benchmarks.run_extended_eval import (
    compute_calibration_metrics,
    evaluate_samples,
    run_extended_eval,
)
from arbiter_omni import ArbiterOmniEngine, ArbiterOmniModel, MockMultimodalEncoder
from arbiter_omni.data.scienceqa import create_mock_scienceqa_samples
from arbiter_omni.data.seedbench import create_mock_seedbench_samples


def test_compute_calibration_metrics_perfect():
    # Perfect calibration: confidence 1.0, all correct
    confidences = np.array([1.0, 1.0, 1.0, 1.0])
    predictions = np.array([0, 1, 2, 0])
    targets = np.array([0, 1, 2, 0])

    ece, mce, bin_stats = compute_calibration_metrics(confidences, predictions, targets, n_bins=10)
    assert abs(ece - 0.0) < 1e-4
    assert abs(mce - 0.0) < 1e-4
    assert len(bin_stats) == 10
    # Bin 10 should have 4 samples with accuracy 1.0 and confidence 1.0
    assert bin_stats[-1]["count"] == 4
    assert bin_stats[-1]["accuracy"] == 1.0


def test_compute_calibration_metrics_miscalibrated():
    # Miscalibrated: 100% confidence, but 0% accuracy
    confidences = np.array([0.95, 0.95, 0.95, 0.95])
    predictions = np.array([0, 0, 0, 0])
    targets = np.array([1, 1, 1, 1])

    ece, mce, bin_stats = compute_calibration_metrics(confidences, predictions, targets, n_bins=10)
    assert ece > 0.9
    assert mce > 0.9


def test_evaluate_samples_structure():
    model = ArbiterOmniModel(encoder=MockMultimodalEncoder())
    engine = ArbiterOmniEngine(model=model)
    samples = create_mock_scienceqa_samples(num_samples=6)

    res = evaluate_samples(engine, samples, dataset_name="TestScienceQA", n_bins=10)
    assert res["dataset_name"] == "TestScienceQA"
    assert res["sample_count"] == 6
    assert 0.0 <= res["top1_accuracy"] <= 1.0
    assert 0.0 <= res["ece_10bin"] <= 1.0
    assert res["latency_mean_ms"] > 0
    assert res["mean_entropy_nats"] >= 0
    assert len(res["calibration_bins"]) == 10
    assert len(res["category_breakdown"]) > 0


def test_run_extended_eval_pipeline():
    results = run_extended_eval(
        scienceqa_samples=4,
        seedbench_samples=4,
        use_mock=True,
        output_json=None,
    )
    assert "telemetry" in results
    assert "scienceqa" in results
    assert "seedbench" in results
    assert results["scienceqa"]["sample_count"] == 4
    assert results["seedbench"]["sample_count"] == 4
