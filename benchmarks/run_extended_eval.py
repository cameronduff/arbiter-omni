"""
Extended Real-World Evaluation Suite for ArbiterOmni.
Streams multimodal benchmarks (ScienceQA & SEED-Bench-2),
evaluates zero-shot / trained decision performance,
and computes 10-bin ECE calibration curves and categorical breakdowns.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from arbiter_omni import (
    ArbiterOmniEngine,
    ArbiterOmniModel,
    MockMultimodalEncoder,
    OpenCLIPMultimodalEncoder,
    get_device_telemetry,
    resolve_device,
)
from arbiter_omni.data.scienceqa import load_scienceqa_dataset, create_mock_scienceqa_samples
from arbiter_omni.data.seedbench import load_seedbench_dataset, create_mock_seedbench_samples
from arbiter_omni.types import MultimodalSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and per-bin reliability diagram statistics.
    """
    accuracies = (predictions == targets).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats: List[Dict[str, Any]] = []

    ece = 0.0
    mce = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            gap = abs(bin_conf - bin_acc)
            weight = count / total_samples
            ece += weight * gap
            mce = max(mce, gap)
        else:
            bin_acc = 0.0
            bin_conf = (lower + upper) / 2.0
            gap = 0.0

        bin_stats.append({
            "bin_index": i + 1,
            "range": f"[{lower:.1f}, {upper:.1f}]",
            "count": count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
            "gap": gap,
        })

    return float(ece), float(mce), bin_stats


def evaluate_samples(
    engine: ArbiterOmniEngine,
    samples: List[MultimodalSample],
    dataset_name: str = "Benchmark",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Evaluates a collection of MultimodalSample items with ground truth labels.
    """
    if not samples:
        return {"error": "No samples to evaluate"}

    confidences: List[float] = []
    predictions: List[int] = []
    targets: List[int] = []
    latencies_ms: List[float] = []
    entropies: List[float] = []
    category_data: Dict[str, Dict[str, Any]] = {}

    for i, sample in enumerate(samples):
        if sample.target_idx is None:
            continue

        t0 = time.perf_counter()
        result = engine.decide(
            question=sample.question,
            candidates=sample.candidates,
            text=sample.text,
            image=sample.image,
            video=sample.video,
            audio=sample.audio,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        confidences.append(result.confidence)
        predictions.append(result.winner_index)
        targets.append(sample.target_idx)
        latencies_ms.append(elapsed_ms)
        entropies.append(result.entropy)

        # Category/subject breakdown
        category = (
            sample.metadata.get("subject")
            or sample.metadata.get("dimension")
            or sample.metadata.get("category")
            or "General"
        )
        if category not in category_data:
            category_data[category] = {"correct": 0, "total": 0, "confidences": [], "entropies": []}

        is_correct = int(result.winner_index == sample.target_idx)
        category_data[category]["correct"] += is_correct
        category_data[category]["total"] += 1
        category_data[category]["confidences"].append(result.confidence)
        category_data[category]["entropies"].append(result.entropy)

    if not targets:
        return {"error": "No labeled samples found"}

    conf_arr = np.array(confidences)
    pred_arr = np.array(predictions)
    targ_arr = np.array(targets)
    lat_arr = np.array(latencies_ms)
    ent_arr = np.array(entropies)

    overall_acc = float(np.mean(pred_arr == targ_arr))
    ece, mce, bin_stats = compute_calibration_metrics(conf_arr, pred_arr, targ_arr, n_bins=n_bins)

    # Category summaries
    categories_summary: Dict[str, Dict[str, float]] = {}
    for cat, data in category_data.items():
        tot = data["total"]
        categories_summary[cat] = {
            "count": tot,
            "accuracy": float(data["correct"] / tot) if tot > 0 else 0.0,
            "mean_confidence": float(np.mean(data["confidences"])) if tot > 0 else 0.0,
            "mean_entropy": float(np.mean(data["entropies"])) if tot > 0 else 0.0,
        }

    return {
        "dataset_name": dataset_name,
        "sample_count": len(targets),
        "top1_accuracy": overall_acc,
        "ece_10bin": ece,
        "mce_10bin": mce,
        "mean_confidence": float(np.mean(conf_arr)),
        "mean_entropy_nats": float(np.mean(ent_arr)),
        "std_entropy_nats": float(np.std(ent_arr)),
        "latency_mean_ms": float(np.mean(lat_arr)),
        "latency_p50_ms": float(np.percentile(lat_arr, 50)),
        "latency_p95_ms": float(np.percentile(lat_arr, 95)),
        "throughput_fps": float(1000.0 / np.mean(lat_arr)) if np.mean(lat_arr) > 0 else 0.0,
        "calibration_bins": bin_stats,
        "category_breakdown": categories_summary,
    }


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Prints a structured ASCII report of evaluation results."""
    d_name = results["dataset_name"]
    print("\n" + "=" * 82)
    print(f"📊 EXTENDED EVALUATION REPORT: {d_name.upper()}")
    print("=" * 82)
    print(f" Total Evaluated Samples : {results['sample_count']}")
    print(f" Top-1 Decision Accuracy : {results['top1_accuracy']*100:.2f}%")
    print(f" Mean Confidence Score   : {results['mean_confidence']*100:.2f}%")
    print(f" Expected Calib. Error   : {results['ece_10bin']:.4f} ({results['ece_10bin']*100:.2f}%)")
    print(f" Max Calibration Error   : {results['mce_10bin']:.4f} ({results['mce_10bin']*100:.2f}%)")
    print(f" Mean Decision Entropy   : {results['mean_entropy_nats']:.3f} nats (± {results['std_entropy_nats']:.3f})")
    print(f" Single-Pass Latency     : p50: {results['latency_p50_ms']:.2f} ms | p95: {results['latency_p95_ms']:.2f} ms")
    print(f" Decision Throughput     : {results['throughput_fps']:.1f} decisions/sec")

    print("\n--- 10-Bin Reliability Calibration Diagram ---")
    print(f"{'Bin':<5} | {'Range':<12} | {'Count':<6} | {'Accuracy':<10} | {'Confidence':<12} | {'Gap (|Acc-Conf|)':<16}")
    print("-" * 75)
    for b in results["calibration_bins"]:
        acc_str = f"{b['accuracy']*100:.1f}%" if b['count'] > 0 else "-"
        conf_str = f"{b['confidence']*100:.1f}%" if b['count'] > 0 else "-"
        gap_str = f"{b['gap']*100:.2f}%" if b['count'] > 0 else "-"
        print(f"{b['bin_index']:<5} | {b['range']:<12} | {b['count']:<6} | {acc_str:<10} | {conf_str:<12} | {gap_str:<16}")

    print("\n--- Categorical / Sub-domain Breakdown ---")
    print(f"{'Category / Dimension':<35} | {'Count':<6} | {'Top-1 Acc':<10} | {'Confidence':<10} | {'Entropy':<10}")
    print("-" * 80)
    for cat, data in results["category_breakdown"].items():
        print(
            f"{cat[:34]:<35} | {data['count']:<6} | {data['accuracy']*100:6.1f}%   | "
            f"{data['mean_confidence']*100:6.1f}%   | {data['mean_entropy']:6.3f} nats"
        )
    print("=" * 82)


def run_extended_eval(
    scienceqa_samples: int = 100,
    seedbench_samples: int = 100,
    use_mock: bool = False,
    checkpoint: Optional[str] = "checkpoints/arbiter_omni_v1.pt",
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs extended real-world evaluation across ScienceQA and SEED-Bench-2."""
    device = resolve_device()
    telemetry = get_device_telemetry(device)
    logger.info(f"Initializing ArbiterOmniEngine on {telemetry['device']} ({telemetry['gpu_name']})...")

    if not use_mock and checkpoint and os.path.exists(checkpoint):
        try:
            engine = ArbiterOmniEngine.from_pretrained(checkpoint, encoder_type="openclip", device=str(device))
            logger.info(f"Loaded pretrained production checkpoint from {checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}); initializing baseline model.")
            encoder = OpenCLIPMultimodalEncoder(device=str(device))
            model = ArbiterOmniModel(encoder=encoder).to(device)
            engine = ArbiterOmniEngine(model=model, device=str(device))
    elif use_mock:
        encoder = MockMultimodalEncoder()
        model = ArbiterOmniModel(encoder=encoder).to(device)
        engine = ArbiterOmniEngine(model=model, device=str(device))
    else:
        try:
            encoder = OpenCLIPMultimodalEncoder(device=str(device))
            model = ArbiterOmniModel(encoder=encoder).to(device)
        except Exception as e:
            logger.warning(f"Could not load OpenCLIP encoder ({e}); using MockMultimodalEncoder.")
            encoder = MockMultimodalEncoder()
            model = ArbiterOmniModel(encoder=encoder).to(device)
        engine = ArbiterOmniEngine(model=model, device=str(device))

    # 1. ScienceQA
    logger.info(f"Preparing ScienceQA ({scienceqa_samples} samples)...")
    if use_mock:
        sqa_data = create_mock_scienceqa_samples(num_samples=scienceqa_samples)
    else:
        sqa_data = load_scienceqa_dataset(
            split="validation",
            max_samples=scienceqa_samples,
            only_multimodal=True,
            streaming=True,
            use_mock_fallback=True,
        )

    sqa_results = evaluate_samples(engine, sqa_data, dataset_name="ScienceQA (derek-thomas/ScienceQA)")
    print_evaluation_report(sqa_results)

    # 2. SEED-Bench-2
    logger.info(f"Preparing SEED-Bench-2 ({seedbench_samples} samples)...")
    if use_mock:
        seed_data = create_mock_seedbench_samples(num_samples=seedbench_samples)
    else:
        seed_data = load_seedbench_dataset(
            max_samples=seedbench_samples,
            use_mock_fallback=True,
        )

    seed_results = evaluate_samples(engine, seed_data, dataset_name="SEED-Bench-2 (Visual & Temporal)")
    print_evaluation_report(seed_results)

    combined_results = {
        "telemetry": telemetry,
        "scienceqa": sqa_results,
        "seedbench": seed_results,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(combined_results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_json}")

    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArbiterOmni Extended Benchmark Evaluation")
    parser.add_argument("--scienceqa-samples", type=int, default=100, help="Number of ScienceQA samples")
    parser.add_argument("--seedbench-samples", type=int, default=100, help="Number of SEED-Bench samples")
    parser.add_argument("--use-mock", action="store_true", help="Force synthetic mock samples")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/arbiter_omni_v1.pt", help="Path to pretrained model checkpoint")
    parser.add_argument("--output-json", type=str, default="benchmarks/extended_eval_results.json", help="Path to write JSON results")

    args = parser.parse_args()
    try:
        run_extended_eval(
            scienceqa_samples=args.scienceqa_samples,
            seedbench_samples=args.seedbench_samples,
            use_mock=args.use_mock,
            checkpoint=args.checkpoint,
            output_json=args.output_json,
        )
    finally:
        # Exit cleanly to kill any hanging HuggingFace / aiohttp thread pools
        os._exit(0)
