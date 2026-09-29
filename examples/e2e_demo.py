"""
End-to-End Demonstration: Multimodal Decision Model inspired by Jev.

Demonstrates:
1. Building ArbiterOmni with frozen multimodal encoders and trainable fusion.
2. Generating a synthetic cross-modal decision dataset with missing modalities.
3. Training the lightweight fusion and dynamic candidate decision head.
4. Performing live inference across different modality combinations and candidate sets.
"""

import logging
import os
import sys
import numpy as np
from PIL import Image

# Ensure local src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from arbiter_omni import (
    ArbiterOmniEngine,
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    TrainingConfig,
    create_synthetic_audio,
    create_synthetic_image,
    generate_synthetic_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_demo")


def run_e2e_demo():
    print("=" * 75)
    print("  ARBITER-OMNI: PROTOTYPE MULTIMODAL SYSTEM 1 DECISION ENGINE (JEV-STYLE)")
    print("=" * 75)

    # 1. Synthesize Dataset
    print("\n[Step 1] Generating Synthetic Multimodal Dataset...")
    samples = generate_synthetic_dataset(num_samples=120, missing_modality_prob=0.35, seed=42)
    train_samples = samples[:100]
    val_samples = samples[100:]

    train_ds = MultimodalDecisionDataset(train_samples)
    val_ds = MultimodalDecisionDataset(val_samples)
    print(f"Generated {len(train_ds)} training samples and {len(val_ds)} validation samples.")
    print("Sample Modalities breakdown:")
    for i in range(3):
        active = [m.value for m in train_samples[i].present_modalities()]
        print(f"  Sample #{i}: Active modalities: {active} | Question: {train_samples[i].question[:45]}...")

    # 2. Build Model with Frozen Encoders & Trainable Fusion
    print("\n[Step 2] Initializing ArbiterOmni Neural Architecture...")
    encoder = MockMultimodalEncoder(embed_dim=128)
    model = ArbiterOmniModel(
        encoder=encoder,
        hidden_dim=128,
        scoring_dim=128,
    )

    frozen_params = sum(p.numel() for p in model.encoder.parameters())
    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"  Frozen Encoder Parameters:    {frozen_params:,} (requires_grad = False)")
    print(f"  Trainable Fusion/Head Params:  {trainable_params:,} (lightweight optimization)")

    # 3. Train Fusion & Decision Head
    print("\n[Step 3] Training Lightweight Fusion & Dynamic Decision Network...")
    config = TrainingConfig(
        learning_rate=3e-3,
        weight_decay=1e-4,
        batch_size=16,
        num_epochs=6,
        label_smoothing=0.05,
    )
    trainer = ArbiterOmniTrainer(model=model, config=config)
    history = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    final_train_acc = history["accuracy"][-1] * 100
    final_val_acc = history["val_accuracy"][-1] * 100
    print(f"\nTraining Complete: Train Acc = {final_train_acc:.1f}% | Val Acc = {final_val_acc:.1f}%")

    # 4. End-to-End Inference API
    print("\n[Step 4] Running Live Inference via ArbiterOmniEngine API...")
    engine = ArbiterOmniEngine(model=model)

    # Test Case A: Full Quad-Modal Input (Robotics Emergency)
    print("\n--- Test Case A: Full Quad-Modal Input (Text + Image + Video + Audio) ---")
    question_a = "What is the immediate action for the autonomous mobile robot?"
    candidates_a = [
        "halt immediately and apply brakes",
        "proceed forward at nominal speed",
        "steer left around obstacle",
        "request remote operator assistance",
    ]
    img_a = create_synthetic_image(color=(220, 20, 20), pattern="cross")
    vid_a = [img_a, img_a]
    aud_a = create_synthetic_audio(freq=1200.0)
    text_a = "CRITICAL: LiDAR detected static barrier at 0.4 meters."

    result_a = engine.decide(
        question=question_a,
        candidates=candidates_a,
        text=text_a,
        image=img_a,
        video=vid_a,
        audio=aud_a,
    )
    print(f"Query:              '{result_a.question}'")
    print(f"Active Modalities:  {result_a.active_modalities}")
    print(f"Winning Decision:   --> '{result_a.winner}'")
    print(f"Confidence:         {result_a.confidence * 100:.2f}%")
    print(f"Decision Entropy:   {result_a.entropy:.3f} nats")
    print("Probability Distribution:")
    for cand, prob in result_a.probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"  - {cand:<38} : {prob * 100:6.2f}% {bar}")

    # Test Case B: Graceful Missing Modality Handling (Audio Only - No Image, Video, or Text)
    print("\n--- Test Case B: Missing Modality (Audio Only - Bearing Failure Screech) ---")
    question_b = "What is the appropriate triage command for the assembly line?"
    candidates_b = [
        "trigger emergency facility shutdown",
        "maintain nominal line operation",
        "reroute component to quality inspection",
        "dispatch maintenance technician",
    ]
    aud_b = create_synthetic_audio(freq=1800.0)  # High screech

    result_b = engine.decide(
        question=question_b,
        candidates=candidates_b,
        audio=aud_b,  # Only audio supplied
    )
    print(f"Query:              '{result_b.question}'")
    print(f"Active Modalities:  {result_b.active_modalities} (Text, Image, Video missing)")
    print(f"Winning Decision:   --> '{result_b.winner}'")
    print(f"Confidence:         {result_b.confidence * 100:.2f}%")
    print("Probability Distribution:")
    for cand, prob in result_b.probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"  - {cand:<38} : {prob * 100:6.2f}% {bar}")

    # Test Case C: Dynamic Candidate Count (Binary Jev-Style Verification)
    print("\n--- Test Case C: Dynamic Candidate Count (2 Candidates - Jev Boolean Noul) ---")
    result_c = engine.decide(
        question="Is the assembly line operating safely within normal parameters?",
        candidates=["Normal Safe Operation", "Hazardous Anomaly Detected"],
        audio=create_synthetic_audio(freq=150.0),  # Low quiet hum
        text="SYSTEM OK: Vibration telemetry within +/- 2% baseline tolerances.",
    )
    print(f"Winning Decision:   --> '{result_c.winner}' ({result_c.confidence * 100:.2f}%)")
    print(f"Jev Boolean Noul:   {result_c.boolean_noul}")

    print("\n" + "=" * 75)
    print("  DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    run_e2e_demo()
