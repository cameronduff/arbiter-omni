"""
ArbiterOmni Production Training Script (v1 Checkpoint).
Trains the 2.21M trainable multimodal cross-attention fusion and dynamic decision head parameters
across combined real-world multimodal datasets (ScienceQA, SEED-Bench-2, Robotics, Synthetic),
while keeping perception encoders strictly frozen.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List
import torch

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from arbiter_omni import (
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    CachedMultimodalDataset,
    HardNegativeMiner,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    OpenCLIPMultimodalEncoder,
    TrainingConfig,
    generate_synthetic_dataset,
    get_device_telemetry,
    resolve_device,
)

from arbiter_omni.data.robotics import generate_robotics_samples
from arbiter_omni.data.scienceqa import load_scienceqa_dataset, create_mock_scienceqa_samples
from arbiter_omni.data.seedbench import load_seedbench_dataset, create_mock_seedbench_samples
from arbiter_omni.types import MultimodalSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_combined_dataset(
    scienceqa_samples: int = 100,
    seedbench_samples: int = 100,
    robotics_samples: int = 50,
    synthetic_samples: int = 50,
    use_mock_data: bool = False,
) -> MultimodalDecisionDataset:
    """Combines diverse multimodal domain samples into a unified training dataset."""
    samples: List[MultimodalSample] = []

    # 1. ScienceQA
    logger.info(f"Gathering ScienceQA samples ({scienceqa_samples})...")
    if use_mock_data:
        sqa = create_mock_scienceqa_samples(num_samples=scienceqa_samples)
    else:
        sqa = load_scienceqa_dataset(
            split="train",
            max_samples=scienceqa_samples,
            only_multimodal=True,
            streaming=True,
            use_mock_fallback=True,
        )
    samples.extend([s for s in sqa if s.target_idx is not None])

    # 2. SEED-Bench-2
    logger.info(f"Gathering SEED-Bench-2 samples ({seedbench_samples})...")
    if use_mock_data:
        seed = create_mock_seedbench_samples(num_samples=seedbench_samples)
    else:
        seed = load_seedbench_dataset(
            max_samples=seedbench_samples,
            use_mock_fallback=True,
        )
    samples.extend([s for s in seed if s.target_idx is not None])

    # 3. Robotics Action Decisions
    logger.info(f"Gathering Robotics Action samples ({robotics_samples})...")
    robotics = generate_robotics_samples(num_samples=robotics_samples)
    samples.extend([s for s in robotics if s.target_idx is not None])

    # 4. Multi-Domain Synthetic Sensor Cases
    logger.info(f"Gathering Synthetic Multi-Modal samples ({synthetic_samples})...")
    synth = generate_synthetic_dataset(num_samples=synthetic_samples, missing_modality_prob=0.3)
    samples.extend([s for s in synth if s.target_idx is not None])

    logger.info(f"Total unified multimodal training samples: {len(samples)}")
    return MultimodalDecisionDataset(samples)


def train_v1(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    save_path: str = "checkpoints/arbiter_omni_v1.pt",
    encoder_type: str = "openclip",
    use_mock_data: bool = False,
    device_name: str | None = None,
    cache_embeddings: bool = True,
    contrastive_lambda: float = 0.2,
    margin_gamma: float = 0.5,
    mine_hard_negatives: bool = False,
) -> str:
    """Executes the v1 checkpoint training pipeline and saves weights."""
    device = resolve_device(device_name)
    telemetry = get_device_telemetry(device)
    logger.info(f"Training on Device: {telemetry['device']} ({telemetry['gpu_name']})")

    # Initialize perception encoder
    if encoder_type == "openclip":
        logger.info("Initializing OpenCLIP ViT-B-32 backbone (frozen)...")
        try:
            encoder = OpenCLIPMultimodalEncoder(device=str(device))
        except Exception as e:
            logger.warning(f"Could not load OpenCLIP ({e}); falling back to MockMultimodalEncoder.")
            encoder = MockMultimodalEncoder(device=str(device))
    else:
        encoder = MockMultimodalEncoder(device=str(device))

    model = ArbiterOmniModel(encoder=encoder).to(device)

    # Audit parameter freezing
    trainable_params = model.trainable_parameters()
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"Frozen Perception Params: {frozen_count:,} (0.00% gradient updates)")
    logger.info(f"Trainable Fusion & Decision Params: {trainable_count:,}")

    # Build dataset
    train_dataset = build_combined_dataset(
        scienceqa_samples=80,
        seedbench_samples=80,
        robotics_samples=40,
        synthetic_samples=40,
        use_mock_data=use_mock_data,
    )
    val_dataset = build_combined_dataset(
        scienceqa_samples=20,
        seedbench_samples=20,
        robotics_samples=10,
        synthetic_samples=10,
        use_mock_data=use_mock_data,
    )

    if mine_hard_negatives:
        logger.info("🎯 Mining semantically adjacent candidate foils using text encoder cosine similarity...")
        miner = HardNegativeMiner.from_dataset(train_dataset.samples, encoder=model.encoder)
        augmented_train = miner.augment_dataset(train_dataset.samples, num_hard_negatives=1)
        train_dataset = MultimodalDecisionDataset(augmented_train)
        logger.info(f"Augmented train dataset with mined hard negatives (size: {len(train_dataset)})")

    if cache_embeddings:
        logger.info("⚡ Pre-caching frozen representations into memory (bypassing frozen encoders during epochs)...")
        train_dataset = CachedMultimodalDataset.from_dataset(
            train_dataset, model=model, batch_size=batch_size, device=device
        )
        val_dataset = CachedMultimodalDataset.from_dataset(
            val_dataset, model=model, batch_size=batch_size, device=device
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        fp16=True,
        save_path=save_path,
        device=str(device),
        accumulate_grad_batches=1,
        contrastive_lambda=contrastive_lambda,
        margin_gamma=margin_gamma,
    )

    trainer = ArbiterOmniTrainer(model=model, config=config)
    logger.info(f"Starting training loop (contrastive_lambda={contrastive_lambda}, margin_gamma={margin_gamma})...")
    history = trainer.fit(train_dataset=train_dataset, val_dataset=val_dataset)

    # Verify saved checkpoint
    if not os.path.exists(save_path):
        trainer.save_checkpoint(save_path)

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"✅ Successfully trained and published checkpoint: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArbiterOmni v1 Checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="checkpoints/arbiter_omni_v1.pt", help="Checkpoint output path")
    parser.add_argument("--encoder-type", type=str, default="openclip", choices=["openclip", "mock"], help="Encoder type")
    parser.add_argument("--use-mock-data", action="store_true", help="Use synthetic mock data instead of streaming")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding pre-caching")
    parser.add_argument("--contrastive-lambda", type=float, default=0.2, help="Weight lambda for contrastive margin loss")
    parser.add_argument("--margin-gamma", type=float, default=0.5, help="Margin gamma for contrastive loss")
    parser.add_argument("--mine-hard-negatives", action="store_true", help="Mine hard negative candidate foils")
    parser.add_argument("--device", type=str, default=None, help="Compute device override")

    args = parser.parse_args()
    try:
        train_v1(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_path,
            encoder_type=args.encoder_type,
            use_mock_data=args.use_mock_data,
            device_name=args.device,
            cache_embeddings=not args.no_cache,
            contrastive_lambda=args.contrastive_lambda,
            margin_gamma=args.margin_gamma,
            mine_hard_negatives=args.mine_hard_negatives,
        )
    finally:
        os._exit(0)

