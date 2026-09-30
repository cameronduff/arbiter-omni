"""
ArbiterOmni v2 GPU Checkpoint Production Training Script.
Trains the 7.8M parameter fusion transformer and dynamic decision head
with ViT-B-16 (512-dim) backbone on GPU (CUDA / ROCm / DirectML).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from scripts.train_v1 import train_v1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_v2(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    save_path: str = "checkpoints/arbiter_omni_v2.pt",
    device: str | None = None,
    use_mock_data: bool = False,
    cache_embeddings: bool = True,
) -> str:
    """Trains the ArbiterOmni v2 checkpoint with ViT-B-16 backbone."""
    logger.info("=" * 70)
    logger.info("🚀 ARBITEROMNI v2 TRAINING: ViT-B-16 Backbone, 512-Dim Fusion Transformer")
    logger.info("=" * 70)

    return train_v1(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_path=save_path,
        encoder_type="openclip",
        model_name="ViT-B-16",
        hidden_dim=512,
        scoring_dim=512,
        num_layers=3,
        num_heads=8,
        device_name=device,
        use_mock_data=use_mock_data,
        cache_embeddings=cache_embeddings,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArbiterOmni v2 Checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="checkpoints/arbiter_omni_v2.pt", help="Checkpoint save path")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, directml)")
    parser.add_argument("--use-mock-data", action="store_true", help="Use lightweight mock data for testing")
    parser.add_argument("--no-cache", action="store_true", help="Disable representation caching")

    args = parser.parse_args()
    train_v2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        device=args.device,
        use_mock_data=args.use_mock_data,
        cache_embeddings=not args.no_cache,
    )
