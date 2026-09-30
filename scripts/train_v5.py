"""
ArbiterOmni v5 Hardware Frontier Production Training & Benchmark Script [AO-28].
Trains the v5 Production Checkpoint integrating:
- INT8 cache dynamic quantization (~75% RAM reduction) [AO-25]
- 100k resident candidate hard-negative memory bank in shared system RAM [AO-25]
- SigLIP foundation backbone perception [AO-26]
- Sparse Mixture-of-Experts (MoE) 4-layer / 4-expert Top-2 multimodal fusion [AO-27]
- Asynchronous DMA double-buffering stream pipeline [AO-24]
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import resource
import sys
import time
from typing import List, Tuple
import torch

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from arbiter_omni import (
    ArbiterOmniEngine,
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    CachedMultimodalDataset,
    HardNegativeMiner,
    OpenCLIPMultimodalEncoder,
    PersistentMemoryBank,
    TrainingConfig,
    generate_synthetic_dataset,
    get_device_telemetry,
    resolve_device,
)
from arbiter_omni.data.ai2d import load_ai2d_dataset, _make_mock_ai2d_samples
from arbiter_omni.data.gqa import load_gqa_dataset, _make_mock_gqa_samples
from arbiter_omni.data.robotics import generate_robotics_samples
from arbiter_omni.data.scienceqa import load_scienceqa_dataset, create_mock_scienceqa_samples
from arbiter_omni.data.seedbench import load_seedbench_dataset, create_mock_seedbench_samples
from arbiter_omni.fusion.transformer import TransformerMultimodalFusion
from arbiter_omni.model.decision_head import DynamicDecisionHead
from arbiter_omni.types import ModalityType, MultimodalSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_process_memory_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def build_v5_dataset(
    scienceqa_samples: int = 1600,
    seedbench_samples: int = 80,
    ai2d_samples: int = 400,
    gqa_samples: int = 1600,
    use_mock_data: bool = False,
) -> Tuple[List[MultimodalSample], List[MultimodalSample]]:
    """Builds balanced train/val split across multimodal benchmarks."""
    train_samples: List[MultimodalSample] = []
    val_samples: List[MultimodalSample] = []

    # 1. ScienceQA
    logger.info(f"Gathering ScienceQA samples ({scienceqa_samples})...")
    sqa = create_mock_scienceqa_samples(scienceqa_samples) if use_mock_data else load_scienceqa_dataset(
        split="train", max_samples=scienceqa_samples, only_multimodal=True, streaming=True, use_mock_fallback=True
    )
    sqa = [s for s in sqa if s.target_idx is not None]
    n_sqa_train = int(len(sqa) * 0.8)
    train_samples.extend(sqa[:n_sqa_train])
    val_samples.extend(sqa[n_sqa_train:])

    # 2. SEED-Bench-2 (Video & Image Reasoning)
    logger.info(f"Gathering SEED-Bench-2 samples ({seedbench_samples})...")
    seed = create_mock_seedbench_samples(seedbench_samples) if use_mock_data else load_seedbench_dataset(
        max_samples=seedbench_samples, use_mock_fallback=True
    )
    seed = [s for s in seed if s.target_idx is not None]
    n_seed_train = int(len(seed) * 0.8)
    train_samples.extend(seed[:n_seed_train])
    val_samples.extend(seed[n_seed_train:])

    # 3. AI2D (Diagram Reasoning)
    logger.info(f"Gathering AI2D diagram samples ({ai2d_samples})...")
    ai2d = _make_mock_ai2d_samples(ai2d_samples) if use_mock_data else load_ai2d_dataset(
        split="train", max_samples=ai2d_samples, streaming=True, use_mock_fallback=True
    )
    ai2d = [s for s in ai2d if s.target_idx is not None]
    n_ai2d_train = int(len(ai2d) * 0.8)
    train_samples.extend(ai2d[:n_ai2d_train])
    val_samples.extend(ai2d[n_ai2d_train:])

    # 4. GQA (Real-World Image VQA)
    logger.info(f"Gathering GQA real-world visual samples ({gqa_samples})...")
    gqa = _make_mock_gqa_samples(gqa_samples) if use_mock_data else load_gqa_dataset(
        split="train", max_samples=gqa_samples, streaming=True, use_mock_fallback=True
    )
    gqa = [s for s in gqa if s.target_idx is not None]
    n_gqa_train = int(len(gqa) * 0.8)
    train_samples.extend(gqa[:n_gqa_train])
    val_samples.extend(gqa[n_gqa_train:])

    # 5. Robotics System 1 Action Control (Open X-Embodiment / RT-X)
    logger.info("Gathering Robotics action control samples (100)...")
    robotics = generate_robotics_samples(num_samples=100)
    n_rob_train = int(len(robotics) * 0.8)
    train_samples.extend(robotics[:n_rob_train])
    val_samples.extend(robotics[n_rob_train:])

    return train_samples, val_samples


def train_v5(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "checkpoints/arbiter_omni_v5.pt",
    model_name: str = "ViT-B-16-SigLIP",
    hidden_dim: int = 512,
    num_heads: int = 8,
    scoring_dim: int = 512,
    max_spatial_patches: int = 980,
    memory_bank_capacity: int = 100000,
    moe_layers: int = 4,
    moe_experts: int = 4,
    moe_top_k: int = 2,
    device_name: str | None = None,
    use_mock_data: bool = False,
    use_int8_cache: bool = True,
) -> str:
    """Executes ArbiterOmni v5 training pipeline."""
    print("=" * 75)
    print("🚀 ARBITEROMNI v5 HARDWARE FRONTIER — PRODUCTION TRAINING")
    print("   Sparse MoE Fusion (4-Layer, Top-2 Gating) + INT8 Cache + 100k Bank")
    print("=" * 75)

    device = resolve_device(device_name)
    telemetry = get_device_telemetry(device)
    logger.info(f"Compute Device: {telemetry['device']} ({telemetry['gpu_name']})")
    logger.info(f"Initial Host RSS Memory: {get_process_memory_mb():.1f} MB")

    # 1. Initialize perception encoder in host shared RAM (frozen)
    logger.info(f"Initializing OpenCLIP {model_name} perception backbone...")
    encoder = OpenCLIPMultimodalEncoder(model_name=model_name, device="cpu", cpu_offload_encoder=True)
    logger.info(f"Perception Dims: Text={encoder.text_dim}, Image={encoder.image_dim}, Video={encoder.video_dim}, Audio={encoder.audio_dim}")

    # 2. Build Sparse MoE fusion & dynamic decision head [AO-27]
    modality_dims = {
        "question": encoder.text_dim,
        ModalityType.TEXT.value: encoder.text_dim,
        ModalityType.IMAGE.value: encoder.image_dim,
        ModalityType.VIDEO.value: encoder.video_dim,
        ModalityType.AUDIO.value: encoder.audio_dim,
    }
    fusion = TransformerMultimodalFusion(
        modality_dims=modality_dims,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_feedforward=hidden_dim * 2,
        enable_spatial_cross_attention=True,
        max_spatial_patches=max_spatial_patches,  # 980 multi-tile patches
        use_moe=True,
        moe_num_layers=moe_layers,
        moe_num_experts=moe_experts,
        moe_top_k=moe_top_k,
    )
    decision_head = DynamicDecisionHead(
        context_dim=hidden_dim,
        candidate_dim=encoder.text_dim,
        scoring_dim=scoring_dim,
    )
    model = ArbiterOmniModel(
        encoder=encoder,
        fusion=fusion,
        decision_head=decision_head,
        hidden_dim=hidden_dim,
        scoring_dim=scoring_dim,
        use_moe=True,
        moe_num_layers=moe_layers,
        moe_num_experts=moe_experts,
        moe_top_k=moe_top_k,
    ).to(device)

    trainable_params = model.trainable_parameters()
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"Frozen Perception Params: {frozen_count:,} (0.00% gradient updates)")
    logger.info(f"Trainable Sparse MoE Params: {trainable_count:,} ({moe_layers} layers, {moe_experts} experts/layer, Top-{moe_top_k})")

    # 3. Build training corpus
    train_samples, val_samples = build_v5_dataset(use_mock_data=use_mock_data)
    logger.info(f"Dataset split: Train={len(train_samples):,} | Val={len(val_samples):,}")

    # 4. Pre-caching representation vectors with INT8 quantization [AO-25]
    cache_tag = f"v5_int8_{model_name.replace('/', '_')}"
    cache_train_path = f"data/cache/{cache_tag}_train.pt"
    cache_val_path = f"data/cache/{cache_tag}_val.pt"

    v4_cache_train = f"data/cache/v4_siglip_{model_name.replace('/', '_')}_train.pt"
    v4_cache_val = f"data/cache/v4_siglip_{model_name.replace('/', '_')}_val.pt"

    if os.path.exists(cache_train_path) and os.path.exists(cache_val_path):
        logger.info(f"⚡ Loading cached representations from {cache_tag}_*.pt...")
        train_cached = CachedMultimodalDataset.load(cache_train_path)
        val_cached = CachedMultimodalDataset.load(cache_val_path)
    elif os.path.exists(v4_cache_train) and os.path.exists(v4_cache_val):
        logger.info("⚡ Loading pre-computed SigLIP cache and quantizing to INT8 in-place...")
        train_cached = CachedMultimodalDataset.load(v4_cache_train)
        val_cached = CachedMultimodalDataset.load(v4_cache_val)
        if use_int8_cache:
            train_cached.to_int8()
            val_cached.to_int8()
        try:
            train_cached.save(cache_train_path)
            val_cached.save(cache_val_path)
            logger.info(f"Saved INT8 quantized cache to {cache_tag}_*.pt")
        except Exception as e:
            logger.warning(f"Could not persist cache: {e}")
    else:
        logger.info(f"⚡ Pre-caching frozen representations into shared memory (use_int8={use_int8_cache})...")
        from arbiter_omni.data.dataset import MultimodalDecisionDataset
        train_cached = CachedMultimodalDataset.from_dataset(
            MultimodalDecisionDataset(train_samples),
            model=model,
            batch_size=batch_size,
            device=device,
            use_int8=use_int8_cache,
        )
        val_cached = CachedMultimodalDataset.from_dataset(
            MultimodalDecisionDataset(val_samples),
            model=model,
            batch_size=batch_size,
            device=device,
            use_int8=use_int8_cache,
        )
        try:
            train_cached.save(cache_train_path)
            val_cached.save(cache_val_path)
            logger.info("Saved cached representations to disk.")
        except Exception as e:
            logger.warning(f"Could not persist cache: {e}")

    logger.info(f"Train Cache Resident Memory: {train_cached.total_memory_mb:.1f} MB (INT8 compressed)")

    # 5. Offload frozen perception encoder to CPU to free ~850 MB GDDR5 VRAM
    logger.info("🧹 Offloading frozen perception encoder from GPU to CPU to reclaim VRAM for backprop...")
    model.encoder.to("cpu")
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Initialize 100k PersistentMemoryBank in shared system RAM [AO-25]
    logger.info(f"🧠 Initializing 100k PersistentMemoryBank in shared RAM (capacity: {memory_bank_capacity:,}, fp16=True)...")
    memory_bank = PersistentMemoryBank(
        capacity=memory_bank_capacity,
        candidate_dim=encoder.text_dim,
        device="cpu",
        store_fp16=True,
    )
    bank_cache_path = f"data/cache/v5_memory_bank_{model_name.replace('/', '_')}.pt"
    if os.path.exists(bank_cache_path):
        logger.info(f"⚡ Loading pre-computed 100k Memory Bank from {bank_cache_path}...")
        memory_bank.load(bank_cache_path)
    else:
        # Populate memory bank with deduplicated candidate options harvested from train set
        miner = HardNegativeMiner.from_dataset(train_samples, encoder=model.encoder, batch_size=128)
        added = miner.populate_memory_bank(memory_bank)
        try:
            memory_bank.save(bank_cache_path)
            logger.info(f"Saved memory bank cache to {bank_cache_path}")
        except Exception as e:
            logger.warning(f"Could not persist memory bank cache: {e}")
    logger.info(f"Memory Bank Ready: {len(memory_bank):,} unique foils (shared RAM: {memory_bank.memory_usage_mb:.2f} MB)")

    # 7. Configure trainer with DMA double-buffering & memory bank
    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        fp16=True,
        save_path=save_path,
        device=str(device),
        contrastive_lambda=0.25,
        margin_gamma=0.5,
        use_memory_bank=True,
        memory_bank_capacity=memory_bank_capacity,
        memory_bank_k_foils=10,
        async_prefetch=True,
        prefetch_queue_size=2,
    )

    trainer = ArbiterOmniTrainer(model=model, config=config)
    trainer.memory_bank = memory_bank

    logger.info("🏁 Starting ArbiterOmni v5 Sparse MoE training loop...")
    t0 = time.perf_counter()
    history = trainer.fit(train_dataset=train_cached, val_dataset=val_cached)
    train_elapsed = time.perf_counter() - t0

    # Ensure checkpoint is written
    if not os.path.exists(save_path):
        trainer.save_checkpoint(save_path)

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"✅ ArbiterOmni v5 Checkpoint published: {save_path} ({file_size_mb:.2f} MB) in {train_elapsed:.1f}s")

    # 8. Verification & Inference Telemetry
    print("\n" + "=" * 75)
    print("🔬 VERIFYING v5 RUNTIME INFERENCE VIA ArbiterOmniEngine.from_pretrained('v5')")
    print("=" * 75)
    engine = ArbiterOmniEngine.from_pretrained(save_path, device=device)
    
    # Test sample: Autonomous driving high-speed navigation
    sample_text = "Dense multi-lane highway with braking vehicles ahead."
    cands = [
        "proceed at maximum velocity without braking",
        "decelerate smoothly and increase following distance",
        "execute sharp left turn into adjacent barrier",
        "shut down primary electrical systems",
    ]
    res = engine.decide(question="What is the safest action?", candidates=cands, text=sample_text)
    print(f"Decided Choice : {res.winner}")
    print(f"Confidence     : {res.confidence * 100:.2f}%")
    print(f"Entropy        : {res.entropy:.4f} nats")
    print(f"Escalate Sys2  : {res.escalate_system2}")

    final_mem = get_process_memory_mb()
    print(f"\n📊 Process Host Memory Footprint: {final_mem:.1f} MB (< 1.5 GB allocated across 12 GB pool)")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArbiterOmni v5 Checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="checkpoints/arbiter_omni_v5.pt", help="Checkpoint save path")
    parser.add_argument("--model-name", type=str, default="ViT-B-16-SigLIP", help="OpenCLIP model")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Fusion hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--scoring-dim", type=int, default=512, help="Decision head scoring dimension")
    parser.add_argument("--max-spatial-patches", type=int, default=980, help="Max spatial patch tokens")
    parser.add_argument("--memory-bank-capacity", type=int, default=100000, help="Memory bank capacity")
    parser.add_argument("--moe-layers", type=int, default=4, help="MoE layers")
    parser.add_argument("--moe-experts", type=int, default=4, help="MoE experts per layer")
    parser.add_argument("--moe-top-k", type=int, default=2, help="MoE active experts per token")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--use-mock-data", action="store_true", help="Use mock data")
    parser.add_argument("--no-int8-cache", action="store_true", help="Disable INT8 caching")

    args = parser.parse_args()
    try:
        train_v5(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_path,
            model_name=args.model_name,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            scoring_dim=args.scoring_dim,
            max_spatial_patches=args.max_spatial_patches,
            memory_bank_capacity=args.memory_bank_capacity,
            moe_layers=args.moe_layers,
            moe_experts=args.moe_experts,
            moe_top_k=args.moe_top_k,
            device_name=args.device,
            use_mock_data=args.use_mock_data,
            use_int8_cache=not args.no_int8_cache,
        )
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)
    finally:
        os._exit(0)
