"""
ArbiterOmni v6-Max Hardware Saturation Training Script [AO-34, AO-35].

Pushes the AMD Radeon RX 480 (4 GB GDDR5) + 8 GB DDR4 host RAM to full
saturation through three coordinated hardware advances:

  [AO-34] Native 768-Dim Manifold & 8-Expert MoE Scaling
    - hidden_dim / scoring_dim promoted from 512 → 768 (native SigLIP dim)
    - 4 MoE layers × (1 Shared + 7 Domain-Routed) = 32 total experts
    - Active model: ~46 M trainable parameters
    - Estimated VRAM footprint: ~3.2 GB / 4 GB

  [AO-35] 100k Multimodal Memory Bank & DirectML Shader Optimizer
    - DirectML-native AdamW (mul_ + add_) replaces CPU-fallback lerp_
      → 95–100% continuous GPU compute saturation
    - CosineAnnealingLR over 10 epochs + Stochastic Weight Averaging (SWA)
    - Memory Bank harvested from ScienceQA, GQA, AI2D, SEED-Bench (up to 100k foils)
    - Estimated host RAM footprint: ~5.5 GB / 8 GB DDR4

Usage:
  .venv-directml/bin/python scripts/train_v6_max.py \\
      --epochs 10 --batch-size 8 --lr 5e-5 \\
      --save-path checkpoints/arbiter_omni_v6_max.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import resource
import shutil
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

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
from arbiter_omni.model.speculative import SpeculativeDraftHead
from arbiter_omni.training.optim import DirectMLNativeAdamW
from arbiter_omni.training.telemetry import get_rss_mb, memory_summary
from arbiter_omni.types import ModalityType, MultimodalSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset builder — larger crop of ScienceQA + GQA + AI2D + SEED-Bench
# ---------------------------------------------------------------------------

def build_v6_max_dataset(
    scienceqa_samples: int = 2400,
    seedbench_samples: int = 120,
    ai2d_samples: int = 600,
    gqa_samples: int = 2400,
    use_mock_data: bool = False,
) -> Tuple[List[MultimodalSample], List[MultimodalSample]]:
    """Builds balanced train/val split across multimodal benchmarks (v6-Max scale)."""
    train_samples: List[MultimodalSample] = []
    val_samples: List[MultimodalSample] = []

    # 1. ScienceQA
    logger.info(f"[v6-Max] Gathering ScienceQA samples ({scienceqa_samples})...")
    sqa = (
        create_mock_scienceqa_samples(scienceqa_samples)
        if use_mock_data
        else load_scienceqa_dataset(
            split="train", max_samples=scienceqa_samples, only_multimodal=True,
            streaming=True, use_mock_fallback=True,
        )
    )
    sqa = [s for s in sqa if s.target_idx is not None]
    n = int(len(sqa) * 0.8)
    train_samples.extend(sqa[:n])
    val_samples.extend(sqa[n:])

    # 2. SEED-Bench-2
    logger.info(f"[v6-Max] Gathering SEED-Bench-2 samples ({seedbench_samples})...")
    seed = (
        create_mock_seedbench_samples(seedbench_samples)
        if use_mock_data
        else load_seedbench_dataset(max_samples=seedbench_samples, use_mock_fallback=True)
    )
    seed = [s for s in seed if s.target_idx is not None]
    n = int(len(seed) * 0.8)
    train_samples.extend(seed[:n])
    val_samples.extend(seed[n:])

    # 3. AI2D
    logger.info(f"[v6-Max] Gathering AI2D diagram samples ({ai2d_samples})...")
    ai2d = (
        _make_mock_ai2d_samples(ai2d_samples)
        if use_mock_data
        else load_ai2d_dataset(split="train", max_samples=ai2d_samples, streaming=True, use_mock_fallback=True)
    )
    ai2d = [s for s in ai2d if s.target_idx is not None]
    n = int(len(ai2d) * 0.8)
    train_samples.extend(ai2d[:n])
    val_samples.extend(ai2d[n:])

    # 4. GQA
    logger.info(f"[v6-Max] Gathering GQA samples ({gqa_samples})...")
    gqa = (
        _make_mock_gqa_samples(gqa_samples)
        if use_mock_data
        else load_gqa_dataset(split="train", max_samples=gqa_samples, streaming=True, use_mock_fallback=True)
    )
    gqa = [s for s in gqa if s.target_idx is not None]
    n = int(len(gqa) * 0.8)
    train_samples.extend(gqa[:n])
    val_samples.extend(gqa[n:])

    # 5. Robotics
    logger.info("[v6-Max] Gathering Robotics action control samples (150)...")
    robotics = generate_robotics_samples(num_samples=150)
    n = int(len(robotics) * 0.8)
    train_samples.extend(robotics[:n])
    val_samples.extend(robotics[n:])

    return train_samples, val_samples


TRAINABLE_MODULES = ("fusion", "decision_head", "speculative_head")
ARCH_KEYS = ("hidden_dim", "scoring_dim", "num_heads", "moe_num_layers", "moe_num_experts", "moe_top_k")


def log_mem(stage: str) -> None:
    """Logs a labelled process/system memory reading."""
    logger.info(f"[memory] {stage}: {memory_summary()}")


def state_to_cpu(module: nn.Module, dtype: torch.dtype | None = None) -> Dict[str, torch.Tensor]:
    """Copies a module's state dict to CPU one tensor at a time (optionally casting floats)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in module.state_dict().items():
        t = v.detach()
        if dtype is not None and t.is_floating_point():
            out[k] = t.to("cpu", dtype=dtype)
        else:
            out[k] = t.to("cpu").clone() if t.device.type == "cpu" else t.to("cpu")
    return out


def atomic_torch_save(obj, path: str) -> None:
    """
    Writes to <path>.tmp then renames, so an interrupted or killed save can never
    leave a truncated (for example 0-byte) file at the destination.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def to_cpu_tree(obj):
    """
    Recursively moves every tensor in a nested structure to CPU, one tensor at a time.

    torch.save() called directly on DirectML tensors is killed by the OS (SIGKILL) even
    for ~1 GB of state; saving CPU copies is stable. Always route device state through this.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu_tree(v) for v in obj)
    return obj


def save_resume_dir(resume_dir: str, meta: dict, model: nn.Module, optimizer, swa: "RunningWeightAverage") -> None:
    """
    Writes the full resumable training state as a directory of files, one CPU copy alive at
    a time (peak host memory is the largest single file, the optimizer moments), then swaps
    it into place atomically. The previous good state is kept as <dir>.old until the swap ends.
    """
    tmp, old = f"{resume_dir}.tmp", f"{resume_dir}.old"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save({n: state_to_cpu(getattr(model, n)) for n in TRAINABLE_MODULES}, os.path.join(tmp, "weights.pt"))
    torch.save(to_cpu_tree(optimizer.state_dict()), os.path.join(tmp, "optim.pt"))
    torch.save(swa.state_dict(), os.path.join(tmp, "swa.pt"))
    torch.save(meta, os.path.join(tmp, "meta.pt"))
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(resume_dir):
        os.replace(resume_dir, old)
    os.replace(tmp, resume_dir)
    shutil.rmtree(old, ignore_errors=True)


def load_resume_dir(resume_dir: str) -> dict:
    """Loads a resume directory, falling back to <dir>.old if a swap was interrupted."""
    if not os.path.isdir(resume_dir) and os.path.isdir(f"{resume_dir}.old"):
        logger.warning(f"{resume_dir} missing; falling back to {resume_dir}.old (interrupted swap)")
        resume_dir = f"{resume_dir}.old"
    if not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"--resume requested but {resume_dir} does not exist")

    def _load(name):
        return torch.load(os.path.join(resume_dir, name), map_location="cpu", weights_only=False)

    return {"meta": _load("meta.pt"), "weights": _load("weights.pt"), "optim": _load("optim.pt"), "swa": _load("swa.pt")}


class RunningWeightAverage:
    """
    Equal-weight running average of the trainable submodules (Stochastic Weight Averaging).

    Held on CPU in FP32 and restricted to fusion / decision_head / speculative_head, so it
    costs roughly one extra copy of the trainable weights rather than a deep copy of the
    whole model including the frozen encoder.
    """

    def __init__(self) -> None:
        self.n = 0
        self.avg: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        for name in TRAINABLE_MODULES:
            module = getattr(model, name, None)
            if module is None:
                continue
            for k, v in module.state_dict().items():
                key = f"{name}.{k}"
                if not v.is_floating_point():
                    self.avg[key] = v.detach().to("cpu").clone()
                    continue
                cur = v.detach().to("cpu", dtype=torch.float32)
                if key not in self.avg:
                    self.avg[key] = cur.clone()
                else:
                    self.avg[key].mul_(self.n - 1).add_(cur).div_(self.n)

    def module_state_dicts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name in TRAINABLE_MODULES}
        for key, tensor in self.avg.items():
            name, _, k = key.partition(".")
            out[name][k] = tensor
        return out

    def state_dict(self) -> dict:
        return {"n": self.n, "avg": self.avg}

    def load_state_dict(self, state: dict) -> None:
        self.n = int(state["n"])
        self.avg = {k: v.clone() for k, v in state["avg"].items()}


def make_swa_scheduler(optimizer, lr: float, epochs: int, swa_start_epoch: int):
    return torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=lr / 10.0, anneal_epochs=max(1, epochs - swa_start_epoch + 1)
    )


def _delta(series: List[float], scale: float = 1.0, suffix: str = "") -> str:
    """Formats the change between the last two values of a metric series."""
    if len(series) < 2:
        return "first epoch"
    return f"d{(series[-1] - series[-2]) * scale:+.4f}{suffix}"


def setup_file_logging(log_path: str) -> None:
    """Mirrors all log output to a file so a run can be inspected after the terminal is gone."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info(f"Logging to file: {log_path}")


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------

def train_v6_max(
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-5,
    save_path: str = "checkpoints/arbiter_omni_v6_max.pt",
    model_name: str = "ViT-B-16-SigLIP",
    # [AO-34] Native 768-Dim: hidden_dim = scoring_dim = 768
    hidden_dim: int = 768,
    num_heads: int = 8,
    scoring_dim: int = 768,
    max_spatial_patches: int = 980,
    # [AO-34] 8-Expert MoE: 1 Shared + 7 Routed = 32 experts across 4 layers
    moe_layers: int = 4,
    moe_experts: int = 8,
    moe_top_k: int = 2,
    # [AO-35] 100k Memory Bank
    memory_bank_capacity: int = 100_000,
    device_name: str | None = None,
    use_mock_data: bool = False,
    use_int8_cache: bool = True,
    # SWA (Stochastic Weight Averaging)
    swa_start_epoch: int = 7,
    resume: bool = False,
    smoke_test: bool = False,
    log_interval: int = 20,
) -> str:
    """ArbiterOmni v6-Max: full hardware saturation training pipeline [AO-34, AO-35]."""

    if smoke_test:
        if save_path == "checkpoints/arbiter_omni_v6_max.pt":
            save_path = "checkpoints/smoke_v6_max.pt"
        swa_start_epoch = min(swa_start_epoch, max(1, epochs))
        log_interval = 1

    print("=" * 80)
    print("ARBITEROMNI v6-Max HARDWARE SATURATION TRAINING PIPELINE [AO-34 / AO-35]")
    print("   Native 768-Dim Manifold | 8-Expert DeepSeek-V3 MoE | DirectML-Native AdamW")
    print("   CosineAnnealingLR | Stochastic Weight Averaging (SWA) | 100k Memory Bank")
    print("=" * 80)

    device = resolve_device(device_name)
    telemetry = get_device_telemetry(device)
    logger.info(f"Compute Device  : {telemetry['device']} ({telemetry['gpu_name']})")
    logger.info(f"Architecture    : hidden_dim={hidden_dim}, scoring_dim={scoring_dim}, "
                f"MoE {moe_layers}L×{moe_experts}E (Top-{moe_top_k})")
    logger.info(f"Initial Host RSS: {get_rss_mb():.1f} MB")

    # -----------------------------------------------------------------------
    # 1. Perception Encoder — frozen, CPU-resident
    # -----------------------------------------------------------------------
    logger.info(f"Initializing OpenCLIP {model_name} perception backbone (frozen, CPU)...")
    encoder = OpenCLIPMultimodalEncoder(model_name=model_name, device="cpu", cpu_offload_encoder=True)
    logger.info(
        f"Perception Dims : Text={encoder.text_dim}, Image={encoder.image_dim}, "
        f"Video={encoder.video_dim}, Audio={encoder.audio_dim}"
    )
    frozen_count = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Frozen Encoder  : {frozen_count:,} params — 0% gradient updates")

    # -----------------------------------------------------------------------
    # 2. v6-Max Architecture [AO-34]
    #    768-dim manifold (matches SigLIP native space — no down-projection)
    #    4 MoE layers × (1 Shared Invariant + 7 Domain-Routed) = 32 experts
    # -----------------------------------------------------------------------
    modality_dims = {
        "question": encoder.text_dim,
        ModalityType.TEXT.value: encoder.text_dim,
        ModalityType.IMAGE.value: encoder.image_dim,
        ModalityType.VIDEO.value: encoder.video_dim,
        ModalityType.AUDIO.value: encoder.audio_dim,
    }
    fusion = TransformerMultimodalFusion(
        modality_dims=modality_dims,
        hidden_dim=hidden_dim,           # 768 — native SigLIP manifold
        num_heads=num_heads,             # 8 heads × 96 dim = 768
        dim_feedforward=hidden_dim * 2,  # 1536 FFN
        enable_spatial_cross_attention=True,
        max_spatial_patches=max_spatial_patches,
        use_moe=True,
        moe_num_layers=moe_layers,       # 4 MoE transformer layers
        moe_num_experts=moe_experts,     # 8 experts/layer (1 shared + 7 routed)
        moe_top_k=moe_top_k,            # Top-2 soft routing per token
        use_shared_expert=True,
    )
    decision_head = DynamicDecisionHead(
        context_dim=hidden_dim,          # 768
        candidate_dim=encoder.text_dim,  # 768
        scoring_dim=scoring_dim,         # 768
    )
    speculative_head = SpeculativeDraftHead(
        input_dim=encoder.text_dim,
        candidate_dim=encoder.text_dim,
        draft_dim=128,
    )
    model = ArbiterOmniModel(
        encoder=encoder,
        fusion=fusion,
        decision_head=decision_head,
        speculative_head=speculative_head,
        hidden_dim=hidden_dim,
        scoring_dim=scoring_dim,
        use_moe=True,
        moe_num_layers=moe_layers,
        moe_num_experts=moe_experts,
        moe_top_k=moe_top_k,
        use_shared_expert=True,
        enable_speculative_early_exit=True,
    ).to(device)

    trainable_params = model.trainable_parameters()
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable v6-Max Parameters: {trainable_count:,} "
                f"(4 MoE Layers × {moe_experts} Experts + Speculative Head)")
    for name in TRAINABLE_MODULES:
        n_params = sum(p.numel() for p in getattr(model, name).parameters())
        logger.info(f"  {name:<16s}: {n_params:>12,} params")
    log_mem("model built")

    model_config = {
        "hidden_dim": hidden_dim,
        "scoring_dim": scoring_dim,
        "num_layers": moe_layers,
        "num_heads": num_heads,
        "model_name": model_name,
        "use_moe": True,
        "moe_num_layers": moe_layers,
        "moe_num_experts": moe_experts,
        "moe_top_k": moe_top_k,
        "use_shared_expert": True,
        "enable_speculative_early_exit": True,
        "enable_spatial_cross_attention": True,
        "max_spatial_patches": max_spatial_patches,
    }
    logger.info("Run configuration: " + json.dumps(dict(
        model_config, epochs=epochs, batch_size=batch_size, lr=lr, save_path=save_path,
        swa_start_epoch=swa_start_epoch, memory_bank_capacity=memory_bank_capacity,
        resume=resume, smoke_test=smoke_test, log_interval=log_interval,
    ), sort_keys=True))

    # -----------------------------------------------------------------------
    # 3. INT8 Cached Representations [AO-25]
    # -----------------------------------------------------------------------
    cache_tag = f"v6_max_int8_{model_name.replace('/', '_')}"
    cache_train_path = f"data/cache/{cache_tag}_train.pt"
    cache_val_path = f"data/cache/{cache_tag}_val.pt"

    if os.path.exists(cache_train_path) and os.path.exists(cache_val_path):
        logger.info(f"Loading pre-computed cached representations from {cache_tag}_*.pt...")
        train_cached = CachedMultimodalDataset.load(cache_train_path)
        val_cached = CachedMultimodalDataset.load(cache_val_path)
    else:
        logger.info(f"Pre-caching v6-Max representations into shared memory (INT8={use_int8_cache})...")
        train_samples, val_samples = build_v6_max_dataset(use_mock_data=use_mock_data)
        logger.info(f"Dataset split: Train={len(train_samples):,} | Val={len(val_samples):,}")
        from arbiter_omni.data.dataset import MultimodalDecisionDataset

        train_cached = CachedMultimodalDataset.from_dataset(
            MultimodalDecisionDataset(train_samples),
            model=model, batch_size=batch_size, device=device, use_int8=use_int8_cache,
        )
        val_cached = CachedMultimodalDataset.from_dataset(
            MultimodalDecisionDataset(val_samples),
            model=model, batch_size=batch_size, device=device, use_int8=use_int8_cache,
        )
        try:
            train_cached.save(cache_train_path)
            val_cached.save(cache_val_path)
            logger.info(f"Saved v6-Max cache to disk ({cache_tag}_*.pt).")
        except Exception as e:
            logger.warning(f"Could not persist cache: {e}")

    logger.info(f"Train Cache Resident Memory: {train_cached.total_memory_mb:.1f} MB (INT8 compressed)")
    log_mem("caches ready")

    # -----------------------------------------------------------------------
    # 4. Offload frozen encoder → CPU VRAM reclaim
    # -----------------------------------------------------------------------
    logger.info("Offloading frozen perception encoder to CPU to reclaim VRAM for v6-Max...")
    model.encoder.to("cpu")
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 5. 100k Multimodal Memory Bank in shared DDR4 RAM [AO-35]
    # -----------------------------------------------------------------------
    logger.info(f"Initializing {memory_bank_capacity:,}-capacity Memory Bank in shared DDR4 RAM...")
    memory_bank = PersistentMemoryBank(
        capacity=memory_bank_capacity,
        candidate_dim=encoder.text_dim,
        device="cpu",
        store_fp16=True,
    )
    bank_cache_path = f"data/cache/v6_max_memory_bank_{model_name.replace('/', '_')}.pt"
    v5_bank_path = f"data/cache/v5_memory_bank_{model_name.replace('/', '_')}.pt"
    if os.path.exists(bank_cache_path):
        logger.info(f"Loading pre-computed v6-Max Memory Bank from {bank_cache_path}...")
        memory_bank.load(bank_cache_path)
    elif os.path.exists(v5_bank_path):
        # Warm-start from v5 bank, then augment
        logger.info(f"Warm-starting v6-Max Memory Bank from existing v5 bank at {v5_bank_path}...")
        memory_bank.load(v5_bank_path)
    else:
        if not use_mock_data:
            train_samples_for_bank, _ = build_v6_max_dataset(use_mock_data=False)
        else:
            train_samples_for_bank = []
        if train_samples_for_bank:
            miner = HardNegativeMiner.from_dataset(train_samples_for_bank, encoder=model.encoder, batch_size=64)
            miner.populate_memory_bank(memory_bank)
            try:
                memory_bank.save(bank_cache_path)
                logger.info(f"Saved v6-Max memory bank to {bank_cache_path}")
            except Exception as e:
                logger.warning(f"Could not persist memory bank: {e}")

    logger.info(
        f"Memory Bank Ready: {len(memory_bank):,} foils (DDR4 RSS: {memory_bank.memory_usage_mb:.2f} MB)"
    )
    log_mem("memory bank ready")

    # -----------------------------------------------------------------------
    # 6. Optimizer, LR schedule and SWA accumulator [AO-35]
    # -----------------------------------------------------------------------
    logger.info("Initializing DirectML-native AdamW (mul_/add_ only, no lerp CPU fallback)")
    optimizer = DirectMLNativeAdamW(
        model.trainable_parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )

    # CosineAnnealingLR: decay lr from `lr` to `lr/100` over `epochs` steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100.0
    )

    # SWA accumulator: trainable submodules only, FP32, resident on CPU.
    # (The previous implementation deep-copied the entire model including the
    # frozen 212M-parameter encoder, and kept the copy on the accelerator.)
    swa = RunningWeightAverage()
    swa_scheduler = None  # created lazily at the first SWA epoch
    log_mem("optimizer and SWA accumulator ready")

    # -----------------------------------------------------------------------
    # 7. Trainer
    # -----------------------------------------------------------------------
    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        fp16=True,
        save_path=None,  # all checkpointing is handled atomically by this script
        device=str(device),
        contrastive_lambda=0.30,
        margin_gamma=0.5,
        use_memory_bank=True,
        memory_bank_capacity=memory_bank_capacity,
        memory_bank_k_foils=12,
        async_prefetch=True,
        prefetch_queue_size=2,
        accumulate_grad_batches=2,
        log_interval=log_interval,
    )

    trainer = ArbiterOmniTrainer(model=model, config=config)
    trainer.memory_bank = memory_bank
    # Inject DirectML-native optimizer, overriding the default AdamW
    trainer.optimizer = optimizer

    from torch.utils.data import DataLoader, Subset
    from arbiter_omni.data.cached import collate_cached_multimodal_decision

    if smoke_test:
        train_cached = Subset(train_cached, list(range(min(48, len(train_cached)))))
        val_cached = Subset(val_cached, list(range(min(16, len(val_cached)))))
        logger.info(f"SMOKE TEST: using {len(train_cached)} train / {len(val_cached)} val samples")

    train_loader = DataLoader(
        train_cached,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cached_multimodal_decision,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_cached,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cached_multimodal_decision,
        num_workers=0,
    )
    logger.info(
        f"Dataloaders: train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={batch_size} grad_accum=2 effective_batch={batch_size * 2}"
    )

    # -----------------------------------------------------------------------
    # 8. Resume (optional)
    # -----------------------------------------------------------------------
    history: Dict[str, list] = {
        "loss": [], "accuracy": [], "entropy": [], "margin_loss": [],
        "val_loss": [], "val_accuracy": [], "val_entropy": [],
        "lr": [], "epoch_seconds": [],
    }
    start_epoch = 1
    prior_elapsed_s = 0.0
    resume_dir = f"{save_path}.resume"
    metrics_path = f"{save_path}.metrics.jsonl"

    if resume:
        logger.info(f"Resuming from {resume_dir}")
        ckpt = load_resume_dir(resume_dir)
        meta = ckpt["meta"]
        saved_cfg = meta.get("model_config", {})
        for key in ARCH_KEYS:
            if key in saved_cfg and saved_cfg[key] != model_config[key]:
                raise ValueError(
                    f"Resume architecture mismatch for '{key}': checkpoint={saved_cfg[key]} current={model_config[key]}"
                )
        for name in TRAINABLE_MODULES:
            getattr(model, name).load_state_dict(ckpt["weights"][name])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(meta["scheduler"])
        swa.load_state_dict(ckpt["swa"])
        if meta.get("swa_scheduler") is not None:
            swa_scheduler = make_swa_scheduler(optimizer, lr, epochs, swa_start_epoch)
            swa_scheduler.load_state_dict(meta["swa_scheduler"])
        history = meta["history"]
        start_epoch = int(meta["epoch"]) + 1
        prior_elapsed_s = float(meta.get("elapsed_s", 0.0))
        del ckpt, meta
        gc.collect()
        logger.info(
            f"Resume state restored: completed_epochs={start_epoch - 1} "
            f"swa_snapshots={swa.n} lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        log_mem("resume restored")
        if start_epoch > epochs:
            logger.info("All requested epochs already completed; skipping straight to finalisation")
    elif os.path.exists(metrics_path):
        # fresh run: do not append to a stale metrics file from an earlier run
        os.replace(metrics_path, metrics_path + ".old")

    # -----------------------------------------------------------------------
    # 9. Training loop
    # -----------------------------------------------------------------------
    logger.info(
        f"Starting v6-Max training loop: epochs {start_epoch}..{epochs}, SWA from epoch {swa_start_epoch}"
    )
    run_t0 = time.perf_counter()
    epoch_durations: List[float] = list(history.get("epoch_seconds", []))

    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        in_swa = epoch >= swa_start_epoch
        logger.info("=" * 100)
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} START | lr={lr_now:.3e} | "
            f"phase={'SWA' if in_swa else 'cosine'} | {memory_summary()}"
        )

        train_metrics = trainer.train_epoch(train_loader)
        train_s = time.perf_counter() - epoch_t0
        log_mem(f"epoch {epoch:02d} training finished in {train_s / 60.0:.1f} min")

        eval_t0 = time.perf_counter()
        val_metrics = trainer.evaluate(val_loader)
        eval_s = time.perf_counter() - eval_t0
        log_mem(f"epoch {epoch:02d} validation finished in {eval_s:.0f} s")

        history["loss"].append(train_metrics["loss"])
        history["accuracy"].append(train_metrics["accuracy"])
        history["entropy"].append(train_metrics["entropy"])
        history["margin_loss"].append(train_metrics.get("margin_loss", 0.0))
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        history["val_entropy"].append(val_metrics.get("val_entropy", 0.0))
        history["lr"].append(lr_now)

        best_idx = max(range(len(history["val_accuracy"])), key=lambda i: history["val_accuracy"][i])
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} RESULT | "
            f"train_loss={train_metrics['loss']:.4f} ({_delta(history['loss'])}) | "
            f"train_acc={train_metrics['accuracy'] * 100:.2f}% ({_delta(history['accuracy'], 100.0, '%')}) | "
            f"val_loss={val_metrics['val_loss']:.4f} ({_delta(history['val_loss'])}) | "
            f"val_acc={val_metrics['val_accuracy'] * 100:.2f}% ({_delta(history['val_accuracy'], 100.0, '%')}) | "
            f"entropy={train_metrics['entropy']:.3f} ({_delta(history['entropy'])}) | "
            f"margin_loss={history['margin_loss'][-1]:.4f} | "
            f"temperature={train_metrics.get('temperature', float('nan')):.3f} | "
            f"best_val_acc={history['val_accuracy'][best_idx] * 100:.2f}% (epoch {best_idx + 1})"
        )

        # SWA snapshot, then LR schedule step
        if in_swa:
            swa.update(model)
            if swa_scheduler is None:
                swa_scheduler = make_swa_scheduler(optimizer, lr, epochs, swa_start_epoch)
            swa_scheduler.step()
            logger.info(
                f"SWA snapshot {swa.n} accumulated (equal-weight average of epochs "
                f"{epoch - swa.n + 1}..{epoch}); next lr={optimizer.param_groups[0]['lr']:.3e}"
            )
        else:
            scheduler.step()
            logger.info(f"Cosine schedule stepped; next lr={optimizer.param_groups[0]['lr']:.3e}")

        epoch_s = time.perf_counter() - epoch_t0
        epoch_durations.append(epoch_s)
        history["epoch_seconds"].append(epoch_s)
        elapsed_total_s = prior_elapsed_s + (time.perf_counter() - run_t0)
        recent = epoch_durations[-3:]
        eta_h = (sum(recent) / len(recent)) * (epochs - epoch) / 3600.0
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} TIMING | train={train_s / 60.0:.1f}min eval={eval_s / 60.0:.1f}min "
            f"epoch={epoch_s / 60.0:.1f}min | total_elapsed={elapsed_total_s / 3600.0:.2f}h | "
            f"eta_remaining={eta_h:.2f}h"
        )

        # ---- Persist: inference snapshot (FP16) + full resume state + metrics line ----
        persist_t0 = time.perf_counter()
        snapshot = {
            "model_config": dict(model_config, snapshot_epoch=epoch),
            "training_history": history,
            **{name: state_to_cpu(getattr(model, name), torch.float16) for name in TRAINABLE_MODULES},
        }
        atomic_torch_save(snapshot, save_path)
        del snapshot
        log_mem(f"epoch {epoch:02d} inference snapshot written -> {save_path}")

        save_resume_dir(
            resume_dir,
            meta={
                "epoch": epoch,
                "elapsed_s": elapsed_total_s,
                "model_config": model_config,
                "history": history,
                "scheduler": scheduler.state_dict(),
                "swa_scheduler": swa_scheduler.state_dict() if swa_scheduler is not None else None,
            },
            model=model,
            optimizer=optimizer,
            swa=swa,
        )
        gc.collect()
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} CHECKPOINT | snapshot={os.path.getsize(save_path) / 1048576:.1f}MB "
            f"resume_dir={sum(os.path.getsize(os.path.join(resume_dir, f)) for f in os.listdir(resume_dir)) / 1048576:.1f}MB written in "
            f"{time.perf_counter() - persist_t0:.1f}s | {memory_summary()}"
        )

        with open(metrics_path, "a") as mf:
            mf.write(json.dumps({
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_accuracy"],
                "entropy": train_metrics["entropy"],
                "margin_loss": train_metrics.get("margin_loss", 0.0),
                "samples_per_sec": train_metrics.get("samples_per_sec", 0.0),
                "train_seconds": train_s,
                "eval_seconds": eval_s,
                "rss_mb": get_rss_mb(),
                "swa_snapshots": swa.n,
            }) + "\n")

    train_elapsed = prior_elapsed_s + (time.perf_counter() - run_t0)

    # -----------------------------------------------------------------------
    # 10. Finalise: SWA average (or final weights) -> FP16 + FP32 checkpoints
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if swa.n > 0:
        final_sd = swa.module_state_dicts()
        weight_source = f"SWA equal-weight average of {swa.n} epoch snapshots"
    else:
        final_sd = {name: state_to_cpu(getattr(model, name)) for name in TRAINABLE_MODULES}
        weight_source = "final-epoch weights (SWA start epoch not reached)"
    logger.info(f"Finalising checkpoints from: {weight_source}")

    final_config = dict(
        model_config,
        version="v6-Max",
        swa=swa.n > 0,
        swa_snapshots=swa.n,
        swa_start_epoch=swa_start_epoch,
        epochs=epochs,
        optimizer="DirectMLNativeAdamW",
    )
    fp16_state = {
        **{name: {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
           for name, sd in final_sd.items()},
        "training_history": history,
        "model_config": final_config,
    }
    atomic_torch_save(fp16_state, save_path)
    del fp16_state
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(
        f"Final FP16 checkpoint written: {save_path} ({file_size_mb:.2f} MB) "
        f"total training time {train_elapsed / 3600.0:.2f}h"
    )

    fp32_path = save_path.replace(".pt", "_fp32.pt")
    atomic_torch_save(
        {**final_sd, "model_config": final_config, "training_history": history},
        fp32_path,
    )
    logger.info(
        f"Final FP32 checkpoint written: {fp32_path} ({os.path.getsize(fp32_path) / (1024 * 1024):.2f} MB)"
    )
    log_mem("final checkpoints written")

    # Release training state before loading a second encoder for verification
    del trainer, optimizer, scheduler, swa_scheduler, swa, final_sd, train_loader, val_loader
    del train_cached, val_cached
    gc.collect()
    log_mem("training state released")

    # -----------------------------------------------------------------------
    # 11. Verification Inference
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFYING v6-Max RUNTIME INFERENCE VIA ArbiterOmniEngine.from_pretrained()")
    print("=" * 80)
    engine = ArbiterOmniEngine.from_pretrained(save_path, device=device)
    engine.memory_bank = memory_bank

    scenarios = [
        {
            "text": "Dense multi-lane highway, heavy rainfall, vehicle decelerating abruptly 15m ahead.",
            "question": "What is the safest immediate action?",
            "candidates": [
                "decelerate smoothly and increase following distance",
                "proceed at maximum velocity without braking",
                "execute uncontrolled swerve into barrier",
                "turn off all headlights and warning telemetry",
            ],
        },
        {
            "text": "Surgical robot assisting with minimally invasive cardiac procedure. Patient vitals unstable.",
            "question": "What is the correct intervention priority?",
            "candidates": [
                "pause procedure and alert supervising surgeon immediately",
                "continue procedure at increased speed",
                "reduce sedation dosage and proceed",
                "disengage all safety interlocks",
            ],
        },
    ]

    for i, s in enumerate(scenarios):
        res = engine.decide(
            question=s["question"],
            candidates=s["candidates"],
            text=s["text"],
            test_time_deliberate=True,
        )
        print(f"\nScenario {i+1}: {s['question']}")
        print(f"  Decision              : {res.decision}")
        print(f"  Confidence            : {res.confidence * 100:.2f}%")
        print(f"  Entropy               : {res.entropy:.4f} nats")
        print(f"  Speculative Early Exit: {res.speculative_early_exit}")
        print(f"  Conformal Set (95%)   : {res.prediction_set}")
        print(f"  Stability Index       : {res.stability_index:.3f} (Certified: {res.stability_index >= 0.80})")
        print(f"  Escalate System 2     : {res.escalate_system2}")

    final_mem = get_rss_mb()
    logger.info(f"Final Host RSS: {final_mem:.1f} MB (target: <5.5 GB across 8 GB DDR4 pool)")
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArbiterOmni v6-Max [AO-34, AO-35]")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save-path", type=str, default="checkpoints/arbiter_omni_v6_max.pt")
    parser.add_argument("--model-name", type=str, default="ViT-B-16-SigLIP")
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--scoring-dim", type=int, default=768)
    parser.add_argument("--max-spatial-patches", type=int, default=980)
    parser.add_argument("--memory-bank-capacity", type=int, default=100_000)
    parser.add_argument("--moe-layers", type=int, default=4)
    parser.add_argument("--moe-experts", type=int, default=8)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--swa-start-epoch", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--no-int8-cache", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <save-path>.resume (restores weights, optimizer, schedule, SWA, history)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny end-to-end run (48 train / 16 val samples) that exercises every stage, "
                             "including checkpointing, SWA and verification, writing to checkpoints/smoke_v6_max.pt")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="Log a progress line every N training batches (0 disables)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Mirror logs to this file (default: logs/train_v6_max_<timestamp>.log)")
    args = parser.parse_args()

    import os as _os
    log_file = args.log_file or f"logs/train_v6_max_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_file_logging(log_file)

    exit_code = 0
    try:
        train_v6_max(
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
            swa_start_epoch=args.swa_start_epoch,
            device_name=args.device,
            use_mock_data=args.use_mock_data,
            use_int8_cache=not args.no_int8_cache,
            resume=args.resume,
            smoke_test=args.smoke_test,
            log_interval=args.log_interval,
        )
        logger.info("v6-Max training completed successfully")
    except BaseException as e:  # noqa: BLE001 - we must report every failure mode
        logger.exception(f"v6-Max training failed: {type(e).__name__}: {e}")
        exit_code = 1
    finally:
        # os._exit skips interpreter teardown (the DirectML runtime can hang there),
        # so flush everything explicitly and preserve the real exit status.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        _os._exit(exit_code)
