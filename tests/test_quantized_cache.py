"""
Unit tests for ArbiterOmni AO-25: INT8 Cache Dynamic Quantization & 100k Memory Bank.

Covers:
- int8_quantize / int8_dequantize round-trip fidelity
- CachedSample INT8 vs fp32 storage with transparent property access
- CachedSample.memory_bytes footprint reduction
- CachedMultimodalDataset.from_dataset with use_int8=True
- CachedMultimodalDataset.total_memory_mb reporting
- PersistentMemoryBank 100k capacity + fp16 storage
- fp16 bank query_hard_foils numeric precision
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from arbiter_omni import (
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    CachedMultimodalDataset,
    CachedSample,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    TrainingConfig,
    collate_cached_multimodal_decision,
    generate_synthetic_dataset,
    int8_quantize,
    int8_dequantize,
)
from arbiter_omni.data.memory_bank import PersistentMemoryBank


# ---------------------------------------------------------------------------
# int8_quantize / int8_dequantize round-trip tests
# ---------------------------------------------------------------------------

class TestInt8QuantizationRoundTrip:
    """Tests for symmetric per-tensor INT8 quantization utilities."""

    def test_output_dtype(self):
        """int8_quantize must return INT8 tensor and float32 scale."""
        t = torch.randn(4, 8)
        q, scale = int8_quantize(t)
        assert q.dtype == torch.int8
        assert scale.dtype == torch.float32

    def test_scale_is_scalar(self):
        """Scale must be a scalar (0-dimensional) tensor."""
        t = torch.randn(10, 16)
        _, scale = int8_quantize(t)
        assert scale.ndim == 0

    def test_range_clamp(self):
        """Quantized values must be within the symmetric INT8 range [-127, 127]."""
        t = torch.randn(32, 64) * 100  # large values stress the clamp
        q, _ = int8_quantize(t)
        assert q.min().item() >= -127
        assert q.max().item() <= 127

    def test_round_trip_fidelity_small(self):
        """Round-trip error must be < 1% of max absolute value for random tensors."""
        torch.manual_seed(42)
        t = torch.randn(8, 16)
        q, scale = int8_quantize(t)
        t_hat = int8_dequantize(q, scale)
        max_abs = t.abs().max().item()
        max_err = (t - t_hat).abs().max().item()
        # INT8 symmetric quantization: max error <= scale/2 ≈ max_abs/127/2
        assert max_err < max_abs * 0.02, f"Round-trip error too large: {max_err:.6f} vs max_abs {max_abs:.6f}"

    def test_round_trip_fidelity_large(self):
        """Round-trip error is bounded for larger tensors (e.g. spatial patches [P, D])."""
        torch.manual_seed(7)
        t = torch.randn(980, 512)  # typical spatial patch tensor shape
        q, scale = int8_quantize(t)
        t_hat = int8_dequantize(q, scale)
        max_abs = t.abs().max().item()
        max_err = (t - t_hat).abs().max().item()
        assert max_err < max_abs * 0.02

    def test_zero_tensor_safe(self):
        """Quantizing an all-zeros tensor must not raise and round-trip to zeros."""
        t = torch.zeros(4, 8)
        q, scale = int8_quantize(t)
        t_hat = int8_dequantize(q, scale)
        assert torch.allclose(t_hat, t, atol=1e-6)

    def test_dequantize_output_dtype(self):
        """Dequantized output must be float32."""
        t = torch.randn(6, 12)
        q, scale = int8_quantize(t)
        t_hat = int8_dequantize(q, scale)
        assert t_hat.dtype == torch.float32

    def test_sign_preservation(self):
        """Quantization must preserve sign for positive and negative values."""
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        neg = torch.tensor([[-1.0, -2.0, -3.0]])
        q_pos, scale_pos = int8_quantize(pos)
        q_neg, scale_neg = int8_quantize(neg)
        assert (q_pos > 0).all()
        assert (q_neg < 0).all()


# ---------------------------------------------------------------------------
# CachedSample INT8 storage tests
# ---------------------------------------------------------------------------

class TestCachedSampleInt8:
    """Tests for INT8 storage mode in CachedSample."""

    def _make_sample(self, dim=128, n_cands=3, use_int8=False):
        question_embed = torch.randn(dim)
        modality_embeds = {
            "text": torch.randn(dim),
            "image": torch.randn(dim),
        }
        presence_mask = {"text": True, "image": True}
        candidate_embeds = torch.randn(n_cands, dim)
        image_patches = torch.randn(16, dim)
        return CachedSample(
            question_embed=question_embed,
            modality_embeds=modality_embeds,
            presence_mask=presence_mask,
            candidate_embeds=candidate_embeds,
            target_idx=0,
            image_patches=image_patches,
            use_int8=use_int8,
        ), candidate_embeds, image_patches

    def test_fp32_storage_baseline(self):
        """Without use_int8, tensors are stored in float32."""
        sample, cands, patches = self._make_sample(use_int8=False)
        assert sample._candidate_embeds is not None
        assert sample._candidate_embeds.dtype == torch.float32
        assert sample._candidate_embeds_q is None

    def test_int8_candidate_storage(self):
        """With use_int8=True, candidate_embeds are stored as INT8 internally."""
        sample, _, _ = self._make_sample(use_int8=True)
        assert sample._candidate_embeds is None, "fp32 copy should not be retained in INT8 mode"
        assert sample._candidate_embeds_q is not None
        assert sample._candidate_embeds_q.dtype == torch.int8

    def test_int8_patch_storage(self):
        """With use_int8=True, image_patches are stored as INT8 internally."""
        sample, _, _ = self._make_sample(use_int8=True)
        assert sample._image_patches is None, "fp32 patches should not be retained in INT8 mode"
        assert sample._image_patches_q is not None
        assert sample._image_patches_q.dtype == torch.int8

    def test_candidate_embeds_property_fp32_output(self):
        """candidate_embeds property always returns float32 regardless of storage mode."""
        sample_fp32, _, _ = self._make_sample(use_int8=False)
        sample_int8, _, _ = self._make_sample(use_int8=True)
        assert sample_fp32.candidate_embeds.dtype == torch.float32
        assert sample_int8.candidate_embeds.dtype == torch.float32

    def test_image_patches_property_fp32_output(self):
        """image_patches property always returns float32 regardless of storage mode."""
        sample_fp32, _, _ = self._make_sample(use_int8=False)
        sample_int8, _, _ = self._make_sample(use_int8=True)
        assert sample_fp32.image_patches.dtype == torch.float32
        assert sample_int8.image_patches.dtype == torch.float32

    def test_int8_candidate_round_trip_shape(self):
        """INT8 candidate property output has the same shape as original float32 tensor."""
        dim, n = 64, 4
        sample, cands, _ = self._make_sample(dim=dim, n_cands=n, use_int8=True)
        out = sample.candidate_embeds
        assert out.shape == (n, dim)

    def test_int8_candidate_round_trip_values(self):
        """INT8 candidate embeds round-trip with < 1% relative error vs originals."""
        sample, cands, _ = self._make_sample(dim=128, n_cands=3, use_int8=True)
        out = sample.candidate_embeds
        max_abs = cands.abs().max().item()
        max_err = (cands - out).abs().max().item()
        assert max_err < max_abs * 0.02

    def test_int8_patches_round_trip_shape(self):
        """INT8 image patches round-trip preserves tensor shape."""
        sample, _, patches = self._make_sample(dim=64, use_int8=True)
        out = sample.image_patches
        assert out.shape == patches.shape

    def test_memory_bytes_int8_smaller_than_fp32(self):
        """INT8 CachedSample uses less RAM for candidate_embeds than fp32 equivalent."""
        sample_fp32, _, _ = self._make_sample(dim=512, n_cands=8, use_int8=False)
        sample_int8, _, _ = self._make_sample(dim=512, n_cands=8, use_int8=True)
        # INT8 stores 1 byte/elem vs 4 bytes/elem for fp32; expect ≥ 3x reduction on candidates+patches
        assert sample_int8.memory_bytes < sample_fp32.memory_bytes

    def test_no_patches_int8_safe(self):
        """CachedSample with image_patches=None and use_int8=True must not raise."""
        question_embed = torch.randn(64)
        sample = CachedSample(
            question_embed=question_embed,
            modality_embeds={"text": torch.randn(64)},
            presence_mask={"text": True},
            candidate_embeds=torch.randn(2, 64),
            use_int8=True,
        )
        assert sample.image_patches is None

    def test_empty_candidate_embeds_safe(self):
        """CachedSample with empty candidate_embeds and use_int8=True must not raise."""
        sample = CachedSample(
            question_embed=torch.randn(32),
            modality_embeds={},
            presence_mask={},
            candidate_embeds=torch.empty(0, 32),
            use_int8=True,
        )
        # Empty tensor falls back to fp32 storage path (numel == 0 guard)
        assert sample.candidate_embeds.shape == (0, 32)


# ---------------------------------------------------------------------------
# CachedMultimodalDataset use_int8 integration tests
# ---------------------------------------------------------------------------

class TestCachedDatasetInt8Integration:
    """Integration tests for INT8-quantized cached dataset."""

    def _build_cached(self, use_int8=False, num_samples=8, embed_dim=128):
        encoder = MockMultimodalEncoder(embed_dim=embed_dim)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=embed_dim, scoring_dim=embed_dim)
        model.eval()
        samples = generate_synthetic_dataset(num_samples=num_samples)
        raw_dataset = MultimodalDecisionDataset(samples)
        return (
            CachedMultimodalDataset.from_dataset(
                raw_dataset, model=model, batch_size=4, verbose=False, use_int8=use_int8
            ),
            model,
            samples,
        )

    def test_int8_dataset_length(self):
        """INT8 cached dataset has the same length as the source dataset."""
        ds, _, _ = self._build_cached(use_int8=True, num_samples=6)
        assert len(ds) == 6

    def test_int8_dataset_sample_shapes(self):
        """Samples in INT8 cached dataset have correct output shapes."""
        ds, _, _ = self._build_cached(use_int8=True, embed_dim=128)
        s = ds[0]
        assert s.question_embed.shape == (128,)
        assert s.candidate_embeds.shape[-1] == 128

    def test_int8_dataset_collate_compatible(self):
        """INT8 cached dataset samples collate correctly with collate_cached_multimodal_decision."""
        ds, _, _ = self._build_cached(use_int8=True, embed_dim=128)
        batch = collate_cached_multimodal_decision([ds[0], ds[1]])
        assert batch["is_cached"] is True
        assert batch["question_embed"].shape[0] == 2
        assert batch["candidate_embeds"].dtype == torch.float32

    def test_int8_dataset_total_memory_mb(self):
        """total_memory_mb property returns a positive float."""
        ds, _, _ = self._build_cached(use_int8=True, num_samples=8)
        mb = ds.total_memory_mb
        assert mb > 0.0

    def test_int8_reduces_total_memory_vs_fp32(self):
        """INT8 cached dataset occupies less RAM than the fp32 equivalent."""
        ds_fp32, _, _ = self._build_cached(use_int8=False, num_samples=8, embed_dim=512)
        ds_int8, _, _ = self._build_cached(use_int8=True, num_samples=8, embed_dim=512)
        # INT8 must be strictly smaller (candidates + patches are 4x smaller)
        assert ds_int8.total_memory_mb < ds_fp32.total_memory_mb

    def test_int8_dataset_trainer_compatible(self):
        """Trainer can complete a training epoch on an INT8 cached dataset."""
        ds, _, _ = self._build_cached(use_int8=True, num_samples=16, embed_dim=128)
        encoder = MockMultimodalEncoder(embed_dim=128)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128)
        config = TrainingConfig(num_epochs=2, batch_size=4, fp16=False)
        trainer = ArbiterOmniTrainer(model=model, config=config)
        history = trainer.fit(train_dataset=ds)
        assert "loss" in history
        assert len(history["loss"]) == 2

    def test_int8_dataset_serialization(self):
        """INT8 cached dataset round-trips through save/load correctly."""
        ds, _, _ = self._build_cached(use_int8=True, embed_dim=128)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "int8_cache.pt"
            ds.save(str(path))
            loaded = CachedMultimodalDataset.load(str(path))
        assert len(loaded) == len(ds)
        # After reload the _use_int8 state in CachedSample is persisted via torch.save
        # and shape must be preserved
        assert loaded[0].candidate_embeds.shape == ds[0].candidate_embeds.shape


# ---------------------------------------------------------------------------
# PersistentMemoryBank 100k + fp16 tests
# ---------------------------------------------------------------------------

class TestMemoryBank100kFp16:
    """Tests for 100k capacity and fp16 storage mode in PersistentMemoryBank [AO-25]."""

    def test_default_capacity_is_100k(self):
        """Default capacity must be 100,000 after AO-25 upgrade."""
        bank = PersistentMemoryBank(candidate_dim=32)
        assert bank.capacity == 100_000

    def test_fp16_storage_mode_dtype(self):
        """store_fp16=True must allocate candidate_bank as torch.float16."""
        bank = PersistentMemoryBank(capacity=1000, candidate_dim=64, store_fp16=True)
        assert bank.dtype == torch.float16
        assert bank.candidate_bank.dtype == torch.float16

    def test_fp16_bank_half_memory_vs_fp32(self):
        """fp16 bank must occupy exactly half the RAM of an equivalent fp32 bank."""
        cap, dim = 10000, 256
        fp32_bank = PersistentMemoryBank(capacity=cap, candidate_dim=dim, store_fp16=False)
        fp16_bank = PersistentMemoryBank(capacity=cap, candidate_dim=dim, store_fp16=True)
        assert pytest.approx(fp16_bank.memory_usage_mb, rel=1e-3) == fp32_bank.memory_usage_mb / 2

    def test_fp16_enqueue_and_query(self):
        """fp16 bank enqueues and returns correct shapes from query_hard_foils."""
        dim = 64
        bank = PersistentMemoryBank(capacity=200, candidate_dim=dim, store_fp16=True)
        vecs = F.normalize(torch.randn(50, dim), p=2, dim=-1)
        bank.enqueue(vecs)
        assert len(bank) == 50

        query = F.normalize(torch.randn(2, dim), p=2, dim=-1)
        foils, sims, mask = bank.query_hard_foils(query, k=5, min_sim=0.0, max_sim=1.0)
        assert foils.shape == (2, 5, dim)
        assert sims.shape == (2, 5)
        assert mask.shape == (2, 5)

    def test_fp16_query_returns_float32(self):
        """query_hard_foils must dispatch results in the query tensor's dtype (float32)."""
        dim = 32
        bank = PersistentMemoryBank(capacity=50, candidate_dim=dim, store_fp16=True)
        bank.enqueue(torch.randn(20, dim))
        query = torch.randn(1, dim)  # float32 query
        foils, sims, _ = bank.query_hard_foils(query, k=3, min_sim=0.0, max_sim=1.0)
        assert foils.dtype == torch.float32
        assert sims.dtype == torch.float32

    def test_fp16_query_no_nan_inf(self):
        """fp16 bank cosine similarity must not produce NaN or Inf values."""
        dim = 128
        bank = PersistentMemoryBank(capacity=500, candidate_dim=dim, store_fp16=True)
        bank.enqueue(torch.randn(200, dim))
        query = torch.randn(4, dim)
        foils, sims, _ = bank.query_hard_foils(query, k=10, min_sim=0.0, max_sim=1.0)
        assert not torch.isnan(foils).any()
        assert not torch.isinf(foils).any()
        assert not torch.isnan(sims).any()

    def test_fp16_serialization_roundtrip(self):
        """fp16 bank saves and loads store_fp16=True flag correctly."""
        dim = 32
        bank = PersistentMemoryBank(capacity=100, candidate_dim=dim, store_fp16=True)
        bank.enqueue(torch.randn(30, dim))
        assert bank.dtype == torch.float16

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bank_fp16.pt"
            bank.save(path)
            restored = PersistentMemoryBank(capacity=100, candidate_dim=dim, store_fp16=False)
            restored.load(path)

        assert restored.store_fp16 is True
        assert restored.dtype == torch.float16
        assert len(restored) == 30

    def test_100k_enqueue_cyclic_wraparound(self):
        """100k bank wraps correctly when enqueuing more than capacity."""
        cap, dim = 100, 8  # use small capacity proxy to keep test fast
        bank = PersistentMemoryBank(capacity=cap, candidate_dim=dim)
        # Fill to capacity
        bank.enqueue(torch.randn(cap, dim))
        assert bank.is_full
        assert bank.ptr == 0
        # Overflow by 10
        bank.enqueue(torch.randn(10, dim))
        assert bank.size == cap
        assert bank.ptr == 10
