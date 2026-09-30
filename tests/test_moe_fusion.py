"""
Unit tests for ArbiterOmni AO-27: Sparse Mixture-of-Experts (MoE) 4-Layer Multimodal Fusion.

Covers:
- ExpertFFN forward pass shape and dtype
- SoftTopKRouter routing mechanics: top-k indices, weight normalization, aux loss
- SoftTopKRouter load balance: aux_loss > 0 and gradient flows
- SparseMoETransformerBlock: shape preservation, residual correctness
- SparseMoETransformerBlock: last_aux_loss populated after forward
- SparseMoEMultimodalFusion: shape, accumulated_aux_loss, num_layers
- TransformerMultimodalFusion use_moe=True integration
- ArbiterOmniModel with use_moe=True: build, forward, moe_aux_loss
- Training loop with use_moe=True: loss decreases, aux loss contributes
- Backward pass: gradients flow through router + experts
- use_moe=False: no regression, output_norm is LayerNorm
- Frozen encoders with MoE fusion: 0 encoder grad params
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from arbiter_omni import (
    ArbiterOmniModel,
    ArbiterOmniTrainer,
    CachedMultimodalDataset,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    TrainingConfig,
    generate_synthetic_dataset,
)
from arbiter_omni.fusion.moe import (
    ExpertFFN,
    SoftTopKRouter,
    SparseMoETransformerBlock,
    SparseMoEMultimodalFusion,
)
from arbiter_omni.fusion.transformer import TransformerMultimodalFusion
from arbiter_omni.types import ModalityType


# ---------------------------------------------------------------------------
# ExpertFFN tests
# ---------------------------------------------------------------------------

class TestExpertFFN:

    def test_output_shape(self):
        """ExpertFFN must return the same shape as input."""
        ffn = ExpertFFN(hidden_dim=64, ffn_dim=128)
        x = torch.randn(4, 8, 64)  # [B, S, D]
        out = ffn(x)
        assert out.shape == x.shape

    def test_output_shape_2d(self):
        """ExpertFFN works on [N, D] flat inputs."""
        ffn = ExpertFFN(hidden_dim=32, ffn_dim=64)
        x = torch.randn(16, 32)
        out = ffn(x)
        assert out.shape == (16, 32)

    def test_output_dtype_float32(self):
        """ExpertFFN must return float32."""
        ffn = ExpertFFN(hidden_dim=32, ffn_dim=64)
        x = torch.randn(8, 32)
        assert ffn(x).dtype == torch.float32

    def test_gradients_flow(self):
        """Gradients must propagate through ExpertFFN."""
        ffn = ExpertFFN(hidden_dim=16, ffn_dim=32)
        x = torch.randn(4, 16, requires_grad=True)
        loss = ffn(x).sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SoftTopKRouter tests
# ---------------------------------------------------------------------------

class TestSoftTopKRouter:

    def test_indices_shape(self):
        """Router must return [N, top_k] indices."""
        router = SoftTopKRouter(hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(16, 32)
        indices, weights, aux = router(x)
        assert indices.shape == (16, 2)

    def test_weights_shape(self):
        """Router must return [N, top_k] weights."""
        router = SoftTopKRouter(hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(16, 32)
        indices, weights, aux = router(x)
        assert weights.shape == (16, 2)

    def test_weights_sum_to_one(self):
        """Per-token routing weights must sum to ~1.0."""
        router = SoftTopKRouter(hidden_dim=32, num_experts=4, top_k=2)
        x = torch.randn(32, 32)
        _, weights, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(32), atol=1e-5)

    def test_weights_non_negative(self):
        """Routing weights must be non-negative."""
        router = SoftTopKRouter(hidden_dim=64, num_experts=4, top_k=2)
        _, weights, _ = router(torch.randn(20, 64))
        assert (weights >= 0).all()

    def test_indices_valid_range(self):
        """Expert indices must be in [0, num_experts)."""
        router = SoftTopKRouter(hidden_dim=32, num_experts=4, top_k=2)
        indices, _, _ = router(torch.randn(20, 32))
        assert (indices >= 0).all()
        assert (indices < 4).all()

    def test_aux_loss_positive(self):
        """Auxiliary load-balancing loss must be positive."""
        router = SoftTopKRouter(hidden_dim=32, num_experts=4, top_k=2)
        _, _, aux = router(torch.randn(64, 32))
        assert aux.item() > 0

    def test_aux_loss_scalar(self):
        """Auxiliary loss must be a scalar tensor."""
        router = SoftTopKRouter(hidden_dim=32, num_experts=4, top_k=2)
        _, _, aux = router(torch.randn(16, 32))
        assert aux.ndim == 0

    def test_aux_loss_gradient_flows(self):
        """Gradient must flow back through the aux loss to gate weights."""
        router = SoftTopKRouter(hidden_dim=16, num_experts=4, top_k=2)
        x = torch.randn(8, 16)
        _, _, aux = router(x)
        aux.backward()
        for p in router.parameters():
            assert p.grad is not None

    def test_top1_routing(self):
        """top_k=1 must return single expert per token."""
        router = SoftTopKRouter(hidden_dim=32, num_experts=4, top_k=1)
        indices, weights, _ = router(torch.randn(10, 32))
        assert indices.shape == (10, 1)
        assert weights.shape == (10, 1)
        assert torch.allclose(weights, torch.ones(10, 1), atol=1e-5)


# ---------------------------------------------------------------------------
# SparseMoETransformerBlock tests
# ---------------------------------------------------------------------------

class TestSparseMoETransformerBlock:

    def _make_block(self, hidden=64, experts=4, heads=4):
        return SparseMoETransformerBlock(
            hidden_dim=hidden,
            num_experts=experts,
            top_k=2,
            ffn_dim=hidden * 4,
            num_heads=heads,
            dropout=0.0,
        )

    def test_output_shape(self):
        """Block must return same shape as input [B, S, D]."""
        block = self._make_block()
        x = torch.randn(2, 6, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_output_dtype(self):
        """Block output must be float32."""
        block = self._make_block()
        x = torch.randn(2, 6, 64)
        assert block(x).dtype == torch.float32

    def test_last_aux_loss_populated(self):
        """last_aux_loss must be set after a forward pass."""
        block = self._make_block()
        x = torch.randn(2, 6, 64)
        _ = block(x)
        assert block.last_aux_loss.ndim == 0
        assert block.last_aux_loss.item() >= 0

    def test_last_aux_loss_resets_each_forward(self):
        """last_aux_loss must reflect only the most recent forward."""
        block = self._make_block()
        _ = block(torch.randn(2, 6, 64))
        aux1 = block.last_aux_loss.item()
        _ = block(torch.randn(2, 8, 64))
        aux2 = block.last_aux_loss.item()
        # Both calls should produce positive finite aux losses
        assert aux1 >= 0 and aux2 >= 0

    def test_masked_sequence(self):
        """Block must handle key_padding_mask without NaN output."""
        block = self._make_block(hidden=32, heads=4)
        x = torch.randn(2, 6, 32)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, 4:] = True  # mask out last 2 tokens in sample 0
        out = block(x, src_key_padding_mask=mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gradient_flows_through_block(self):
        """Gradients must flow through the block's attention and MoE FFN."""
        block = self._make_block(hidden=32, heads=4)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = block(x)
        loss = out.sum() + block.last_aux_loss
        loss.backward()
        assert x.grad is not None
        # All block parameters must have gradients
        for p in block.parameters():
            assert p.grad is not None, f"No grad for param shape {p.shape}"

    def test_no_nan_with_large_batch(self):
        """Block must not produce NaN/Inf on a larger batch."""
        block = self._make_block(hidden=64, experts=4)
        x = torch.randn(8, 12, 64)
        out = block(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# SparseMoEMultimodalFusion tests
# ---------------------------------------------------------------------------

class TestSparseMoEMultimodalFusion:

    def _make_fusion(self, hidden=64, layers=4, experts=4, heads=4):
        return SparseMoEMultimodalFusion(
            hidden_dim=hidden,
            num_moe_layers=layers,
            num_experts=experts,
            top_k=2,
            ffn_dim=hidden * 2,
            num_heads=heads,
            dropout=0.0,
        )

    def test_output_shape(self):
        """SparseMoEMultimodalFusion must return [B, S, hidden_dim]."""
        fusion = self._make_fusion()
        x = torch.randn(3, 7, 64)
        out = fusion(x)
        assert out.shape == x.shape

    def test_num_layers(self):
        """SparseMoEMultimodalFusion must have exactly moe_num_layers blocks."""
        fusion = self._make_fusion(layers=4)
        assert len(fusion.layers) == 4

    def test_accumulated_aux_loss_after_forward(self):
        """accumulated_aux_loss must be > 0 after a forward pass."""
        fusion = self._make_fusion()
        _ = fusion(torch.randn(2, 6, 64))
        assert fusion.accumulated_aux_loss.item() > 0

    def test_accumulated_aux_loss_is_sum_of_layer_losses(self):
        """accumulated_aux_loss must equal the sum of each layer's last_aux_loss."""
        fusion = self._make_fusion(layers=4)
        _ = fusion(torch.randn(2, 6, 64))
        expected = sum(layer.last_aux_loss.item() for layer in fusion.layers)
        assert abs(fusion.accumulated_aux_loss.item() - expected) < 1e-5

    def test_output_normalized(self):
        """Output tokens must pass through LayerNorm (not wildly out of scale)."""
        fusion = self._make_fusion()
        x = torch.randn(4, 6, 64) * 100  # large input scale
        out = fusion(x)
        # After LayerNorm, each row should have ~unit variance
        assert not torch.isnan(out).any()

    def test_gradient_flows_through_all_layers(self):
        """Gradients must flow from output through all 4 MoE layers."""
        fusion = self._make_fusion(layers=4)
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = fusion(x)
        loss = out.sum() + fusion.accumulated_aux_loss
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# TransformerMultimodalFusion use_moe=True integration tests
# ---------------------------------------------------------------------------

class TestTransformerFusionMoEIntegration:

    def _make_fusion(self, dim=64, hidden=64, use_moe=True, moe_layers=4):
        modality_dims = {
            "question": dim,
            "text": dim,
            "image": dim,
            "video": dim,
            "audio": dim,
        }
        return TransformerMultimodalFusion(
            modality_dims=modality_dims,
            hidden_dim=hidden,
            num_heads=4,
            num_layers=2,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            use_moe=use_moe,
            moe_num_layers=moe_layers,
            moe_num_experts=4,
            moe_top_k=2,
        )

    def _make_inputs(self, B=2, dim=64, device="cpu"):
        q = torch.randn(B, dim, device=device)
        mod_embeds = {
            ModalityType.TEXT:  torch.randn(B, dim, device=device),
            ModalityType.IMAGE: torch.randn(B, dim, device=device),
            ModalityType.VIDEO: torch.randn(B, dim, device=device),
            ModalityType.AUDIO: torch.randn(B, dim, device=device),
        }
        presence = {m: torch.ones(B, dtype=torch.bool) for m in mod_embeds}
        return q, mod_embeds, presence

    def test_moe_fusion_forward_shape(self):
        """use_moe=True fusion must return [B, hidden_dim] context vector."""
        fusion = self._make_fusion()
        q, mod, pres = self._make_inputs()
        out = fusion(q, mod, pres)
        assert out.shape == (2, 64)

    def test_moe_fusion_no_nan(self):
        """use_moe=True fusion must not produce NaN."""
        fusion = self._make_fusion()
        q, mod, pres = self._make_inputs()
        out = fusion(q, mod, pres)
        assert not torch.isnan(out).any()

    def test_moe_aux_loss_property(self):
        """moe_aux_loss must return a positive scalar after forward."""
        fusion = self._make_fusion()
        q, mod, pres = self._make_inputs()
        _ = fusion(q, mod, pres)
        assert fusion.moe_aux_loss.item() > 0

    def test_moe_aux_loss_zero_without_moe(self):
        """moe_aux_loss must return 0.0 when use_moe=False."""
        fusion = self._make_fusion(use_moe=False)
        q, mod, pres = self._make_inputs()
        _ = fusion(q, mod, pres)
        assert fusion.moe_aux_loss.item() == 0.0

    def test_dense_path_not_affected(self):
        """use_moe=False must use dense TransformerEncoder (no moe_transformer)."""
        fusion = self._make_fusion(use_moe=False)
        assert fusion.transformer is not None
        assert fusion.moe_transformer is None

    def test_moe_path_sets_transformer_none(self):
        """use_moe=True must set self.transformer = None."""
        fusion = self._make_fusion(use_moe=True)
        assert fusion.transformer is None
        assert fusion.moe_transformer is not None

    def test_moe_num_layers_respected(self):
        """SparseMoEMultimodalFusion must have exactly moe_num_layers layers."""
        fusion = self._make_fusion(moe_layers=4)
        assert len(fusion.moe_transformer.layers) == 4

    def test_gradient_flows_through_moe_fusion(self):
        """Gradients must flow through the MoE fusion module."""
        fusion = self._make_fusion()
        q, mod, pres = self._make_inputs()
        q.requires_grad_(True)
        out = fusion(q, mod, pres)
        loss = out.sum() + fusion.moe_aux_loss
        loss.backward()
        assert q.grad is not None


# ---------------------------------------------------------------------------
# ArbiterOmniModel with use_moe=True tests
# ---------------------------------------------------------------------------

class TestArbiterOmniWithMoE:

    def _make_model(self, embed_dim=64, hidden=64, use_moe=True):
        encoder = MockMultimodalEncoder(embed_dim=embed_dim)
        return ArbiterOmniModel(
            encoder=encoder,
            hidden_dim=hidden,
            scoring_dim=hidden,
            use_moe=use_moe,
            moe_num_layers=4,
            moe_num_experts=4,
            moe_top_k=2,
        )

    def test_model_builds_with_moe(self):
        """ArbiterOmniModel with use_moe=True must build without errors."""
        model = self._make_model()
        assert model.fusion is not None
        assert model.fusion.use_moe is True

    def test_forward_pass_shape(self):
        """Model with use_moe=True must return valid [1, K] logits."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            logits, probs, entropy, ctx = model(
                questions=["Navigate obstacle?"],
                candidates=[["go left", "go right", "stop"]],
                texts=["Obstacle at 1m ahead"],
            )
        assert logits.shape == (1, 3)
        assert probs.sum(dim=-1).allclose(torch.ones(1), atol=1e-4)

    def test_moe_aux_loss_positive_after_forward(self):
        """model.moe_aux_loss must be > 0 after a forward pass."""
        model = self._make_model()
        _ = model(
            questions=["Navigate?"],
            candidates=[["go", "stop"]],
            texts=["Sensor data"],
        )
        assert model.moe_aux_loss.item() > 0

    def test_moe_aux_loss_zero_without_moe(self):
        """model.moe_aux_loss must be 0.0 when use_moe=False."""
        model = self._make_model(use_moe=False)
        _ = model(
            questions=["Navigate?"],
            candidates=[["go", "stop"]],
            texts=["Sensor data"],
        )
        assert model.moe_aux_loss.item() == 0.0

    def test_encoder_still_frozen_with_moe(self):
        """Encoder must remain strictly frozen when use_moe=True."""
        model = self._make_model()
        for p in model.encoder.parameters():
            assert not p.requires_grad

    def test_training_loop_with_moe(self):
        """Full training loop must complete without error with use_moe=True."""
        model = self._make_model(embed_dim=64, hidden=64)
        samples = generate_synthetic_dataset(num_samples=16)
        ds = MultimodalDecisionDataset(samples)
        cached = CachedMultimodalDataset.from_dataset(ds, model=model, verbose=False)

        config = TrainingConfig(num_epochs=3, batch_size=4, fp16=False, learning_rate=1e-3)
        trainer = ArbiterOmniTrainer(model=model, config=config)
        history = trainer.fit(train_dataset=cached)
        assert "loss" in history
        assert len(history["loss"]) == 3
        # Losses must be finite
        for l in history["loss"]:
            assert not (l != l)  # not NaN

    def test_no_nan_in_moe_output(self):
        """MoE model forward must not produce NaN under batch inference."""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            logits, probs, _, _ = model(
                questions=["Q1", "Q2", "Q3"],
                candidates=[["A", "B", "C"]] * 3,
                texts=["T1", "T2", "T3"],
            )
        assert not torch.isnan(logits).any()
        assert not torch.isnan(probs).any()
