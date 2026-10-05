"""
Unit tests for ArbiterOmni AO-26: SigLIP-SO400M Foundation Backbone Integration.

Covers:
- Pretrained tag auto-resolution for SigLIP and SO400M variants
- CPU offload flag and is_cpu_offloaded property
- _encoder_device routing in encode_text / encode_image / encode_image_patches
- Backbone output dimension resolution (1152 for SO400M)
- ArbiterOmniModel with 1152-dim encoder (dynamic projection sizing)
- Full e2e pipeline: encode → fuse → score with large-dim (1152-dim) embeddings
- cpu_offload_encoder=True forced offload regardless of param count
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from unittest.mock import MagicMock, patch

from arbiter_omni import (
    ArbiterOmniModel,
    MockMultimodalEncoder,
    MultimodalDecisionDataset,
    TrainingConfig,
    ArbiterOmniTrainer,
    generate_synthetic_dataset,
    CachedMultimodalDataset,
)
from arbiter_omni.encoders.openclip import (
    OpenCLIPMultimodalEncoder,
    _SO400M_NAME,
    _SO400M_DIM,
    _CPU_OFFLOAD_PARAM_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_preprocess():
    """Returns a preprocess callable that converts any PIL image to a (3, 224, 224) tensor."""
    def preprocess(img):
        return torch.zeros(3, 224, 224)
    return preprocess


def _make_fake_tokenizer():
    """Returns a tokenizer callable that converts text to (B, 77) token ids."""
    def tokenizer(texts):
        return torch.zeros(len(texts), 77, dtype=torch.long)
    return tokenizer


def _build_mock_oc_model(param_count: int, out_dim: int):
    """Builds a minimal mock open_clip model with real torch parameter tensors.

    Uses FakeParam (subclass of nn.Parameter) so param counting works correctly
    without allocating gigabytes of memory for 200M+ parameter mocks.
    """
    import torch.nn as nn

    class FakeParam(nn.Parameter):
        def __new__(cls, data, count):
            obj = super().__new__(cls, data, requires_grad=False)
            obj._param_count = count
            return obj

        def numel(self):
            return self._param_count

    class _MockOCModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Lightweight parameter that reports param_count without allocating memory
            self.backbone = FakeParam(torch.zeros(1), param_count)
            self.visual = MagicMock()
            self.visual.output_dim = out_dim

        def encode_text(self, tokens):
            B = tokens.shape[0]
            return F.normalize(torch.randn(B, out_dim), dim=-1)

        def encode_image(self, batch):
            B = batch.shape[0]
            return F.normalize(torch.randn(B, out_dim), dim=-1)

        def parameters(self, recurse=True):
            return super().parameters(recurse=recurse)

    return _MockOCModel()


def _build_encoder(
    model_name="ViT-B-32",
    pretrained=None,
    param_count=86_000_000,
    out_dim=512,
    device=None,
    cpu_offload_encoder=False,
):
    """Builds an OpenCLIPMultimodalEncoder with open_clip patched to avoid real downloads."""
    if device is None:
        device = torch.device("cpu")

    mock_model = _build_mock_oc_model(param_count, out_dim)
    fake_preprocess = _make_fake_preprocess()
    fake_tokenizer = _make_fake_tokenizer()

    mock_oc = MagicMock()
    mock_oc.create_model_and_transforms.return_value = (mock_model, None, fake_preprocess)
    mock_oc.get_tokenizer.return_value = fake_tokenizer

    with patch.dict("sys.modules", {"open_clip": mock_oc}):
        enc = OpenCLIPMultimodalEncoder(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            cpu_offload_encoder=cpu_offload_encoder,
        )

    # Store the mock for call inspection
    enc._mock_oc = mock_oc
    return enc


# ---------------------------------------------------------------------------
# Pretrained tag resolution tests
# ---------------------------------------------------------------------------

class TestSO400MPretainedTagResolution:
    """Validates that the constructor auto-resolves correct pretrained tags."""

    def test_siglip_b16_resolves_webli(self):
        """ViT-B-16-SigLIP must auto-resolve to pretrained='webli'."""
        enc = _build_encoder("ViT-B-16-SigLIP", out_dim=768)
        assert enc.pretrained == "webli"

    def test_so400m_resolves_webli(self):
        """ViT-SO400M-14-SigLIP-384 must auto-resolve to pretrained='webli'."""
        enc = _build_encoder(_SO400M_NAME, param_count=_CPU_OFFLOAD_PARAM_THRESHOLD + 1, out_dim=1152)
        assert enc.pretrained == "webli"

    def test_siglip_overrides_laion_tag_to_webli(self):
        """SigLIP with an explicit laion tag must still be overridden to 'webli'."""
        enc = _build_encoder("ViT-B-16-SigLIP", pretrained="laion400m_e31", out_dim=768)
        assert enc.pretrained == "webli"

    def test_vit_b32_default_laion_tag(self):
        """Standard ViT-B-32 must default to laion2b_s34b_b79k."""
        enc = _build_encoder("ViT-B-32", out_dim=512)
        assert "laion" in enc.pretrained.lower()

    def test_vit_b16_resolves_b88k_tag(self):
        """ViT-B-16 must auto-resolve to laion2b_s34b_b88k."""
        enc = _build_encoder("ViT-B-16", out_dim=512)
        assert enc.pretrained == "laion2b_s34b_b88k"

    def test_vit_l14_resolves_b82k_tag(self):
        """ViT-L-14 must auto-resolve to laion2b_s32b_b82k."""
        enc = _build_encoder("ViT-L-14", out_dim=768)
        assert enc.pretrained == "laion2b_s32b_b82k"


# ---------------------------------------------------------------------------
# CPU offload and device routing tests
# ---------------------------------------------------------------------------

class TestCPUOffloadLogic:
    """Tests for automatic CPU offloading of large backbones (AO-26)."""

    def test_small_model_not_offloaded(self):
        """86M param model (ViT-B-32 size) must NOT trigger CPU offload."""
        enc = _build_encoder(param_count=86_000_000, out_dim=512)
        assert not enc.is_cpu_offloaded

    def test_large_model_auto_offloaded(self):
        """435M param model (SO400M size) must trigger automatic CPU offload."""
        enc = _build_encoder(
            model_name=_SO400M_NAME,
            param_count=_CPU_OFFLOAD_PARAM_THRESHOLD + 1,
            out_dim=1152,
        )
        assert enc.is_cpu_offloaded
        assert enc.backbone_device == torch.device("cpu")

    def test_forced_cpu_offload_small_model(self):
        """cpu_offload_encoder=True must force CPU offload regardless of param count."""
        enc = _build_encoder(param_count=86_000_000, cpu_offload_encoder=True)
        assert enc.is_cpu_offloaded
        assert enc.backbone_device == torch.device("cpu")

    def test_encoder_param_count_property(self):
        """encoder_param_count must accurately reflect backbone param count."""
        enc = _build_encoder(param_count=86_000_000)
        # The mock model has one 86M-element parameter tensor
        assert enc.encoder_param_count == 86_000_000

    def test_large_encoder_param_count(self):
        """encoder_param_count for SO400M-sized model must be above 200M threshold."""
        enc = _build_encoder(
            model_name=_SO400M_NAME,
            param_count=_CPU_OFFLOAD_PARAM_THRESHOLD + 1,
            out_dim=1152,
        )
        assert enc.encoder_param_count > _CPU_OFFLOAD_PARAM_THRESHOLD


# ---------------------------------------------------------------------------
# Dimension resolution tests
# ---------------------------------------------------------------------------

class TestSO400MDimensionResolution:
    """Tests that SO400M always resolves to 1152-dim regardless of open_clip output_dim."""

    def test_so400m_always_1152_dim(self):
        """SO400M encoder must report 1152-dim regardless of what open_clip reports."""
        enc = _build_encoder(
            model_name=_SO400M_NAME,
            param_count=_CPU_OFFLOAD_PARAM_THRESHOLD + 1,
            out_dim=768,  # intentionally wrong to test SO400M dim override
        )
        assert enc.text_dim == _SO400M_DIM  # 1152
        assert enc.image_dim == _SO400M_DIM
        assert enc.video_dim == _SO400M_DIM

    def test_so400m_dim_constant_is_1152(self):
        """_SO400M_DIM constant must be 1152."""
        assert _SO400M_DIM == 1152

    def test_siglip_b16_uses_openclip_output_dim(self):
        """Non-SO400M SigLIP models use open_clip's reported output_dim."""
        enc = _build_encoder("ViT-B-16-SigLIP", out_dim=768)
        assert enc.text_dim == 768

    def test_vit_b32_512_dim(self):
        """Standard ViT-B-32 must report 512-dim via open_clip's output_dim."""
        enc = _build_encoder("ViT-B-32", out_dim=512)
        assert enc.text_dim == 512


# ---------------------------------------------------------------------------
# ArbiterOmniModel dynamic projection sizing for 1152-dim inputs
# ---------------------------------------------------------------------------

class TestArbiterOmniWith1152DimEncoder:
    """Tests that ArbiterOmniModel correctly sizes projections for 1152-dim SO400M embeddings."""

    def test_model_builds_with_1152_dim_mock_encoder(self):
        """ArbiterOmniModel must build without error when encoder outputs 1152-dim."""
        encoder = MockMultimodalEncoder(embed_dim=1152)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, scoring_dim=256)
        assert model.fusion is not None
        assert model.decision_head is not None

    def test_fusion_projection_accepts_1152_dim(self):
        """Question modality projection in fusion must accept 1152-dim inputs."""
        encoder = MockMultimodalEncoder(embed_dim=1152)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, scoring_dim=256)
        q = torch.randn(2, 1152)
        out = model.fusion.projections["question"](q)
        assert out.shape == (2, 256)

    def test_decision_head_candidate_projection_for_1152(self):
        """DynamicDecisionHead candidate projection must accept 1152-dim inputs."""
        encoder = MockMultimodalEncoder(embed_dim=1152)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, scoring_dim=256)
        ctx = torch.randn(2, 256)
        cands = torch.randn(2, 3, 1152)
        logits, probs, entropy = model.decision_head(context_embed=ctx, candidate_embeds=cands)
        assert logits.shape == (2, 3)
        assert probs.shape == (2, 3)
        assert not torch.isnan(logits).any()

    def test_e2e_forward_pass_1152_dim(self):
        """Full model forward pass with 1152-dim MockMultimodalEncoder must produce valid outputs."""
        encoder = MockMultimodalEncoder(embed_dim=1152)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, scoring_dim=256)
        model.eval()
        with torch.no_grad():
            logits, probs, entropy, ctx = model(
                questions=["What should the robot do?"],
                candidates=[["move forward", "stop", "turn left"]],
                texts=["Obstacle detected at 2m."],
            )
        assert logits.shape == (1, 3)
        assert probs.sum(dim=-1).allclose(torch.ones(1), atol=1e-4)
        assert not torch.isnan(logits).any()

    def test_training_loop_1152_dim_encoder(self):
        """Complete training epoch must run without error with 1152-dim encoder."""
        encoder = MockMultimodalEncoder(embed_dim=1152)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, scoring_dim=256)
        samples = generate_synthetic_dataset(num_samples=16)
        raw_dataset = MultimodalDecisionDataset(samples)
        cached = CachedMultimodalDataset.from_dataset(raw_dataset, model=model, verbose=False)

        config = TrainingConfig(num_epochs=2, batch_size=4, fp16=False, learning_rate=1e-3)
        trainer = ArbiterOmniTrainer(model=model, config=config)
        history = trainer.fit(train_dataset=cached)
        assert "loss" in history
        assert len(history["loss"]) == 2

    def test_encoder_gradients_frozen_at_1152_dim(self):
        """All encoder parameters must be frozen (requires_grad=False) regardless of embed dim."""
        encoder = MockMultimodalEncoder(embed_dim=1152)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, scoring_dim=256)
        encoder_params = list(model.encoder.parameters())
        frozen_count = sum(1 for p in encoder_params if not p.requires_grad)
        assert frozen_count == len(encoder_params), (
            f"Expected all {len(encoder_params)} encoder params frozen, "
            f"but only {frozen_count} are frozen"
        )

    def test_trainable_params_exclude_encoder(self):
        """ArbiterOmniModel.trainable_parameters() must not include encoder parameters."""
        encoder = MockMultimodalEncoder(embed_dim=1152)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=256, scoring_dim=256)
        trainable_ids = {id(p) for p in model.trainable_parameters()}
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        overlap = trainable_ids & encoder_ids
        assert len(overlap) == 0, f"Encoder params leaked into trainable: {len(overlap)} params overlap"


# ---------------------------------------------------------------------------
# Integration: encode_text / encode_image with _encoder_device routing
# ---------------------------------------------------------------------------

class TestEncoderDeviceRouting:
    """Tests that encode_text / encode_image route correctly to _encoder_device."""

    def test_encode_text_returns_correct_shape(self):
        """encode_text must return [B, text_dim] tensor."""
        enc = _build_encoder(out_dim=512)
        out = enc.encode_text(["hello", "world"])
        assert out.shape == (2, 512)

    def test_encode_text_returns_float_tensor(self):
        """encode_text must return a float32 tensor."""
        enc = _build_encoder(out_dim=512)
        out = enc.encode_text(["hello"])
        assert out.dtype == torch.float32

    def test_encode_text_with_cpu_offload_device_correct(self):
        """encode_text with cpu_offload must still return tensor on self.device."""
        enc = _build_encoder(cpu_offload_encoder=True, out_dim=512)
        out = enc.encode_text(["hello"])
        assert out.device == enc.device

    def test_encode_image_returns_correct_shape(self):
        """encode_image returns [B, dim] tensor."""
        enc = _build_encoder(out_dim=512)
        imgs = [Image.new("RGB", (224, 224)) for _ in range(3)]
        out = enc.encode_image(imgs)
        assert out.shape == (3, 512)

    def test_encode_image_with_cpu_offload_returns_self_device(self):
        """encode_image with cpu_offload returns features on self.device."""
        enc = _build_encoder(cpu_offload_encoder=True, out_dim=512)
        imgs = [Image.new("RGB", (224, 224)) for _ in range(2)]
        out = enc.encode_image(imgs)
        assert out.device == enc.device

    def test_backbone_device_property_cpu(self):
        """backbone_device must return cpu for CPU-offloaded encoder."""
        enc = _build_encoder(cpu_offload_encoder=True)
        assert enc.backbone_device == torch.device("cpu")

    def test_backbone_device_small_model(self):
        """backbone_device for a small model on cpu device must return cpu."""
        enc = _build_encoder(param_count=86_000_000)
        # device=cpu in test, so small model stays on cpu
        assert enc.backbone_device.type == "cpu"
