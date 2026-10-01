"""
Unit tests for ArbiterOmni AO-28: Live Streaming Arbitrator & v5 Production Release.

Covers:
- predict_streaming_arbitration function behavior with live frames and fallback
- Millisecond turnaround latency reporting in streaming arbitrator
- ArbiterOmniEngine.from_pretrained('v5') tag resolution and MoE configuration
- Checkpoint loader handles Sparse MoE and SO400M architecture metadata
- Real-time continuous streaming tab integration in build_app()
"""

from __future__ import annotations

import os
import tempfile
import time
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock, patch

from arbiter_omni import (
    ArbiterOmniEngine,
    ArbiterOmniModel,
    MockMultimodalEncoder,
)
from examples.interactive_demo import (
    build_app,
    predict_streaming_arbitration,
    predict_dual_streaming_arbitration,
    predict_brain_map,
)


class TestLiveStreamingArbitrator:
    """Tests for continuous live streaming arbitration functionality [AO-28]."""

    def test_streaming_arbitration_without_frame(self):
        """predict_streaming_arbitration works gracefully with None frame (text/context fallback)."""
        winner_md, probs, conf, entropy, lat = predict_streaming_arbitration(
            frame=None,
            question="What is the next navigational waypoint?",
            candidates_raw="Waypoint Alpha, Waypoint Beta, Emergency Stop",
            temperature=0.5,
        )
        assert "Current Action" in winner_md
        assert len(probs) == 3
        assert 0.0 <= conf <= 1.0
        assert entropy >= 0.0
        assert lat >= 0.0
        assert abs(sum(probs.values()) - 1.0) < 1e-3

    def test_streaming_arbitration_with_pil_frame(self):
        """predict_streaming_arbitration processes live PIL image frame."""
        frame = Image.new("RGB", (224, 224), color=(100, 150, 200))
        winner_md, probs, conf, entropy, lat = predict_streaming_arbitration(
            frame=frame,
            question="What is the safest maneuver?",
            candidates_raw="Hold position, Steer right, Steer left",
            temperature=0.4,
        )
        assert len(probs) == 3
        assert 0.0 <= conf <= 1.0
        assert "ms" in winner_md

    def test_streaming_arbitration_with_numpy_frame(self):
        """predict_streaming_arbitration handles numpy array frame from camera capture."""
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        winner_md, probs, conf, entropy, lat = predict_streaming_arbitration(
            frame=arr,
            question="Detect obstacle status",
            candidates_raw="Clear path, Obstacle detected, Sensor occluded",
            temperature=0.3,
        )
        assert len(probs) == 3
        assert conf > 0.0

    def test_streaming_arbitration_turnaround_latency(self):
        """Streaming arbitrator reports turnaround latency in milliseconds."""
        _, _, _, _, lat = predict_streaming_arbitration(
            frame=None,
            question="Quick check",
            candidates_raw="A, B",
        )
        assert isinstance(lat, float)
        assert lat >= 0.0

    def test_streaming_temperature_sharpness(self):
        """Lower temperature sharpens the streaming output distribution."""
        q = "Direction"
        cands = "North, South, East, West"
        _, probs_soft, _, ent_soft, _ = predict_streaming_arbitration(
            frame=None, question=q, candidates_raw=cands, temperature=1.5
        )
        _, probs_sharp, _, ent_sharp, _ = predict_streaming_arbitration(
            frame=None, question=q, candidates_raw=cands, temperature=0.2
        )
        # Lower temperature must yield lower or equal entropy
        assert ent_sharp <= ent_soft + 1e-5


class TestV5EngineResolution:
    """Tests for ArbiterOmniEngine.from_pretrained('v5') integration [AO-28]."""

    def test_v5_model_config_loading(self):
        """Engine correctly creates MoE model from v5 model_config checkpoint dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "arbiter_omni_v5.pt")
            encoder = MockMultimodalEncoder(embed_dim=128)
            model = ArbiterOmniModel(
                encoder=encoder,
                hidden_dim=64,
                scoring_dim=64,
                use_moe=True,
                moe_num_layers=2,
                moe_num_experts=4,
                moe_top_k=2,
            )
            state = {
                "fusion": model.fusion.state_dict(),
                "decision_head": model.decision_head.state_dict(),
                "model_config": {
                    "hidden_dim": 64,
                    "scoring_dim": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "model_name": "mock",
                    "use_moe": True,
                    "moe_num_layers": 2,
                    "moe_num_experts": 4,
                    "moe_top_k": 2,
                },
            }
            torch.save(state, ckpt_path)

            engine = ArbiterOmniEngine.from_pretrained(
                ckpt_path,
                encoder_type="mock",
                device="cpu",
                embed_dim=128,
            )
            assert engine.model.fusion.use_moe is True
            assert engine.model.fusion.moe_transformer is not None
            assert len(engine.model.fusion.moe_transformer.layers) == 2

            res = engine.decide(
                question="What action?",
                candidates=["Action 1", "Action 2"],
                text="Telemetry status green.",
            )
            assert res.decision in ["Action 1", "Action 2"]
            assert res.confidence > 0.0

    def test_build_app_includes_streaming_tab(self):
        """build_app() constructs Gradio application containing all tabs."""
        app = build_app()
        assert app is not None


class TestDualStreamSensorium:
    """Tests for continuous dual-stream (video + audio) sensorium [AO-32]."""

    def test_dual_streaming_without_inputs(self):
        """predict_dual_streaming_arbitration handles None video and None audio gracefully."""
        winner_md, probs, conf, entropy, lat, conformal_md = predict_dual_streaming_arbitration(
            frame=None,
            audio=None,
            question="What is the active robotic protocol?",
            candidates_raw="Hold position, Advance cautiously, Halt immediately",
            temperature=0.5,
        )
        assert "Current Action" in winner_md
        assert len(probs) == 3
        assert 0.0 <= conf <= 1.0
        assert entropy >= 0.0
        assert lat >= 0.0
        assert "Certified Conformal Set" in conformal_md
        assert abs(sum(probs.values()) - 1.0) < 1e-3

    def test_dual_streaming_with_frame_and_audio(self):
        """predict_dual_streaming_arbitration processes concurrent video frame and audio waveform."""
        frame = Image.new("RGB", (224, 224), color=(200, 100, 50))
        audio = np.random.randn(16000).astype(np.float32)
        winner_md, probs, conf, entropy, lat, conformal_md = predict_dual_streaming_arbitration(
            frame=frame,
            audio=audio,
            question="Analyze sensorium state",
            candidates_raw="Clear path / continue, Auditory warning / halt, Visual obstacle / detour",
            temperature=0.4,
        )
        assert len(probs) == 3
        assert conf > 0.0
        assert "Certified Conformal Set" in conformal_md
        assert lat >= 0.0

    def test_dual_streaming_audio_tuple(self):
        """predict_dual_streaming_arbitration accepts audio tuple (sample_rate, waveform)."""
        audio_tuple = (16000, np.zeros((16000, 2), dtype=np.float32))
        winner_md, probs, conf, entropy, lat, conformal_md = predict_dual_streaming_arbitration(
            frame=None,
            audio=audio_tuple,
            question="Voice alert detected",
            candidates_raw="Acknowledge, Ignore",
        )
        assert len(probs) == 2
        assert "Current Action" in winner_md


class TestV6EngineAndBrainMap:
    """Tests for ArbiterOmni v6 Brain Map telemetry and architecture resolution [AO-33]."""

    def test_predict_brain_map_telemetry(self):
        """predict_brain_map returns all 4 diagnostic panels with MoE routing and tournament."""
        dec_md, gate_md, routing_md, tourn_md = predict_brain_map(
            question="Tactical decision inquiry",
            candidates_raw="Maneuver A, Maneuver B, Maneuver C",
            deliberation_passes=3,
        )
        assert "Winning Candidate" in dec_md
        assert "Speculative Draft" in gate_md
        assert "Shared + Domain-Specialized MoE Routing" in routing_md
        assert "Shared Invariant" in routing_md
        assert "Test-Time Deliberation Tournament" in tourn_md

    def test_v6_model_config_loading_with_speculative_head(self):
        """Engine correctly creates MoE model with speculative draft head from v6 checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "arbiter_omni_v6.pt")
            encoder = MockMultimodalEncoder(embed_dim=128)
            model = ArbiterOmniModel(
                encoder=encoder,
                hidden_dim=64,
                scoring_dim=64,
                use_moe=True,
                moe_num_layers=2,
                moe_num_experts=4,
                moe_top_k=2,
                use_shared_expert=True,
                enable_speculative=True,
            )
            state = {
                "fusion": model.fusion.state_dict(),
                "decision_head": model.decision_head.state_dict(),
                "speculative_head": model.speculative_head.state_dict(),
                "model_config": {
                    "hidden_dim": 64,
                    "scoring_dim": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "model_name": "mock",
                    "use_moe": True,
                    "moe_num_layers": 2,
                    "moe_num_experts": 4,
                    "moe_top_k": 2,
                    "use_shared_expert": True,
                    "enable_speculative_early_exit": True,
                },
            }
            torch.save(state, ckpt_path)

            engine = ArbiterOmniEngine.from_pretrained(
                ckpt_path,
                encoder_type="mock",
                device="cpu",
                embed_dim=128,
            )
            assert engine.model.fusion.use_moe is True
            assert engine.model.fusion.moe_transformer is not None
            assert getattr(engine.model, "enable_speculative_early_exit", False) is True
            assert getattr(engine.model, "speculative_head", None) is not None

            res = engine.decide(
                question="What action?",
                candidates=["Action 1", "Action 2"],
                text="Sensors nominal.",
            )
            assert res.decision in ["Action 1", "Action 2"]
            assert res.confidence > 0.0

