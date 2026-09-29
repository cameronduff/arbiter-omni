"""
ArbiterOmni General-Purpose Interactive Decision Playground (Gradio Web UI).
Enables real-time System 1 multimodal decision arbitration over arbitrary open-domain queries,
dynamic candidate options, live probability distributions, Shannon entropy gauges,
and Jev Boolean Noul verification.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from arbiter_omni import (
    ArbiterOmniEngine,
    ArbiterOmniModel,
    MockMultimodalEncoder,
    OpenCLIPMultimodalEncoder,
    create_synthetic_audio,
    create_synthetic_image,
    get_device_telemetry,
    resolve_device,
)

logger = logging.getLogger(__name__)

# General-Purpose Open-Domain Scenarios
PRESETS: Dict[str, Dict[str, Any]] = {
    "Open Visual Reasoning: State of Matter": {
        "question": "Which physical state of matter is depicted under standard atmospheric pressure?",
        "text": "Temperature sensor registers 102°C with continuous vapor plume generation.",
        "candidates": "Gaseous phase (water vapor)\nLiquid phase (condensed water)\nSolid crystalline ice\nSupercritical fluid state",
        "image_color": (70, 140, 220),
        "sound_freq": 800.0,
        "description": "General scientific deduction from visual and thermal context.",
    },
    "Cross-Modal Audio-Visual: Environmental Event": {
        "question": "What environmental phenomenon matches the acoustic and visual observation?",
        "text": "Sky illumination spiked rapidly followed by immediate acoustic pressure wave.",
        "candidates": "Severe thunderstorm lightning discharge\nRapid transit train entering station\nHigh-wind atmospheric turbulence\nGentle forest rainfall",
        "image_color": (30, 30, 85),
        "sound_freq": 120.0,
        "description": "Multi-sensory event classification resolving acoustic-visual agreement.",
    },
    "Truth & Safety Verification: Jev Boolean Noul": {
        "question": "Is the observed physical system operating within safe nominal parameters?",
        "text": "Thermal telemetry indicates core operating at 94°C with nominal cooling flow.",
        "candidates": "True: System is operating within acceptable safety limits\nFalse: Critical safety threshold exceeded",
        "image_color": (40, 180, 80),
        "sound_freq": 440.0,
        "description": "Jev-style calibrated binary certainty and truth verification.",
    },
    "Dynamic Priority Rating: Continuous Score Manifold": {
        "question": "What priority tier should be assigned to this incoming operational alert?",
        "text": "Primary communications link degraded to 15% bandwidth; secondary backup link active.",
        "candidates": "Critical Priority (Immediate intervention required)\nElevated Priority (Inspect within 1 hour)\nRoutine Advisory (Monitor during scheduled review)\nInformational (Log and suppress alert)",
        "image_color": (230, 140, 30),
        "sound_freq": 1200.0,
        "description": "Continuous score prediction and multi-tier priority arbitration.",
    },
    "Zero-Shot Ambiguity Deliberation: Multi-Foil Triage": {
        "question": "What is the primary visual entity observed in the center of the scene?",
        "text": "Low-light camera captures an elongated silhouette traversing uneven terrain.",
        "candidates": "Canine domestic quadruped\nFeline predatory animal\nMechanical terrestrial drone\nOptical camera lens flare artifact",
        "image_color": (120, 115, 110),
        "sound_freq": 300.0,
        "description": "Demonstrating entropy divergence when sensory cues are ambiguous.",
    },
}

# Global singleton engine for low-latency interactive serving
_ENGINE: Optional[ArbiterOmniEngine] = None


def get_engine() -> ArbiterOmniEngine:
    """Initializes or returns cached ArbiterOmniEngine with pretrained v1 weights when available."""
    global _ENGINE
    if _ENGINE is None:
        device = resolve_device()
        v1_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "checkpoints", "arbiter_omni_v1.pt")
        )

        if os.path.exists(v1_path):
            try:
                logger.info(f"Loading official pretrained v1 weights from {v1_path}...")
                _ENGINE = ArbiterOmniEngine.from_pretrained(v1_path, encoder_type="openclip", device=device)
                logger.info("Successfully loaded v1 checkpoint for interactive demo.")
                return _ENGINE
            except Exception as e:
                logger.warning(f"Could not load v1 checkpoint with OpenCLIP ({e}); falling back to mock engine.")

        # Fallback to responsive mock engine
        encoder = MockMultimodalEncoder(embed_dim=128, device=device)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128).to(device)
        _ENGINE = ArbiterOmniEngine(model=model, device=device)
    return _ENGINE


def arbitrate_decision(
    question: str,
    candidates_text: str,
    text_context: Optional[str] = None,
    image_input: Optional[Any] = None,
    audio_input: Optional[Any] = None,
) -> Tuple[str, Dict[str, float], str, str, str]:
    """
    Core arbitration logic called by Gradio and test suites.
    """
    engine = get_engine()

    # Parse candidate lines
    lines = [c.strip() for c in candidates_text.strip().split("\n") if len(c.strip()) > 0]
    if len(lines) == 0:
        lines = ["Candidate A", "Candidate B"]

    # Handle audio format from Gradio (can be (sample_rate, numpy_array) or filepath)
    audio_data = None
    if audio_input is not None:
        if isinstance(audio_input, tuple):
            sr, arr = audio_input
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            audio_data = arr.astype(np.float32) / (np.max(np.abs(arr)) + 1e-8)
        else:
            audio_data = audio_input

    t0 = time.perf_counter()
    result = engine.decide(
        question=question or "What is the optimal decision?",
        candidates=lines,
        text=text_context if text_context and len(text_context.strip()) > 0 else None,
        image=image_input,
        audio=audio_data,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Format winner banner
    winner_str = f"## 🏆 Winning Decision: **{result.winner}**\n### Confidence: **{result.confidence*100:.1f}%** (Single-Pass Latency: {elapsed_ms:.2f} ms)"

    # Format probability distribution dict for Gradio Label
    probs_dict = {cand: float(prob) for cand, prob in result.probabilities.items()}

    # Format entropy and calibration stats
    entropy_str = f"**{result.entropy:.3f} nats** ({'Crisp Consensus' if result.entropy < 0.4 else 'Active Deliberation / Ambiguity'})"

    # Jev Boolean Noul certainty
    if result.boolean_noul:
        cert = result.boolean_noul.get("calibrated_certainty", 0.0)
        true_p = result.boolean_noul.get("true_probability", 0.5)
        noul_str = f"**Certainty:** {cert*100:.1f}% | True Prob: {true_p*100:.1f}%"
    else:
        noul_str = "**Certainty:** N/A"

    # Jev Continuous Score
    score_val = result.score if result.score is not None else 0.0
    score_str = f"**Score:** {score_val:+.3f}"

    return winner_str, probs_dict, entropy_str, noul_str, score_str


def get_preset_payload(preset_name: str) -> Tuple[str, str, str, Image.Image, Tuple[int, np.ndarray]]:
    """Constructs the full pre-populated payload for a preset scenario."""
    sc = PRESETS.get(preset_name, list(PRESETS.values())[0])
    img = create_synthetic_image(color=sc["image_color"])
    aud_wave = create_synthetic_audio(freq=sc.get("sound_freq", 440.0), duration=0.5)
    return sc["question"], sc["candidates"], sc["text"], img, (16000, aud_wave)


def load_preset(preset_name: str):
    """Event handler for scenario dropdown selection."""
    return get_preset_payload(preset_name)


def build_app():
    """Builds the generalized Gradio user interface."""
    import gradio as gr

    telemetry = get_device_telemetry()
    first_preset_name = list(PRESETS.keys())[0]
    default_q, default_cands, default_text, default_img, default_audio = get_preset_payload(first_preset_name)

    with gr.Blocks(title="ArbiterOmni: Multimodal System 1 Decision Engine") as demo:
        gr.Markdown(
            f"""
            # ⚡ ArbiterOmni: General Multimodal System 1 Decision Engine
            ### Open-Domain Dynamic Candidate Arbitration inspired by Jev | Active Device: `{telemetry['device']}` ({telemetry['gpu_name']})
            > **Non-Autoregressive Decision Making:** Evaluates multimodal sensory context (text, vision, audio) in a single forward pass, dynamically projecting and scoring runtime candidate options into a calibrated probability distribution.
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                preset_selector = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value=first_preset_name,
                    label="📂 Load Pre-configured General Scenario",
                )

                question_input = gr.Textbox(
                    label="❓ Decision Query / Prompt",
                    value=default_q,
                    lines=2,
                    placeholder="Enter any open-domain question...",
                )

                candidates_input = gr.Textbox(
                    label="🎯 Dynamic Candidate Options (Enter 2 to N choices, one per line)",
                    value=default_cands,
                    lines=4,
                    placeholder="Option 1\nOption 2\nOption 3...",
                )

                text_context = gr.Textbox(
                    label="📝 Text Context / Sensor Log (Optional)",
                    value=default_text,
                    lines=2,
                    placeholder="Optional background text, sensor telemetry, or observational notes...",
                )

                with gr.Row():
                    image_input = gr.Image(
                        value=default_img,
                        type="pil",
                        label="📷 Visual Observation (Image / Keyframe)",
                    )
                    audio_input = gr.Audio(
                        value=default_audio,
                        label="🔊 Acoustic Telemetry (Audio Waveform)",
                    )

                arbitrate_btn = gr.Button("⚡ Arbitrate Decision", variant="primary", size="lg")

            with gr.Column(scale=5):
                winner_output = gr.Markdown("### ⏳ Click '⚡ Arbitrate Decision' to compute distribution.")
                probs_output = gr.Label(label="📊 Calibrated Decision Probability Distribution", num_top_classes=6)

                with gr.Row():
                    entropy_output = gr.Textbox(label="🌀 Shannon Entropy (Uncertainty)", interactive=False)
                    noul_output = gr.Textbox(label="⚖️ Jev Boolean Noul", interactive=False)
                    score_output = gr.Textbox(label="📈 Continuous Score", interactive=False)

                gr.Markdown(
                    r"""
                    ---
                    ### 💡 Architectural Highlights
                    - **Zero Class Hardcoding**: Candidates are projected dynamically onto the fused decision manifold.
                    - **Missing Modality Robustness**: Omit any input (vision, audio, or text); attention masks eliminate phantom activations.
                    - **Jev Calibration**: Softmax probabilities $\sum p_k = 1.0$ with Shannon entropy monitoring deliberation tension.
                    """
                )

        # Wire interactive events
        preset_selector.change(
            fn=load_preset,
            inputs=[preset_selector],
            outputs=[question_input, candidates_input, text_context, image_input, audio_input],
        )

        arbitrate_btn.click(
            fn=arbitrate_decision,
            inputs=[question_input, candidates_input, text_context, image_input, audio_input],
            outputs=[winner_output, probs_output, entropy_output, noul_output, score_output],
        )

        # One-Click Clickable Examples
        example_data = []
        for name in PRESETS.keys():
            q, cands, txt, img, aud = get_preset_payload(name)
            example_data.append([q, cands, txt, img, aud])

        gr.Examples(
            examples=example_data,
            inputs=[question_input, candidates_input, text_context, image_input, audio_input],
            outputs=[winner_output, probs_output, entropy_output, noul_output, score_output],
            fn=arbitrate_decision,
            cache_examples=False,
            label="⚡ 1-Click General Decision Scenarios",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
