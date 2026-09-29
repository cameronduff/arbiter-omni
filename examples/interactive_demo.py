"""
ArbiterOmni Interactive Decision Playground (Gradio Web UI).
Enables real-time System 1 multimodal decision arbitration with dynamic candidates,
live probability distributions, Shannon entropy gauges, and Jev Boolean Noul verification.
"""

from __future__ import annotations

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
    create_synthetic_audio,
    create_synthetic_image,
    get_device_telemetry,
    resolve_device,
)

PRESETS = {
    "Robotics: Collision Hazard Triage": {
        "question": "What is the immediate action for the autonomous mobile robot?",
        "text": "LiDAR detects dynamic obstacle at 1.2 meters rapidly closing.",
        "candidates": "halt immediately and apply brakes\nproceed forward at nominal speed\nsteer left around obstacle\nrequest remote operator assistance",
        "image_color": (220, 30, 30),
        "sound_type": "screech",
    },
    "Industrial: Acoustic Anomaly Triage": {
        "question": "What is the appropriate triage command for the assembly line conveyor?",
        "text": "Acoustic sensors detect abnormal high-frequency screech in primary roller bearing.",
        "candidates": "trigger emergency facility shutdown\nmaintain nominal line operation\nreroute component to quality inspection\ndispatch maintenance technician",
        "image_color": (160, 160, 160),
        "sound_type": "screech",
    },
    "Logistics: Clear Navigation Path": {
        "question": "What is the next navigation maneuver for the automated guided vehicle?",
        "text": "Sensors report clear runway with nominal floor friction.",
        "candidates": "proceed forward at cruising velocity\nemergency halt\nturn 90 degrees right\nwait for traffic clearance",
        "image_color": (30, 200, 30),
        "sound_type": "hum",
    },
    "Jev Verification: Binary Noul Certainty": {
        "question": "Is the observed subsystem operating within safety tolerances?",
        "text": "Thermal telemetry indicates core operating at 94 degrees Celsius.",
        "candidates": "Hazardous anomaly confirmed\nNominal operational parameters",
        "image_color": (240, 180, 20),
        "sound_type": "screech",
    },
}

# Global singleton engine for low-latency interactive serving
_ENGINE: Optional[ArbiterOmniEngine] = None


def get_engine() -> ArbiterOmniEngine:
    """Initializes or returns cached ArbiterOmniEngine."""
    global _ENGINE
    if _ENGINE is None:
        device = resolve_device()
        encoder = MockMultimodalEncoder(embed_dim=128)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128).to(device)
        _ENGINE = ArbiterOmniEngine(model=model)
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
        lines = ["Candidate 1", "Candidate 2"]

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
    entropy_str = f"**{result.entropy:.3f} nats** ({'Crisp Consensus' if result.entropy < 0.3 else 'Active Deliberation / Uncertainty'})"

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


def load_preset(preset_name: str):
    """Loads a pre-configured scenario into the form."""
    sc = PRESETS.get(preset_name, list(PRESETS.values())[0])
    img = create_synthetic_image(color=sc["image_color"])
    freq = 2000.0 if sc["sound_type"] == "screech" else 120.0
    aud_wave = create_synthetic_audio(freq=freq, duration=0.5)
    return sc["question"], sc["candidates"], sc["text"], img, (16000, aud_wave)


def build_app():
    """Builds the Gradio user interface."""
    import gradio as gr

    telemetry = get_device_telemetry()

    with gr.Blocks(title="ArbiterOmni: Multimodal System 1 Decision Engine") as demo:
        gr.Markdown(
            f"""
            # ⚡ ArbiterOmni: Multimodal System 1 Decision Engine
            ### Real-Time Dynamic Candidate Arbitration inspired by Jev | Active Device: `{telemetry['device']}` ({telemetry['gpu_name']})
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                preset_selector = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value=list(PRESETS.keys())[0],
                    label="📂 Load Pre-configured Scenario Preset",
                )

                question_input = gr.Textbox(
                    label="❓ Prompt / Decision Query",
                    value=PRESETS[list(PRESETS.keys())[0]]["question"],
                    lines=2,
                )

                candidates_input = gr.Textbox(
                    label="🎯 Dynamic Candidate Choices (One per line)",
                    value=PRESETS[list(PRESETS.keys())[0]]["candidates"],
                    lines=4,
                )

                text_context = gr.Textbox(
                    label="📝 Text Context / Sensor Stream (Optional)",
                    value=PRESETS[list(PRESETS.keys())[0]]["text"],
                    lines=2,
                )

                with gr.Row():
                    image_input = gr.Image(type="pil", label="📷 Visual Input (Image / Diagram)")
                    audio_input = gr.Audio(label="🔊 Acoustic Input (Waveform / Telemetry)")

                arbitrate_btn = gr.Button("⚡ Arbitrate Decision", variant="primary", size="lg")

            with gr.Column(scale=5):
                winner_output = gr.Markdown("### ⏳ Click 'Arbitrate Decision' to compute distribution.")
                probs_output = gr.Label(label="📊 Calibrated Decision Probability Distribution", num_top_classes=6)

                with gr.Row():
                    entropy_output = gr.Textbox(label="🌀 Shannon Entropy (Uncertainty)", interactive=False)
                    noul_output = gr.Textbox(label="⚖️ Jev Boolean Noul", interactive=False)
                    score_output = gr.Textbox(label="📈 Continuous Score", interactive=False)

        # Wire events
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

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
