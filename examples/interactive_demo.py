"""
ArbiterOmni Interactive Decision Playground (Gradio Web UI).

General-purpose, open-domain multimodal decision arbitration.
Drop in any image, type any question, list any candidates — one forward pass returns
a calibrated probability distribution over your options.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from arbiter_omni import (
    ArbiterOmniEngine,
    ArbiterOmniModel,
    MockMultimodalEncoder,
    get_device_telemetry,
    resolve_device,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values shown when the UI first loads
# ---------------------------------------------------------------------------
DEFAULT_QUESTION = "What animal is in this image?"
DEFAULT_CANDIDATES = "Cat\nDog\nHorse\nRabbit\nBird"
DEFAULT_TEXT = ""  # optional — leave blank by default

# A simple placeholder image (a warm orange patch) so the UI isn't empty on launch
def _default_image() -> Image.Image:
    arr = np.full((224, 224, 3), (230, 140, 60), dtype=np.uint8)
    return Image.fromarray(arr)

# ---------------------------------------------------------------------------
# One-click examples — everyday, domain-agnostic tasks
# ---------------------------------------------------------------------------
# Each entry: [question, candidates (newline-separated), text_context, image, audio]
# Images are generated as solid-colour placeholders; users can swap them for real photos.
def _solid(r: int, g: int, b: int, size: int = 224) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), (r, g, b), dtype=np.uint8))

EXAMPLES: List[List[Any]] = [
    [
        "What animal is in this image?",
        "Cat\nDog\nHorse\nRabbit\nBird",
        "",
        _solid(210, 180, 140),   # tan — neutral animal placeholder
        None,
    ],
    [
        "What colour is the object?",
        "Red\nBlue\nGreen\nYellow\nPurple\nOrange",
        "",
        _solid(60, 120, 220),    # blue patch
        None,
    ],
    [
        "Is this image taken indoors or outdoors?",
        "Indoors\nOutdoors",
        "",
        _solid(135, 185, 130),   # muted green — suggests outside
        None,
    ],
    [
        "What type of vehicle is shown?",
        "Car\nMotorcycle\nBicycle\nBus\nTruck\nTrain",
        "",
        _solid(80, 80, 90),      # dark grey — vehicle-neutral
        None,
    ],
    [
        "What is the weather like in this scene?",
        "Sunny\nCloudy\nRainy\nSnowy\nFoggy",
        "",
        _solid(200, 215, 235),   # pale blue-grey — overcast sky
        None,
    ],
    [
        "What emotion does this person appear to be expressing?",
        "Happy\nSad\nAngry\nSurprised\nNeutral\nFearful",
        "The subject is facing the camera directly.",
        _solid(240, 210, 185),   # skin-tone placeholder
        None,
    ],
    [
        "What meal of the day does this food most resemble?",
        "Breakfast\nLunch\nDinner\nSnack\nDessert",
        "",
        _solid(220, 160, 60),    # golden-yellow — food-like warmth
        None,
    ],
    [
        "Is this true or false? The sky is blue.",
        "True\nFalse",
        "",
        _solid(100, 160, 230),   # blue sky placeholder
        None,
    ],
]

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------
_ENGINE: Optional[ArbiterOmniEngine] = None


def get_engine() -> ArbiterOmniEngine:
    """Initialises or returns cached ArbiterOmniEngine (v1 checkpoint → mock fallback)."""
    global _ENGINE
    if _ENGINE is None:
        device = resolve_device()
        v1_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "checkpoints", "arbiter_omni_v1.pt")
        )
        if os.path.exists(v1_path):
            try:
                from arbiter_omni import OpenCLIPMultimodalEncoder  # noqa: F401
                _ENGINE = ArbiterOmniEngine.from_pretrained(v1_path, encoder_type="openclip", device=device)
                logger.info("Loaded pretrained v1 checkpoint.")
                return _ENGINE
            except Exception as e:
                logger.warning(f"Could not load v1 checkpoint ({e}); falling back to mock engine.")

        encoder = MockMultimodalEncoder(embed_dim=128, device=device)
        model = ArbiterOmniModel(encoder=encoder, hidden_dim=128, scoring_dim=128).to(device)
        _ENGINE = ArbiterOmniEngine(model=model, device=device)
    return _ENGINE


# ---------------------------------------------------------------------------
# Core arbitration function
# ---------------------------------------------------------------------------
def arbitrate_decision(
    question: str,
    candidates_text: str,
    text_context: Optional[str] = None,
    image_input: Optional[Any] = None,
    audio_input: Optional[Any] = None,
) -> Tuple[str, Dict[str, float], str, str, str]:
    """Called by Gradio on button click and by the test suite directly."""
    engine = get_engine()

    # Parse one candidate per non-empty line
    lines = [c.strip() for c in (candidates_text or "").strip().split("\n") if c.strip()]
    if len(lines) < 2:
        lines = ["Option A", "Option B"]

    # Normalise audio from Gradio's (sample_rate, array) tuple
    audio_data = None
    if audio_input is not None:
        if isinstance(audio_input, tuple):
            _, arr = audio_input
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            audio_data = arr.astype(np.float32) / (np.max(np.abs(arr)) + 1e-8)
        else:
            audio_data = audio_input

    t0 = time.perf_counter()
    result = engine.decide(
        question=question or "What is the best option?",
        candidates=lines,
        text=text_context if text_context and text_context.strip() else None,
        image=image_input,
        audio=audio_data,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    winner_str = (
        f"## 🏆 **{result.winner}**\n"
        f"### Confidence: **{result.confidence * 100:.1f}%** &nbsp;·&nbsp; "
        f"Latency: **{elapsed_ms:.2f} ms**"
    )

    probs_dict = {cand: float(prob) for cand, prob in result.probabilities.items()}

    label = "Crisp consensus" if result.entropy < 0.4 else "Deliberating / ambiguous"
    entropy_str = f"{result.entropy:.3f} nats  ({label})"

    if result.boolean_noul:
        cert = result.boolean_noul.get("calibrated_certainty", 0.0)
        true_p = result.boolean_noul.get("true_probability", 0.5)
        noul_str = f"Certainty: {cert * 100:.1f}%  |  True prob: {true_p * 100:.1f}%"
    else:
        noul_str = "N/A"

    score_val = result.score if result.score is not None else 0.0
    score_str = f"{score_val:+.3f}"

    return winner_str, probs_dict, entropy_str, noul_str, score_str


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_app():
    """Builds the open-domain Gradio interface."""
    import gradio as gr

    telemetry = get_device_telemetry()

    with gr.Blocks(title="ArbiterOmni · Multimodal Decision Engine") as demo:

        gr.Markdown(
            f"""
# ⚡ ArbiterOmni — Multimodal Decision Engine
**Device:** `{telemetry['device']}` ({telemetry['gpu_name']}) &nbsp;·&nbsp;
Single non-autoregressive forward pass · Calibrated probabilities · Works with any question and any candidates
            """
        )

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────
            with gr.Column(scale=5):

                question_input = gr.Textbox(
                    label="❓ Question",
                    value=DEFAULT_QUESTION,
                    lines=2,
                    placeholder="e.g.  What animal is this?  ·  Is the room tidy?  ·  What genre is this music?",
                )

                candidates_input = gr.Textbox(
                    label="🎯 Candidates  (one per line, 2 – N)",
                    value=DEFAULT_CANDIDATES,
                    lines=5,
                    placeholder="Cat\nDog\nHorse\n...",
                )

                text_context = gr.Textbox(
                    label="📝 Extra context  (optional)",
                    value=DEFAULT_TEXT,
                    lines=2,
                    placeholder="Any supporting text — a caption, a description, sensor data, …",
                )

                with gr.Row():
                    image_input = gr.Image(
                        value=_default_image(),
                        type="pil",
                        label="📷 Image  (optional — drop or upload any photo)",
                    )
                    audio_input = gr.Audio(
                        value=None,
                        label="🔊 Audio  (optional)",
                    )

                arbitrate_btn = gr.Button("⚡ Run Arbitration", variant="primary", size="lg")

            # ── Right column: outputs ─────────────────────────────────────
            with gr.Column(scale=5):
                winner_output = gr.Markdown(
                    "### ⏳ Fill in your question and candidates, then click **Run Arbitration**."
                )
                probs_output = gr.Label(
                    label="📊 Probability Distribution",
                    num_top_classes=10,
                )

                with gr.Row():
                    entropy_output = gr.Textbox(label="🌀 Shannon Entropy", interactive=False)
                    noul_output   = gr.Textbox(label="⚖️ Boolean Noul",     interactive=False)
                    score_output  = gr.Textbox(label="📈 Score",             interactive=False)

                gr.Markdown(
                    """
---
**How it works**
- Type *any* question — animal, object, sentiment, fact-checking, preference, …
- List *any* candidates, one per line. No retraining needed.
- Optionally attach an image and/or audio clip for multimodal context.
- Missing inputs are gracefully masked — the model never hallucinates from absent modalities.
                    """
                )

        # Wire button
        arbitrate_btn.click(
            fn=arbitrate_decision,
            inputs=[question_input, candidates_input, text_context, image_input, audio_input],
            outputs=[winner_output, probs_output, entropy_output, noul_output, score_output],
        )

        # One-click example gallery (no dropdown, no scenarios)
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input, candidates_input, text_context, image_input, audio_input],
            outputs=[winner_output, probs_output, entropy_output, noul_output, score_output],
            fn=arbitrate_decision,
            cache_examples=False,
            label="💡 Click any example to load it instantly",
        )

    return demo


if __name__ == "__main__":
    import gradio as gr
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
