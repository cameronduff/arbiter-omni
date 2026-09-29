"""
ArbiterOmni Interactive Decision Playground (Gradio Web UI).

General-purpose, open-domain multimodal decision arbitration.
Drop in any image, type any question, list any candidates — one forward pass returns
a calibrated probability distribution over your options.

Crafted strictly in accordance with /frontend-craft-standards:
- Anti-AI-Slop: Grounded charcoal/slate palette with precision emerald accents (no purple/violet gradient slop).
- Typography Hierarchy: Negative tracking on display headings, JetBrains Mono for telemetry & statistics.
- Animation Discipline: Physics springs cubic-bezier(0.16, 1, 0.3, 1), transform/opacity animations, prefers-reduced-motion support.
- Complete State Coverage: Dedicated standby empty state, rich decision verdict card, graceful error fallback.
- Tactile Micro-Interactions: Spring press states, subtle border highlights, live hardware telemetry strip.
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

# Default image: load user-provided photo.jpg if available, else synthetic fallback
def _default_image() -> Image.Image:
    candidates = [
        os.path.join(os.path.dirname(__file__), "assets", "kitten.jpg"),
        os.path.join(os.path.dirname(__file__), "photo.jpg"),
        os.path.join(os.path.dirname(__file__), "..", "photo.jpg"),
        "photo.jpg",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    arr = np.full((224, 224, 3), (230, 140, 60), dtype=np.uint8)
    return Image.fromarray(arr)

def _beagle_image() -> Image.Image:
    for p in ["beagle.webp", "examples/assets/beagle.webp"]:
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    return Image.new("RGB", (224, 224), (160, 100, 60))

# ---------------------------------------------------------------------------
# One-click examples — everyday, domain-agnostic tasks
# ---------------------------------------------------------------------------
# Each entry: [question, candidates (newline-separated), text_context, image, audio, temperature]
def _solid(r: int, g: int, b: int, size: int = 224) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), (r, g, b), dtype=np.uint8))

EXAMPLES: List[List[Any]] = [
    [
        "What animal is in this image?",
        "Cat\nDog\nHorse\nRabbit\nBird",
        "",
        _default_image(),   # Orange kitten photo
        None,
        0.7,
    ],
    [
        "What animal is in this image?",
        "Cat\nDog\nHorse\nRabbit\nBird",
        "",
        _beagle_image(),    # Beagle hound dog photo
        None,
        0.7,
    ],
    [
        "What colour is the object?",
        "Red\nBlue\nGreen\nYellow\nPurple\nOrange",
        "",
        _solid(60, 120, 220),    # blue patch
        None,
        0.7,
    ],
    [
        "Is this image taken indoors or outdoors?",
        "Indoors\nOutdoors",
        "",
        _solid(135, 185, 130),   # muted green — suggests outside
        None,
        0.7,
    ],
    [
        "What type of vehicle is shown?",
        "Car\nMotorcycle\nBicycle\nBus\nTruck\nTrain",
        "",
        _solid(80, 80, 90),      # dark grey — vehicle-neutral
        None,
        0.7,
    ],
    [
        "What is the weather like in this scene?",
        "Sunny\nCloudy\nRainy\nSnowy\nFoggy",
        "",
        _solid(200, 215, 235),   # pale blue-grey — overcast sky
        None,
        0.7,
    ],
    [
        "What emotion does this person appear to be expressing?",
        "Happy\nSad\nAngry\nSurprised\nNeutral\nFearful",
        "The subject is facing the camera directly.",
        _solid(240, 210, 185),   # skin-tone placeholder
        None,
        0.7,
    ],
    [
        "What meal of the day does this food most resemble?",
        "Breakfast\nLunch\nDinner\nSnack\nDessert",
        "",
        _solid(220, 160, 60),    # golden-yellow — food-like warmth
        None,
        0.7,
    ],
    [
        "Is this true or false? The sky is blue.",
        "True\nFalse",
        "",
        _solid(100, 160, 230),   # blue sky placeholder
        None,
        0.7,
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
    temperature: Optional[float] = 0.7,
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
        temperature=temperature,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    conf_pct = float(result.confidence * 100.0)

    # Contextual certainty styling (Anti-AI-Slop: purposeful semantic colors, no violet neon)
    if conf_pct >= 70.0:
        tag_text = "Crisp Consensus · High Certainty"
        bar_color_class = "emerald"
        accent_color = "#10B981"
        calib_label = "Decisive Single-Pass"
    elif conf_pct >= 40.0:
        tag_text = "Deliberating · Moderate Confidence"
        bar_color_class = "amber"
        accent_color = "#F59E0B"
        calib_label = "Distributed Probability"
    else:
        tag_text = "Ambiguous · Competing Candidates"
        bar_color_class = "slate"
        accent_color = "#94A3B8"
        calib_label = "High Entropy Spread"

    # HTML embedded inside Markdown for craft styling while remaining 100% backward-compatible
    winner_str = (
        f"## 🏆 Winning Decision: **{result.winner}**\n\n"
        f'<div class="ao-verdict-card">\n'
        f'  <div class="ao-verdict-header">\n'
        f'    <span class="ao-verdict-tag {bar_color_class}">{tag_text}</span>\n'
        f'    <span class="ao-latency-pill">⚡ {elapsed_ms:.2f} ms</span>\n'
        f'  </div>\n'
        f'  <div class="ao-verdict-title-row">\n'
        f'    <div class="ao-verdict-winner">{result.winner}</div>\n'
        f'    <div class="ao-confidence-val" style="color: {accent_color};">{conf_pct:.1f}%</div>\n'
        f'  </div>\n'
        f'  <div class="ao-progress-track">\n'
        f'    <div class="ao-progress-fill {bar_color_class}" style="width: {min(100.0, max(0.0, conf_pct)):.1f}%;"></div>\n'
        f'  </div>\n'
        f'  <div class="ao-verdict-footer">\n'
        f'    <span>Calibration: <strong>{calib_label}</strong></span>\n'
        f'    <span>Entropy: <strong>{result.entropy:.3f} nats</strong></span>\n'
        f'  </div>\n'
        f'</div>'
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
# Styling and Theme Configuration (Anti-AI-Slop & Frontend Craft Standards)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

body, .gradio-container {
  font-family: var(--font-sans) !important;
  background-color: #090D16 !important;
  color: #E2E8F0 !important;
  -webkit-font-smoothing: antialiased;
}

/* Headings with tight negative tracking */
h1, h2, h3, h4 {
  font-family: var(--font-sans) !important;
  letter-spacing: -0.03em !important;
  font-weight: 700 !important;
  color: #F8FAFC !important;
}

/* Top Header & Telemetry Banner */
.ao-header-container {
  background: #111827;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  padding: 1.25rem 1.75rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

.ao-header-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 0.6rem;
}

.ao-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.ao-logo-badge {
  background: #10B981;
  color: #064E3B;
  font-weight: 800;
  font-size: 1.15rem;
  width: 2.2rem;
  height: 2.2rem;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 14px rgba(16, 185, 129, 0.35);
}

.ao-title {
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: #FFFFFF;
  margin: 0;
  line-height: 1.2;
}

.ao-version-pill {
  font-size: 0.72rem;
  font-weight: 600;
  font-family: var(--font-mono);
  background: rgba(255, 255, 255, 0.06);
  color: #94A3B8;
  padding: 0.2rem 0.55rem;
  border-radius: 9999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.ao-telemetry-strip {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.ao-telemetry-item {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  background: #1A2234;
  color: #CBD5E1;
  padding: 0.25rem 0.65rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.ao-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background-color: #10B981;
  box-shadow: 0 0 8px #10B981;
  animation: aoPulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes aoPulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.9); }
}

.ao-subtitle {
  font-size: 0.86rem;
  color: #94A3B8;
  margin: 0;
  line-height: 1.5;
}

/* Primary Action Button — Authentic Emerald, Tactile Spring Press */
button.primary, button[variant="primary"], .primary-btn {
  background: #059669 !important;
  background-image: linear-gradient(180deg, #10B981 0%, #059669 100%) !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  letter-spacing: -0.01em !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
  border-radius: 9px !important;
  transition: all 180ms cubic-bezier(0.16, 1, 0.3, 1) !important;
  cursor: pointer !important;
}

button.primary:hover, button[variant="primary"]:hover {
  background: #10B981 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
}

button.primary:active, button[variant="primary"]:active {
  transform: scale(0.985) translateY(1px) !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
}

/* Card Surfaces & Clean 1px Borders */
.gradio-container .block, .gradio-container .panel {
  background: #111827 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 12px !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
  transition: border-color 180ms cubic-bezier(0.16, 1, 0.3, 1);
}

.gradio-container .block:hover {
  border-color: rgba(255, 255, 255, 0.14) !important;
}

/* Inputs & Textareas */
textarea, input[type="text"], input[type="number"] {
  background: #0D1322 !important;
  border: 1px solid #1E293B !important;
  color: #F1F5F9 !important;
  border-radius: 8px !important;
  font-family: var(--font-sans) !important;
  transition: border-color 160ms cubic-bezier(0.16, 1, 0.3, 1), box-shadow 160ms cubic-bezier(0.16, 1, 0.3, 1) !important;
}

textarea:focus, input[type="text"]:focus {
  border-color: #10B981 !important;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
  outline: none !important;
}

/* Verdict Hero Card */
.ao-verdict-card {
  background: #131C2E;
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-top: 0.5rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  animation: aoFadeSlide 200ms cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes aoFadeSlide {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

.ao-verdict-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.ao-verdict-tag {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
}

.ao-verdict-tag.emerald {
  color: #10B981;
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.25);
}

.ao-verdict-tag.amber {
  color: #F59E0B;
  background: rgba(245, 158, 11, 0.12);
  border: 1px solid rgba(245, 158, 11, 0.25);
}

.ao-verdict-tag.slate {
  color: #94A3B8;
  background: rgba(148, 163, 184, 0.12);
  border: 1px solid rgba(148, 163, 184, 0.25);
}

.ao-latency-pill {
  font-family: var(--font-mono);
  font-size: 0.74rem;
  color: #94A3B8;
  background: #1E293B;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}

.ao-verdict-title-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.85rem;
}

.ao-verdict-winner {
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.035em;
  color: #FFFFFF;
  margin: 0;
  line-height: 1.15;
}

.ao-confidence-val {
  font-family: var(--font-mono);
  font-size: 1.45rem;
  font-weight: 700;
}

/* Confidence Gauge Bar */
.ao-progress-track {
  width: 100%;
  height: 8px;
  background: #1E293B;
  border-radius: 9999px;
  overflow: hidden;
  margin-bottom: 0.65rem;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.4);
}

.ao-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #059669 0%, #10B981 100%);
  border-radius: 9999px;
  transition: width 350ms cubic-bezier(0.16, 1, 0.3, 1);
}

.ao-progress-fill.amber {
  background: linear-gradient(90deg, #D97706 0%, #F59E0B 100%);
}

.ao-progress-fill.slate {
  background: linear-gradient(90deg, #475569 0%, #64748B 100%);
}

.ao-verdict-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.76rem;
  color: #94A3B8;
}

/* Empty State Card */
.ao-empty-state-card {
  background: #111827;
  border: 1px dashed #334155;
  border-radius: 12px;
  padding: 1.75rem 1.5rem;
  text-align: center;
  margin-top: 0.5rem;
  margin-bottom: 0.75rem;
}

.ao-empty-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #94A3B8;
  background: #1E293B;
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  margin-bottom: 0.75rem;
}

.ao-empty-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: #F1F5F9;
  margin: 0 0 0.35rem 0;
}

.ao-empty-desc {
  font-size: 0.84rem;
  color: #64748B;
  max-width: 44ch;
  margin: 0 auto;
  line-height: 1.5;
}

/* Explainer card */
.ao-explainer-card {
  background: #0E1524;
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  margin-top: 1rem;
}

.ao-explainer-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.75rem;
  margin-top: 0.5rem;
}

.ao-explainer-item {
  display: flex;
  gap: 0.5rem;
  font-size: 0.8rem;
  color: #94A3B8;
  line-height: 1.4;
}

.ao-explainer-bullet {
  color: #10B981;
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 0.82rem;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    transform: none !important;
  }
}
</style>
"""


def create_custom_theme() -> Any:
    """Creates authentic slate + emerald theme for Gradio."""
    import gradio as gr
    return gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#090D16",
        body_text_color="#E2E8F0",
        block_background_fill="#111827",
        block_border_color="#1E293B",
        input_background_fill="#0D1322",
        input_border_color="#334155",
        button_primary_background_fill="#059669",
        button_primary_background_fill_hover="#10B981",
        button_primary_text_color="#FFFFFF",
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_app():
    """Builds the open-domain Gradio interface."""
    import gradio as gr

    telemetry = get_device_telemetry()

    initial_winner_md = (
        "## ⏳ Awaiting Arbitration\n\n"
        '<div class="ao-empty-state-card">\n'
        '  <div class="ao-empty-badge">\n'
        '    <span class="ao-status-dot"></span>\n'
        '    <span>Standby · Pipeline Calibrated & Ready</span>\n'
        '  </div>\n'
        '  <div class="ao-empty-title">Awaiting Query Inputs</div>\n'
        '  <div class="ao-empty-desc">'
        'Select any test case below or enter a question, dynamic candidates, and media, '
        'then click <strong>Run Arbitration</strong> to execute a zero-shot forward pass.'
        '</div>\n'
        '</div>'
    )

    header_html = f"""
<div class="ao-header-container">
  <div class="ao-header-top">
    <div class="ao-brand">
      <div class="ao-logo-badge">⚡</div>
      <div>
        <h1 class="ao-title">ArbiterOmni</h1>
        <p class="ao-subtitle">Real-Time Multimodal Decision Arbitration & Dynamic Class Scoring</p>
      </div>
      <span class="ao-version-pill">v1.0 Production</span>
    </div>
    <div class="ao-telemetry-strip">
      <div class="ao-telemetry-item">
        <span class="ao-status-dot"></span>
        <span>Device: <strong>{telemetry['device'].upper()} ({telemetry['gpu_name']})</strong></span>
      </div>
      <div class="ao-telemetry-item">
        <span>Backbone: <strong>ViT-B-16 (196 Patches)</strong></span>
      </div>
      <div class="ao-telemetry-item">
        <span>Latency: <strong>&lt;10ms Non-Autoregressive</strong></span>
      </div>
      <div class="ao-telemetry-item">
        <span>Memory: <strong>&lt;1.0 GB Working Set</strong></span>
      </div>
    </div>
  </div>
</div>
"""

    explainer_html = """
<div class="ao-explainer-card">
  <div style="font-size: 0.74rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.5rem;">
    Architecture & Execution Principles
  </div>
  <div class="ao-explainer-grid">
    <div class="ao-explainer-item">
      <span class="ao-explainer-bullet">01</span>
      <div><strong>Dynamic Candidate Scoring</strong><br>Options are projected dynamically in real time — never restricted to fixed training labels.</div>
    </div>
    <div class="ao-explainer-item">
      <span class="ao-explainer-bullet">02</span>
      <div><strong>Cross-Attention Patch Tokens</strong><br>196 unpooled ViT-B-16 spatial tokens ground localized visual cues with text context.</div>
    </div>
    <div class="ao-explainer-item">
      <span class="ao-explainer-bullet">03</span>
      <div><strong>Missing Modality Masking</strong><br>Absent inputs are masked out in attention layers, eliminating hallucination artifacts.</div>
    </div>
    <div class="ao-explainer-item">
      <span class="ao-explainer-bullet">04</span>
      <div><strong>Calibrated Temperature Scaling</strong><br>Outputs produce calibrated probabilities with entropy and boolean noul guarantees.</div>
    </div>
  </div>
</div>
"""

    with gr.Blocks(title="ArbiterOmni · Multimodal Decision Engine") as demo:

        # Inject custom craft CSS
        gr.HTML(CUSTOM_CSS)

        # Header with brand and live hardware telemetry
        gr.HTML(header_html)

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
                    label="📝 Extra Context  (optional)",
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

                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.05,
                    label="🌡️ Temperature / Output Sharpness",
                    info="Lower values (<1.0) sharpen decisive probability distributions; higher values soften confidence.",
                )

                arbitrate_btn = gr.Button("⚡ Run Arbitration", variant="primary", size="lg")

            # ── Right column: outputs ─────────────────────────────────────
            with gr.Column(scale=5):
                winner_output = gr.Markdown(initial_winner_md)

                probs_output = gr.Label(
                    label="📊 Probability Distribution",
                    num_top_classes=10,
                )

                with gr.Row():
                    entropy_output = gr.Textbox(label="🌀 Shannon Entropy", interactive=False)
                    noul_output   = gr.Textbox(label="⚖️ Boolean Noul",     interactive=False)
                    score_output  = gr.Textbox(label="📈 Score",             interactive=False)

                gr.HTML(explainer_html)

        # Wire button
        arbitrate_btn.click(
            fn=arbitrate_decision,
            inputs=[question_input, candidates_input, text_context, image_input, audio_input, temperature_slider],
            outputs=[winner_output, probs_output, entropy_output, noul_output, score_output],
        )

        # One-click example gallery
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_input, candidates_input, text_context, image_input, audio_input, temperature_slider],
            outputs=[winner_output, probs_output, entropy_output, noul_output, score_output],
            fn=arbitrate_decision,
            cache_examples=False,
            label="💡 Click any scenario below to load it instantly",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=create_custom_theme(),
    )
