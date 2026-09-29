<div align="center">

# ArbiterOmni ⚡
### Open-Source Multimodal System 1 Decision Engine
> **State in, calibrated probabilistic decisions out.** Single forward-pass arbitration across text, images, video, and audio with dynamic candidate scoring and zero-leakage missing modality handling.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![uv](https://img.shields.io/badge/Environment-uv-DE5FE9?style=flat-square&logo=astral&logoColor=white)](https://astral.sh/uv)
[![OpenCLIP](https://img.shields.io/badge/Encoders-OpenCLIP%20%2B%20Spectral-059669?style=flat-square)](https://github.com/mlfoundations/open_clip)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-10B981?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![Architecture](https://img.shields.io/badge/Paradigm-System%201%20Decision-8B5CF6?style=flat-square)](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

[Architecture](#-architecture) • [Quickstart](#-quickstart) • [Mathematical-Grounding](#-mathematical-formulation) • [Jev-Primitives](#-jev-system-1-primitives) • [Benchmarks](#-benchmarks--empirical-evaluation) • [Extensibility](#-extending-encoders--fusion)

</div>

---

## ⚡ What is ArbiterOmni?

Traditional Vision-Language Models (VLMs) operate as **System 2 conversationalists**: they generate answers token-by-token autoregressively. While flexible, this design incurs severe liabilities when deployed in robotics, autonomous triage, safety interlocks, and agentic routers:
- **High Latency & Compute**: Decoding 100 tokens takes 800–3,000 ms.
- **Syntactic Drift & Hallucination**: Output parsing requires complex JSON regex or constrained grammars.
- **Uncalibrated Confidence**: Logprobs over generated prose do not reflect true posterior task certainty.

**ArbiterOmni** is an open-source, lightweight **System 1 Multimodal Decision Engine** inspired by the architecture of **Jev** (TypeSafe AI). Given multimodal context (text, vision, temporal video frames, audio) and a query with an arbitrary list of candidate options, ArbiterOmni evaluates the entire sensory state in a **single non-autoregressive forward pass** (10–40 ms) and returns a **calibrated softmax probability distribution** directly over the candidates.

```
                              AUTOREGRESSIVE VLM
Sensory State ──> [ 7B-70B LLM Backbone ] ──> Token ... Token ... Token ──> Regex / JSON Parse (800-3000ms)
                                                                             ❌ Drift / Hallucination

                                 ARBITER-OMNI
Sensory State ──> [ Frozen Encoders ] ──> [ Attention Fusion ] ──> Calibrated P(Candidate) (10-40ms)
                                                                    ✅ Deterministic / Typed
```

---

## 🌟 Core Highlights

- **Quad-Modal Perception**: Accepts raw text, PIL/tensor images, video keyframe sequences, and audio waveforms.
- **Graceful Missing-Modality Handling**: Modalities can be dropped dynamically. Attention key-padding masks and learned modality-type embeddings eliminate zero-vector skew and phantom activations.
- **Dynamic Candidate Scoring**: Candidate options are **never** hardcoded into fixed classifier classes. Any arbitrary set of $K \ge 2$ text options can be evaluated on-the-fly at inference time.
- **Frozen Encoders + Lightweight Fusion**: Multimodal encoders (OpenCLIP, spectral transformers) remain frozen. Only a lightweight cross-attention fusion network and decision scoring head (~590k parameters) are trained, converging in minutes even on modest CPUs.
- **Jev-Compatible Decision Primitives**: Native support for **Choice** (categorical softmax), **Boolean / Noul** (calibrated binary certainty), and **Score** (continuous ordinal regression).
- **Hardware Agnostic**: Runs seamlessly on CPU, Apple Silicon, NVIDIA GPUs (CUDA), and AMD GPUs (ROCm / Radeon RX 480). Memory footprint remains $<1$ GB.

---

## 📐 Architecture

ArbiterOmni decouples sensory feature extraction from decision arbitration:

```mermaid
flowchart TD
    subgraph Inputs["1. Multimodal Input Channels"]
        T["Text Context"]
        I["RGB Image"]
        V["Video Keyframes"]
        A["Audio Waveform"]
        Q["Decision Question"]
        C["Candidate Options [C_1, ..., C_K]"]
    end

    subgraph Encoders["2. Frozen Multimodal Encoders"]
        E_T["Frozen Text Encoder"]
        E_I["Frozen Vision Encoder"]
        E_V["Temporal Frame Pooling"]
        E_A["Spectral Audio Projection"]
        E_C["Frozen Candidate Tokenizer"]
    end

    subgraph Fusion["3. Transformer Cross-Attention Fusion"]
        Mask["Dynamic Modality Presence Mask\n(Key-Padding Mask)"]
        TypeEmb["Learned Modality Type Embeddings\n[Query, Q, Text, Img, Vid, Aud]"]
        Query["Learned [DECISION_QUERY] Latent Token"]
        Transformer["Multi-Head Cross Attention Layers"]
    end

    subgraph DecisionHead["4. Dynamic Decision Interaction Head"]
        ScoreManifold["Bilinear + Cross-MLP Interaction\ns_k = (u_ctx · u_cand_k)/√d + MLP([u_ctx || u_cand_k])"]
        Temp["Learnable Temperature Scaling (τ)"]
    end

    subgraph Outputs["5. Calibrated System 1 Outputs"]
        Choice["Categorical Softmax: P(C_k)"]
        Entropy["Shannon Entropy: H(P)"]
        Noul["Jev Boolean Noul: Calibrated Certainty"]
        Winner["Top-1 Winning Decision"]
    end

    T --> E_T
    I --> E_I
    V --> E_V
    A --> E_A
    Q --> E_T
    C --> E_C

    E_T & E_I & E_V & E_A --> Mask
    Mask & TypeEmb & Query --> Transformer
    E_T -.->|Question Embed| Transformer

    Transformer -->|Fused Context Vector| ScoreManifold
    E_C -->|Candidate Embeds| ScoreManifold
    ScoreManifold --> Temp
    Temp --> Choice & Entropy & Noul & Winner
```

---

## 🔬 Mathematical Formulation

### 1. Missing-Modality Masking & Token Assembly
Let $\mathcal{M} = \{\text{text}, \text{image}, \text{video}, \text{audio}\}$ represent the input modalities. Each present modality $m \in \mathcal{M}$ produces an embedding $\mathbf{x}_m \in \mathbb{R}^{d_m}$. Missing modalities are assigned arbitrary zero vectors and marked in the boolean key-padding mask $\mathbf{M} \in \{0, 1\}^{L}$:

$$\mathbf{t}_m = \text{LayerNorm}(\mathbf{W}_m \mathbf{x}_m) + \mathbf{e}_{\text{type}}(m)$$

$$\mathbf{M}_j = \begin{cases} 0 & \text{if token } j \text{ is valid and present} \\ 1 & \text{if modality } j \text{ is missing (masked out)} \end{cases}$$

The sequence $\mathbf{T} = [\mathbf{t}_{\text{query}}, \mathbf{t}_{\text{question}}, \mathbf{t}_{\text{text}}, \mathbf{t}_{\text{image}}, \mathbf{t}_{\text{video}}, \mathbf{t}_{\text{audio}}] \in \mathbb{R}^{6 \times d_h}$ is processed by a multi-head transformer with scaled dot-product attention:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{attn}}\right) \mathbf{V}$$

The updated representation at index 0 yields the unified multimodal context state $\mathbf{z}_{\text{context}} \in \mathbb{R}^{d_h}$.

### 2. Dynamic Candidate Interaction
Given $K$ runtime candidate strings $\{c_1, \dots, c_K\}$, candidate embeddings $\mathbf{e}_{c_k}$ interact with $\mathbf{z}_{\text{context}}$ via dual projection:

$$\mathbf{u}_{\text{context}} = \text{LayerNorm}(\mathbf{W}_u \mathbf{z}_{\text{context}}), \quad \mathbf{u}_{c_k} = \text{LayerNorm}(\mathbf{W}_c \mathbf{e}_{c_k})$$

The unnormalized compatibility logit $s_k$ combines bilinear similarity with non-linear cross-MLP interaction:

$$s_k = \frac{\mathbf{u}_{\text{context}}^\top \mathbf{u}_{c_k}}{\sqrt{d_s}} + \text{MLP}_{\text{cross}}([\mathbf{u}_{\text{context}} \,\|\, \mathbf{u}_{c_k}])$$

The calibrated probability distribution is computed via learnable temperature $\tau$:

$$P(c_k \mid \mathbf{X}_{\text{multimodal}}, Q, \{c_j\}) = \frac{\exp(s_k / \tau)}{\sum_{j=1}^K \exp(s_j / \tau)}$$

Decision uncertainty is monitored directly using Shannon Entropy:

$$H(P) = -\sum_{k=1}^K P(c_k) \ln P(c_k)$$

---

## 🚀 Quickstart

### Installation

Clone the repository and install dependencies with [`uv`](https://astral.sh/uv) (recommended) or `pip`:

```bash
git clone https://github.com/cameronduff/arbiter-omni.git
cd arbiter-omni

# Create virtualenv and sync dependencies
uv sync

# Or with pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 1. Minimal Inference (Single Call)

```python
from arbiter_omni import ArbiterOmniEngine
from PIL import Image

# Initialize engine (uses OpenCLIP or lightweight mock for offline testing)
engine = ArbiterOmniEngine.create(encoder_type="mock")

# Arbitrate a decision over dynamic candidates with an image and sensor note
result = engine.decide(
    question="What is the immediate action for the autonomous mobile robot?",
    candidates=[
        "halt immediately and apply brakes",
        "proceed forward at nominal speed",
        "steer left around obstacle",
        "request remote operator assistance",
    ],
    image=Image.new("RGB", (224, 224), color=(220, 20, 20)), # Visual hazard
    text="CRITICAL: LiDAR detected static obstacle at 0.3 meters.",
)

print(f"Winner:     {result.winner}")
print(f"Confidence: {result.confidence * 100:.2f}%")
print(f"Entropy:    {result.entropy:.3f} nats")
print(f"Full Distribution: {result.probabilities}")
```

### 2. Missing Modalities in Action

ArbiterOmni seamlessly adapts if sensory channels drop. Here, **only audio is supplied** (no image, video, or text context):

```python
import numpy as np

# Audio waveform (e.g. 1800Hz industrial bearing screech)
t = np.linspace(0, 0.5, 8000, endpoint=False)
screech_audio = (0.5 * np.sin(2 * np.pi * 1800 * t)).astype(np.float32)

result = engine.decide(
    question="What is the appropriate triage command for the assembly line?",
    candidates=[
        "trigger emergency facility shutdown",
        "maintain nominal line operation",
        "reroute component to quality inspection",
        "dispatch maintenance technician",
    ],
    audio=screech_audio, # Image, video, and text are omitted
)

print(f"Active Channels: {result.active_modalities}") # ['audio']
print(f"Decision:        {result.winner} ({result.confidence * 100:.1f}%)")
```

---

## 🎯 Jev System 1 Primitives

ArbiterOmni supports all three canonical System 1 decision types documented in the [Jev Specification](https://docs.typesafe.ai/primitives):

| Primitive | Description | Output Structure |
| :--- | :--- | :--- |
| **`Choice`** | Multi-class arbitration over $K \ge 2$ dynamic options. | `probabilities: Dict[str, float]`, `winner: str`, `confidence: float` |
| **`Noul`** | Binary true/false verification with calibrated certainty score. | `boolean_noul: {"true_probability": 0.94, "calibrated_certainty": 0.88}` |
| **`Score`** | Continuous regression along an ordinal / continuous rating scale. | `score: float` |

---

## 📊 Benchmarks & Empirical Evaluation

We benchmarked ArbiterOmni against a synthetic cross-modal triage benchmark spanning robotics navigation and industrial equipment safety:

### Verification Results (`examples/e2e_demo.py`)

- **Training Scale**: 100 training samples, 20 validation samples with **35% random modality dropout**.
- **Optimization**: 6 epochs of AdamW ($\text{lr}=3\times 10^{-3}$) on CPU.
- **Convergence**:
  - **Train Accuracy**: **100.0%**
  - **Validation Accuracy**: **85.0%**
  - **Average Decision Latency**: **~12 ms / query** (single thread CPU)

| Test Scenario | Active Modalities | Ground Truth | ArbiterOmni Prediction | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **Robotics Obstacle** | `[Text, Image, Video, Audio]` | Emergency Brake | `halt immediately and apply brakes` | **98.41%** |
| **Bearing Failure** | `[Audio]` *(Vis/Text missing)* | Line Shutdown | `trigger emergency facility shutdown` | **96.40%** |
| **Safety Check** | `[Text, Audio]` | Normal Operation | `Normal Safe Operation` | **50.31%** (calibrated) |

### Autoregressive VLM vs. ArbiterOmni

| Dimension | Autoregressive VLM (e.g. LLaVA-1.5, Qwen-VL) | ArbiterOmni (System 1) |
| :--- | :--- | :--- |
| **Forward Passes** | 50 – 250 iterative auto-regressive steps | **1 deterministic pass** |
| **Latency** | 800 ms – 3,500 ms | **10 ms – 45 ms** |
| **Output Type** | Unstructured tokens (strings) | **Calibrated probability distribution** |
| **Missing Modalities** | Requires explicit prompt engineering / N/A tokens | **Zero-leakage hardware attention mask** |
| **Candidate Flexibility**| Hallucinates options outside candidate list | **Mathematically constrained to candidate set** |
| **VRAM Footprint** | 8 GB – 24 GB | **< 1.0 GB** |

---

## 🔧 Extending Encoders & Fusion

ArbiterOmni is strictly modular. Swapping the vision backbone or fusion layer requires no changes to the decision head or inference API:

### Plugging in a Custom Encoder
Inherit from [`BaseMultimodalEncoder`](file:///home/cd_server/repositories/arbiter-omni/src/arbiter_omni/encoders/base.py):

```python
from arbiter_omni.encoders.base import BaseMultimodalEncoder
import torch

class CustomRoboticsEncoder(BaseMultimodalEncoder):
    @property
    def text_dim(self) -> int: return 768
    @property
    def image_dim(self) -> int: return 768
    @property
    def video_dim(self) -> int: return 768
    @property
    def audio_dim(self) -> int: return 768

    def encode_text(self, texts): ...
    def encode_image(self, images): ...
    def encode_video(self, videos, num_frames=8): ...
    def encode_audio(self, audios, sample_rate=16000): ...
```

### Plugging in a Custom Fusion Layer
Inherit from [`BaseMultimodalFusion`](file:///home/cd_server/repositories/arbiter-omni/src/arbiter_omni/fusion/base.py) or use the built-in [`GatedMultimodalFusion`](file:///home/cd_server/repositories/arbiter-omni/src/arbiter_omni/fusion/gated.py):

```python
from arbiter_omni.fusion import GatedMultimodalFusion

gmu_fusion = GatedMultimodalFusion(
    modality_dims={"question": 512, "text": 512, "image": 512, "video": 512, "audio": 512},
    hidden_dim=256,
)
```

---

## 📂 Repository Structure

```
arbiter-omni/
├── pyproject.toml               # PEP 621 dependencies & uv configuration
├── README.md                    # Comprehensive documentation & architecture guide
├── AGENTS.md                    # Environment, safety, and hardware rules
├── PROGRESS.md                  # Experiment logs and milestone tracking
├── src/
│   └── arbiter_omni/
│       ├── __init__.py          # Top-level exports
│       ├── types.py             # Pydantic models (MultimodalSample, DecisionResult)
│       ├── encoders/
│       │   ├── base.py          # BaseMultimodalEncoder abstract interface
│       │   ├── openclip.py      # OpenCLIP + STFT audio spectral projection
│       │   └── mock.py          # Fast zero-download deterministic mock encoder
│       ├── fusion/
│       │   ├── base.py          # BaseMultimodalFusion interface
│       │   ├── transformer.py   # Perceiver-style cross-attention with padding masks
│       │   └── gated.py         # Gated Multimodal Unit (GMU) alternative
│       ├── model/
│       │   ├── decision_head.py # Dynamic candidate bilinear + MLP interaction head
│       │   └── arbiter.py       # ArbiterOmniModel uniting encoders, fusion, & heads
│       ├── data/
│       │   ├── dataset.py       # MultimodalDecisionDataset & variable collate_fn
│       │   └── synthetic.py     # Robotics & industrial safety scenario generator
│       ├── training/
│       │   ├── config.py        # TrainingConfig dataclass
│       │   └── trainer.py       # Lightweight AdamW trainer with temperature tracking
│       └── api/
│           └── engine.py        # ArbiterOmniEngine (decide, decide_batch)
├── examples/
│   └── e2e_demo.py              # End-to-end training and inference walkthrough
└── tests/
    ├── test_types.py            # Pydantic schema validation
    ├── test_encoders.py         # Shape, normalization, and freeze checks
    ├── test_fusion.py           # Missing-modality masking verification
    ├── test_decision_head.py    # Variable candidate scoring ($K=2,3,5,10$)
    ├── test_dataset.py          # Generator and collator integrity
    └── test_e2e_pipeline.py     # Checkpoint save/load and inference tests
```

---

## 🧪 Testing

Run the test suite via `uv`:

```bash
uv run pytest
```

All 12 test suites execute in under 12 seconds on standard CPU hardware.

---

## 📜 Citation & Lineage

ArbiterOmni builds upon ideas from:
1. **TypeSafe AI Jev**: The System 1 decision-making paradigm, discrete typed primitives (`Choice`, `Score`, `Noul`), and RLCD calibration ([TypeSafe Blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).
2. **Awesome JEV Gallery**: Open-source reproductions, benchmarks, and lineage analysis ([Awesome JEV Gallery](https://github.com/OmniJev/awesome-jev-gallery)).
3. **Perceiver & Perceiver IO**: Latent cross-attention mechanisms for arbitrary multimodal inputs ([Jaegle et al., 2021](https://arxiv.org/abs/2103.03206)).

```bibtex
@misc{arbiteromni2026,
  author = {Cameron Duff and Contributors},
  title = {ArbiterOmni: An Open-Source Multimodal System 1 Decision Engine},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/cameronduff/arbiter-omni}}
}
```

---

## 📄 License

ArbiterOmni is licensed under the [Apache 2.0 License](LICENSE).
