# ArbiterOmni

ArbiterOmni is a prototype multimodal System 1 decision engine inspired by Jev (TypeSafe AI).

## Features
- **Multimodal Inputs**: Accepts text, images, video (temporal frames), and audio.
- **Graceful Missing Modality Handling**: Missing inputs are masked with zero-leakage attention masking and learned modality-type indicators.
- **Dynamic Candidate Scoring**: Scores arbitrary candidate decision sets passed at runtime via deep multimodal state-to-candidate interactions.
- **Calibrated Softmax Probability Distributions**: Outputs discrete categorical confidence, Jev boolean/noul certainty, and decision entropy rather than autoregressive prose.
- **Frozen Encoders & Lightweight Fusion**: Encoders remain frozen; only cross-attention/fusion projections and scoring heads are trained.
- **Hardware Agnostic**: Seamlessly switches between CPU, CUDA, and ROCm (AMD Radeon RX 480).
