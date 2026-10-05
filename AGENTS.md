# ArbiterOmni: Agent Rules & Guidelines

## 1. Virtual Environment Isolation
- All commands, tests, and examples MUST be executed within `.venv` using `uv run ...` or `.venv/bin/python`.
- Never execute global python without uv virtualenv isolation.

## 2. Hardware Agnostic Device Dispatch
- Support CPU, CUDA, and AMD ROCm (e.g. desktop Radeon RX 480) dynamically:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Keep memory usage lean (<1 GB working memory during inference) to ensure seamless execution on 4 GB RAM servers and 8-16 GB desktop environments.

## 3. Modular Architecture Standards
- **Frozen Encoders**: Multimodal encoders (vision, text, audio) remain strictly frozen during fusion and decision training.
- **Dynamic Candidate Scoring**: Candidate decisions are never hardcoded class indices; they are dynamic text options projected and scored via interaction with the fused multimodal state.
- **Graceful Missing Modality Handling**: Missing inputs must be explicitly masked out in the attention/fusion layers using attention masks and learned modality embeddings, preventing leakage or NaN artifacts.
- **Calibrated Outputs**: Return probability distributions with softmax confidence, decision entropy, and Jev-style boolean/noul confidence.

## 4. Live Documentation & Progress Tracking
- Maintain live updates in `PROGRESS.md` tracking completed components, test coverage, and benchmark results.
