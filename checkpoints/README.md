# ArbiterOmni Checkpoints

This directory contains trained checkpoints for the ArbiterOmni System 1 decision engine.

## Production `v1` Checkpoint (`arbiter_omni_v1.pt`)

- **Parameters Trained**: 2,210,052 (Fusion + Dynamic Decision Head)
- **Frozen Perception**: OpenCLIP ViT-B-32 (151.27M params, frozen)
- **Trained Datasets**: ScienceQA, SEED-Bench-2, Open X-Embodiment Robotics Actions, Synthetic Sensor Telemetry.
- **Accuracy**: 85.0% Training Accuracy | 86.7% Validation Accuracy
- **File Size**: ~8.46 MB

### How to Generate
```bash
uv run python scripts/train_v1.py --epochs 3 --batch-size 32
```

### How to Load
```python
from arbiter_omni import ArbiterOmniEngine

engine = ArbiterOmniEngine.from_pretrained("v1", encoder_type="openclip")
result = engine.decide(question="...", candidates=["Choice A", "Choice B"])
```
