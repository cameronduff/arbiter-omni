"""
Synthetic Multimodal Decision Dataset Generator.
Generates realistic, cross-modal decision tasks (e.g. robotics action selection,
safety triage, autonomous routing) with controlled missing modality rates.
"""

from __future__ import annotations

import random
from typing import List, Optional
import numpy as np
from PIL import Image

from arbiter_omni.types import MultimodalSample


def create_synthetic_image(color: tuple, pattern: str = "flat", size: tuple = (64, 64)) -> Image.Image:
    """Creates a synthetic colored image representing sensory perception."""
    arr = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    arr[:, :] = color
    if pattern == "cross":
        arr[size[0] // 2 - 4 : size[0] // 2 + 4, :] = 255
        arr[:, size[1] // 2 - 4 : size[1] // 2 + 4] = 255
    elif pattern == "stripe":
        arr[::8, :] = 255
    return Image.fromarray(arr)


def create_synthetic_audio(freq: float, duration: float = 0.5, sr: int = 16000) -> np.ndarray:
    """Generates synthetic audio waveform with pure tones and harmonic noise."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = 0.05 * np.random.randn(len(t))
    waveform = (sine + noise).astype(np.float32)
    return waveform


def generate_synthetic_dataset(
    num_samples: int = 100,
    missing_modality_prob: float = 0.35,
    seed: int = 42,
) -> List[MultimodalSample]:
    """
    Generates a generic multimodal decision dataset with dynamic candidate choices
    and controlled missing modalities.
    
    Tasks span:
    1. Robotics / VLA Action Selection (obstacle vs clear vs target item)
    2. Facility Safety & Triage (normal vs fire hazard vs intrusion)
    3. Industrial Equipment Monitoring (bearing failure vs normal run)
    """
    random.seed(seed)
    np.random.seed(seed)

    scenarios = [
        # Scenario 1: Robotics Obstacle vs Clear Path
        {
            "question": "What is the immediate action for the autonomous mobile robot?",
            "candidates": [
                "halt immediately and apply brakes",
                "proceed forward at nominal speed",
                "steer left around obstacle",
                "request remote operator assistance",
            ],
            "actions": [
                {
                    "target_idx": 0,
                    "text": "CRITICAL: LiDAR detected static barrier at 0.4 meters.",
                    "img_color": (220, 20, 20),  # Red alert
                    "img_pattern": "cross",
                    "audio_freq": 1200.0,        # High-pitch alarm
                },
                {
                    "target_idx": 1,
                    "text": "STATUS: Path clear. Corridor unobstructed.",
                    "img_color": (20, 180, 20),  # Green clear
                    "img_pattern": "flat",
                    "audio_freq": 220.0,         # Low hum
                },
                {
                    "target_idx": 2,
                    "text": "WARNING: Obstacle located on starboard side.",
                    "img_color": (220, 180, 20), # Yellow steer
                    "img_pattern": "stripe",
                    "audio_freq": 600.0,         # Mid chime
                },
            ],
        },
        # Scenario 2: Industrial Safety & Anomaly Triage
        {
            "question": "What is the appropriate triage command for the assembly line?",
            "candidates": [
                "trigger emergency facility shutdown",
                "maintain nominal line operation",
                "reroute component to quality inspection",
                "dispatch maintenance technician",
            ],
            "actions": [
                {
                    "target_idx": 0,
                    "text": "THERMAL HAZARD: Bearing temperature exceeding 140C with rapid friction rise.",
                    "img_color": (240, 40, 10),
                    "img_pattern": "cross",
                    "audio_freq": 1800.0,
                },
                {
                    "target_idx": 1,
                    "text": "SYSTEM OK: Vibration telemetry within +/- 2% baseline tolerances.",
                    "img_color": (30, 200, 80),
                    "img_pattern": "flat",
                    "audio_freq": 150.0,
                },
                {
                    "target_idx": 2,
                    "text": "SURFACE DEFECT: Optical camera notes dimensional variance on part #841.",
                    "img_color": (80, 120, 220),
                    "img_pattern": "stripe",
                    "audio_freq": 440.0,
                },
            ],
        },
    ]

    samples: List[MultimodalSample] = []

    for i in range(num_samples):
        scenario = random.choice(scenarios)
        action_spec = random.choice(scenario["actions"])

        target_idx = action_spec["target_idx"]
        question = scenario["question"]
        candidates = list(scenario["candidates"])

        # Create modality assets
        text_content = action_spec["text"]
        img_content = create_synthetic_image(action_spec["img_color"], pattern=action_spec["img_pattern"])
        # Video: 3 synthetic frames with slight color shifting
        video_content = [
            img_content,
            create_synthetic_image(action_spec["img_color"], pattern=action_spec["img_pattern"]),
            create_synthetic_image(action_spec["img_color"], pattern="flat"),
        ]
        audio_content = create_synthetic_audio(action_spec["audio_freq"])

        # Randomly drop modalities to enforce graceful missing modality handling
        # Ensure at least ONE non-question modality is retained
        keep_text = random.random() > missing_modality_prob
        keep_img = random.random() > missing_modality_prob
        keep_vid = random.random() > missing_modality_prob
        keep_aud = random.random() > missing_modality_prob

        # If all dropped, force keep at least one
        if not (keep_text or keep_img or keep_vid or keep_aud):
            keep_img = True

        sample = MultimodalSample(
            question=question,
            candidates=candidates,
            target_idx=target_idx,
            text=text_content if keep_text else None,
            image=img_content if keep_img else None,
            video=video_content if keep_vid else None,
            audio=audio_content if keep_aud else None,
            metadata={"sample_id": i, "scenario": scenario["question"][:25]},
        )
        samples.append(sample)

    return samples
