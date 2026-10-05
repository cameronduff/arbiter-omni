"""
Robotics Action Decision Adapter inspired by Open X-Embodiment (RT-X / BridgeData).
Formulates continuous robotic perception and dynamic candidate action arbitration
for real-time System 1 robot control and safety triage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from PIL import Image

from arbiter_omni.types import MultimodalSample

logger = logging.getLogger(__name__)

STANDARD_ROBOTICS_ACTIONS = [
    "emergency_halt",
    "navigate_forward",
    "turn_left",
    "turn_right",
    "align_gripper",
    "grasp_target",
    "lift_object",
    "place_object",
]


class RoboticsActionAdapter:
    """
    Adapter converting robotic observations and task instructions into MultimodalSample records
    over dynamic candidate action sets.
    """

    @staticmethod
    def observation_to_sample(
        camera_image: Any,
        task_instruction: str,
        candidates: Optional[Sequence[str]] = None,
        target_action_idx: Optional[int] = None,
        audio_telemetry: Optional[Any] = None,
        video_sequence: Optional[Sequence[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalSample:
        """
        Creates a MultimodalSample for robotic decision arbitration.
        """
        cand_list = list(candidates) if candidates is not None else list(STANDARD_ROBOTICS_ACTIONS)
        return MultimodalSample(
            question=task_instruction,
            candidates=cand_list,
            target_idx=target_action_idx,
            image=camera_image,
            video=video_sequence,
            audio=audio_telemetry,
            metadata=metadata or {"dataset": "OpenX_Robotics"},
        )


def generate_robotics_samples(num_samples: int = 40, seed: int = 42) -> List[MultimodalSample]:
    """
    Synthesizes a realistic robotics evaluation set covering navigation, manipulation,
    and acoustic safety emergency triage.
    """
    rng = np.random.default_rng(seed)
    scenarios = [
        {
            "instruction": "Emergency! Obstacle rapidly approaching robot path.",
            "color": (255, 30, 30),  # Red barrier
            "sound": "screech",      # High alert acoustic anomaly
            "choices": ["emergency_halt", "navigate_forward", "grasp_target"],
            "target": 0,             # emergency_halt
        },
        {
            "instruction": "Corridor is clear. Proceed with autonomous navigation to goal.",
            "color": (30, 200, 30),  # Green clear path
            "sound": "hum",          # Low normal motor hum
            "choices": ["navigate_forward", "emergency_halt", "turn_left"],
            "target": 0,             # navigate_forward
        },
        {
            "instruction": "Path blocked ahead. Turn left around corner.",
            "color": (230, 140, 20), # Orange corner marker
            "sound": "hum",
            "choices": ["turn_left", "turn_right", "navigate_forward", "emergency_halt"],
            "target": 0,             # turn_left
        },
        {
            "instruction": "Red cube is centered in gripper view. Execute grasp.",
            "color": (200, 40, 40),  # Red cube
            "sound": "hum",
            "choices": ["grasp_target", "lift_object", "place_object", "navigate_forward"],
            "target": 0,             # grasp_target
        },
        {
            "instruction": "Object secured in gripper. Lift to transport height.",
            "color": (50, 50, 220),  # Blue container/table
            "sound": "hum",
            "choices": ["lift_object", "place_object", "emergency_halt"],
            "target": 0,             # lift_object
        },
    ]

    samples: List[MultimodalSample] = []
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)

    for i in range(num_samples):
        sc = scenarios[i % len(scenarios)]
        # Synthetic camera frame
        base_color = sc["color"]
        img = Image.new("RGB", (224, 224), color=base_color)
        arr = np.array(img).astype(np.float32) + rng.integers(-15, 15, size=(224, 224, 3))
        noisy_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # Acoustic waveform
        if sc["sound"] == "screech":
            # 2 kHz screech tone + noise
            wave = 0.5 * np.sin(2 * np.pi * 2000 * t) + 0.1 * rng.normal(size=len(t))
        else:
            # 120 Hz low hum
            wave = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.05 * rng.normal(size=len(t))
        wave = wave.astype(np.float32)

        sample = RoboticsActionAdapter.observation_to_sample(
            camera_image=noisy_img,
            task_instruction=sc["instruction"],
            candidates=sc["choices"],
            target_action_idx=sc["target"],
            audio_telemetry=wave,
            metadata={"dataset": "OpenX_Robotics_Sim", "sample_id": i},
        )
        samples.append(sample)

    return samples
