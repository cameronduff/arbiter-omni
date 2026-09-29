"""
Unit tests for Robotics Action Decision Adapter and Open X-Embodiment System 1 arbitration.
"""

from PIL import Image
import pytest

from arbiter_omni.data.robotics import (
    RoboticsActionAdapter,
    generate_robotics_samples,
    STANDARD_ROBOTICS_ACTIONS,
)
from arbiter_omni.types import MultimodalSample


def test_robotics_adapter_observation_to_sample():
    img = Image.new("RGB", (64, 64), color="red")
    sample = RoboticsActionAdapter.observation_to_sample(
        camera_image=img,
        task_instruction="Emergency stop required immediately",
        candidates=["emergency_halt", "navigate_forward"],
        target_action_idx=0,
    )

    assert isinstance(sample, MultimodalSample)
    assert sample.question == "Emergency stop required immediately"
    assert sample.candidates == ["emergency_halt", "navigate_forward"]
    assert sample.target_idx == 0
    assert sample.image is not None


def test_generate_robotics_samples():
    samples = generate_robotics_samples(num_samples=25, seed=123)
    assert len(samples) == 25
    for s in samples:
        assert isinstance(s, MultimodalSample)
        assert len(s.candidates) >= 2
        assert 0 <= s.target_idx < len(s.candidates)
        assert s.image is not None
        assert s.audio is not None
