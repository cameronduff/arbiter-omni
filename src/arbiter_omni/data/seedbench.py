"""
SEED-Bench & SEED-Bench-2 Multimodal Dataset Adapter.
Supports Image and Video multi-choice question answering with dynamic 4-candidate options.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from PIL import Image

from arbiter_omni.types import MultimodalSample

logger = logging.getLogger(__name__)


class SEEDBenchAdapter:
    """
    Adapter converting SEED-Bench records into ArbiterOmni MultimodalSample format.
    Supports both static Image questions and continuous Video Clip questions.
    """

    LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

    @classmethod
    def record_to_sample(
        cls,
        record: Dict[str, Any],
        image: Optional[Any] = None,
        video_frames: Optional[Sequence[Any]] = None,
    ) -> MultimodalSample:
        """
        Converts a SEED-Bench record into a MultimodalSample.
        """
        question = record.get("question", "")
        choices = [
            record.get("choice_a", ""),
            record.get("choice_b", ""),
            record.get("choice_c", ""),
            record.get("choice_d", ""),
        ]

        raw_answer = str(record.get("answer", "A")).strip().upper()
        target_idx = cls.LETTER_TO_INDEX.get(raw_answer, 0)
        data_type = record.get("data_type", "image")

        metadata = {
            "dataset": "SEED-Bench-2",
            "data_type": data_type,
            "question_id": str(record.get("question_id", "")),
            "data_id": str(record.get("data_id", "")),
            "dimension": record.get("dimension", ""),
        }

        # Assign media according to data_type
        sample_img = image if data_type == "image" else None
        sample_vid = video_frames if data_type == "video" else None

        return MultimodalSample(
            question=question,
            candidates=choices,
            target_idx=target_idx,
            image=sample_img,
            video=sample_vid,
            metadata=metadata,
        )


def create_mock_seedbench_samples(num_samples: int = 20, seed: int = 42) -> List[MultimodalSample]:
    """
    Generates realistic synthetic SEED-Bench samples covering both static visual reasoning
    and multi-frame video action sequences.
    """
    rng = np.random.default_rng(seed)
    archetypes = [
        # Image Comprehension: Spatial Relations
        {
            "data_type": "image",
            "question": "Where is the red mug located relative to the laptop on the desk?",
            "choice_a": "To the right of the laptop",
            "choice_b": "Behind the laptop screen",
            "choice_c": "Directly on top of the keyboard",
            "choice_d": "Underneath the desk",
            "answer": "A",
            "color": (210, 50, 40),
            "dimension": "Spatial Comprehension",
        },
        # Video Action Reasoning: Sequential Motion Direction
        {
            "data_type": "video",
            "question": "In this video clip, in which direction is the robotic arm moving the pallet?",
            "choice_a": "Moving forward and lowering into bin",
            "choice_b": "Spinning clockwise 360 degrees",
            "choice_c": "Retracting backwards towards home position",
            "choice_d": "Remaining completely stationary",
            "answer": "A",
            "color": (40, 120, 220),
            "dimension": "Action Recognition",
        },
        # Image Comprehension: Scene Recognition
        {
            "data_type": "image",
            "question": "What primary industrial environment is depicted in the surveillance feed?",
            "choice_a": "Automated warehouse sorting facility",
            "choice_b": "Residential kitchen dining area",
            "choice_c": "Outdoor public park",
            "choice_d": "Commercial airline cockpit",
            "answer": "A",
            "color": (140, 140, 150),
            "dimension": "Scene Understanding",
        },
        # Video Action Reasoning: Dynamic Causal Event
        {
            "data_type": "video",
            "question": "What happens after the conveyor belt stops moving in the video?",
            "choice_a": "A safety technician inspects the roller bearings",
            "choice_b": "The assembly line catches on fire",
            "choice_c": "Boxes spill off the side of the platform",
            "choice_d": "The machine immediately accelerates to maximum speed",
            "answer": "A",
            "color": (60, 180, 80),
            "dimension": "Dynamic Causal Reasoning",
        },
    ]

    samples: List[MultimodalSample] = []
    for i in range(num_samples):
        arch = archetypes[i % len(archetypes)]
        base_color = arch["color"]

        if arch["data_type"] == "video":
            # Generate a 4-frame video sequence with slight shift
            frames = []
            for f in range(4):
                col = tuple(min(255, max(0, c + (f * 15))) for c in base_color)
                f_img = Image.new("RGB", (224, 224), color=col)
                frames.append(f_img)
            sample = SEEDBenchAdapter.record_to_sample(
                record=arch,
                video_frames=frames,
            )
        else:
            img = Image.new("RGB", (224, 224), color=base_color)
            sample = SEEDBenchAdapter.record_to_sample(
                record=arch,
                image=img,
            )

        samples.append(sample)

    return samples


def load_seedbench_dataset(
    max_samples: Optional[int] = 50,
    use_mock_fallback: bool = True,
) -> List[MultimodalSample]:
    """
    Loads SEED-Bench samples or falls back to synthetic mock samples when offline.
    """
    try:
        import urllib.request
        logger.info("Attempting to load SEED-Bench question metadata from HuggingFace...")
        url = "https://huggingface.co/datasets/AILab-CVC/SEED-Bench/raw/main/SEED-Bench.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            questions = data.get("questions", [])
            samples: List[MultimodalSample] = []
            for q in questions[: (max_samples or 50)]:
                # Generate blank/proxy image for metadata evaluation
                dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                sample = SEEDBenchAdapter.record_to_sample(q, image=dummy_img)
                samples.append(sample)
            logger.info(f"Loaded {len(samples)} SEED-Bench records from HuggingFace.")
            return samples
    except Exception as e:
        if use_mock_fallback:
            logger.warning(
                f"Could not load remote SEED-Bench ({e}). Using {max_samples or 20} mock SEED-Bench samples."
            )
            return create_mock_seedbench_samples(num_samples=max_samples or 20)
        raise e
