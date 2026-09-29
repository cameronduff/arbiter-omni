"""
ScienceQA Multimodal Multiple-Choice Dataset Adapter.
Loads and adapts multimodal questions with dynamic 2-to-5 candidate options
into ArbiterOmni MultimodalSample records.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np

from arbiter_omni.types import MultimodalSample

logger = logging.getLogger(__name__)


class ScienceQAAdapter:
    """
    Adapter converting HuggingFace ScienceQA records into ArbiterOmni MultimodalSample format.
    """

    @staticmethod
    def record_to_sample(record: Dict[str, Any]) -> MultimodalSample:
        """
        Converts a single ScienceQA record into a MultimodalSample.
        """
        question = record.get("question", "")
        choices = record.get("choices", [])
        answer_idx = record.get("answer", 0)
        image = record.get("image", None)
        hint = record.get("hint", "") or record.get("lecture", "")

        metadata = {
            "dataset": "ScienceQA",
            "subject": record.get("subject", ""),
            "topic": record.get("topic", ""),
            "category": record.get("category", ""),
            "skill": record.get("skill", ""),
        }

        return MultimodalSample(
            question=question,
            candidates=choices,
            target_idx=int(answer_idx) if answer_idx is not None else None,
            image=image,
            text=hint if (hint and len(str(hint).strip()) > 0) else None,
            metadata=metadata,
        )


def create_mock_scienceqa_samples(num_samples: int = 10, seed: int = 42) -> List[MultimodalSample]:
    """
    Generates realistic synthetic ScienceQA multimodal samples for offline validation and tests.
    """
    rng = np.random.default_rng(seed)
    archetypes = [
        {
            "question": "Which animal group does this organism belong to?",
            "choices": ["Amphibian", "Mammal", "Reptile", "Bird"],
            "answer": 0,
            "hint": "Notice the moist, scaleless skin and amphibious lifecycle.",
            "color": (40, 180, 80),  # Green frog proxy
        },
        {
            "question": "Identify the state of matter shown in the observation.",
            "choices": ["Solid", "Liquid", "Gas"],
            "answer": 1,
            "hint": "The substance takes the shape of its container with a defined volume.",
            "color": (50, 100, 220),  # Blue liquid proxy
        },
        {
            "question": "What primary energy transformation occurs in this solar apparatus?",
            "choices": [
                "Light energy to electrical energy",
                "Chemical energy to thermal energy",
                "Nuclear energy to mechanical energy",
                "Kinetic energy to potential energy",
                "Electrical energy to sound energy",
            ],
            "answer": 0,
            "hint": "Photovoltaic cells absorb photons to stimulate electron flow.",
            "color": (240, 210, 40),  # Yellow solar proxy
        },
        {
            "question": "Is this geological rock formation igneous or sedimentary?",
            "choices": ["Sedimentary with visible strata layers", "Extrusive igneous basalt", "Metamorphic marble"],
            "answer": 0,
            "hint": "Distinct horizontal stratification layers are clearly visible.",
            "color": (160, 120, 80),  # Brown sandstone proxy
        },
    ]

    samples: List[MultimodalSample] = []
    for i in range(num_samples):
        arch = archetypes[i % len(archetypes)]
        # Generate color image proxy
        img = Image.new("RGB", (224, 224), color=arch["color"])
        # Slight noise
        arr = np.array(img).astype(np.float32) + rng.integers(-10, 10, size=(224, 224, 3))
        noisy_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        sample = MultimodalSample(
            question=arch["question"],
            candidates=arch["choices"],
            target_idx=arch["answer"],
            image=noisy_img,
            text=arch["hint"],
            metadata={"dataset": "ScienceQA_Mock", "sample_id": i},
        )
        samples.append(sample)

    return samples


def load_scienceqa_dataset(
    split: str = "validation",
    max_samples: Optional[int] = 500,
    only_multimodal: bool = True,
    streaming: bool = True,
    use_mock_fallback: bool = True,
) -> List[MultimodalSample]:
    """
    Loads ScienceQA samples from HuggingFace datasets (derek-thomas/ScienceQA) or falls back to mock samples if offline.
    Supports streaming to avoid downloading massive local archives.
    """
    try:
        from datasets import load_dataset  # type: ignore

        logger.info(
            f"Loading ScienceQA split='{split}' (streaming={streaming}) from HuggingFace derek-thomas/ScienceQA..."
        )
        ds = load_dataset("derek-thomas/ScienceQA", split=split, streaming=streaming)
        samples: List[MultimodalSample] = []

        for record in ds:
            if only_multimodal and record.get("image") is None:
                continue

            sample = ScienceQAAdapter.record_to_sample(record)
            samples.append(sample)

            if max_samples is not None and len(samples) >= max_samples:
                break

        logger.info(f"Successfully loaded {len(samples)} real ScienceQA samples.")
        return samples
    except Exception as e:
        if use_mock_fallback:
            logger.warning(
                f"Could not download ScienceQA from HuggingFace ({e}). Using {max_samples or 20} mock ScienceQA samples."
            )
            return create_mock_scienceqa_samples(num_samples=max_samples or 20)
        raise e
