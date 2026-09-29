"""
AI2D (Allen Institute for AI Diagrams) Dataset Adapter.

Adapts the `lmms-lab/ai2d` HuggingFace dataset into ArbiterOmni MultimodalSample records.
Each record contains a science diagram image and 4 multiple-choice answer candidates,
making it an ideal source of visually-grounded scientific decision samples.

HuggingFace dataset: lmms-lab/ai2d  (no authentication required)
Task type: Multi-choice visual question answering over scientific diagrams.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np

from arbiter_omni.types import MultimodalSample

logger = logging.getLogger(__name__)

# Answer letter → index map
_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "a": 0, "b": 1, "c": 2, "d": 3}


class AI2DAdapter:
    """
    Adapter converting HuggingFace lmms-lab/ai2d records into ArbiterOmni MultimodalSample format.

    Record fields used:
        question  (str)       : The question about the diagram.
        options   (List[str]) : Four answer options ["A. ...", "B. ...", "C. ...", "D. ..."].
        answer    (str)       : Correct answer letter e.g. "A" or "B".
        image     (PIL.Image) : Diagram image.
    """

    @staticmethod
    def record_to_sample(record: Dict[str, Any]) -> MultimodalSample:
        """Converts a single AI2D record into a MultimodalSample."""
        question = str(record.get("question", "")).strip()
        options = record.get("options", [])
        answer = str(record.get("answer", "A")).strip()
        image = record.get("image", None)

        # Strip leading letter+dot prefix from options if present (e.g. "A. cat" → "cat")
        candidates: List[str] = []
        for opt in options:
            opt_str = str(opt).strip()
            # Remove "A. " / "A) " / "(A) " style prefixes
            if len(opt_str) >= 2 and opt_str[0].upper() in "ABCD" and opt_str[1] in ".):":
                opt_str = opt_str[2:].strip()
            candidates.append(opt_str)

        if not candidates:
            candidates = ["Option A", "Option B", "Option C", "Option D"]

        # Resolve target index from answer (can be int, digit string '1', letter 'B', or option text)
        target_idx: Optional[int] = None
        if isinstance(answer, int):
            target_idx = answer
        elif str(answer).isdigit():
            target_idx = int(answer)
        else:
            target_idx = _LETTER_TO_IDX.get(answer)

        if target_idx is None:
            # Sometimes answer is a full string matching one of the options
            for i, c in enumerate(candidates):
                if str(answer).strip().lower() == c.strip().lower():
                    target_idx = i
                    break

        if target_idx is not None and (target_idx < 0 or target_idx >= len(candidates)):
            target_idx = None

        return MultimodalSample(
            question=question or "What does the diagram depict?",
            candidates=candidates,
            target_idx=target_idx,
            image=image,
            text=None,
            metadata={"dataset": "AI2D"},
        )


def _make_mock_ai2d_samples(num_samples: int = 10, seed: int = 0) -> List[MultimodalSample]:
    """Generates synthetic AI2D-style samples for offline testing."""
    rng = np.random.default_rng(seed)
    archetypes = [
        {
            "question": "What phase of the life cycle is shown at the top of the diagram?",
            "candidates": ["Larva", "Pupa", "Adult", "Egg"],
            "answer": 2,
            "color": (180, 220, 90),
        },
        {
            "question": "Which layer of Earth's interior is depicted in the innermost ring?",
            "candidates": ["Crust", "Mantle", "Outer core", "Inner core"],
            "answer": 3,
            "color": (210, 100, 60),
        },
        {
            "question": "What process does the arrow pointing upward from the ocean represent?",
            "candidates": ["Precipitation", "Evaporation", "Condensation", "Runoff"],
            "answer": 1,
            "color": (80, 150, 220),
        },
    ]
    samples: List[MultimodalSample] = []
    for i in range(num_samples):
        arch = archetypes[i % len(archetypes)]
        arr = np.full((224, 224, 3), arch["color"], dtype=np.uint8)
        arr = arr + rng.integers(-15, 15, size=arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        samples.append(
            MultimodalSample(
                question=arch["question"],
                candidates=arch["candidates"],
                target_idx=arch["answer"],
                image=img,
                metadata={"dataset": "AI2D_Mock", "sample_id": i},
            )
        )
    return samples


def load_ai2d_dataset(
    split: str = "test",
    max_samples: Optional[int] = 5000,
    streaming: bool = True,
    use_mock_fallback: bool = True,
) -> List[MultimodalSample]:
    """
    Loads AI2D samples from HuggingFace `lmms-lab/ai2d`.
    Note: lmms-lab/ai2d hosts all ~15k samples under the 'test' split.

    Args:
        split:            HuggingFace split name (defaults to 'test'; 'train' automatically maps to 'test').
        max_samples:      Cap on number of loaded samples (None = unlimited).
        streaming:        Use streaming mode to avoid large local downloads.
        use_mock_fallback: Fall back to synthetic mock samples if download fails.

    Returns:
        List[MultimodalSample]: Loaded & converted samples ready for training.
    """
    try:
        from datasets import load_dataset  # type: ignore

        # Map train/val to test split if requested, as lmms-lab/ai2d only defines 'test'
        hf_split = "test" if split in ("train", "val", "validation") else split
        logger.info(f"Loading AI2D split='{hf_split}' (requested '{split}', streaming={streaming}) from lmms-lab/ai2d...")
        ds = load_dataset("lmms-lab/ai2d", split=hf_split, streaming=streaming)

        samples: List[MultimodalSample] = []
        adapter = AI2DAdapter()
        try:
            for record in ds:
                sample = adapter.record_to_sample(record)
                # Only keep samples that have an image and a valid label
                if sample.target_idx is not None and len(sample.candidates) >= 2:
                    samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    break
        except Exception as iter_e:
            logger.warning(f"Interruption while streaming AI2D: {iter_e}")
            if not samples:
                raise iter_e

        logger.info(f"Loaded {len(samples)} AI2D samples.")
        return samples

    except Exception as e:
        if use_mock_fallback:
            n = max_samples or 20
            logger.warning(f"Could not download AI2D ({e}). Using {n} mock AI2D samples.")
            return _make_mock_ai2d_samples(num_samples=n)
        raise
