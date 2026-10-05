"""
GQA (Compositional Questions About Real-World Images) Dataset Adapter.

Adapts the `lmms-lab/GQA` HuggingFace dataset into ArbiterOmni MultimodalSample records.
GQA provides compositionally rich, real-image grounded questions with verified answers.
Since GQA is an open-ended VQA dataset (not inherently multi-choice), we reframe it as
a multi-candidate System 1 decision task by dynamically constructing distractor candidates
from a pool of plausible semantic foils drawn from the same batch.

HuggingFace dataset: lmms-lab/GQA  (balanced split recommended).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np

from arbiter_omni.types import MultimodalSample

logger = logging.getLogger(__name__)


# Semantic category foil pools used to build distractors when within-batch foils
# are unavailable.  Keyed loosely by answer type so foils remain plausible.
_FOIL_POOLS: Dict[str, List[str]] = {
    "color": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "gray"],
    "count": ["1", "2", "3", "4", "5", "6", "none", "several", "many", "a few"],
    "bool": ["yes", "no"],
    "material": ["wood", "metal", "plastic", "glass", "fabric", "stone", "rubber", "paper"],
    "location": ["left", "right", "top", "bottom", "center", "background", "foreground"],
    "size": ["small", "large", "medium", "tiny", "huge"],
    "default": [
        "yes", "no", "left", "right", "red", "blue", "green", "wood",
        "metal", "glass", "large", "small", "2", "3", "none", "indoor", "outdoor",
    ],
}


def _guess_foil_pool(answer: str) -> List[str]:
    """Heuristically selects a foil pool by answer content."""
    a = answer.strip().lower()
    if a in ("yes", "no"):
        return _FOIL_POOLS["bool"]
    if a.isdigit():
        return _FOIL_POOLS["count"]
    if a in _FOIL_POOLS["color"]:
        return _FOIL_POOLS["color"]
    if a in _FOIL_POOLS["material"]:
        return _FOIL_POOLS["material"]
    if a in _FOIL_POOLS["location"]:
        return _FOIL_POOLS["location"]
    if a in _FOIL_POOLS["size"]:
        return _FOIL_POOLS["size"]
    return _FOIL_POOLS["default"]


def _build_candidates(
    answer: str,
    foil_pool: Optional[List[str]] = None,
    num_distractors: int = 3,
    rng: Optional[random.Random] = None,
) -> tuple[List[str], int]:
    """
    Builds a shuffled candidate list from the correct answer + distractors.

    Returns:
        candidates (List[str]), target_idx (int)
    """
    if rng is None:
        rng = random.Random()
    pool = foil_pool or _guess_foil_pool(answer)
    # Pick distractors that differ from answer (case-insensitive)
    foils = [f for f in pool if f.strip().lower() != answer.strip().lower()]
    foils = list(dict.fromkeys(foils))  # deduplicate, preserve order
    rng.shuffle(foils)
    distractors = foils[:num_distractors]
    # Pad if pool is too small
    while len(distractors) < num_distractors:
        distractors.append(f"option {len(distractors) + 1}")

    candidates = [answer] + distractors
    rng.shuffle(candidates)
    target_idx = candidates.index(answer)
    return candidates, target_idx


class GQAAdapter:
    """
    Adapter converting HuggingFace lmms-lab/GQA records into ArbiterOmni MultimodalSample format.

    Record fields used:
        question       (str)       : The natural language question.
        answer         (str)       : Short open-ended answer string (e.g. "yes", "red", "3").
        image          (PIL.Image) : The reference image.
        question_id    (str)       : Optional unique ID for logging.
    """

    def __init__(self, num_distractors: int = 3, seed: int = 42):
        self._rng = random.Random(seed)
        self.num_distractors = num_distractors

    def record_to_sample(self, record: Dict[str, Any]) -> MultimodalSample:
        """Converts a single GQA record into a MultimodalSample."""
        question = str(record.get("question", "")).strip()
        raw_ans = record.get("answer") or record.get("multiple_choice_answer") or ""
        answer = str(raw_ans).strip().lower()
        image = record.get("image", None)
        qid = str(record.get("question_id", ""))

        if not answer:
            return None  # type: ignore  # skip unresolvable records

        foil_pool = _guess_foil_pool(answer)
        candidates, target_idx = _build_candidates(
            answer, foil_pool=foil_pool, num_distractors=self.num_distractors, rng=self._rng
        )

        return MultimodalSample(
            question=question or "What is shown in the image?",
            candidates=candidates,
            target_idx=target_idx,
            image=image,
            text=None,
            metadata={"dataset": "GQA", "question_id": qid},
        )


def _make_mock_gqa_samples(num_samples: int = 10, seed: int = 7) -> List[MultimodalSample]:
    """Generates synthetic GQA-style samples for offline testing."""
    rng_np = np.random.default_rng(seed)
    rng_r = random.Random(seed)
    archetypes = [
        {"question": "What color is the object on the left?", "answer": "red", "color": (200, 40, 40)},
        {"question": "Is there a person in the image?", "answer": "yes", "color": (100, 160, 200)},
        {"question": "How many objects are on the table?", "answer": "3", "color": (230, 200, 100)},
        {"question": "What material is the bench made of?", "answer": "wood", "color": (140, 90, 50)},
        {"question": "Is the room indoors or outdoors?", "answer": "indoor", "color": (180, 180, 200)},
    ]
    samples: List[MultimodalSample] = []
    adapter = GQAAdapter(seed=seed)
    for i in range(num_samples):
        arch = archetypes[i % len(archetypes)]
        arr = np.full((224, 224, 3), arch["color"], dtype=np.uint8)
        arr = arr + rng_np.integers(-15, 15, size=arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        foil_pool = _guess_foil_pool(arch["answer"])
        candidates, target_idx = _build_candidates(arch["answer"], foil_pool=foil_pool, rng=rng_r)
        samples.append(
            MultimodalSample(
                question=arch["question"],
                candidates=candidates,
                target_idx=target_idx,
                image=img,
                metadata={"dataset": "GQA_Mock", "sample_id": i},
            )
        )
    return samples


def load_gqa_dataset(
    split: str = "train_balanced",
    max_samples: Optional[int] = 10000,
    streaming: bool = True,
    num_distractors: int = 3,
    use_mock_fallback: bool = True,
    seed: int = 42,
) -> List[MultimodalSample]:
    """
    Loads GQA samples from HuggingFace `lmms-lab/GQA`.

    Args:
        split:             HuggingFace split name.  Recommended: "train_balanced".
        max_samples:       Cap on samples loaded (None = unlimited).
        streaming:         Avoids large local downloads via HuggingFace streaming.
        num_distractors:   Number of distractor candidates added per sample (default 3 → 4-way choice).
        use_mock_fallback: Fall back to synthetic mock if download fails.
        seed:              RNG seed for repeatable distractor construction.

    Returns:
        List[MultimodalSample]: Loaded & converted samples ready for training.
    """
    try:
        from datasets import load_dataset  # type: ignore

        is_val = any(k in str(split).lower() for k in ("val", "test", "dev"))
        config_name = "val_balanced_instructions" if is_val else "train_balanced_instructions"
        split_name = "val" if is_val else "train"

        logger.info(f"Loading GQA config='{config_name}', split='{split_name}' (streaming={streaming}) from lmms-lab/GQA...")
        ds = load_dataset("lmms-lab/GQA", config_name, split=split_name, streaming=streaming)

        adapter = GQAAdapter(num_distractors=num_distractors, seed=seed)
        samples: List[MultimodalSample] = []
        try:
            for record in ds:
                sample = adapter.record_to_sample(record)
                if (
                    sample is not None
                    and sample.target_idx is not None
                    and len(sample.candidates) >= 2
                ):
                    samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    break
        except Exception as iter_e:
            logger.warning(f"Interruption while streaming GQA: {iter_e}")
            if not samples:
                raise iter_e

        logger.info(f"Loaded {len(samples)} GQA samples.")
        return samples

    except Exception as e:
        if use_mock_fallback:
            n = max_samples or 20
            logger.warning(f"Could not download GQA ({e}). Using {n} mock GQA samples.")
            return _make_mock_gqa_samples(num_samples=n, seed=seed)
        raise
