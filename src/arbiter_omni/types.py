"""
ArbiterOmni Core Type Definitions and Pydantic Schemas.
Inspired by Jev System 1 Decision Architecture.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ModalityType(str, Enum):
    """Supported multimodal input channels."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class DecisionResult(BaseModel):
    """
    Structured System 1 decision result returned by ArbiterOmni.
    
    Rather than generating token-by-token prose, returns a calibrated probability
    distribution over the discrete candidate decisions provided at runtime.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str = Field(..., description="Decision prompt or query.")
    winner: str = Field(..., description="Top-1 candidate decision with highest probability.")
    winner_index: int = Field(..., description="Index of the chosen decision.")
    confidence: float = Field(..., description="Calibrated confidence score (top-1 probability).")
    probabilities: Dict[str, float] = Field(
        ..., description="Full categorical probability distribution over candidate decisions."
    )
    entropy: float = Field(..., description="Shannon entropy (uncertainty) in the decision distribution.")
    active_modalities: List[str] = Field(
        default_factory=list, description="List of modality channels present in this query."
    )
    boolean_noul: Optional[Dict[str, float]] = Field(
        default=None,
        description="Jev-style binary certainty evaluation if query is a true/false verification.",
    )
    score: Optional[float] = Field(
        default=None,
        description="Continuous expected decision rating along an ordinal scale, if applicable.",
    )
    latent_embedding: Optional[List[float]] = Field(
        default=None, description="Optional pooled multimodal decision latent vector."
    )
    conformal_set: List[str] = Field(
        default_factory=list,
        description="Conformal prediction set guaranteeing 1-alpha statistical coverage.",
    )
    escalate_system2: bool = Field(
        default=False,
        description="Flag indicating that decision ambiguity or conformal set size exceeds tolerance, requiring System 2 escalation.",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Diagnostic reason for System 2 escalation if triggered.",
    )

    @property
    def decision(self) -> str:
        """Alias for top-1 winner decision."""
        return self.winner



class MultimodalSample(BaseModel):
    """
    Single multimodal sample containing multimodal inputs, a question, candidate decisions,
    and an optional ground-truth decision label for training/evaluation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    candidates: List[str]
    target_idx: Optional[int] = None
    text: Optional[str] = None
    image: Optional[Any] = None
    video: Optional[Any] = None
    audio: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def present_modalities(self) -> List[ModalityType]:
        """Detects which non-question modalities are supplied."""
        present: List[ModalityType] = []
        if self.text is not None and len(self.text.strip()) > 0:
            present.append(ModalityType.TEXT)
        if self.image is not None:
            present.append(ModalityType.IMAGE)
        if self.video is not None:
            present.append(ModalityType.VIDEO)
        if self.audio is not None:
            present.append(ModalityType.AUDIO)
        return present
