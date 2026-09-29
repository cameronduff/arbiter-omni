"""
ArbiterOmni High-Level Inference API Engine.
Inspired by Jev System 1 Decision Architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import os
import torch
import torch.nn.functional as F

from arbiter_omni.calibration.conformal import ConformalCalibrator, System2EscalationGate
from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder
from arbiter_omni.fusion.base import BaseMultimodalFusion
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.model.decision_head import DynamicDecisionHead
from arbiter_omni.types import DecisionResult, ModalityType, MultimodalSample



class ArbiterOmniEngine:
    """
    User-facing Inference Engine for Multimodal Decision Making.
    
    Provides single-call and batched decision inference over arbitrary dynamic
    candidate sets, outputting calibrated probability distributions and decision metrics.
    """

    def __init__(
        self,
        model: ArbiterOmniModel,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.model.eval()

        # Calibration & System 2 Escalation Gate
        self.conformal_calibrator = ConformalCalibrator()
        self.escalation_gate = System2EscalationGate()


    @classmethod
    def create(
        cls,
        encoder_type: str = "mock",
        hidden_dim: int = 256,
        scoring_dim: int = 256,
        openclip_model: str = "ViT-B-32",
        pretrained_dataset: str = "laion2b_s34b_b79k",
        device: Optional[Union[str, torch.device]] = None,
    ) -> ArbiterOmniEngine:
        """
        Factory constructor for ArbiterOmniEngine.
        
        Args:
            encoder_type: 'mock' for lightweight instant testing, or 'openclip' for production CLIP.
            hidden_dim: Fusion latent dimension.
            scoring_dim: Decision interaction dimension.
            openclip_model: OpenCLIP architecture name.
            pretrained_dataset: OpenCLIP checkpoint tag.
            device: Optional torch device.
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if encoder_type == "openclip":
            encoder = OpenCLIPMultimodalEncoder(
                model_name=openclip_model, pretrained=pretrained_dataset, device=dev
            )
        elif encoder_type == "mock":
            encoder = MockMultimodalEncoder(embed_dim=128, device=dev)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'openclip' or 'mock'.")

        model = ArbiterOmniModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            scoring_dim=scoring_dim,
        )
        return cls(model=model, device=dev)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_name_or_path: str = "v1",
        encoder_type: str = "openclip",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> ArbiterOmniEngine:
        """
        Loads an ArbiterOmniEngine instance initialized with pretrained weights.

        Args:
            checkpoint_name_or_path: Checkpoint tag ('v1') or filepath to .pt checkpoint.
            encoder_type: 'openclip' or 'mock'.
            device: Target torch device or device string.
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path = checkpoint_name_or_path
        if path == "v1":
            candidates = [
                "checkpoints/arbiter_omni_v1.pt",
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints", "arbiter_omni_v1.pt"),
                os.path.join(os.getcwd(), "checkpoints", "arbiter_omni_v1.pt"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    path = cand
                    break

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_name_or_path}' could not be resolved at path: {path}. "
                "Ensure checkpoints/arbiter_omni_v1.pt exists or run scripts/train_v1.py."
            )

        engine = cls.create(encoder_type=encoder_type, device=dev, **kwargs)
        engine.load_weights(path)
        return engine

    def load_weights(self, weights_path: str):
        """Loads trained fusion and decision head weights."""
        checkpoint = torch.load(weights_path, map_location=self.device)
        if "fusion" in checkpoint and "decision_head" in checkpoint:
            self.model.fusion.load_state_dict(checkpoint["fusion"], strict=False)
            self.model.decision_head.load_state_dict(checkpoint["decision_head"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

    def calibrate_conformal(
        self,
        samples: Sequence[MultimodalSample],
        alpha: float = 0.05,
        method: str = "lac",
    ) -> float:
        """
        Calibrates finite-sample (1 - alpha) conformal prediction sets using held-out samples.

        Args:
            samples: Calibration dataset containing MultimodalSample instances with target_idx.
            alpha: Significance level (default 0.05 for 95% statistical coverage guarantee).
            method: 'lac' (Least Ambiguous Classifier) or 'aps' (Adaptive Prediction Sets).

        Returns:
            q_hat: Calibrated non-conformity threshold quantile.
        """
        self.conformal_calibrator = ConformalCalibrator(alpha=alpha, method=method)
        results = self.decide_batch(samples)
        
        valid_probs = []
        targets = []
        cand_lists = []
        for s, r in zip(samples, results):
            if s.target_idx is not None:
                valid_probs.append(r.probabilities)
                targets.append(s.target_idx)
                cand_lists.append(s.candidates)

        return self.conformal_calibrator.calibrate(
            probabilities=valid_probs,
            targets=targets,
            candidate_lists=cand_lists,
        )

    def configure_escalation_gate(
        self,
        entropy_threshold: float = 0.95,
        max_conformal_size: int = 1,
        min_confidence: float = 0.50,
        enabled: bool = True,
    ):
        """
        Configures criteria for escalating ambiguous System 1 decisions to System 2.

        Args:
            entropy_threshold: Max permissible Shannon entropy before escalation (default 0.95 nats).
            max_conformal_size: Max permissible conformal prediction set size (default 1 candidate).
            min_confidence: Minimum top-1 confidence required before escalation (default 0.50).
            enabled: Whether the escalation gate is actively monitored.
        """
        self.escalation_gate = System2EscalationGate(
            entropy_threshold=entropy_threshold,
            max_conformal_size=max_conformal_size,
            min_confidence=min_confidence,
            enabled=enabled,
        )

    def decide(

        self,
        question: str,
        candidates: Sequence[str],
        text: Optional[str] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        audio: Optional[Any] = None,
        return_embedding: bool = False,
    ) -> DecisionResult:
        """
        Evaluates multimodal state and returns calibrated probability distribution over candidates.
        
        Args:
            question: Decision query or prompt.
            candidates: Sequence of at least two dynamic candidate decision strings.
            text: Optional descriptive context or sensor log.
            image: Optional PIL Image, numpy array, or image path.
            video: Optional list of frames, video tensor, or video path.
            audio: Optional audio waveform, array, or audio path.
            return_embedding: If True, includes the fused latent vector in DecisionResult.
            
        Returns:
            DecisionResult containing top choice, full probabilities, entropy, and metrics.
        """
        if len(candidates) < 2:
            raise ValueError("At least 2 candidate decisions are required for decision arbitration.")

        # Determine active modalities
        active: List[str] = []
        if text is not None and len(str(text).strip()) > 0:
            active.append(ModalityType.TEXT.value)
        if image is not None:
            active.append(ModalityType.IMAGE.value)
        if video is not None:
            active.append(ModalityType.VIDEO.value)
        if audio is not None:
            active.append(ModalityType.AUDIO.value)

        self.model.eval()
        with torch.no_grad():
            logits, probs, entropy, fused_context = self.model(
                questions=[question],
                candidates=[list(candidates)],
                texts=[text],
                images=[image],
                videos=[video],
                audios=[audio],
            )

            p_vec = probs[0].cpu().numpy().tolist()
            ent = float(entropy[0].item())
            fused_vec = fused_context[0].cpu().numpy().tolist() if return_embedding else None

            # Auxiliary Jev primitives
            noul_cert = float(self.model.decision_head.predict_boolean_noul(fused_context)[0].item())
            cont_score = float(self.model.decision_head.predict_score(fused_context)[0].item())

        prob_dict = {cand: float(p) for cand, p in zip(candidates, p_vec)}
        winner_idx = int(torch.argmax(probs[0]).item())
        winner = candidates[winner_idx]
        confidence = float(p_vec[winner_idx])

        # Formulate boolean noul dict (probability of affirmative/valid state)
        boolean_noul = {
            "true_probability": noul_cert,
            "false_probability": 1.0 - noul_cert,
            "calibrated_certainty": abs(noul_cert - 0.5) * 2.0,
        }

        # Conformal prediction set & System 2 escalation check
        conformal_set = self.conformal_calibrator.predict_set(prob_dict)
        escalate, reason = self.escalation_gate.evaluate(
            confidence=confidence,
            entropy=ent,
            conformal_set=conformal_set,
        )

        return DecisionResult(
            question=question,
            winner=winner,
            winner_index=winner_idx,
            confidence=confidence,
            probabilities=prob_dict,
            entropy=ent,
            active_modalities=active,
            boolean_noul=boolean_noul,
            score=cont_score,
            latent_embedding=fused_vec,
            conformal_set=conformal_set,
            escalate_system2=escalate,
            escalation_reason=reason,
        )


    def decide_batch(
        self,
        samples: Sequence[MultimodalSample],
        return_embedding: bool = False,
    ) -> List[DecisionResult]:
        """Runs batched multimodal arbitration over a list of MultimodalSamples."""
        results: List[DecisionResult] = []
        for s in samples:
            res = self.decide(
                question=s.question,
                candidates=s.candidates,
                text=s.text,
                image=s.image,
                video=s.video,
                audio=s.audio,
                return_embedding=return_embedding,
            )
            results.append(res)
        return results
