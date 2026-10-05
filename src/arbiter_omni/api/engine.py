"""
ArbiterOmni High-Level Inference API Engine.
Inspired by Jev System 1 Decision Architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import os
import time
import torch
import torch.nn.functional as F

from arbiter_omni.calibration.conformal import ConformalCalibrator, System2EscalationGate
from arbiter_omni.calibration.deliberator import (
    TestTimeDeliberator,
    TestTimeDeliberationSummary,
    TournamentBracket,
)
from arbiter_omni.calibration.temperature import CalibrationSummary, TemperatureCalibrator
from arbiter_omni.data.memory_bank import PersistentMemoryBank
from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder
from arbiter_omni.fusion.base import BaseMultimodalFusion
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.model.decision_head import DynamicDecisionHead
from arbiter_omni.types import DecisionResult, ModalityType, MultimodalSample
from arbiter_omni.device import resolve_device



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
            self.device = resolve_device()

        self.model = model.to(self.device)
        self.model.eval()

        # Calibration & System 2 Escalation Gate
        self.conformal_calibrator = ConformalCalibrator()
        self.escalation_gate = System2EscalationGate()
        self.temperature_calibrator = TemperatureCalibrator()
        self.temperature: Optional[float] = None

        # Test-Time Compute (TTC) Deliberator [AO-31]
        self.memory_bank: Optional[PersistentMemoryBank] = None
        self.deliberator = TestTimeDeliberator()
        self.enable_test_time_deliberation: bool = False

    def attach_memory_bank(self, memory_bank: PersistentMemoryBank) -> None:
        """Attaches a resident memory bank to the engine and deliberator for adversarial foil testing [AO-23, AO-31]."""
        self.memory_bank = memory_bank
        self.deliberator.memory_bank = memory_bank

    def configure_deliberator(
        self,
        num_passes: int = 3,
        router_noise_std: float = 0.05,
        foil_k: int = 6,
        min_foil_margin: float = 0.20,
        stability_threshold: float = 0.70,
        enabled: bool = True,
    ) -> None:
        """Configures Test-Time Compute (TTC) deliberation parameters [AO-31]."""
        self.deliberator.num_passes = num_passes
        self.deliberator.router_noise_std = router_noise_std
        self.deliberator.foil_k = foil_k
        self.deliberator.min_foil_margin = min_foil_margin
        self.deliberator.stability_threshold = stability_threshold
        self.enable_test_time_deliberation = enabled


    @classmethod
    def create(
        cls,
        encoder_type: str = "mock",
        hidden_dim: int = 256,
        scoring_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        enable_spatial_cross_attention: bool = False,
        openclip_model: str = "ViT-B-32",
        pretrained_dataset: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> ArbiterOmniEngine:
        """
        Factory constructor for ArbiterOmniEngine.
        
        Args:
            encoder_type: 'mock' for lightweight instant testing, or 'openclip' for production CLIP.
            hidden_dim: Fusion latent dimension.
            scoring_dim: Decision interaction dimension.
            num_layers: Number of transformer fusion layers.
            num_heads: Number of attention heads.
            enable_spatial_cross_attention: Enable question-conditioned spatial cross attention.
            openclip_model: OpenCLIP architecture name ('ViT-B-32' or 'ViT-B-16').
            pretrained_dataset: Optional OpenCLIP checkpoint tag (auto-resolved if None).
            device: Optional torch device.
        """
        dev = torch.device(device) if device else resolve_device()

        cpu_offload = kwargs.pop("cpu_offload_encoder", False)
        if encoder_type == "openclip":
            encoder = OpenCLIPMultimodalEncoder(
                model_name=openclip_model,
                pretrained=pretrained_dataset,
                device=dev,
                cpu_offload_encoder=cpu_offload,
            )
        elif encoder_type == "mock":
            encoder = MockMultimodalEncoder(embed_dim=kwargs.get("embed_dim", 128), device=dev)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'openclip' or 'mock'.")

        model = ArbiterOmniModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            scoring_dim=scoring_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            enable_spatial_cross_attention=enable_spatial_cross_attention,
            **kwargs,
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
            checkpoint_name_or_path: Checkpoint tag ('v1', 'v2', 'v3', 'v4', 'v5') or filepath to .pt checkpoint.
            encoder_type: 'openclip' or 'mock'.
            device: Target torch device or device string.
        """
        dev = torch.device(device) if device else resolve_device()

        path = checkpoint_name_or_path
        if path in ("v1", "v2", "v3", "v4", "v5", "v6", "v6_max"):
            tag_name = f"arbiter_omni_{path}.pt"
            candidates = [
                f"checkpoints/{tag_name}",
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints", tag_name),
                os.path.join(os.getcwd(), "checkpoints", tag_name),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    path = cand
                    break

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_name_or_path}' could not be resolved at path: {path}. "
                "Ensure checkpoints/arbiter_omni_v6_max.pt (or v6/v5/v4/v1) exists or run training scripts."
            )

        # Inspect checkpoint for architectural parameters if present
        try:
            try:
                raw_ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                raw_ckpt = torch.load(path, map_location="cpu")
            m_cfg = raw_ckpt.get("model_config", {}) if isinstance(raw_ckpt, dict) else {}
            fusion_sd = raw_ckpt.get("fusion", {}) if isinstance(raw_ckpt, dict) else {}
            head_sd = raw_ckpt.get("decision_head", {}) if isinstance(raw_ckpt, dict) else {}
        except Exception:
            m_cfg = {}
            fusion_sd = {}
            head_sd = {}

        if "hidden_dim" not in kwargs:
            if "hidden_dim" in m_cfg:
                kwargs["hidden_dim"] = m_cfg["hidden_dim"]
            elif "projections.question.0.weight" in fusion_sd:
                kwargs["hidden_dim"] = fusion_sd["projections.question.0.weight"].shape[0]

        if "scoring_dim" not in kwargs:
            if "scoring_dim" in m_cfg:
                kwargs["scoring_dim"] = m_cfg["scoring_dim"]
            elif "scoring_net.0.weight" in head_sd:
                kwargs["scoring_dim"] = head_sd["scoring_net.0.weight"].shape[0]

        # MoE configuration [AO-27, AO-28, AO-30]
        if "use_moe" not in kwargs:
            if "use_moe" in m_cfg:
                kwargs["use_moe"] = m_cfg["use_moe"]
            elif any(k.startswith("moe_transformer.") for k in fusion_sd.keys()) or "v5" in str(path) or "v6" in str(path):
                kwargs["use_moe"] = True

        if kwargs.get("use_moe", False):
            if "moe_num_layers" not in kwargs:
                kwargs["moe_num_layers"] = m_cfg.get("moe_num_layers", 4)
            if "moe_num_experts" not in kwargs:
                kwargs["moe_num_experts"] = m_cfg.get("moe_num_experts", 4)
            if "moe_top_k" not in kwargs:
                kwargs["moe_top_k"] = m_cfg.get("moe_top_k", 2)
            if "use_shared_expert" not in kwargs:
                kwargs["use_shared_expert"] = m_cfg.get("use_shared_expert", ("v6" in str(path)))

        # Speculative draft configuration [AO-29]
        if "enable_speculative_early_exit" not in kwargs:
            if "enable_speculative_early_exit" in m_cfg:
                kwargs["enable_speculative_early_exit"] = m_cfg["enable_speculative_early_exit"]
            elif "v6" in str(path):
                kwargs["enable_speculative_early_exit"] = True

        if "num_layers" not in kwargs:
            if "num_layers" in m_cfg:
                kwargs["num_layers"] = m_cfg["num_layers"]
            else:
                layer_indices = [
                    int(k.split(".")[2])
                    for k in fusion_sd.keys()
                    if k.startswith("transformer.layers.") and k.split(".")[2].isdigit()
                ]
                if layer_indices:
                    kwargs["num_layers"] = max(layer_indices) + 1

        if "num_heads" not in kwargs:
            if "num_heads" in m_cfg:
                kwargs["num_heads"] = m_cfg["num_heads"]

        if "enable_spatial_cross_attention" not in kwargs:
            if "enable_spatial_cross_attention" in m_cfg:
                kwargs["enable_spatial_cross_attention"] = m_cfg["enable_spatial_cross_attention"]
            elif any(k.startswith("spatial_cross_attn") for k in fusion_sd.keys()) or "v6" in str(path):
                kwargs["enable_spatial_cross_attention"] = True

        if "max_spatial_patches" not in kwargs:
            if "max_spatial_patches" in m_cfg:
                kwargs["max_spatial_patches"] = m_cfg["max_spatial_patches"]
            elif "v4" in str(path) or "v5" in str(path) or "v6" in str(path):
                kwargs["max_spatial_patches"] = 980

        if "openclip_model" not in kwargs:
            if "model_name" in m_cfg:
                kwargs["openclip_model"] = m_cfg["model_name"]
            elif "v6" in str(path) or "v5" in str(path):
                kwargs["openclip_model"] = "ViT-SO400M-14-SigLIP-384"
            elif "v4" in str(path) or "v3" in str(path):
                kwargs["openclip_model"] = "ViT-B-16-SigLIP"
            elif "v2" in str(path):
                kwargs["openclip_model"] = "ViT-B-16"

        engine = cls.create(encoder_type=encoder_type, device=dev, **kwargs)
        engine.load_weights(path)
        return engine

    def load_weights(self, weights_path: str):
        """Loads trained fusion, decision head, and speculative weights."""
        # Always deserialize on CPU: load_state_dict copies into the (possibly DirectML)
        # device-resident parameters, whereas restoring storages straight onto a DirectML
        # device fails for checkpoints that were saved from CPU tensors.
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location="cpu")
        if "fusion" in checkpoint and "decision_head" in checkpoint:
            self.model.fusion.load_state_dict(checkpoint["fusion"], strict=False)
            self.model.decision_head.load_state_dict(checkpoint["decision_head"], strict=False)
            if "speculative_head" in checkpoint and getattr(self.model, "speculative_head", None) is not None:
                self.model.speculative_head.load_state_dict(checkpoint["speculative_head"], strict=False)
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

    def set_temperature(self, temperature: float) -> None:
        """Sets the global post-hoc scaling temperature for output sharpening and calibration."""
        self.temperature = float(temperature)
        self.model.decision_head.set_temperature(temperature)

    def calibrate_temperature(
        self,
        samples: Sequence[MultimodalSample],
        lr: float = 0.05,
        max_iter: int = 50,
        n_bins: int = 10,
    ) -> CalibrationSummary:
        """
        Calibrates post-hoc temperature scaling on held-out validation samples to minimize ECE.
        
        Args:
            samples: Held-out validation samples with target_idx specified.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Max iterations.
            n_bins: Number of bins for reliability diagram / ECE calculation.
            
        Returns:
            CalibrationSummary with initial/calibrated ECE, NLL, and optimal temperature.
        """
        valid_samples = [s for s in samples if s.target_idx is not None]
        if not valid_samples:
            raise ValueError("Temperature calibration requires samples with ground truth target_idx.")

        self.model.eval()
        logits_list = []
        targets = []

        with torch.no_grad():
            for s in valid_samples:
                logits, _, _, _ = self.model(
                    questions=[s.question],
                    candidates=[list(s.candidates)],
                    texts=[s.text],
                    images=[s.image],
                    videos=[s.video],
                    audios=[s.audio],
                    temperature=1.0,
                )
                logits_list.append(logits[0])
                targets.append(s.target_idx)

        summary = self.temperature_calibrator.fit(
            logits_list=logits_list,
            targets=targets,
            lr=lr,
            max_iter=max_iter,
            n_bins=n_bins,
        )
        self.set_temperature(summary.optimal_temperature)
        return summary

    def decide(
        self,
        question: str,
        candidates: Sequence[str],
        text: Optional[str] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        audio: Optional[Any] = None,
        return_embedding: bool = False,
        use_prompt_ensembling: Optional[bool] = None,
        prompt_templates: Optional[Sequence[str]] = None,
        temperature: Optional[float] = None,
        test_time_deliberate: Optional[bool] = None,
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
            use_prompt_ensembling: If True (or None with visual input), averages candidate embeddings
                                  across descriptive templates to sharpen zero-shot visual alignment.
            prompt_templates: Optional custom templates (e.g. ['a photo of a {}', '{}']).
            temperature: Optional inference temperature for output sharpening (<1.0) or softening (>1.0).
            test_time_deliberate: If True, triggers Test-Time Compute (TTC) multi-pass deliberation and foil stress-testing [AO-31].
            
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

        # Prompt ensembling defaults to active when visual inputs are provided
        if use_prompt_ensembling is None:
            use_prompt_ensembling = (image is not None or video is not None)

        temp_to_use = temperature if temperature is not None else self.temperature

        # Tier-0 Speculative Draft Arbitration check [AO-29]
        draft_telemetry = None
        if getattr(self.model, "enable_speculative_early_exit", False) and getattr(self.model, "speculative_head", None) is not None:
            t_spec_start = time.perf_counter()
            with torch.no_grad():
                q_embed, mod_embeds, pres_mask, _ = self.model.encode_inputs(
                    questions=[question], texts=[text], images=[image], videos=[video], audios=[audio]
                )
                cnd_embeds, cnd_mask = self.model.encode_candidates(
                    [candidates],
                    prompt_templates=prompt_templates,
                    use_prompt_ensembling=use_prompt_ensembling,
                )
                _, draft_probs, draft_ent, can_exit, margins, _ = self.model.forward_speculative(
                    question_embed=q_embed,
                    candidate_embeds=cnd_embeds,
                    candidate_mask=cnd_mask,
                    modality_embeds=mod_embeds,
                    presence_mask=pres_mask,
                    temperature=temp_to_use,
                )
                draft_lat_ms = (time.perf_counter() - t_spec_start) * 1000.0

            dp_vec = draft_probs[0].cpu().numpy().tolist()
            d_prob_dict = {cand: float(p) for cand, p in zip(candidates, dp_vec)}
            w_idx = int(torch.argmax(draft_probs[0]).item())

            draft_telemetry = {
                "early_exit_taken": bool(can_exit[0].item()),
                "draft_winner": candidates[w_idx],
                "draft_confidence": float(dp_vec[w_idx]),
                "draft_margin": float(margins[0].item()),
                "draft_entropy": float(draft_ent[0].item()),
                "draft_latency_ms": draft_lat_ms,
            }

            if can_exit[0].item():
                # Certified high-margin early exit: bypass deep MoE layers entirely (<0.5 ms)!
                conformal_set = self.conformal_calibrator.predict_set(d_prob_dict)
                return DecisionResult(
                    question=question,
                    winner=candidates[w_idx],
                    winner_index=w_idx,
                    confidence=float(dp_vec[w_idx]),
                    probabilities=d_prob_dict,
                    entropy=float(draft_ent[0].item()),
                    active_modalities=active,
                    speculative_early_exit=True,
                    draft_telemetry=draft_telemetry,
                    deliberation_passes=1,
                    active_experts=[{"Shared-Invariant": 1.0, "Tier-0 Speculative Draft": 1.0}],
                    conformal_set=conformal_set,
                    escalate_system2=False,
                )

        self.model.eval()
        with torch.no_grad():
            logits, probs, entropy, fused_context = self.model(
                questions=[question],
                candidates=[list(candidates)],
                texts=[text],
                images=[image],
                videos=[video],
                audios=[audio],
                use_prompt_ensembling=use_prompt_ensembling,
                prompt_templates=prompt_templates,
                temperature=temp_to_use,
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

        # Test-Time Compute (TTC) Deliberation Tournament [AO-31]
        delib_summary = None
        should_deliberate = (test_time_deliberate if test_time_deliberate is not None else self.enable_test_time_deliberation)
        if should_deliberate and self.deliberator is not None:
            cnd_embeds, cnd_mask = self.model.encode_candidates(
                [candidates],
                prompt_templates=prompt_templates,
                use_prompt_ensembling=use_prompt_ensembling,
            )
            _, mod_embeds, pres_mask, _ = self.model.encode_inputs(
                questions=[question], texts=[text], images=[image], videos=[video], audios=[audio]
            )
            delib_summary = self.deliberator.deliberate(
                context_embed=fused_context,
                candidate_embeds=cnd_embeds,
                candidates=candidates,
                decision_head=self.model.decision_head,
                candidate_mask=cnd_mask,
                modality_embeds=mod_embeds,
                presence_mask=pres_mask,
                temperature=temp_to_use,
            )
            prob_dict = delib_summary.calibrated_probabilities
            winner = max(prob_dict, key=prob_dict.get)
            winner_idx = list(candidates).index(winner)
            confidence = prob_dict[winner]

        # Adaptive Conformal Risk Control (CRC) prediction set & escalation check [AO-08, AO-32]
        stab_idx = delib_summary.stability_index if delib_summary is not None else None
        conformal_set = self.conformal_calibrator.predict_set(prob_dict, stability_index=stab_idx)
        escalate, reason = self.escalation_gate.evaluate(
            confidence=confidence,
            entropy=ent,
            conformal_set=conformal_set,
        )
        if delib_summary is not None and not delib_summary.certified_stable:
            escalate = True
            reason = delib_summary.escalate_reason or "DELIBERATION_FRAGILITY"

        active_experts = self.model.get_routing_distribution() if hasattr(self.model, "get_routing_distribution") else None

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
            speculative_early_exit=False,
            draft_telemetry=draft_telemetry,
            active_experts=active_experts,
            deliberation_passes=delib_summary.deliberation_passes if delib_summary is not None else 1,
            tournament_bracket=delib_summary.tournament_bracket.model_dump() if (delib_summary and delib_summary.tournament_bracket) else None,
            stability_index=stab_idx,
            deliberation_summary=delib_summary,
        )


    def decide_batch(
        self,
        samples: Sequence[MultimodalSample],
        return_embedding: bool = False,
        use_prompt_ensembling: Optional[bool] = None,
        prompt_templates: Optional[Sequence[str]] = None,
        temperature: Optional[float] = None,
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
                use_prompt_ensembling=use_prompt_ensembling,
                prompt_templates=prompt_templates,
                temperature=temperature,
            )
            results.append(res)
        return results
