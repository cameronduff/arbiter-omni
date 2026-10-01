"""
ArbiterOmni Unified Decision Model.
Combines frozen multimodal encoders, multimodal attention/gated fusion,
and dynamic candidate decision scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.fusion.base import BaseMultimodalFusion
from arbiter_omni.fusion.transformer import TransformerMultimodalFusion
from arbiter_omni.model.decision_head import DynamicDecisionHead
from arbiter_omni.model.speculative import SpeculativeDraftHead, SpeculativeGate
from arbiter_omni.types import ModalityType

DEFAULT_PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of a {}",
    "a picture of a {}",
    "an image showing {}",
    "{}",
)


class ArbiterOmniModel(nn.Module):
    """
    ArbiterOmni Neural Architecture.
    
    Coordinates:
    - Pretrained, frozen multimodal encoders (Text, Image, Video, Audio)
    - Multimodal fusion network with explicit missing-modality masking
    - Dynamic candidate scoring head returning calibrated decision distributions
    """

    def __init__(
        self,
        encoder: BaseMultimodalEncoder,
        fusion: Optional[BaseMultimodalFusion] = None,
        decision_head: Optional[DynamicDecisionHead] = None,
        hidden_dim: int = 256,
        scoring_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        enable_spatial_cross_attention: bool = False,
        use_spatial_patches: bool = True,
        modality_dropout_prob: float = 0.0,
        max_spatial_patches: int = 196,
        use_moe: bool = False,
        moe_num_layers: int = 4,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        use_shared_expert: bool = False,
        enable_speculative_early_exit: bool = False,
        speculative_head: Optional[SpeculativeDraftHead] = None,
        speculative_margin_threshold: float = 0.45,
        speculative_entropy_threshold: float = 0.40,
        speculative_min_confidence: float = 0.65,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.freeze()  # Guarantee encoders are frozen
        self.use_spatial_patches = use_spatial_patches
        self.modality_dropout_prob = modality_dropout_prob
        self.enable_speculative_early_exit = enable_speculative_early_exit or kwargs.get("enable_speculative", False)

        # Setup default fusion if none supplied
        if fusion is None:
            modality_dims = {
                "question": self.encoder.text_dim,
                ModalityType.TEXT.value: self.encoder.text_dim,
                ModalityType.IMAGE.value: self.encoder.image_dim,
                ModalityType.VIDEO.value: self.encoder.video_dim,
                ModalityType.AUDIO.value: self.encoder.audio_dim,
            }
            self.fusion = TransformerMultimodalFusion(
                modality_dims=modality_dims,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=hidden_dim * 2,
                enable_spatial_cross_attention=enable_spatial_cross_attention,
                max_spatial_patches=max_spatial_patches,
                use_moe=use_moe,
                moe_num_layers=moe_num_layers,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                use_shared_expert=use_shared_expert,
            )
        else:
            self.fusion = fusion

        # Setup dynamic decision head
        if decision_head is None:
            self.decision_head = DynamicDecisionHead(
                context_dim=self.fusion.hidden_dim,
                candidate_dim=self.encoder.text_dim,
                scoring_dim=scoring_dim,
            )
        else:
            self.decision_head = decision_head

        # Setup Tier-0 Speculative Draft Head and Early-Exit Gate [AO-29]
        if speculative_head is not None:
            self.speculative_head = speculative_head
        elif self.enable_speculative_early_exit:
            self.speculative_head = SpeculativeDraftHead(
                input_dim=self.encoder.text_dim,
                candidate_dim=self.encoder.text_dim,
                draft_dim=128,
            )
        else:
            self.speculative_head = None

        self.speculative_gate = SpeculativeGate(
            margin_threshold=speculative_margin_threshold,
            entropy_threshold=speculative_entropy_threshold,
            min_confidence=speculative_min_confidence,
            enabled=enable_speculative_early_exit,
        )

    @property
    def moe_aux_loss(self) -> torch.Tensor:
        """Returns the MoE load-balancing auxiliary loss from the fusion module.

        Returns 0.0 when use_moe=False. Add to training loss with coefficient ~0.01
        to prevent routing collapse without dominating the task objective.
        """
        if hasattr(self.fusion, "moe_aux_loss"):
            return self.fusion.moe_aux_loss
        return torch.tensor(0.0)

    def get_routing_distribution(self) -> List[Dict[str, float]]:
        """Returns layer-by-layer MoE expert routing distribution for Brain Map telemetry [AO-30, AO-33]."""
        if hasattr(self.fusion, "get_routing_distribution"):
            return self.fusion.get_routing_distribution()
        return []

    @property
    def device(self) -> torch.device:
        return next(self.decision_head.parameters()).device

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns only trainable parameters (fusion + decision head + speculative head), excluding frozen encoders."""
        params = list(self.fusion.parameters()) + list(self.decision_head.parameters())
        if self.speculative_head is not None:
            params += list(self.speculative_head.parameters())
        return [p for p in params if p.requires_grad]

    def encode_inputs(
        self,
        questions: Sequence[str],
        texts: Optional[Sequence[Optional[str]]] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        audios: Optional[Sequence[Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[ModalityType, torch.Tensor], Dict[ModalityType, torch.Tensor], Optional[torch.Tensor]]:
        """
        Runs frozen encoders over inputs and builds presence masks.
        """
        B = len(questions)
        device = self.device

        # Encode questions
        q_embed = self.encoder.encode_text(questions).to(device)

        modality_embeds: Dict[ModalityType, torch.Tensor] = {}
        presence_mask: Dict[ModalityType, torch.Tensor] = {}
        image_patches: Optional[torch.Tensor] = None

        # 1. Text modality
        if texts is not None:
            text_present = [t is not None and len(str(t).strip()) > 0 for t in texts]
            presence_mask[ModalityType.TEXT] = torch.tensor(text_present, dtype=torch.bool, device=device)
            if any(text_present):
                valid_texts = [t if p else "" for t, p in zip(texts, text_present)]
                modality_embeds[ModalityType.TEXT] = self.encoder.encode_text(valid_texts).to(device)
            else:
                modality_embeds[ModalityType.TEXT] = torch.zeros((B, self.encoder.text_dim), device=device)
        else:
            presence_mask[ModalityType.TEXT] = torch.zeros(B, dtype=torch.bool, device=device)
            modality_embeds[ModalityType.TEXT] = torch.zeros((B, self.encoder.text_dim), device=device)

        # 2. Image modality
        if images is not None:
            img_present = [img is not None for img in images]
            presence_mask[ModalityType.IMAGE] = torch.tensor(img_present, dtype=torch.bool, device=device)
            if any(img_present):
                valid_imgs = [img if p else None for img, p in zip(images, img_present)]
                modality_embeds[ModalityType.IMAGE] = self.encoder.encode_image(valid_imgs).to(device)
                if self.use_spatial_patches:
                    image_patches = self.encoder.encode_image_patches(valid_imgs).to(device)
            else:
                modality_embeds[ModalityType.IMAGE] = torch.zeros((B, self.encoder.image_dim), device=device)
        else:
            presence_mask[ModalityType.IMAGE] = torch.zeros(B, dtype=torch.bool, device=device)
            modality_embeds[ModalityType.IMAGE] = torch.zeros((B, self.encoder.image_dim), device=device)

        # 3. Video modality
        if videos is not None:
            vid_present = [
                vid is not None and (len(vid) > 0 if isinstance(vid, (list, tuple)) else True)
                for vid in videos
            ]
            presence_mask[ModalityType.VIDEO] = torch.tensor(vid_present, dtype=torch.bool, device=device)
            if any(vid_present):
                valid_vids = [vid if p else [] for vid, p in zip(videos, vid_present)]
                modality_embeds[ModalityType.VIDEO] = self.encoder.encode_video(valid_vids).to(device)
            else:
                modality_embeds[ModalityType.VIDEO] = torch.zeros((B, self.encoder.video_dim), device=device)
        else:
            presence_mask[ModalityType.VIDEO] = torch.zeros(B, dtype=torch.bool, device=device)
            modality_embeds[ModalityType.VIDEO] = torch.zeros((B, self.encoder.video_dim), device=device)

        # 4. Audio modality
        if audios is not None:
            aud_present = [aud is not None for aud in audios]
            presence_mask[ModalityType.AUDIO] = torch.tensor(aud_present, dtype=torch.bool, device=device)
            if any(aud_present):
                valid_auds = [aud if p else None for aud, p in zip(audios, aud_present)]
                modality_embeds[ModalityType.AUDIO] = self.encoder.encode_audio(valid_auds).to(device)
            else:
                modality_embeds[ModalityType.AUDIO] = torch.zeros((B, self.encoder.audio_dim), device=device)
        else:
            presence_mask[ModalityType.AUDIO] = torch.zeros(B, dtype=torch.bool, device=device)
            modality_embeds[ModalityType.AUDIO] = torch.zeros((B, self.encoder.audio_dim), device=device)

        return q_embed, modality_embeds, presence_mask, image_patches


    def encode_candidates(
        self,
        candidates_batch: Sequence[Sequence[str]],
        prompt_templates: Optional[Sequence[str]] = None,
        use_prompt_ensembling: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes variable numbers of candidate decision strings across a batch.
        Pads to maximum candidates in the batch and creates candidate_mask.
        If use_prompt_ensembling is True, averages normalized embeddings across prompt_templates
        to reduce prompt variance and boost zero-shot visual discrimination.
        
        Returns:
            candidate_embeds: [B, max_K, candidate_dim]
            candidate_mask: [B, max_K] boolean tensor
        """
        B = len(candidates_batch)
        max_K = max((len(cands) for cands in candidates_batch), default=0)
        device = self.device

        candidate_embeds = torch.zeros((B, max_K, self.encoder.text_dim), device=device)
        candidate_mask = torch.zeros((B, max_K), dtype=torch.bool, device=device)

        templates = prompt_templates if prompt_templates is not None else DEFAULT_PROMPT_TEMPLATES

        if not (use_prompt_ensembling and templates):
            # High-efficiency batched text encoding: flatten all candidates into single forward pass
            flat_cands: List[str] = []
            cand_indices: List[Tuple[int, int]] = []
            for i, cands in enumerate(candidates_batch):
                for j, c in enumerate(cands):
                    flat_cands.append(c)
                    cand_indices.append((i, j))
            if flat_cands:
                all_encoded = self.encoder.encode_text(flat_cands).to(device)
                for idx, (i, j) in enumerate(cand_indices):
                    candidate_embeds[i, j, :] = all_encoded[idx]
                    candidate_mask[i, j] = True
            return candidate_embeds, candidate_mask

        # Prompt template ensembling
        for i, cands in enumerate(candidates_batch):
            k = len(cands)
            if k == 0:
                continue

            all_prompts: List[str] = []
            for c in cands:
                for t in templates:
                    all_prompts.append(t.format(c) if "{}" in t else f"{t} {c}")
            all_encoded = self.encoder.encode_text(all_prompts).to(device)
            num_t = len(templates)
            reshaped = all_encoded.view(k, num_t, -1)
            avg_encoded = reshaped.mean(dim=1)
            avg_encoded = F.normalize(avg_encoded, p=2, dim=-1)
            candidate_embeds[i, :k, :] = avg_encoded
            candidate_mask[i, :k] = True

        return candidate_embeds, candidate_mask

    def apply_modality_dropout(
        self, presence_mask: Dict[Any, torch.Tensor]
    ) -> Dict[Any, torch.Tensor]:
        """
        Randomly drops present sensory modalities with probability modality_dropout_prob during training.
        Prevents text/visual shortcuts and forces the fusion engine to learn invariant multi-sensory representations.
        """
        if not self.training or self.modality_dropout_prob <= 0.0:
            return presence_mask

        dropped: Dict[Any, torch.Tensor] = {}
        for mod, mask in presence_mask.items():
            if not isinstance(mask, torch.Tensor):
                mask_t = torch.tensor(mask, device=self.device)
            else:
                mask_t = mask
            rand = torch.rand_like(mask_t.float())
            keep = rand >= self.modality_dropout_prob
            dropped[mod] = mask_t & keep
        return dropped

    def forward(
        self,
        questions: Sequence[str],
        candidates: Sequence[Sequence[str]],
        texts: Optional[Sequence[Optional[str]]] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        audios: Optional[Sequence[Any]] = None,
        use_prompt_ensembling: bool = False,
        prompt_templates: Optional[Sequence[str]] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass:
        1. Encodes questions, modalities, and candidates with frozen encoders.
        2. Fuses present modalities into unified context state.
        3. Dynamically scores candidates and computes probability distribution.
        
        Returns:
            logits: [B, max_K]
            probs: [B, max_K]
            entropy: [B]
            fused_context: [B, hidden_dim]
        """
        q_embed, mod_embeds, presence_mask, image_patches = self.encode_inputs(
            questions=questions, texts=texts, images=images, videos=videos, audios=audios
        )

        if self.training and self.modality_dropout_prob > 0.0:
            presence_mask = self.apply_modality_dropout(presence_mask)

        fused_context = self.fusion(
            question_embed=q_embed,
            modality_embeds=mod_embeds,
            presence_mask=presence_mask,
            image_patches=image_patches,
        )

        cnd_embeds, cnd_mask = self.encode_candidates(
            candidates,
            prompt_templates=prompt_templates,
            use_prompt_ensembling=use_prompt_ensembling,
        )

        logits, probs, entropy = self.decision_head(
            context_embed=fused_context,
            candidate_embeds=cnd_embeds,
            candidate_mask=cnd_mask,
            modality_embeds=mod_embeds,
            presence_mask=presence_mask,
            temperature=temperature,
        )

        return logits, probs, entropy, fused_context

    def forward_cached(
        self,
        question_embed: torch.Tensor,
        modality_embeds: Dict[str, torch.Tensor],
        presence_mask: Dict[str, torch.Tensor],
        candidate_embeds: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes fusion and decision scoring directly on pre-computed encoder embeddings,
        bypassing perception forward passes.
        """
        if self.training and self.modality_dropout_prob > 0.0:
            presence_mask = self.apply_modality_dropout(presence_mask)

        fused_context = self.fusion(
            question_embed=question_embed,
            modality_embeds=modality_embeds,
            presence_mask=presence_mask,
            image_patches=image_patches,
        )

        logits, probs, entropy = self.decision_head(
            context_embed=fused_context,
            candidate_embeds=candidate_embeds,
            candidate_mask=candidate_mask,
            modality_embeds=modality_embeds,
            presence_mask=presence_mask,
            temperature=temperature,
        )

        return logits, probs, entropy, fused_context

    def forward_speculative(
        self,
        question_embed: torch.Tensor,
        candidate_embeds: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        modality_embeds: Optional[Dict[Any, torch.Tensor]] = None,
        presence_mask: Optional[Dict[Any, Any]] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fast Tier-0 Speculative Draft forward pass [AO-29].

        Evaluates lightweight draft compatibility in GPU L1/L2 cache in <0.5 ms.
        Returns:
            logits: [B, K] draft decision logits
            probs: [B, K] draft probability distribution
            entropy: [B] draft entropy
            can_exit: [B] boolean mask of whether early exit criteria are satisfied
            margins: [B] difference between top-1 and top-2 probabilities
            top1_conf: [B] top-1 confidence score
        """
        if self.speculative_head is None:
            raise RuntimeError("Speculative head is not initialized. Pass enable_speculative_early_exit=True.")

        img_embed = None
        img_present = None
        if modality_embeds is not None and presence_mask is not None:
            for k in [ModalityType.IMAGE, "image", ModalityType.IMAGE.value]:
                if k in modality_embeds:
                    img_embed = modality_embeds[k]
                    break
            for k in [ModalityType.IMAGE, "image", ModalityType.IMAGE.value]:
                if k in presence_mask:
                    img_present = presence_mask[k]
                    break

        logits, probs, entropy = self.speculative_head(
            input_embed=question_embed,
            candidate_embeds=candidate_embeds,
            candidate_mask=candidate_mask,
            residual_visual_embed=img_embed,
            visual_present=img_present,
            temperature=temperature,
        )
        can_exit, margins, top1 = self.speculative_gate.should_exit(probs, entropy)
        return logits, probs, entropy, can_exit, margins, top1

