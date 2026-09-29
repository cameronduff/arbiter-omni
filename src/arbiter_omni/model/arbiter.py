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
        use_spatial_patches: bool = True,
        modality_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.freeze()  # Guarantee encoders are frozen
        self.use_spatial_patches = use_spatial_patches
        self.modality_dropout_prob = modality_dropout_prob


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

    @property
    def device(self) -> torch.device:
        return next(self.decision_head.parameters()).device

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns only trainable parameters (fusion + decision head), excluding frozen encoders."""
        params = list(self.fusion.parameters()) + list(self.decision_head.parameters())
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
            valid_texts = [t if p else "" for t, p in zip(texts, text_present)]
            modality_embeds[ModalityType.TEXT] = self.encoder.encode_text(valid_texts).to(device)
        else:
            presence_mask[ModalityType.TEXT] = torch.zeros(B, dtype=torch.bool, device=device)
            modality_embeds[ModalityType.TEXT] = torch.zeros((B, self.encoder.text_dim), device=device)

        # 2. Image modality
        if images is not None:
            img_present = [img is not None for img in images]
            presence_mask[ModalityType.IMAGE] = torch.tensor(img_present, dtype=torch.bool, device=device)
            valid_imgs = [img if p else None for img, p in zip(images, img_present)]
            modality_embeds[ModalityType.IMAGE] = self.encoder.encode_image(valid_imgs).to(device)
            if self.use_spatial_patches and any(img_present):
                image_patches = self.encoder.encode_image_patches(valid_imgs).to(device)
        else:
            presence_mask[ModalityType.IMAGE] = torch.zeros(B, dtype=torch.bool, device=device)
            modality_embeds[ModalityType.IMAGE] = torch.zeros((B, self.encoder.image_dim), device=device)

        # 3. Video modality
        if videos is not None:
            vid_present = [vid is not None for vid in videos]
            presence_mask[ModalityType.VIDEO] = torch.tensor(vid_present, dtype=torch.bool, device=device)
            valid_vids = [vid if p else [] for vid, p in zip(videos, vid_present)]
            modality_embeds[ModalityType.VIDEO] = self.encoder.encode_video(valid_vids).to(device)
        else:
            presence_mask[ModalityType.VIDEO] = torch.zeros(B, dtype=torch.bool, device=device)
            modality_embeds[ModalityType.VIDEO] = torch.zeros((B, self.encoder.video_dim), device=device)

        # 4. Audio modality
        if audios is not None:
            aud_present = [aud is not None for aud in audios]
            presence_mask[ModalityType.AUDIO] = torch.tensor(aud_present, dtype=torch.bool, device=device)
            valid_auds = [aud if p else None for aud, p in zip(audios, aud_present)]
            modality_embeds[ModalityType.AUDIO] = self.encoder.encode_audio(valid_auds).to(device)
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

        # Flatten all unique candidates or per-sample candidates
        for i, cands in enumerate(candidates_batch):
            k = len(cands)
            if k == 0:
                continue

            if use_prompt_ensembling and templates:
                # Prompt template ensembling
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
            else:
                encoded = self.encoder.encode_text(cands).to(device)
                candidate_embeds[i, :k, :] = encoded

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

