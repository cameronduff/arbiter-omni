"""
OpenCLIP Multimodal Pretrained Encoder with Advanced Extensions.
Encodes Text, Images, Video (via Spatio-Temporal Video Attention), and Audio (via CLAP).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union
import numpy as np
import torch
from PIL import Image

from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.encoders.clap import CLAPAudioEncoder
from arbiter_omni.encoders.temporal import SpatioTemporalVideoAttention


class OpenCLIPMultimodalEncoder(BaseMultimodalEncoder):
    """
    Multimodal encoder backed by OpenCLIP (e.g. ViT-B-32).
    Maps text, images, video frame sequences, and audio into a 512-dimensional joint vector space.
    Employs Spatio-Temporal Video Attention for video sequences and CLAP for audio.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: Optional[str] = None,
        device: Optional[torch.device] = None,
        enable_clap_weights: bool = False,
        use_temporal_attention: bool = True,
    ):
        super().__init__(device=device)
        self.model_name = model_name

        # Auto-resolve pretrained dataset tag if not specified or default is passed
        if pretrained is None or pretrained == "laion2b_s34b_b79k":
            if model_name == "ViT-B-16":
                pretrained = "laion2b_s34b_b88k"
            else:
                pretrained = "laion2b_s34b_b79k"
        elif model_name == "ViT-B-32" and pretrained == "laion2b_s34b_b88k":
            pretrained = "laion2b_s34b_b79k"

        self.pretrained = pretrained
        self._dim = 512
        self.use_temporal_attention = use_temporal_attention

        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._dim = getattr(self.model, "visual", self.model).output_dim if hasattr(self.model, "visual") else 512

        # Temporal Video Attention Transformer
        self.temporal_attention = SpatioTemporalVideoAttention(
            embed_dim=self._dim, max_frames=32, num_heads=8
        ).to(self.device)

        # CLAP Audio Encoder
        self.audio_encoder = CLAPAudioEncoder(
            device=self.device,
            output_dim=self._dim,
            load_pretrained=enable_clap_weights,
        )

        self.freeze()

    def freeze(self) -> OpenCLIPMultimodalEncoder:
        """Freezes vision-text backbone, temporal attention module, and audio encoder."""
        super().freeze()
        if hasattr(self, "temporal_attention") and self.temporal_attention is not None:
            self.temporal_attention.eval()
            for p in self.temporal_attention.parameters():
                p.requires_grad = False
        if hasattr(self, "audio_encoder") and self.audio_encoder is not None:
            self.audio_encoder.freeze()
        return self

    @property
    def text_dim(self) -> int:
        return self._dim

    @property
    def image_dim(self) -> int:
        return self._dim

    @property
    def video_dim(self) -> int:
        return self._dim

    @property
    def audio_dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Embeds text prompts, questions, or candidate decisions."""
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features

    def encode_image(self, images: Sequence[Any]) -> torch.Tensor:
        """Embeds single images or keyframes."""
        processed_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            processed_tensors.append(self.preprocess(img))

        batch = torch.stack(processed_tensors).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features

    def encode_image_patches(self, images: Sequence[Any]) -> torch.Tensor:
        """
        Extracts unpooled 2D spatial patch tokens from OpenCLIP visual transformer.
        Returns:
            [B, P, image_dim] tensor of normalized patch embeddings
            (P = 49 for ViT-B-32 [7x7 grid], P = 196 for ViT-B-16 [14x14 grid]).
        """
        processed_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            processed_tensors.append(self.preprocess(img))

        batch = torch.stack(processed_tensors).to(self.device)
        with torch.no_grad():
            visual = getattr(self.model, "visual", None)
            if visual is not None and hasattr(visual, "_embeds") and hasattr(visual, "transformer"):
                x_embeds = visual._embeds(batch)
                x_trans = visual.transformer(x_embeds)
                _, tokens = visual._pool(x_trans)
                if getattr(visual, "proj", None) is not None:
                    tokens_proj = tokens @ visual.proj
                else:
                    tokens_proj = tokens
                patch_tokens = tokens_proj / (tokens_proj.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                pooled = self.encode_image(images)
                patch_tokens = pooled.unsqueeze(1)
        return patch_tokens


    def encode_video(self, videos: Sequence[Any], num_frames: int = 8) -> torch.Tensor:
        """
        Encodes video clips by sampling frames and aggregating them using Spatio-Temporal Video Attention.
        """
        batch_video_features = []
        for vid in videos:
            if isinstance(vid, (list, tuple)):
                if len(vid) == 0:
                    pooled = torch.zeros(self._dim, device=self.device)
                elif len(vid) == 1:
                    pooled = self.encode_image(vid)[0]
                else:
                    step = max(1, len(vid) // num_frames)
                    sampled = vid[::step][:num_frames]
                    frame_feats = self.encode_image(sampled)  # [num_frames, dim]
                    if self.use_temporal_attention:
                        with torch.no_grad():
                            pooled = self.temporal_attention(frame_feats)
                    else:
                        pooled = frame_feats.mean(dim=0)
            elif isinstance(vid, str):
                try:
                    img = Image.open(vid).convert("RGB")
                    pooled = self.encode_image([img])[0]
                except Exception:
                    pooled = torch.zeros(self._dim, device=self.device)
            else:
                pooled = torch.zeros(self._dim, device=self.device)

            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
            batch_video_features.append(pooled)

        return torch.stack(batch_video_features).to(self.device)

    def encode_audio(self, audios: Sequence[Any], sample_rate: int = 16000) -> torch.Tensor:
        """
        Embeds raw audio waveforms into the shared multimodal space using CLAP.
        """
        return self.audio_encoder.encode_audio(audios, sample_rate=sample_rate)
