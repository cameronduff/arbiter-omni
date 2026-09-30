"""
OpenCLIP Multimodal Pretrained Encoder with Advanced Extensions.
Encodes Text, Images, Video (via Spatio-Temporal Video Attention), and Audio (via CLAP).

AO-26: SigLIP-SO400M Integration
Adds support for ViT-SO400M-14-SigLIP-384 (1152-dim, 435M parameters), Google's largest
SigLIP backbone. At 435M parameters it exceeds the 4 GB GDDR5 VRAM budget when combined
with the 64-frame temporal attention pipeline, so it is automatically offloaded to CPU
shared RAM (8 GB DDR4 pool) while keeping the lightweight fusion+head on GPU.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union
import numpy as np
import torch
from PIL import Image

from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.encoders.clap import CLAPAudioEncoder
from arbiter_omni.encoders.temporal import SpatioTemporalVideoAttention


# SO400M: 435M param SigLIP backbone with 1152-dim output and 384×384 input resolution
_SO400M_NAME = "ViT-SO400M-14-SigLIP-384"
_SO400M_DIM = 1152
# Threshold: models with weight count above this trigger automatic CPU offload to shared RAM
_CPU_OFFLOAD_PARAM_THRESHOLD = 200_000_000  # 200M params


class OpenCLIPMultimodalEncoder(BaseMultimodalEncoder):
    """
    Multimodal encoder backed by OpenCLIP.
    Maps text, images, video frame sequences, and audio into a joint vector space.
    Employs Spatio-Temporal Video Attention for video sequences and CLAP for audio.

    AO-26: Supports ViT-SO400M-14-SigLIP-384 (1152-dim, 435M parameters).
    Large backbones (>200M params) are automatically offloaded to CPU shared RAM
    to prevent VRAM OOM on the RX 480 (4 GB GDDR5) while keeping the fusion
    and decision head on GPU.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: Optional[str] = None,
        device: Optional[torch.device] = None,
        enable_clap_weights: bool = False,
        use_temporal_attention: bool = True,
        max_frames: int = 64,
        cpu_offload_encoder: bool = False,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.max_frames = max_frames

        # ---- AO-26: Pretrained tag auto-resolution ----
        is_siglip = "siglip" in model_name.lower()
        is_so400m = _SO400M_NAME.lower() in model_name.lower() or "so400m" in model_name.lower()

        if is_siglip or is_so400m:
            if pretrained is None or pretrained.startswith("laion"):
                pretrained = "webli"
        elif pretrained is None or pretrained == "laion2b_s34b_b79k":
            if model_name == "ViT-B-16":
                pretrained = "laion2b_s34b_b88k"
            elif model_name == "ViT-L-14":
                pretrained = "laion2b_s32b_b82k"
            else:
                pretrained = "laion2b_s34b_b79k"
        elif model_name == "ViT-B-32" and pretrained == "laion2b_s34b_b88k":
            pretrained = "laion2b_s34b_b79k"

        self.pretrained = pretrained
        self._dim = 512
        self.use_temporal_attention = use_temporal_attention

        import open_clip

        # AO-26: Load model initially on CPU so we can count params before deciding
        # where to place it (avoids OOM from loading 435M param SO400M directly to VRAM)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device="cpu"
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # ---- Resolve output embedding dimension ----
        visual_mod = getattr(self.model, "visual", None)
        text_mod = getattr(self.model, "text", None)
        if is_so400m:
            # SO400M always outputs 1152-dim regardless of open_clip output_dim reporting
            self._dim = _SO400M_DIM
        elif visual_mod is not None and getattr(visual_mod, "output_dim", None) is not None:
            self._dim = visual_mod.output_dim
        elif text_mod is not None and getattr(text_mod, "output_dim", None) is not None:
            self._dim = text_mod.output_dim
        else:
            self._dim = 768 if is_siglip else 512

        # ---- AO-26: CPU offload decision ----
        # Count total backbone params; automatically offload large models (>200M)
        # to CPU shared RAM (8 GB DDR4) to avoid exhausting the 4 GB GDDR5 VRAM budget.
        total_params = sum(p.numel() for p in self.model.parameters())
        self._encoder_on_cpu = cpu_offload_encoder or (total_params > _CPU_OFFLOAD_PARAM_THRESHOLD)

        if self._encoder_on_cpu:
            # Backbone stays on CPU shared RAM; fusion+head live on GPU
            self._encoder_device = torch.device("cpu")
        else:
            # Smaller backbones (ViT-B/L) fit in VRAM alongside the fusion network
            self._encoder_device = self.device
            self.model = self.model.to(self._encoder_device)

        # Temporal Video Attention Transformer (64-frame dense long horizon)
        # Always resides on main device for GPU-accelerated temporal pooling
        self.temporal_attention = SpatioTemporalVideoAttention(
            embed_dim=self._dim, max_frames=max_frames, num_heads=8
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

    def _apply(self, fn, recurse=True):
        """Controls submodule parameter dispatch to guarantee backbone stays on CPU when offloaded."""
        if getattr(self, "_encoder_on_cpu", False):
            if hasattr(self, "temporal_attention") and self.temporal_attention is not None:
                self.temporal_attention._apply(fn, recurse=recurse)
            if hasattr(self, "audio_encoder") and self.audio_encoder is not None:
                self.audio_encoder._apply(fn, recurse=recurse)
            return self
        return super()._apply(fn, recurse=recurse)

    def to(self, *args, **kwargs) -> OpenCLIPMultimodalEncoder:
        """Dispatches encoder submodules while respecting CPU offload settings."""
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
            if hasattr(self, "temporal_attention") and self.temporal_attention is not None:
                self.temporal_attention.to(*args, **kwargs)
            if hasattr(self, "audio_encoder") and self.audio_encoder is not None:
                self.audio_encoder.to(*args, **kwargs)
            if not getattr(self, "_encoder_on_cpu", False):
                if hasattr(self, "model") and self.model is not None:
                    self.model.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

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

    @property
    def encoder_param_count(self) -> int:
        """Returns total trainable + frozen parameter count for the vision-language backbone."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def is_cpu_offloaded(self) -> bool:
        """Returns True if the backbone is resident in CPU shared RAM (SO400M offload mode)."""
        return getattr(self, "_encoder_on_cpu", False)

    @property
    def backbone_device(self) -> torch.device:
        """Returns the device where backbone parameters reside (cpu or cuda/directml)."""
        if getattr(self, "_encoder_on_cpu", False):
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return getattr(self, "_encoder_device", self.device)

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Embeds text prompts, questions, or candidate decisions.

        AO-26: Routes tokens to ``backbone_device`` (CPU for large backbones like SO400M),
        then returns features on ``self.device`` so downstream fusion stays on GPU.
        """
        enc_dev = self.backbone_device
        tokens = self.tokenizer(list(texts)).to(enc_dev)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features.to(self.device)

    def encode_image(self, images: Sequence[Any]) -> torch.Tensor:
        """Embeds single images or keyframes.

        AO-26: Sends preprocessed batch to ``backbone_device`` (CPU for large backbones)
        and returns normalized features on ``self.device``.
        """
        enc_dev = self.backbone_device
        processed_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            processed_tensors.append(self.preprocess(img))

        batch = torch.stack(processed_tensors).to(enc_dev)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features.to(self.device)

    def encode_image_patches(self, images: Sequence[Any]) -> torch.Tensor:
        """
        Extracts unpooled 2D spatial patch tokens from OpenCLIP visual transformer.
        Returns:
            [B, P, image_dim] tensor of normalized patch embeddings
            (P = 49 for ViT-B-32 [7x7 grid], P = 196 for ViT-B-16 [14x14 grid]).
        """
        enc_dev = self.backbone_device
        processed_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            processed_tensors.append(self.preprocess(img))

        batch = torch.stack(processed_tensors).to(enc_dev)
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
            elif visual is not None and hasattr(visual, "trunk") and hasattr(visual.trunk, "forward_features"):
                feat = visual.trunk.forward_features(batch)
                if hasattr(visual, "head") and getattr(visual.head, "proj", None) is not None:
                    tokens_proj = visual.head.proj(feat)
                else:
                    tokens_proj = feat
                patch_tokens = tokens_proj / (tokens_proj.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                pooled = self.encode_image(images)
                patch_tokens = pooled.unsqueeze(1)
        return patch_tokens.to(self.device)

    def encode_tiled_image_patches(self, images: Sequence[Any], force_tiling: bool = False) -> torch.Tensor:
        """
        Extracts multi-scale spatial patch tokens (1 global overview + 4 quadrant tiles = up to 980 patches)
        for high-resolution photos and diagrams.
        
        Returns:
            [B, total_P, D] tensor where total_P is up to 980 spatial patch tokens.
        """
        from arbiter_omni.encoders.tiling import DynamicImageTiler
        tiler = DynamicImageTiler()
        batch_tiled_patches = []
        for img in images:
            tiles = tiler.tile_image(img, force_tiling=force_tiling)
            tile_patches = self.encode_image_patches(tiles) # [num_tiles, 196, D]
            concatenated = tile_patches.reshape(1, tile_patches.shape[0] * tile_patches.shape[1], -1)
            batch_tiled_patches.append(concatenated)

        max_P = max(tp.shape[1] for tp in batch_tiled_patches)
        B = len(images)
        D = batch_tiled_patches[0].shape[-1]
        output = torch.zeros((B, max_P, D), device=self.device)
        for i, tp in enumerate(batch_tiled_patches):
            p_len = tp.shape[1]
            output[i, :p_len, :] = tp[0]
        return output


    def encode_video(
        self,
        videos: Sequence[Any],
        num_frames: int = 16,
        sub_batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Encodes video clips by sampling frames and aggregating them using Spatio-Temporal Video Attention.
        Supports dense long-horizon temporal buffering (up to 64 frames) via sub-batched chunking to
        guarantee <1 GB peak working memory during perception.
        Supports lists of PIL images, numpy arrays, or file paths (.mp4, .webm, .gif, .avi, etc.).
        """
        batch_video_features = []
        for vid in videos:
            frames: List[Image.Image] = []
            if isinstance(vid, (list, tuple)):
                frames = list(vid)
            elif isinstance(vid, str) and os.path.exists(vid):
                ext = os.path.splitext(vid)[1].lower()
                if ext in (".mp4", ".avi", ".mov", ".webm", ".mkv", ".gif"):
                    try:
                        import imageio.v3 as iio
                        raw = iio.imread(vid)
                        step = max(1, len(raw) // num_frames)
                        frames = [Image.fromarray(raw[i]) for i in range(0, len(raw), step)[:num_frames]]
                    except Exception:
                        try:
                            frames = [Image.open(vid).convert("RGB")]
                        except Exception:
                            frames = []
                else:
                    try:
                        frames = [Image.open(vid).convert("RGB")]
                    except Exception:
                        frames = []
            elif isinstance(vid, Image.Image):
                frames = [vid]

            if len(frames) == 0:
                pooled = torch.zeros(self._dim, device=self.device)
            elif len(frames) == 1:
                pooled = self.encode_image(frames)[0]
            else:
                step = max(1, len(frames) // num_frames)
                sampled = frames[::step][:num_frames]

                # Chunked sub-batch processing for dense temporal sequences (e.g. 32–64 frames)
                frame_feats_chunks = []
                frame_patches_chunks = []
                for chunk_start in range(0, len(sampled), sub_batch_size):
                    chunk = sampled[chunk_start : chunk_start + sub_batch_size]
                    f_chunk = self.encode_image(chunk)
                    frame_feats_chunks.append(f_chunk)
                    if self.use_temporal_attention:
                        with torch.no_grad():
                            p_chunk = self.encode_image_patches(chunk)
                            frame_patches_chunks.append(p_chunk)

                frame_feats = torch.cat(frame_feats_chunks, dim=0)  # [num_frames, dim]

                if self.use_temporal_attention:
                    frame_patches = torch.cat(frame_patches_chunks, dim=0)  # [num_frames, P, dim]
                    pooled = self.temporal_attention(frame_feats, patch_features=frame_patches)
                else:
                    pooled = frame_feats.mean(dim=0)

            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
            batch_video_features.append(pooled)

        return torch.stack(batch_video_features).to(self.device)

    def encode_audio(self, audios: Sequence[Any], sample_rate: int = 16000) -> torch.Tensor:
        """
        Embeds raw audio waveforms into the shared multimodal space using CLAP.
        """
        return self.audio_encoder.encode_audio(audios, sample_rate=sample_rate)
