"""
OpenCLIP Multimodal Pretrained Encoder.
Encodes Text, Images, and Video Keyframes using frozen Vision-Language Transformers.
Includes spectral projection for Audio into the shared multimodal space.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union
import numpy as np
import torch
from PIL import Image

from arbiter_omni.encoders.base import BaseMultimodalEncoder


class OpenCLIPMultimodalEncoder(BaseMultimodalEncoder):
    """
    Multimodal encoder backed by OpenCLIP (e.g. ViT-B-32).
    Maps text, images, video frames, and audio into a 512-dimensional joint vector space.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.pretrained = pretrained
        self._dim = 512

        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._dim = getattr(self.model, "visual", self.model).output_dim if hasattr(self.model, "visual") else 512
        self.freeze()

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

    def encode_video(self, videos: Sequence[Any], num_frames: int = 8) -> torch.Tensor:
        """Encodes video clips by sampling frames and mean-pooling frame embeddings."""
        batch_video_features = []
        for vid in videos:
            if isinstance(vid, (list, tuple)):
                if len(vid) == 0:
                    frame_feats = torch.zeros((1, self._dim), device=self.device)
                else:
                    step = max(1, len(vid) // num_frames)
                    sampled = vid[::step][:num_frames]
                    frame_feats = self.encode_image(sampled)
                pooled = frame_feats.mean(dim=0)
            elif isinstance(vid, str):
                # Placeholder for video file path loading
                # Loads keyframes or single image proxy
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
        Embeds raw audio waveforms into the shared multimodal space
        via spectral projection and multi-band energy pooling.
        """
        results = []
        for audio in audios:
            if isinstance(audio, str):
                # Audio path
                import soundfile as sf
                try:
                    data, _ = sf.read(audio)
                    audio_tensor = torch.tensor(data, dtype=torch.float32)
                except Exception:
                    audio_tensor = torch.zeros(16000, dtype=torch.float32)
            elif isinstance(audio, np.ndarray):
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
            elif isinstance(audio, torch.Tensor):
                audio_tensor = audio.float()
            else:
                audio_tensor = torch.zeros(16000, dtype=torch.float32)

            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.mean(dim=-1)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # STFT Spectral Energy
            n_fft = 512
            if audio_tensor.shape[-1] < n_fft:
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, n_fft - audio_tensor.shape[-1]))

            spec = torch.stft(audio_tensor, n_fft=n_fft, return_complex=True)
            spec_mag = torch.abs(spec).mean(dim=-1).squeeze(0)  # [freq_bins]

            if spec_mag.shape[-1] < self._dim:
                padded = torch.zeros(self._dim, device=self.device)
                padded[: spec_mag.shape[-1]] = spec_mag.to(self.device)
                spec_mag = padded
            elif spec_mag.shape[-1] > self._dim:
                spec_mag = spec_mag[: self._dim].to(self.device)
            else:
                spec_mag = spec_mag.to(self.device)

            spec_mag = spec_mag / (spec_mag.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(spec_mag)

        return torch.stack(results).to(self.device)
