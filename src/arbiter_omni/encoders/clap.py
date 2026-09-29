"""
Pretrained CLAP (Contrastive Language-Audio Pretraining) Encoder.
Embeds raw audio waveforms into a 512-dim joint semantic space aligned with language and vision.
Includes robust fallback to spectral STFT projection when offline or uninitialized.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn

from arbiter_omni.encoders.base import BaseMultimodalEncoder

logger = logging.getLogger(__name__)


class CLAPAudioEncoder(BaseMultimodalEncoder):
    """
    CLAP Pretrained Audio Feature Extractor.
    Extracts fixed 512-dimensional joint audio-language embeddings.
    
    If transformers CLAP checkpoint cannot be downloaded (e.g. offline/isolated tests),
    it seamlessly falls back to spectral STFT projection to prevent disruption.
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: Optional[torch.device] = None,
        load_pretrained: bool = True,
        output_dim: int = 512,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self._output_dim = output_dim
        self.model = None
        self.processor = None
        self.use_clap_backend = False

        # Spectral fallback projection matrix
        self.spectral_proj = nn.Linear(512, output_dim, bias=False)
        nn.init.orthogonal_(self.spectral_proj.weight)

        if load_pretrained:
            try:
                from transformers import ClapAudioModel, ClapProcessor

                self.processor = ClapProcessor.from_pretrained(model_name)
                self.model = ClapAudioModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                self.use_clap_backend = True
                logger.info(f"Loaded pretrained CLAP audio encoder: {model_name}")
            except Exception as e:
                logger.warning(
                    f"Could not load CLAP checkpoint ({e}). Operating in spectral STFT fallback mode."
                )
                self.use_clap_backend = False

        self.freeze()

    @property
    def text_dim(self) -> int:
        return self._output_dim

    @property
    def image_dim(self) -> int:
        return self._output_dim

    @property
    def video_dim(self) -> int:
        return self._output_dim

    @property
    def audio_dim(self) -> int:
        return self._output_dim

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Fallback or zero embedding for text (CLAPAudioEncoder focuses on audio)."""
        B = len(texts)
        return torch.zeros((B, self._output_dim), device=self.device)

    def encode_image(self, images: Sequence[Any]) -> torch.Tensor:
        """Fallback or zero embedding for images."""
        B = len(images)
        return torch.zeros((B, self._output_dim), device=self.device)

    def encode_video(self, videos: Sequence[Any], num_frames: int = 8) -> torch.Tensor:
        """Fallback or zero embedding for video."""
        B = len(videos)
        return torch.zeros((B, self._output_dim), device=self.device)

    def encode_audio(self, audios: Sequence[Any], sample_rate: int = 16000) -> torch.Tensor:
        """
        Embeds audio waveforms or audio file paths into [B, audio_dim].
        Uses pretrained CLAP if initialized, otherwise falls back to spectral STFT.
        """
        raw_waveforms = []
        for audio in audios:
            if isinstance(audio, str):
                import soundfile as sf
                try:
                    data, sr = sf.read(audio)
                    if data.ndim > 1:
                        data = data.mean(axis=-1)
                    raw_waveforms.append(np.array(data, dtype=np.float32))
                except Exception:
                    raw_waveforms.append(np.zeros(16000, dtype=np.float32))
            elif isinstance(audio, np.ndarray):
                arr = audio
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                raw_waveforms.append(arr.astype(np.float32))
            elif isinstance(audio, torch.Tensor):
                arr = audio.detach().cpu().float().numpy()
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                raw_waveforms.append(arr)
            else:
                raw_waveforms.append(np.zeros(16000, dtype=np.float32))

        if self.use_clap_backend and self.model is not None and self.processor is not None:
            try:
                inputs = self.processor(
                    audios=raw_waveforms,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # CLAP pooled audio feature embedding
                    features = outputs.pooler_output
                    features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                return features
            except Exception as e:
                logger.debug(f"CLAP inference error ({e}), falling back to spectral projection.")

        # STFT Spectral Fallback
        results = []
        for wave in raw_waveforms:
            wave_tensor = torch.tensor(wave, dtype=torch.float32, device="cpu")
            if wave_tensor.ndim == 1:
                wave_tensor = wave_tensor.unsqueeze(0)

            n_fft = 512
            if wave_tensor.shape[-1] < n_fft:
                wave_tensor = torch.nn.functional.pad(
                    wave_tensor, (0, n_fft - wave_tensor.shape[-1])
                )

            # Compute complex STFT on CPU to ensure hardware portability (e.g. DirectML lacks ComplexFloat HLSL support)
            window = torch.hann_window(n_fft, device="cpu")
            spec = torch.stft(wave_tensor, n_fft=n_fft, window=window, return_complex=True)
            spec_mag = torch.abs(spec).mean(dim=-1).squeeze(0).to(self.device)  # [freq_bins]

            padded = torch.zeros(512, device=self.device)
            valid_len = min(512, spec_mag.shape[-1])
            padded[:valid_len] = spec_mag[:valid_len]

            with torch.no_grad():
                proj = self.spectral_proj(padded)
                normed = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(normed)

        return torch.stack(results).to(self.device)
