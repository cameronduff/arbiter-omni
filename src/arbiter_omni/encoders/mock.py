"""
Mock / Lightweight Synthetic Multimodal Encoder.
Provides deterministic, zero-download, instant embeddings for unit testing,
CI pipelines, and resource-constrained environments.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from arbiter_omni.encoders.base import BaseMultimodalEncoder


class MockMultimodalEncoder(BaseMultimodalEncoder):
    """
    Lightweight, deterministic multimodal encoder for testing and quick prototyping.
    Maps Text, Images, Videos, and Audio to a unified embedding space of specified dimension
    without downloading external model weights.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__(device=device)
        self._dim = embed_dim
        # Register a dummy parameter so nn.Module parameters() works
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
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

    def _hash_to_vec(self, text: str) -> np.ndarray:
        """Deterministically hashes text into a pseudo-random unit vector."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Deterministically generates normalized vectors from text."""
        vecs = [self._hash_to_vec(t) for t in texts]
        tensor = torch.tensor(np.stack(vecs), dtype=torch.float32, device=self.device)
        return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)

    def encode_image(self, images: Sequence[Any]) -> torch.Tensor:
        """
        Extracts lightweight color and spatial statistics from images,
        projected deterministically to [B, image_dim].
        """
        results = []
        for img in images:
            if isinstance(img, str):
                # Simulated text hash if path
                results.append(self._hash_to_vec(f"image_path:{img}"))
                continue

            if isinstance(img, Image.Image):
                arr = np.array(img.convert("RGB"), dtype=np.float32)
            elif isinstance(img, np.ndarray):
                arr = img.astype(np.float32)
            elif isinstance(img, torch.Tensor):
                arr = img.detach().cpu().numpy().astype(np.float32)
            else:
                arr = np.ones((64, 64, 3), dtype=np.float32)

            # Compute channel means and variance as pseudo-features
            features = np.zeros(self._dim, dtype=np.float32)
            ch_means = arr.mean(axis=(0, 1)) if arr.ndim >= 3 else [arr.mean()] * 3
            ch_stds = arr.std(axis=(0, 1)) if arr.ndim >= 3 else [arr.std()] * 3

            for i in range(min(len(ch_means), self._dim // 2)):
                features[i] = ch_means[i] / 255.0
                features[i + self._dim // 2] = ch_stds[i] / 255.0

            norm = np.linalg.norm(features) + 1e-8
            results.append(features / norm)

        tensor = torch.tensor(np.stack(results), dtype=torch.float32, device=self.device)
        return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)

    def encode_image_patches(self, images: Sequence[Any], num_patches: int = 49) -> torch.Tensor:
        """
        Generates synthetic unpooled spatial patch tokens [B, num_patches, image_dim].
        Deterministic per-patch spatial offsets simulating 2D visual grid locations.
        """
        base_pooled = self.encode_image(images)  # [B, dim]
        B = base_pooled.size(0)
        # Create deterministic pseudo-spatial offsets across patches
        grid_offsets = []
        for p in range(num_patches):
            px = (p % 7) / 7.0 - 0.5
            py = (p // 7) / 7.0 - 0.5
            offset = self._hash_to_vec(f"patch_offset:{px:.3f}:{py:.3f}")
            grid_offsets.append(offset)
        offsets_tensor = torch.tensor(np.stack(grid_offsets), dtype=torch.float32, device=self.device)  # [P, dim]
        
        # Combine base pooled with spatial grid offset: [B, P, dim]
        patches = base_pooled.unsqueeze(1) + 0.15 * offsets_tensor.unsqueeze(0)
        return patches / (patches.norm(dim=-1, keepdim=True) + 1e-8)

    def encode_video(self, videos: Sequence[Any], num_frames: int = 8) -> torch.Tensor:

        """Encodes video by sampling frame representations and mean-pooling."""
        results = []
        for vid in videos:
            if isinstance(vid, (list, tuple)):
                if len(vid) == 0:
                    frame_feats = torch.zeros((1, self._dim), device=self.device)
                else:
                    # Sample up to num_frames
                    step = max(1, len(vid) // num_frames)
                    sampled = vid[::step][:num_frames]
                    frame_feats = self.encode_image(sampled)
                video_feat = frame_feats.mean(dim=0)
            elif isinstance(vid, str):
                video_feat = torch.tensor(
                    self._hash_to_vec(f"video_path:{vid}"),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                video_feat = torch.zeros(self._dim, dtype=torch.float32, device=self.device)

            video_feat = video_feat / (video_feat.norm(dim=-1, keepdim=True) + 1e-8)
            results.append(video_feat)

        return torch.stack(results).to(self.device)

    def encode_audio(self, audios: Sequence[Any], sample_rate: int = 16000) -> torch.Tensor:
        """Encodes audio waveform statistics into a normalized vector."""
        results = []
        for aud in audios:
            if isinstance(aud, str):
                results.append(self._hash_to_vec(f"audio_path:{aud}"))
                continue

            if isinstance(aud, torch.Tensor):
                arr = aud.detach().cpu().numpy().astype(np.float32).flatten()
            elif isinstance(aud, np.ndarray):
                arr = aud.astype(np.float32).flatten()
            else:
                arr = np.zeros(1600, dtype=np.float32)

            features = np.zeros(self._dim, dtype=np.float32)
            if len(arr) > 0:
                features[0] = float(np.mean(arr))
                features[1] = float(np.std(arr))
                features[2] = float(np.max(np.abs(arr)))
                # Energy across frequency bins (FFT)
                fft = np.abs(np.fft.rfft(arr[:1024]))
                num_bins = min(len(fft), self._dim - 3)
                if num_bins > 0:
                    features[3 : 3 + num_bins] = fft[:num_bins]

            norm = np.linalg.norm(features) + 1e-8
            results.append(features / norm)

        tensor = torch.tensor(np.stack(results), dtype=torch.float32, device=self.device)
        return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)
