"""
Abstract Base Class and Registry for Pretrained Multimodal Encoders.
All encoders remain strictly frozen during fusion and decision training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn


class BaseMultimodalEncoder(nn.Module, ABC):
    """
    Abstract Interface for Multimodal Feature Extraction.
    
    Extracts fixed, normalized embeddings for Text, Image, Video, and Audio.
    Concrete implementations can wrap OpenCLIP, HuggingFace Transformers,
    CLAP, or lightweight synthetic models.
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, val: Union[str, torch.device]):
        self._device = torch.device(val)

    @property
    @abstractmethod
    def text_dim(self) -> int:
        """Output embedding dimension for text."""
        pass

    @property
    @abstractmethod
    def image_dim(self) -> int:
        """Output embedding dimension for images."""
        pass

    @property
    @abstractmethod
    def video_dim(self) -> int:
        """Output embedding dimension for video."""
        pass

    @property
    @abstractmethod
    def audio_dim(self) -> int:
        """Output embedding dimension for audio."""
        pass

    def freeze(self) -> BaseMultimodalEncoder:
        """Freezes all encoder weights and sets evaluation mode."""
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        return self

    def to(self, *args, **kwargs) -> BaseMultimodalEncoder:
        """Transfers encoder to device and updates internal device pointer."""
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                self._device = torch.device(arg)
                break
        if "device" in kwargs and kwargs["device"] is not None:
            self._device = torch.device(kwargs["device"])
        return super().to(*args, **kwargs)

    @abstractmethod
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """
        Embeds a batch of text strings into [B, text_dim].
        Must return normalized embeddings.
        """
        pass

    @abstractmethod
    def encode_image(self, images: Sequence[Any]) -> torch.Tensor:
        """
        Embeds a batch of images (PIL Images, numpy arrays, or paths) into [B, image_dim].
        Must return normalized embeddings.
        """
        pass

    def encode_image_patches(self, images: Sequence[Any]) -> torch.Tensor:
        """
        Embeds a batch of images into unpooled spatial patch tokens [B, P, image_dim].
        Default fallback replicates/reshapes pooled image embedding to [B, 1, image_dim].
        """
        pooled = self.encode_image(images)
        return pooled.unsqueeze(1)


    @abstractmethod
    def encode_video(self, videos: Sequence[Any], num_frames: int = 8) -> torch.Tensor:
        """
        Embeds a batch of videos (list of frames or paths) into [B, video_dim].
        Samples keyframes, encodes them, and aggregates temporally.
        Must return normalized embeddings.
        """
        pass

    @abstractmethod
    def encode_audio(self, audios: Sequence[Any], sample_rate: int = 16000) -> torch.Tensor:
        """
        Embeds a batch of audio waveforms or audio file paths into [B, audio_dim].
        Must return normalized embeddings.
        """
        pass
