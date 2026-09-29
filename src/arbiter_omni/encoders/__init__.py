"""
Modular multimodal encoders package.
"""

from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.encoders.clap import CLAPAudioEncoder
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder
from arbiter_omni.encoders.temporal import SpatioTemporalVideoAttention

__all__ = [
    "BaseMultimodalEncoder",
    "CLAPAudioEncoder",
    "MockMultimodalEncoder",
    "OpenCLIPMultimodalEncoder",
    "SpatioTemporalVideoAttention",
]
