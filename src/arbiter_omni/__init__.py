"""
ArbiterOmni: Multimodal System 1 Decision Engine inspired by Jev.
"""

from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.data.dataset import (
    MultimodalDecisionDataset,
    collate_multimodal_decision,
)
from arbiter_omni.data.synthetic import (
    create_synthetic_audio,
    create_synthetic_image,
    generate_synthetic_dataset,
)
from arbiter_omni.device import (
    configure_cpu_threads,
    get_device_telemetry,
    print_device_diagnostics,
    resolve_device,
)
from arbiter_omni.encoders.base import BaseMultimodalEncoder
from arbiter_omni.encoders.clap import CLAPAudioEncoder
from arbiter_omni.encoders.mock import MockMultimodalEncoder
from arbiter_omni.encoders.openclip import OpenCLIPMultimodalEncoder
from arbiter_omni.encoders.temporal import SpatioTemporalVideoAttention
from arbiter_omni.fusion.base import BaseMultimodalFusion
from arbiter_omni.fusion.gated import GatedMultimodalFusion
from arbiter_omni.fusion.transformer import TransformerMultimodalFusion
from arbiter_omni.model.arbiter import ArbiterOmniModel
from arbiter_omni.model.decision_head import DynamicDecisionHead
from arbiter_omni.training.config import TrainingConfig
from arbiter_omni.training.trainer import ArbiterOmniTrainer
from arbiter_omni.types import DecisionResult, ModalityType, MultimodalSample

__version__ = "0.1.0"

__all__ = [
    "ArbiterOmniEngine",
    "ArbiterOmniModel",
    "ArbiterOmniTrainer",
    "BaseMultimodalEncoder",
    "BaseMultimodalFusion",
    "CLAPAudioEncoder",
    "DecisionResult",
    "DynamicDecisionHead",
    "GatedMultimodalFusion",
    "MockMultimodalEncoder",
    "ModalityType",
    "MultimodalDecisionDataset",
    "MultimodalSample",
    "OpenCLIPMultimodalEncoder",
    "SpatioTemporalVideoAttention",
    "TrainingConfig",
    "TransformerMultimodalFusion",
    "collate_multimodal_decision",
    "configure_cpu_threads",
    "create_synthetic_audio",
    "create_synthetic_image",
    "generate_synthetic_dataset",
    "get_device_telemetry",
    "print_device_diagnostics",
    "resolve_device",
]
