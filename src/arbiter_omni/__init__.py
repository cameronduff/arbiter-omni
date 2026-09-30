"""
ArbiterOmni: Multimodal System 1 Decision Engine inspired by Jev.
"""

from arbiter_omni.api.engine import ArbiterOmniEngine
from arbiter_omni.calibration.conformal import (
    ConformalCalibrator,
    System2EscalationGate,
)
from arbiter_omni.data.cached import (
    CachedMultimodalDataset,
    collate_cached_multimodal_decision,
)
from arbiter_omni.data.dataset import (
    MultimodalDecisionDataset,
    collate_multimodal_decision,
)
from arbiter_omni.data.memory_bank import PersistentMemoryBank

from arbiter_omni.data.miner import (
    DEFAULT_CANDIDATE_POOL,
    HardNegativeMiner,
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
from arbiter_omni.model.decision_head import (
    DynamicDecisionHead,
    contrastive_margin_loss,
)
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
    "CachedMultimodalDataset",
    "ConformalCalibrator",
    "DEFAULT_CANDIDATE_POOL",
    "DecisionResult",
    "DynamicDecisionHead",
    "GatedMultimodalFusion",
    "HardNegativeMiner",
    "MockMultimodalEncoder",
    "ModalityType",
    "MultimodalDecisionDataset",
    "MultimodalSample",
    "OpenCLIPMultimodalEncoder",
    "PersistentMemoryBank",
    "SpatioTemporalVideoAttention",
    "System2EscalationGate",

    "TrainingConfig",
    "TransformerMultimodalFusion",
    "collate_cached_multimodal_decision",
    "collate_multimodal_decision",
    "configure_cpu_threads",
    "contrastive_margin_loss",

    "create_synthetic_audio",
    "create_synthetic_image",
    "generate_synthetic_dataset",
    "get_device_telemetry",
    "print_device_diagnostics",
    "resolve_device",
]
