"""
ArbiterOmni data loading and dataset generation package.
"""

from arbiter_omni.data.ai2d import (
    AI2DAdapter,
    load_ai2d_dataset,
)
from arbiter_omni.data.dataset import (
    MultimodalDecisionDataset,
    collate_multimodal_decision,
)
from arbiter_omni.data.gqa import (
    GQAAdapter,
    load_gqa_dataset,
)
from arbiter_omni.data.memory_bank import PersistentMemoryBank
from arbiter_omni.data.robotics import (
    RoboticsActionAdapter,
    generate_robotics_samples,
)
from arbiter_omni.data.scienceqa import (
    ScienceQAAdapter,
    create_mock_scienceqa_samples,
    load_scienceqa_dataset,
)
from arbiter_omni.data.seedbench import (
    SEEDBenchAdapter,
    create_mock_seedbench_samples,
    load_seedbench_dataset,
)
from arbiter_omni.data.synthetic import (
    create_synthetic_audio,
    create_synthetic_image,
    generate_synthetic_dataset,
)

__all__ = [
    "AI2DAdapter",
    "GQAAdapter",
    "MultimodalDecisionDataset",
    "PersistentMemoryBank",
    "RoboticsActionAdapter",
    "SEEDBenchAdapter",
    "ScienceQAAdapter",
    "collate_multimodal_decision",
    "create_mock_scienceqa_samples",
    "create_mock_seedbench_samples",
    "create_synthetic_audio",
    "create_synthetic_image",
    "generate_robotics_samples",
    "generate_synthetic_dataset",
    "load_ai2d_dataset",
    "load_gqa_dataset",
    "load_scienceqa_dataset",
    "load_seedbench_dataset",
]
