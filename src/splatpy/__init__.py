from .api import (
    video_to_splat,
    video_to_splat_advanced,
    video_to_gsplat,  # legacy
)
from .config import TrainingConfig, QualityPreset, get_quality_preset
from .trainer import Trainer

__all__ = [
    "video_to_splat",
    "video_to_splat_advanced",
    "video_to_gsplat",
    "TrainingConfig",
    "QualityPreset",
    "get_quality_preset",
    "Trainer",
]

__version__ = "0.1.0"