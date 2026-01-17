"""Copy-Paste synthesis pipeline for anime segmentation training.

This module provides tools for generating synthetic training data
by compositing foreground character cutouts onto background images.
"""

from anime_segmentation.training.synthesis.blending import (
    BlendingStrategy,
    BoundaryRGBRandomizer,
    FeatherBlending,
    HardPasteBlending,
)
from anime_segmentation.training.synthesis.compositor import (
    CompositorConfig,
    CopyPasteCompositor,
)
from anime_segmentation.training.synthesis.consistency import (
    ColorToneMatching,
    ConsistencyPipeline,
    LightWrap,
    NoiseGrainConsistency,
    SimpleShadow,
)
from anime_segmentation.training.synthesis.degradation import QualityDegradation
from anime_segmentation.training.synthesis.transforms import InstanceTransform
from anime_segmentation.training.synthesis.validation import (
    DataValidator,
    DistributionMonitor,
    ValidationResult,
)

__all__ = [
    # Blending
    "BlendingStrategy",
    "BoundaryRGBRandomizer",
    # Consistency
    "ColorToneMatching",
    # Compositor
    "CompositorConfig",
    "ConsistencyPipeline",
    "CopyPasteCompositor",
    # Validation
    "DataValidator",
    "DistributionMonitor",
    "FeatherBlending",
    "HardPasteBlending",
    # Transforms
    "InstanceTransform",
    "LightWrap",
    "NoiseGrainConsistency",
    # Degradation
    "QualityDegradation",
    "SimpleShadow",
    "ValidationResult",
]
