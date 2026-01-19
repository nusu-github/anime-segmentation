"""Centralized configuration dataclasses for synthesis pipeline.

This module provides a single source of truth for all synthesis-related
configuration values, eliminating duplication across datamodule.py,
compositor.py, and YAML config files.
"""

from dataclasses import dataclass, field


@dataclass
class CompositorConfig:
    """Configuration for CopyPasteCompositor.

    Attributes:
        k_probs: Probability distribution for number of characters (k).
        min_area_ratio: Minimum area ratio for a character (vs canvas).
        max_area_ratio: Maximum area ratio for a single character.
        max_total_coverage: Maximum total coverage of all characters.
        max_iou_overlap: Maximum IoU overlap allowed between characters.
        blending_probs: Probability distribution for blending strategies.
        boundary_randomize_prob: Probability of boundary RGB randomization.
        boundary_randomize_width: Boundary width in pixels.
        boundary_randomize_noise_std: Noise std for boundary randomization.
    """

    k_probs: dict[int, float] = field(
        default_factory=lambda: {
            0: 0.05,  # Negative examples (background only)
            1: 0.35,
            2: 0.35,
            3: 0.20,
            4: 0.05,
        },
    )

    min_area_ratio: float = 0.02
    max_area_ratio: float = 0.60
    max_total_coverage: float = 0.85

    max_iou_overlap: float = 0.30

    blending_probs: dict[str, float] = field(
        default_factory=lambda: {
            "hard": 0.35,
            "feather": 0.55,
            "seamless": 0.10,
        },
    )

    boundary_randomize_prob: float = 0.3
    boundary_randomize_width: int = 3
    boundary_randomize_noise_std: float = 0.05

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate k_probs sum to 1
        prob_sum = sum(self.k_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            msg = f"k_probs must sum to 1.0, got {prob_sum}"
            raise ValueError(msg)

        # Validate blending_probs sum to 1
        blend_sum = sum(self.blending_probs.values())
        if abs(blend_sum - 1.0) > 1e-6:
            msg = f"blending_probs must sum to 1.0, got {blend_sum}"
            raise ValueError(msg)

        # Validate ranges
        if not 0 < self.min_area_ratio < self.max_area_ratio <= 1:
            msg = "Invalid area ratio range"
            raise ValueError(msg)
        if not 0 < self.max_total_coverage <= 1:
            msg = "Invalid max_total_coverage"
            raise ValueError(msg)
        if not 0 <= self.max_iou_overlap <= 1:
            msg = "Invalid max_iou_overlap"
            raise ValueError(msg)
        if not 0.0 <= self.boundary_randomize_prob <= 1.0:
            msg = "Invalid boundary_randomize_prob"
            raise ValueError(msg)
        if self.boundary_randomize_width < 1:
            msg = "boundary_randomize_width must be >= 1"
            raise ValueError(msg)
        if self.boundary_randomize_noise_std < 0:
            msg = "boundary_randomize_noise_std must be >= 0"
            raise ValueError(msg)


@dataclass
class ConsistencyConfig:
    """Configuration for consistency processing pipeline.

    Attributes:
        enabled: Whether consistency processing is enabled.
        color_prob: Probability of applying color matching.
        light_wrap_prob: Probability of applying light wrap effect.
        shadow_prob: Probability of applying shadow rendering.
        noise_prob: Probability of applying noise matching.
    """

    enabled: bool = True
    color_prob: float = 0.5
    light_wrap_prob: float = 0.3
    shadow_prob: float = 0.3
    noise_prob: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        for name in ("color_prob", "light_wrap_prob", "shadow_prob", "noise_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1], got {value}"
                raise ValueError(msg)


@dataclass
class DegradationConfig:
    """Configuration for quality degradation pipeline.

    Attributes:
        enabled: Whether degradation is enabled.
        jpeg_prob: Probability of JPEG compression artifacts.
        blur_prob: Probability of Gaussian blur.
        noise_prob: Probability of noise addition.
    """

    enabled: bool = True
    jpeg_prob: float = 0.3
    blur_prob: float = 0.1
    noise_prob: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration values."""
        for name in ("jpeg_prob", "blur_prob", "noise_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1], got {value}"
                raise ValueError(msg)


@dataclass
class SynthesisConfig:
    """Top-level configuration for the entire synthesis pipeline.

    Combines all synthesis-related settings into a single configuration object.

    Attributes:
        enabled: Whether synthesis is enabled.
        ratio: Ratio of synthetic samples in mixed dataset (0.0 to 1.0).
        length: Number of synthetic samples per epoch.
        compositor: Compositor configuration.
        consistency: Consistency processing configuration.
        degradation: Degradation pipeline configuration.
        strict_validation: If True, raise on validation failure instead of warning.
    """

    enabled: bool = False
    ratio: float = 0.5
    length: int = 1000

    compositor: CompositorConfig = field(default_factory=CompositorConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    strict_validation: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.ratio <= 1.0:
            msg = f"synthesis ratio must be in [0, 1], got {self.ratio}"
            raise ValueError(msg)
        if self.length < 1:
            msg = f"synthesis length must be >= 1, got {self.length}"
            raise ValueError(msg)
