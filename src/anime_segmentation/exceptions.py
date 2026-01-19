"""Typed exceptions for anime-segmentation.

This module provides a hierarchy of exceptions that enable:
1. Specific error handling at call sites
2. Clear error messages for debugging
3. Type-safe exception catching
"""


class AnimeSegmentationError(Exception):
    """Base exception for all anime-segmentation errors."""


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(AnimeSegmentationError):
    """Base for model-related errors."""


class ModelNotInitializedError(ModelError):
    """Raised when model is accessed before initialization."""


class BackboneNotFoundError(ModelError):
    """Raised when specified backbone is not available in the registry."""


class InvalidBackboneOutputError(ModelError):
    """Raised when backbone returns unexpected output format."""


class DecoderBlockNotFoundError(ModelError):
    """Raised when specified decoder block type is not available."""


class InvalidForwardInputError(ModelError):
    """Raised when forward pass receives invalid input dimensions or types."""


# =============================================================================
# Data Errors
# =============================================================================


class DataError(AnimeSegmentationError):
    """Base for data-related errors."""


class DatasetNotFoundError(DataError):
    """Raised when dataset path does not exist."""


class InvalidImageError(DataError):
    """Raised when image file is corrupt or unreadable."""


class MaskDimensionMismatchError(DataError):
    """Raised when mask dimensions don't match image dimensions."""


class EmptyPoolError(DataError):
    """Raised when foreground/background pool is empty or exhausted."""


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(AnimeSegmentationError):
    """Base for configuration-related errors."""


class InvalidLossWeightsError(ConfigurationError):
    """Raised when loss weights configuration is invalid."""


class InvalidCompositorConfigError(ConfigurationError):
    """Raised when compositor configuration is invalid."""


class InvalidActivationError(ConfigurationError):
    """Raised when specified activation function is not available."""


# =============================================================================
# Synthesis Errors
# =============================================================================


class SynthesisError(AnimeSegmentationError):
    """Base for synthesis pipeline errors."""


class SynthesisValidationError(SynthesisError):
    """Raised when synthesized data fails validation."""


class CompositorError(SynthesisError):
    """Raised when compositor fails to generate valid output."""


# =============================================================================
# Inference Errors
# =============================================================================


class InferenceError(AnimeSegmentationError):
    """Base for inference-related errors."""


class InvalidInputError(InferenceError):
    """Raised when predictor receives invalid input."""


class InvalidTargetSizeError(InferenceError):
    """Raised when target size for inference is invalid."""
