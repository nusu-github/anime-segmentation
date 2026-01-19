"""Compatibility shim for synthesis configuration dataclasses."""

from __future__ import annotations

from anime_segmentation.training.config import (
    CompositorConfig,
    ConsistencyConfig,
    DegradationConfig,
    SynthesisConfig,
)

__all__ = [
    "CompositorConfig",
    "ConsistencyConfig",
    "DegradationConfig",
    "SynthesisConfig",
]
