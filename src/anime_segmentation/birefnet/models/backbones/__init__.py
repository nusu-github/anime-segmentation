"""Backbone wrappers for BiRefNet.

This module provides a unified interface for timm-based backbones
with a registry pattern for dynamic backbone selection.
"""

from .base import BackboneRegistry, NHWCWrapper, StandardWrapper, TimmBackboneWrapper
from .build_backbone import build_backbone

__all__ = [
    "BackboneRegistry",
    "NHWCWrapper",
    "StandardWrapper",
    "TimmBackboneWrapper",
    "build_backbone",
]
