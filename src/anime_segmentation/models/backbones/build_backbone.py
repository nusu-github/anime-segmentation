"""Backbone factory using the unified registry system.

This module provides the main entry point for backbone construction,
using the BackboneRegistry for dynamic backbone selection.
"""

from typing import Any

from torch import nn

# Import registry to trigger all backbone registrations
from . import registry as _registry  # noqa: F401
from .base import BackboneRegistry


def build_backbone(bb_name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
    """Build a backbone by name.

    This function maintains backward compatibility with existing code
    while using the new unified registry system.

    Args:
        bb_name: Registered backbone name (e.g., "convnext_atto", "swin_v1_l")
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments passed to timm.create_model

    Returns:
        A backbone module (TimmBackboneWrapper subclass)

    Raises:
        NotImplementedError: If backbone name is not registered

    Example:
        >>> backbone = build_backbone("convnext_atto", pretrained=False)
        >>> outputs = backbone(torch.randn(1, 3, 256, 256))
        >>> len(outputs)  # 4-level feature pyramid
        4

    """
    return BackboneRegistry.build(bb_name, pretrained=pretrained, **kwargs)
