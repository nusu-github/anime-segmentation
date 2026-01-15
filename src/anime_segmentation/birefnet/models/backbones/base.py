"""Unified backbone wrapper and registry system.

This module provides a base class for wrapping timm backbones and a registry
pattern for backbone registration, eliminating code duplication across
individual backbone files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import timm
import torch
from einops import rearrange
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


class BackboneRegistry:
    """Registry for backbone factory functions.

    This class maintains a mapping of backbone names to factory functions,
    allowing dynamic backbone selection without hardcoded imports.
    """

    _registry: ClassVar[dict[str, Callable[..., nn.Module]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        """Decorator to register a backbone factory function.

        Args:
            name: The name to register the backbone under (e.g., "convnext_atto")

        Returns:
            Decorator function that registers the factory

        Example:
            @BackboneRegistry.register("my_backbone")
            def my_backbone(pretrained=True, **kwargs):
                return MyBackboneWrapper(...)

        """

        def decorator(factory_fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            cls._registry[name] = factory_fn
            return factory_fn

        return decorator

    @classmethod
    def build(cls, name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        """Build a backbone by registered name.

        Args:
            name: Registered backbone name
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments passed to the factory

        Returns:
            Instantiated backbone module

        Raises:
            NotImplementedError: If backbone name is not registered

        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            msg = f"Backbone {name} is not supported. Available: {available}"
            raise NotImplementedError(msg)
        return cls._registry[name](pretrained=pretrained, **kwargs)

    @classmethod
    def list_backbones(cls) -> list[str]:
        """Return list of all registered backbone names."""
        return sorted(cls._registry.keys())


class TimmBackboneWrapper(nn.Module):
    """Base wrapper for timm backbones with features_only output.

    This class provides a common interface for all timm-based backbones,
    extracting multi-scale feature pyramids.

    Subclasses can override `output_transform` for output format conversions
    (e.g., Swin's NHWC -> NCHW).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        **kwargs: Any,
    ) -> None:
        """Initialize the backbone wrapper.

        Args:
            model_name: timm model name (e.g., "convnextv2_atto")
            pretrained: Whether to load pretrained weights
            out_indices: Feature level indices to extract
            **kwargs: Additional arguments passed to timm.create_model

        """
        super().__init__()
        self.model = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=out_indices,
            **kwargs,
        )
        self._out_channels: list[int] | None = None

    @property
    def out_channels(self) -> list[int]:
        """Return output channels for each feature level.

        Lazily computed from timm's feature_info.
        """
        if self._out_channels is None:
            self._out_channels = [info["num_chs"] for info in self.model.feature_info]
        return self._out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass with optional output transformation.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of feature tensors at each pyramid level

        """
        outs = self.model(x)
        return self.output_transform(outs)

    def output_transform(self, outs: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Transform outputs. Override in subclasses for format conversion.

        Default implementation converts list to tuple without shape changes.

        Args:
            outs: List of feature tensors from timm model

        Returns:
            Tuple of feature tensors

        """
        return tuple(outs)


class StandardWrapper(TimmBackboneWrapper):
    """Standard wrapper for backbones with NCHW output (ConvNeXt, PVT, DINOv3)."""


class NHWCWrapper(TimmBackboneWrapper):
    """Wrapper for backbones that output NHWC format (Swin Transformer).

    Converts NHWC to NCHW format for compatibility with the decoder.
    """

    def output_transform(self, outs: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Convert NHWC to NCHW format.

        Args:
            outs: List of feature tensors in NHWC format

        Returns:
            Tuple of feature tensors in NCHW format

        """
        return tuple(rearrange(o, "n h w c -> n c h w") for o in outs)
