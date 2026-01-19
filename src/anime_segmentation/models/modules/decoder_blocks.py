"""Decoder blocks with unified base class and registry pattern.

This module provides decoder block implementations for BiRefNet:
- BaseDecoderBlock: Abstract base with shared initialization logic
- BasicDecBlk: Simple feed-forward decoder block
- ResBlk: Decoder block with residual/skip connection
- DecoderBlockRegistry: Registry for extensible block types
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from timm.layers import ConvNormAct
from torch import nn

from .aspp import ASPP, ASPPDeformable
from .norms import adaptive_group_norm_act


class DecoderBlockRegistry:
    """Registry for decoder block types.

    Allows extensible block type registration without modifying
    the core BiRefNet code.
    """

    _registry: ClassVar[dict[str, type["BaseDecoderBlock"]]] = {}

    @classmethod
    def register(cls, name: str):
        """Register a decoder block class with the given name."""

        def decorator(block_cls: type["BaseDecoderBlock"]):
            cls._registry[name] = block_cls
            return block_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type["BaseDecoderBlock"]:
        """Get a decoder block class by name.

        Args:
            name: Name of the registered block type.

        Returns:
            The registered block class.

        Raises:
            ValueError: If the block type is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown decoder block type '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        """Return list of available block type names."""
        return sorted(cls._registry.keys())


def _build_attention(
    attention_type: str | None,
    inter_channels: int,
    use_norm: bool,
    act_layer: type[nn.Module],
    act_kwargs: dict | None,
) -> nn.Module | None:
    """Build attention module based on type.

    Args:
        attention_type: Type of attention ("ASPP", "ASPPDeformable", or None).
        inter_channels: Number of intermediate channels.
        use_norm: Whether to use normalization.
        act_layer: Activation layer class.
        act_kwargs: Activation layer kwargs.

    Returns:
        Attention module or None if attention_type is None.
    """
    match attention_type:
        case "ASPP":
            return ASPP(
                in_channels=inter_channels,
                use_norm=use_norm,
                act_layer=act_layer,
                act_kwargs=act_kwargs,
            )
        case "ASPPDeformable":
            return ASPPDeformable(
                in_channels=inter_channels,
                use_norm=use_norm,
                act_layer=act_layer,
                act_kwargs=act_kwargs,
            )
        case _:
            return None


class BaseDecoderBlock(nn.Module, ABC):
    """Abstract base class for decoder blocks with shared initialization.

    Provides common setup for:
    - Input convolution (conv_in)
    - Attention module (dec_att)
    - Output convolution (conv_out)

    Subclasses implement the forward method to define data flow.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        inter_channels: int = 64,
        dec_channels_inter: str = "fixed",
        attention_type: str | None = None,
        use_norm: bool = True,
        act_layer: type[nn.Module] = nn.ReLU,
        act_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        # Compute intermediate channels
        self.inter_channels = in_channels // 4 if dec_channels_inter == "adap" else inter_channels

        # Input convolution
        self.conv_in = ConvNormAct(
            in_channels,
            self.inter_channels,
            3,
            padding=1,
            norm_layer=adaptive_group_norm_act,
            apply_norm=use_norm,
            act_layer=act_layer,
            act_kwargs=act_kwargs,
        )

        # Attention module (optional)
        self.dec_att = _build_attention(
            attention_type,
            self.inter_channels,
            use_norm,
            act_layer,
            act_kwargs,
        )

        # Output convolution
        self.conv_out = ConvNormAct(
            self.inter_channels,
            out_channels,
            3,
            padding=1,
            norm_layer=adaptive_group_norm_act,
            apply_norm=use_norm,
            apply_act=False,
            act_layer=act_layer,
            act_kwargs=act_kwargs,
        )

    def _apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention if configured."""
        if self.dec_att is not None:
            return self.dec_att(x)
        return x

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - implemented by subclasses."""
        ...


@DecoderBlockRegistry.register("BasicDecBlk")
class BasicDecBlk(BaseDecoderBlock):
    """Basic decoder block with optional ASPP attention.

    Simple feed-forward block: conv_in -> attention -> conv_out
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self._apply_attention(x)
        return self.conv_out(x)


@DecoderBlockRegistry.register("ResBlk")
class ResBlk(BaseDecoderBlock):
    """Residual decoder block with skip connection and optional ASPP attention.

    Adds a residual connection: output = conv_out(attention(conv_in(x))) + conv_resi(x)
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int | None = None,
        inter_channels: int = 64,
        dec_channels_inter: str = "fixed",
        attention_type: str | None = None,
        use_norm: bool = True,
        act_layer: type[nn.Module] = nn.ReLU,
        act_kwargs: dict | None = None,
    ) -> None:
        if out_channels is None:
            out_channels = in_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            inter_channels=inter_channels,
            dec_channels_inter=dec_channels_inter,
            attention_type=attention_type,
            use_norm=use_norm,
            act_layer=act_layer,
            act_kwargs=act_kwargs,
        )

        # Residual projection
        self.conv_resi = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv_resi(x)
        x = self.conv_in(x)
        x = self._apply_attention(x)
        x = self.conv_out(x)
        return x + residual
