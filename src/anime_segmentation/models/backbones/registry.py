"""Backbone registrations using a unified declarative configuration system.

This module registers all supported backbones with the BackboneRegistry
using a data-driven approach that eliminates repeated factory functions.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .base import BackboneRegistry, NHWCWrapper, StandardWrapper


class WrapperType(Enum):
    """Type of wrapper to use for the backbone."""

    STANDARD = auto()
    NHWC = auto()


@dataclass(frozen=True)
class BackboneConfig:
    """Configuration for a backbone variant.

    Attributes:
        timm_name: Name of the model in timm registry.
        wrapper: Type of wrapper to use (STANDARD or NHWC).
        out_indices: Optional custom output indices for ViT-based models.
        extra_kwargs: Additional kwargs to pass to the wrapper.
    """

    timm_name: str
    wrapper: WrapperType = WrapperType.STANDARD
    out_indices: tuple[int, ...] | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def _create_backbone_factory(config: BackboneConfig):
    """Create a backbone factory function from configuration.

    Args:
        config: BackboneConfig defining the backbone.

    Returns:
        Factory function that creates the backbone instance.
    """
    wrapper_cls = NHWCWrapper if config.wrapper == WrapperType.NHWC else StandardWrapper

    def factory(pretrained: bool = True, **kwargs):
        merged_kwargs = {**config.extra_kwargs, **kwargs}
        if config.out_indices is not None:
            merged_kwargs["out_indices"] = config.out_indices
        return wrapper_cls(config.timm_name, pretrained=pretrained, **merged_kwargs)

    return factory


# =============================================================================
# Backbone Configuration Registry
# =============================================================================

BACKBONE_CONFIGS: dict[str, BackboneConfig] = {
    # -------------------------------------------------------------------------
    # ConvNeXt Variants
    # -------------------------------------------------------------------------
    "convnext_atto": BackboneConfig("convnext_atto"),
    "convnext_femto": BackboneConfig("convnext_femto"),
    "convnext_pico": BackboneConfig("convnext_pico"),
    "convnext_nano": BackboneConfig("convnext_nano"),
    "convnext_tiny": BackboneConfig("convnext_tiny"),
    "convnextv2_atto": BackboneConfig("convnextv2_atto"),
    "convnextv2_femto": BackboneConfig("convnextv2_femto"),
    "convnextv2_pico": BackboneConfig("convnextv2_pico"),
    "convnextv2_nano": BackboneConfig("convnextv2_nano"),
    "convnextv2_tiny": BackboneConfig("convnextv2_tiny"),
    # -------------------------------------------------------------------------
    # Swin Transformer v1 Variants (requires NHWC wrapper)
    # -------------------------------------------------------------------------
    "swin_v1_t": BackboneConfig(
        "swin_tiny_patch4_window7_224",
        wrapper=WrapperType.NHWC,
        extra_kwargs={"strict_img_size": False},
    ),
    "swin_v1_s": BackboneConfig(
        "swin_small_patch4_window7_224",
        wrapper=WrapperType.NHWC,
        extra_kwargs={"strict_img_size": False},
    ),
    "swin_v1_b": BackboneConfig(
        "swin_base_patch4_window12_384",
        wrapper=WrapperType.NHWC,
        extra_kwargs={"strict_img_size": False},
    ),
    "swin_v1_l": BackboneConfig(
        "swin_large_patch4_window12_384",
        wrapper=WrapperType.NHWC,
        extra_kwargs={"strict_img_size": False},
    ),
    # -------------------------------------------------------------------------
    # PVT v2 Variants
    # -------------------------------------------------------------------------
    "pvt_v2_b0": BackboneConfig("pvt_v2_b0"),
    "pvt_v2_b1": BackboneConfig("pvt_v2_b1"),
    "pvt_v2_b2": BackboneConfig("pvt_v2_b2"),
    "pvt_v2_b5": BackboneConfig("pvt_v2_b5"),
    # -------------------------------------------------------------------------
    # DINOv3 Variants (ViT-based with custom out_indices)
    # -------------------------------------------------------------------------
    "dino_v3_s": BackboneConfig(
        "vit_small_patch16_dinov3.lvd1689m",
        out_indices=(3, 5, 7, 11),
        extra_kwargs={"dynamic_img_size": True},
    ),
    "dino_v3_s_plus": BackboneConfig(
        "vit_small_plus_patch16_dinov3.lvd1689m",
        out_indices=(3, 5, 7, 11),
        extra_kwargs={"dynamic_img_size": True},
    ),
    "dino_v3_b": BackboneConfig(
        "vit_base_patch16_dinov3.lvd1689m",
        out_indices=(3, 5, 7, 11),
        extra_kwargs={"dynamic_img_size": True},
    ),
    "dino_v3_l": BackboneConfig(
        "vit_large_patch16_dinov3.lvd1689m",
        out_indices=(5, 11, 17, 23),
        extra_kwargs={"dynamic_img_size": True},
    ),
    "dino_v3_h_plus": BackboneConfig(
        "vit_huge_plus_patch16_dinov3.lvd1689m",
        out_indices=(7, 15, 23, 31),
        extra_kwargs={"dynamic_img_size": True},
    ),
    "dino_v3_7b": BackboneConfig(
        "vit_7b_patch16_dinov3.lvd1689m",
        out_indices=(9, 19, 29, 39),
        extra_kwargs={"dynamic_img_size": True},
    ),
    # -------------------------------------------------------------------------
    # CAFormer Variants (MetaFormer with SepConv + Attention)
    # -------------------------------------------------------------------------
    "caformer_s18": BackboneConfig("caformer_s18"),
    "caformer_s36": BackboneConfig("caformer_s36"),
    "caformer_m36": BackboneConfig("caformer_m36"),
    "caformer_b36": BackboneConfig("caformer_b36"),
}

# =============================================================================
# Register all backbones
# =============================================================================

for name, config in BACKBONE_CONFIGS.items():
    BackboneRegistry.register(name)(_create_backbone_factory(config))
