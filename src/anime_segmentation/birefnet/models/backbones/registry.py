"""Backbone registrations using the unified wrapper system.

This module registers all supported backbones with the BackboneRegistry,
consolidating what was previously spread across multiple files.
"""

from .base import BackboneRegistry, NHWCWrapper, StandardWrapper

# =============================================================================
# ConvNeXt v2 Variants
# =============================================================================

CONVNEXT_MODELS = {
    "convnext_atto": "convnext_atto",
    "convnext_femto": "convnext_femto",
    "convnext_pico": "convnext_pico",
    "convnext_nano": "convnext_nano",
    "convnext_tiny": "convnext_tiny",
    "convnextv2_atto": "convnextv2_atto",
    "convnextv2_femto": "convnextv2_femto",
    "convnextv2_pico": "convnextv2_pico",
    "convnextv2_nano": "convnextv2_nano",
    "convnextv2_tiny": "convnextv2_tiny",
}


def _make_convnext_factory(timm_name: str):
    """Create a factory function for a ConvNeXt variant."""

    def factory(pretrained: bool = True, **kwargs):
        return StandardWrapper(timm_name, pretrained=pretrained, **kwargs)

    return factory


for name, timm_name in CONVNEXT_MODELS.items():
    BackboneRegistry.register(name)(_make_convnext_factory(timm_name))


# =============================================================================
# Swin Transformer v1 Variants
# =============================================================================

SWIN_MODELS = {
    "swin_v1_t": "swin_tiny_patch4_window7_224",
    "swin_v1_s": "swin_small_patch4_window7_224",
    "swin_v1_b": "swin_base_patch4_window12_384",
    "swin_v1_l": "swin_large_patch4_window12_384",
}


def _make_swin_factory(timm_name: str):
    """Create a factory function for a Swin Transformer variant."""

    def factory(pretrained: bool = True, **kwargs):
        return NHWCWrapper(
            timm_name,
            pretrained=pretrained,
            strict_img_size=False,
            **kwargs,
        )

    return factory


for name, timm_name in SWIN_MODELS.items():
    BackboneRegistry.register(name)(_make_swin_factory(timm_name))


# =============================================================================
# PVT v2 Variants
# =============================================================================

PVT_MODELS = {
    "pvt_v2_b0": "pvt_v2_b0",
    "pvt_v2_b1": "pvt_v2_b1",
    "pvt_v2_b2": "pvt_v2_b2",
    "pvt_v2_b5": "pvt_v2_b5",
}


def _make_pvt_factory(timm_name: str):
    """Create a factory function for a PVT v2 variant."""

    def factory(pretrained: bool = True, **kwargs):
        return StandardWrapper(timm_name, pretrained=pretrained, **kwargs)

    return factory


for name, timm_name in PVT_MODELS.items():
    BackboneRegistry.register(name)(_make_pvt_factory(timm_name))


# =============================================================================
# DINOv3 Variants (ViT-based with custom out_indices)
# =============================================================================

DINO_MODELS = {
    "dino_v3_s": ("vit_small_patch16_dinov3.lvd1689m", (3, 5, 7, 11)),
    "dino_v3_s_plus": ("vit_small_plus_patch16_dinov3.lvd1689m", (3, 5, 7, 11)),
    "dino_v3_b": ("vit_base_patch16_dinov3.lvd1689m", (3, 5, 7, 11)),
    "dino_v3_l": ("vit_large_patch16_dinov3.lvd1689m", (5, 11, 17, 23)),
    "dino_v3_h_plus": ("vit_huge_plus_patch16_dinov3.lvd1689m", (7, 15, 23, 31)),
    "dino_v3_7b": ("vit_7b_patch16_dinov3.lvd1689m", (9, 19, 29, 39)),
}


def _make_dino_factory(timm_name: str, out_indices: tuple[int, ...]):
    """Create a factory function for a DINOv3 variant."""

    def factory(pretrained: bool = True, **kwargs):
        return StandardWrapper(
            timm_name,
            pretrained=pretrained,
            out_indices=out_indices,
            dynamic_img_size=True,
            **kwargs,
        )

    return factory


for name, (timm_name, out_indices) in DINO_MODELS.items():
    BackboneRegistry.register(name)(_make_dino_factory(timm_name, out_indices))


# =============================================================================
# CAFormer Variants (MetaFormer with SepConv + Attention)
# =============================================================================

CAFORMER_MODELS = {
    "caformer_s18": "caformer_s18",
    "caformer_s36": "caformer_s36",
    "caformer_m36": "caformer_m36",
    "caformer_b36": "caformer_b36",
}


def _make_caformer_factory(timm_name: str):
    """Create a factory function for a CAFormer variant."""

    def factory(pretrained: bool = True, **kwargs):
        return StandardWrapper(timm_name, pretrained=pretrained, **kwargs)

    return factory


for name, timm_name in CAFORMER_MODELS.items():
    BackboneRegistry.register(name)(_make_caformer_factory(timm_name))
