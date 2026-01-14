"""BiRefNet model configuration with backbone channel specifications.

This module externalizes the hardcoded channel maps from BiRefNet,
enabling channel auto-inference from backbones.
"""

from .backbones.base import TimmBackboneWrapper

# Lateral channels map: backbone name -> [x4, x3, x2, x1] (deepest to shallowest)
# This was previously hardcoded in BiRefNet.__init__
LATERAL_CHANNELS_MAP: dict[str, list[int]] = {
    # ConvNeXt v2
    "convnext_atto": [320, 160, 80, 40],
    "convnext_femto": [384, 192, 96, 48],
    "convnext_pico": [512, 256, 128, 64],
    "convnext_nano": [640, 320, 160, 80],
    "convnext_tiny": [768, 384, 192, 96],
    # DINOv3 (uniform channels)
    "dino_v3_7b": [4096, 4096, 4096, 4096],
    "dino_v3_h_plus": [1280, 1280, 1280, 1280],
    "dino_v3_l": [1024, 1024, 1024, 1024],
    "dino_v3_b": [768, 768, 768, 768],
    "dino_v3_s_plus": [384, 384, 384, 384],
    "dino_v3_s": [384, 384, 384, 384],
    # Swin Transformer v1
    "swin_v1_l": [1536, 768, 384, 192],
    "swin_v1_b": [1024, 512, 256, 128],
    "swin_v1_s": [768, 384, 192, 96],
    "swin_v1_t": [768, 384, 192, 96],
    # PVT v2
    "pvt_v2_b5": [512, 320, 128, 64],
    "pvt_v2_b2": [512, 320, 128, 64],
    "pvt_v2_b1": [512, 320, 128, 64],
    "pvt_v2_b0": [256, 160, 64, 32],
}


def get_lateral_channels(
    bb_name: str,
    backbone: TimmBackboneWrapper | None = None,
) -> list[int]:
    """Get lateral channels for a backbone.

    Channels are ordered from deepest to shallowest: [x4, x3, x2, x1].

    Priority:
    1. Auto-infer from backbone.out_channels (if backbone provided)
    2. Fallback to LATERAL_CHANNELS_MAP

    Args:
        bb_name: Backbone name
        backbone: Optional backbone instance for channel inference

    Returns:
        List of channel counts [x4, x3, x2, x1]

    Raises:
        ValueError: If backbone not found and no inference available

    """
    # Try to infer from backbone
    if backbone is not None and hasattr(backbone, "out_channels"):
        # out_channels is [x1, x2, x3, x4], reverse for lateral channels
        return backbone.out_channels[::-1]

    # Fallback to known map
    if bb_name in LATERAL_CHANNELS_MAP:
        return LATERAL_CHANNELS_MAP[bb_name]

    msg = (
        f"Backbone '{bb_name}' not found in channel map. "
        "Either provide a backbone instance with out_channels property "
        "or add the backbone to LATERAL_CHANNELS_MAP."
    )
    raise ValueError(msg)


def infer_channels_from_timm(model_name: str) -> list[int]:
    """Utility to infer channels from any timm model.

    Useful for adding new backbones without hardcoding.

    Args:
        model_name: timm model name (e.g., "efficientnet_b0")

    Returns:
        List of output channels [x1, x2, x3, x4] (shallowest to deepest)

    Example:
        >>> channels = infer_channels_from_timm("convnextv2_atto")
        >>> print(channels)
        [40, 80, 160, 320]

    """
    import timm

    model = timm.create_model(
        model_name,
        features_only=True,
        pretrained=False,
    )
    return [info["num_chs"] for info in model.feature_info]
