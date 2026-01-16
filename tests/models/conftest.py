"""Shared fixtures for BiRefNet tests."""

import pytest
import torch


@pytest.fixture
def sample_input():
    """Create a standard test input tensor (256x256)."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_input_small():
    """Smaller input for faster tests (64x64)."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def sample_input_training():
    """Input for training mode tests (128x128).

    Training mode requires larger inputs due to BatchNorm constraints
    when feature maps become too small.
    """
    return torch.randn(2, 3, 128, 128)  # batch_size=2 for BatchNorm


# Expected output channels for each backbone (shallowest to deepest: x1, x2, x3, x4)
BACKBONE_CHANNELS = {
    # ConvNeXt v2
    "convnext_atto": [40, 80, 160, 320],
    "convnext_femto": [48, 96, 192, 384],
    "convnext_pico": [64, 128, 256, 512],
    "convnext_nano": [80, 160, 320, 640],
    "convnext_tiny": [96, 192, 384, 768],
    # Swin Transformer v1
    "swin_v1_t": [96, 192, 384, 768],
    "swin_v1_s": [96, 192, 384, 768],
    "swin_v1_b": [128, 256, 512, 1024],
    "swin_v1_l": [192, 384, 768, 1536],
    # PVT v2
    "pvt_v2_b0": [32, 64, 160, 256],
    "pvt_v2_b1": [64, 128, 320, 512],
    "pvt_v2_b2": [64, 128, 320, 512],
    "pvt_v2_b5": [64, 128, 320, 512],
    # DINOv3 (uniform channels across all levels)
    "dino_v3_s": [384, 384, 384, 384],
    "dino_v3_s_plus": [384, 384, 384, 384],
    "dino_v3_b": [768, 768, 768, 768],
    "dino_v3_l": [1024, 1024, 1024, 1024],
    "dino_v3_h_plus": [1280, 1280, 1280, 1280],
    "dino_v3_7b": [4096, 4096, 4096, 4096],
    # CAFormer (MetaFormer with SepConv + Attention)
    "caformer_s18": [64, 128, 320, 512],
    "caformer_s36": [64, 128, 320, 512],
    "caformer_m36": [96, 192, 384, 576],
    "caformer_b36": [128, 256, 512, 768],
}

# Lateral channels used in BiRefNet (deepest to shallowest: x4, x3, x2, x1)
LATERAL_CHANNELS = {bb_name: channels[::-1] for bb_name, channels in BACKBONE_CHANNELS.items()}


@pytest.fixture
def backbone_channels():
    """Expected channels for each backbone (shallowest to deepest)."""
    return BACKBONE_CHANNELS


@pytest.fixture
def lateral_channels():
    """Expected lateral channels for BiRefNet (deepest to shallowest)."""
    return LATERAL_CHANNELS


# Small backbones for quick tests (avoid large models in CI)
QUICK_TEST_BACKBONES = [
    "convnext_atto",
    "swin_v1_t",
    "pvt_v2_b0",
    "caformer_s18",
]

# All registered backbones
ALL_BACKBONES = list(BACKBONE_CHANNELS.keys())
