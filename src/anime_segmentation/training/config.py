"""Centralized configuration dataclasses for training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataLoaderConfig:
    """Configuration for data loaders.

    Attributes:
        batch_size: Batch size for training and evaluation.
        num_workers: Number of worker processes for DataLoader.
        pin_memory: Whether to pin memory.
        prefetch_factor: Optional prefetch factor for workers.
    """

    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int | None = None


@dataclass
class AugmentationConfig:
    """GPU augmentation configuration."""

    enabled: bool = True
    hflip_prob: float = 0.5
    rotation_degrees: float = 10.0
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1


@dataclass
class CompositorConfig:
    """Copy-Paste compositor configuration."""

    k_probs: dict[int, float] = field(
        default_factory=lambda: {0: 0.05, 1: 0.35, 2: 0.35, 3: 0.20, 4: 0.05},
    )
    min_area_ratio: float = 0.02
    max_area_ratio: float = 0.60
    max_total_coverage: float = 0.85
    max_iou_overlap: float = 0.30
    blending_probs: dict[str, float] = field(
        default_factory=lambda: {"hard": 0.35, "feather": 0.55, "seamless": 0.10},
    )
    boundary_randomize_prob: float = 0.3
    boundary_randomize_width: int = 3
    boundary_randomize_noise_std: float = 0.05

    def __post_init__(self) -> None:
        prob_sum = sum(self.k_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            msg = f"k_probs must sum to 1.0, got {prob_sum}"
            raise ValueError(msg)

        blend_sum = sum(self.blending_probs.values())
        if abs(blend_sum - 1.0) > 1e-6:
            msg = f"blending_probs must sum to 1.0, got {blend_sum}"
            raise ValueError(msg)

        if not 0 < self.min_area_ratio < self.max_area_ratio <= 1:
            msg = "Invalid area ratio range"
            raise ValueError(msg)
        if not 0 < self.max_total_coverage <= 1:
            msg = "Invalid max_total_coverage"
            raise ValueError(msg)
        if not 0 <= self.max_iou_overlap <= 1:
            msg = "Invalid max_iou_overlap"
            raise ValueError(msg)
        if not 0.0 <= self.boundary_randomize_prob <= 1.0:
            msg = "Invalid boundary_randomize_prob"
            raise ValueError(msg)
        if self.boundary_randomize_width < 1:
            msg = "boundary_randomize_width must be >= 1"
            raise ValueError(msg)
        if self.boundary_randomize_noise_std < 0:
            msg = "boundary_randomize_noise_std must be >= 0"
            raise ValueError(msg)


@dataclass
class ConsistencyConfig:
    """Consistency processing configuration."""

    enabled: bool = True
    color_prob: float = 0.5
    light_wrap_prob: float = 0.3
    shadow_prob: float = 0.3
    noise_prob: float = 0.3

    def __post_init__(self) -> None:
        for name in ("color_prob", "light_wrap_prob", "shadow_prob", "noise_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1], got {value}"
                raise ValueError(msg)


@dataclass
class DegradationConfig:
    """Degradation pipeline configuration."""

    enabled: bool = True
    jpeg_prob: float = 0.3
    blur_prob: float = 0.1
    noise_prob: float = 0.1

    def __post_init__(self) -> None:
        for name in ("jpeg_prob", "blur_prob", "noise_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1], got {value}"
                raise ValueError(msg)


@dataclass
class SynthesisConfig:
    """Top-level synthesis pipeline configuration."""

    enabled: bool = False
    ratio: float = 0.5
    length: int = 1000
    compositor: CompositorConfig = field(default_factory=CompositorConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    strict_validation: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            msg = f"synthesis ratio must be in [0, 1], got {self.ratio}"
            raise ValueError(msg)
        if self.length < 1:
            msg = f"synthesis length must be >= 1, got {self.length}"
            raise ValueError(msg)


@dataclass
class BackboneConfig:
    """Backbone encoder configuration."""

    name: str = "swin_v1_t"
    pretrained: bool = True


@dataclass
class DecoderConfig:
    """Decoder architecture configuration."""

    out_ref: bool = True
    ms_supervision: bool = True
    dec_ipt: bool = True
    dec_ipt_split: bool = True
    cxt_num: int = 3
    mul_scl_ipt: Literal["", "add", "cat"] = "cat"
    dec_att: Literal["", "ASPP", "ASPPDeformable"] = "ASPPDeformable"
    squeeze_block: str = "BasicDecBlk_x1"
    dec_blk: Literal["BasicDecBlk", "ResBlk"] = "BasicDecBlk"
    lat_blk: Literal["BasicLatBlk"] = "BasicLatBlk"
    dec_channels_inter: Literal["fixed", "adap"] = "fixed"
    use_norm: bool = True


@dataclass
class LossConfig:
    """Pixel loss weighting configuration."""

    bce: float = 30.0
    iou: float = 0.5
    ssim: float = 10.0
    mae: float = 0.0
    mse: float = 0.0
    reg: float = 0.0
    iou_patch: float = 0.0
    cnt: float = 0.0
    structure: float = 0.0


@dataclass
class ClassificationLossConfig:
    """Classification loss weighting configuration."""

    ce: float = 5.0
