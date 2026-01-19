"""Centralized configuration classes for training.

This module uses Pydantic v2 BaseModel for type-safe configuration with
declarative validation, leveraging jsonargparse's native Pydantic support.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Type aliases for common constraints
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class DataLoaderConfig(BaseModel):
    """Configuration for data loaders.

    Attributes:
        batch_size: Batch size for training and evaluation.
        num_workers: Number of worker processes for DataLoader.
        pin_memory: Whether to pin memory.
        prefetch_factor: Optional prefetch factor for workers.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt = 8
    num_workers: NonNegativeInt = 4
    pin_memory: bool = True
    prefetch_factor: PositiveInt | None = None


class AugmentationConfig(BaseModel):
    """GPU augmentation configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    hflip_prob: Probability = 0.5
    rotation_degrees: NonNegativeFloat = 10.0
    color_jitter: bool = True
    brightness: NonNegativeFloat = 0.2
    contrast: NonNegativeFloat = 0.2
    saturation: NonNegativeFloat = 0.2
    hue: Annotated[float, Field(ge=0.0, le=0.5)] = 0.1


class CompositorConfig(BaseModel):
    """Copy-Paste compositor configuration."""

    model_config = ConfigDict(extra="forbid")

    k_probs: dict[int, float] = Field(default_factory=lambda: {0: 0.1, 1: 0.9})
    max_instances: NonNegativeInt = 1
    min_area_ratio: Annotated[float, Field(gt=0.0, le=1.0)] = 0.02
    max_area_ratio: Annotated[float, Field(gt=0.0, le=1.0)] = 0.60
    max_total_coverage: Annotated[float, Field(gt=0.0, le=1.0)] = 0.85
    max_iou_overlap: Probability = 0.30
    blending_probs: dict[str, float] = Field(
        default_factory=lambda: {"hard": 0.35, "feather": 0.55, "seamless": 0.10},
    )
    boundary_randomize_prob: Probability = 0.3
    boundary_randomize_width: PositiveInt = 3
    boundary_randomize_noise_std: NonNegativeFloat = 0.05

    @model_validator(mode="after")
    def validate_constraints(self) -> CompositorConfig:
        """Validate probability distributions and area ratio ordering."""
        # Validate probability distributions sum to 1.0
        for name, probs in [
            ("k_probs", self.k_probs),
            ("blending_probs", self.blending_probs),
        ]:
            total = sum(probs.values())
            if abs(total - 1.0) > 1e-6:
                msg = f"{name} must sum to 1.0, got {total}"
                raise ValueError(msg)

        # Validate area ratio ordering
        if not self.min_area_ratio < self.max_area_ratio:
            msg = (
                f"min_area_ratio ({self.min_area_ratio}) "
                f"must be < max_area_ratio ({self.max_area_ratio})"
            )
            raise ValueError(msg)

        return self


class ConsistencyConfig(BaseModel):
    """Consistency processing configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    color_prob: Probability = 0.5
    light_wrap_prob: Probability = 0.3
    shadow_prob: Probability = 0.3
    noise_prob: Probability = 0.3


class DegradationConfig(BaseModel):
    """Degradation pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    jpeg_prob: Probability = 0.3
    blur_prob: Probability = 0.1
    noise_prob: Probability = 0.1


class SynthesisConfig(BaseModel):
    """Top-level synthesis pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    ratio: Probability = 0.5
    length: PositiveInt = 1000
    compositor: CompositorConfig = Field(default_factory=CompositorConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    strict_validation: bool = False


class BackboneConfig(BaseModel):
    """Backbone encoder configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "swin_v1_t"
    pretrained: bool = True


class DecoderConfig(BaseModel):
    """Decoder architecture configuration."""

    model_config = ConfigDict(extra="forbid")

    out_ref: bool = True
    ms_supervision: bool = True
    dec_ipt: bool = True
    dec_ipt_split: bool = True
    cxt_num: PositiveInt = 3
    mul_scl_ipt: Literal["", "add", "cat"] = "cat"
    dec_att: Literal["", "ASPP", "ASPPDeformable"] = "ASPPDeformable"
    squeeze_block: str = "BasicDecBlk_x1"
    dec_blk: Literal["BasicDecBlk", "ResBlk"] = "BasicDecBlk"
    lat_blk: Literal["BasicLatBlk"] = "BasicLatBlk"
    dec_channels_inter: Literal["fixed", "adap"] = "fixed"
    use_norm: bool = True


class LossConfig(BaseModel):
    """Pixel loss weighting configuration."""

    model_config = ConfigDict(extra="forbid")

    bce: NonNegativeFloat = 30.0
    iou: NonNegativeFloat = 0.5
    ssim: NonNegativeFloat = 10.0
    mae: NonNegativeFloat = 0.0
    mse: NonNegativeFloat = 0.0
    reg: NonNegativeFloat = 0.0
    iou_patch: NonNegativeFloat = 0.0
    cnt: NonNegativeFloat = 0.0
    structure: NonNegativeFloat = 0.0


class ClassificationLossConfig(BaseModel):
    """Classification loss weighting configuration."""

    model_config = ConfigDict(extra="forbid")

    ce: NonNegativeFloat = 5.0
