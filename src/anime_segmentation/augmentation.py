"""GPU-accelerated data augmentation pipeline using Kornia.

This module provides a configurable augmentation pipeline that leverages Kornia
for GPU-accelerated image transformations. The pipeline separates geometric
transforms (applied to both image and mask) from intensity transforms (applied
only to image) to ensure mask integrity.

Typical usage:
    pipeline = KorniaAugmentationPipeline(config)
    augmented_img, augmented_mask = pipeline(img_batch, mask_batch)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kornia.augmentation as K
import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from anime_segmentation.data_loader import AugmentationConfig


class KorniaAugmentationPipeline(nn.Module):
    """GPU-accelerated augmentation pipeline using Kornia.

    This pipeline handles both geometric transforms (applied to both image and mask)
    and intensity transforms (applied only to image).

    Args:
        config: AugmentationConfig specifying which augmentations to apply.
        normalize_mean: RGB normalization mean. Defaults to ImageNet mean.
        normalize_std: RGB normalization std. Defaults to ImageNet std.
    """

    def __init__(
        self,
        config: AugmentationConfig,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.geometric_aug = self._build_geometric_transforms()
        self.intensity_aug = self._build_intensity_transforms()

        # Default to ImageNet statistics for transfer learning compatibility
        if normalize_mean is None:
            normalize_mean = [0.485, 0.456, 0.406]
        if normalize_std is None:
            normalize_std = [0.229, 0.224, 0.225]
        self.normalize = K.Normalize(
            mean=torch.tensor(normalize_mean),
            std=torch.tensor(normalize_std),
        )

    def _build_geometric_transforms(self) -> K.AugmentationSequential:
        """Build geometric transforms that preserve spatial correspondence.

        Geometric transforms are applied identically to both image and mask
        to maintain pixel-level alignment for segmentation tasks.

        Returns:
            K.AugmentationSequential: Configured geometric augmentation pipeline.
        """
        transforms = []

        if self.config.horizontal_flip_prob > 0:
            transforms.append(K.RandomHorizontalFlip(p=self.config.horizontal_flip_prob))

        if self.config.vertical_flip_prob > 0:
            transforms.append(K.RandomVerticalFlip(p=self.config.vertical_flip_prob))

        if self.config.enable_geometric:
            has_affine = any([
                self.config.rotation_prob > 0,
                self.config.scale_jitter_prob > 0,
                self.config.shear_prob > 0,
                self.config.translate_prob > 0,
            ])

            if has_affine:
                degrees = self.config.rotation_angle_range if self.config.rotation_prob > 0 else 0.0
                translate = self.config.translate_range if self.config.translate_prob > 0 else None
                scale = self.config.scale_range if self.config.scale_jitter_prob > 0 else None
                shear = self.config.shear_range if self.config.shear_prob > 0 else None

                # Use max probability across affine params as a unified trigger.
                # Kornia's RandomAffine applies all transforms together, so we combine
                # them for computational efficiency rather than separate calls.
                max_prob = max(
                    self.config.rotation_prob,
                    self.config.scale_jitter_prob,
                    self.config.shear_prob,
                    self.config.translate_prob,
                )

                if max_prob > 0:
                    transforms.append(
                        K.RandomAffine(
                            degrees=degrees,
                            translate=translate,
                            scale=scale,
                            shear=shear,
                            p=max_prob,
                            padding_mode="zeros",
                            keepdim=True,
                        )
                    )

        return K.AugmentationSequential(
            *transforms, data_keys=["input", "mask"], same_on_batch=False
        )

    def _build_intensity_transforms(self) -> K.AugmentationSequential:
        """Build intensity transforms applied only to images.

        Intensity transforms modify pixel values without changing spatial layout,
        so they are only applied to images, not segmentation masks.

        Returns:
            K.AugmentationSequential: Configured intensity augmentation pipeline.
        """
        transforms = []

        if self.config.enable_color:
            brightness = self.config.brightness_range if self.config.brightness_prob > 0 else 0.0
            contrast = self.config.contrast_range if self.config.contrast_prob > 0 else 0.0
            saturation = self.config.saturation_range if self.config.saturation_prob > 0 else 0.0
            hue = self.config.hue_range if self.config.hue_prob > 0 else 0.0

            has_jitter = any([
                self.config.brightness_prob > 0,
                self.config.contrast_prob > 0,
                self.config.saturation_prob > 0,
                self.config.hue_prob > 0,
            ])

            if has_jitter:
                max_prob = max(
                    self.config.brightness_prob,
                    self.config.contrast_prob,
                    self.config.saturation_prob,
                    self.config.hue_prob,
                )

                transforms.append(
                    K.ColorJitter(
                        brightness=brightness if self.config.brightness_prob > 0 else 0.0,
                        contrast=contrast if self.config.contrast_prob > 0 else 0.0,
                        saturation=saturation if self.config.saturation_prob > 0 else 0.0,
                        hue=hue if self.config.hue_prob > 0 else 0.0,
                        p=max_prob,
                    )
                )

            if self.config.grayscale_prob > 0:
                transforms.append(K.RandomGrayscale(p=self.config.grayscale_prob))

        if self.config.enable_blur and self.config.gaussian_blur_prob > 0:
            kernel_size = (7, 7)  # Kornia requires odd kernel sizes
            transforms.append(
                K.RandomGaussianBlur(
                    kernel_size=kernel_size,
                    sigma=self.config.gaussian_sigma_range,
                    p=self.config.gaussian_blur_prob,
                )
            )

        if self.config.enable_blur and self.config.motion_blur_prob > 0:
            transforms.append(
                K.RandomMotionBlur(
                    kernel_size=5,
                    angle=35.0,
                    direction=0.5,
                    p=self.config.motion_blur_prob,
                )
            )

        if self.config.enable_cutout and self.config.cutout_prob > 0:
            transforms.append(
                K.RandomErasing(
                    scale=self.config.cutout_size_range,
                    ratio=(0.3, 3.3),
                    value=0.0,
                    p=self.config.cutout_prob,
                )
            )

        return K.AugmentationSequential(*transforms, data_keys=["input"], same_on_batch=False)

    def forward(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the full augmentation pipeline to image-mask pairs.

        The pipeline applies transforms in order:
        1. Geometric transforms (to both image and mask for spatial consistency)
        2. Intensity transforms (to image only)
        3. Normalization (to image only)

        Args:
            img: Batch of images with shape (B, C, H, W).
            mask: Batch of masks with shape (B, C, H, W) or (B, 1, H, W).

        Returns:
            Tuple of (augmented_img, augmented_mask) with same shapes as input.
        """
        # Ensure float32 for Kornia compatibility
        if img.dtype != torch.float32:
            img = img.float()
        if mask.dtype != torch.float32:
            mask = mask.float()

        img_geo, mask_geo = self.geometric_aug(img, mask)
        img_final = self.intensity_aug(img_geo)
        img_final = self.normalize(img_final)

        return img_final, mask_geo
