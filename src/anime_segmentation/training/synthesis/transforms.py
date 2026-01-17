"""Instance-level transforms for Copy-Paste synthesis.

Provides random geometric transforms applied to foreground instances
before compositing onto backgrounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

if TYPE_CHECKING:
    from torch import Generator, Tensor


class InstanceTransform:
    """Random transforms for foreground instances.

    Applies synchronized random geometric transforms to both RGB and mask
    tensors. Mask uses nearest neighbor interpolation to preserve binary values.

    Transforms applied:
        - Horizontal flip (probability-based)
        - Random rotation within range
        - Random scale within range
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        rotation_range: tuple[float, float] = (-15.0, 15.0),
        scale_range: tuple[float, float] = (0.5, 1.5),
    ) -> None:
        """Initialize instance transform.

        Args:
            hflip_prob: Probability of horizontal flip (0.0-1.0).
            rotation_range: (min, max) rotation angle in degrees.
            scale_range: (min, max) scale factor.

        Raises:
            ValueError: If parameters are invalid.

        """
        if not 0.0 <= hflip_prob <= 1.0:
            msg = f"hflip_prob must be in [0, 1], got {hflip_prob}"
            raise ValueError(msg)
        if rotation_range[0] > rotation_range[1]:
            msg = f"rotation_range min > max: {rotation_range}"
            raise ValueError(msg)
        if scale_range[0] <= 0 or scale_range[1] <= 0:
            msg = f"scale_range values must be positive, got {scale_range}"
            raise ValueError(msg)
        if scale_range[0] > scale_range[1]:
            msg = f"scale_range min > max: {scale_range}"
            raise ValueError(msg)

        self.hflip_prob = hflip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range

    def _random_uniform(
        self,
        low: float,
        high: float,
        rng: Generator | None = None,
    ) -> float:
        """Sample uniform random value."""
        if rng is not None:
            return low + (high - low) * torch.rand(1, generator=rng).item()
        return low + (high - low) * torch.rand(1).item()

    def _random_bool(self, prob: float, rng: Generator | None = None) -> bool:
        """Sample boolean with given probability."""
        if rng is not None:
            return torch.rand(1, generator=rng).item() < prob
        return torch.rand(1).item() < prob

    def __call__(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply random transforms to foreground RGB and mask.

        Args:
            fg_rgb: RGB tensor [3, H, W] in [0, 1] range.
            fg_mask: Mask tensor [1, H, W] in [0, 1] range (binary).
            rng: Optional torch.Generator for reproducible transforms.

        Returns:
            Tuple of (transformed_rgb, transformed_mask) with same shapes.

        """
        # Horizontal flip
        if self._random_bool(self.hflip_prob, rng):
            fg_rgb = TF.hflip(fg_rgb)
            fg_mask = TF.hflip(fg_mask)

        # Random rotation
        angle = self._random_uniform(*self.rotation_range, rng)
        if abs(angle) > 0.1:  # Skip if angle is negligible
            fg_rgb = TF.rotate(fg_rgb, angle, interpolation=TF.InterpolationMode.BILINEAR)
            fg_mask = TF.rotate(fg_mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            # Re-binarize mask after rotation
            fg_mask = (fg_mask > 0.5).float()

        # Random scale
        scale = self._random_uniform(*self.scale_range, rng)
        if abs(scale - 1.0) > 0.01:  # Skip if scale is close to 1
            _, h, w = fg_rgb.shape
            new_h, new_w = int(h * scale), int(w * scale)

            if new_h > 0 and new_w > 0:
                fg_rgb = F.interpolate(
                    fg_rgb.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                fg_mask = F.interpolate(
                    fg_mask.unsqueeze(0),
                    size=(new_h, new_w),
                    mode="nearest",
                ).squeeze(0)

        return fg_rgb, fg_mask


def compute_bounding_box(mask: Tensor) -> tuple[int, int, int, int] | None:
    """Compute tight bounding box of non-zero mask region.

    Args:
        mask: Binary mask tensor [1, H, W] or [H, W].

    Returns:
        (y_min, x_min, y_max, x_max) or None if mask is empty.

    """
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    nonzero = torch.nonzero(mask > 0.5)
    if nonzero.numel() == 0:
        return None

    y_min = int(nonzero[:, 0].min().item())
    y_max = int(nonzero[:, 0].max().item())
    x_min = int(nonzero[:, 1].min().item())
    x_max = int(nonzero[:, 1].max().item())

    return y_min, x_min, y_max, x_max


def crop_to_content(
    rgb: Tensor,
    mask: Tensor,
    padding: int = 0,
) -> tuple[Tensor, Tensor]:
    """Crop RGB and mask to tight bounding box with optional padding.

    Args:
        rgb: RGB tensor [3, H, W].
        mask: Mask tensor [1, H, W].
        padding: Padding to add around bounding box.

    Returns:
        Tuple of (cropped_rgb, cropped_mask).

    """
    bbox = compute_bounding_box(mask)
    if bbox is None:
        return rgb, mask

    y_min, x_min, y_max, x_max = bbox
    _, h, w = rgb.shape

    # Add padding with bounds check
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(h, y_max + padding + 1)
    x_max = min(w, x_max + padding + 1)

    return rgb[:, y_min:y_max, x_min:x_max], mask[:, y_min:y_max, x_min:x_max]


def compute_mask_area_ratio(mask: Tensor, total_area: int) -> float:
    """Compute ratio of mask foreground area to total area.

    Args:
        mask: Binary mask tensor.
        total_area: Total canvas area (H * W).

    Returns:
        Area ratio in [0, 1].

    """
    fg_area = (mask > 0.5).sum().item()
    return fg_area / total_area if total_area > 0 else 0.0


def compute_mask_iou(mask1: Tensor, mask2: Tensor) -> float:
    """Compute IoU between two binary masks.

    Args:
        mask1: First binary mask tensor.
        mask2: Second binary mask tensor.

    Returns:
        IoU value in [0, 1].

    """
    m1 = (mask1 > 0.5).float()
    m2 = (mask2 > 0.5).float()

    intersection = (m1 * m2).sum().item()
    union = m1.sum().item() + m2.sum().item() - intersection

    return intersection / union if union > 0 else 0.0
