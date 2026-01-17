"""Copy-Paste compositor for synthetic data generation.

Orchestrates the composition of multiple foreground characters onto
backgrounds with various blending strategies and placement constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from anime_segmentation.training.synthesis.blending import (
    BlendingStrategy,
    FeatherBlending,
    HardPasteBlending,
)
from anime_segmentation.training.synthesis.transforms import (
    InstanceTransform,
    compute_mask_area_ratio,
    compute_mask_iou,
    crop_to_content,
)

if TYPE_CHECKING:
    from torch import Generator, Tensor

    from anime_segmentation.data.pools import BackgroundPool, ForegroundPool

logger = logging.getLogger(__name__)


@dataclass
class CompositorConfig:
    """Configuration for CopyPasteCompositor.

    Attributes:
        k_probs: Probability distribution for number of characters (k).
        min_area_ratio: Minimum area ratio for a character (vs canvas).
        max_area_ratio: Maximum area ratio for a single character.
        max_total_coverage: Maximum total coverage of all characters.
        max_iou_overlap: Maximum IoU overlap allowed between characters.
        blending_probs: Probability distribution for blending strategies.

    """

    k_probs: dict[int, float] = field(
        default_factory=lambda: {
            0: 0.05,  # Negative examples (background only)
            1: 0.35,
            2: 0.35,
            3: 0.20,
            4: 0.05,
        },
    )

    min_area_ratio: float = 0.02
    max_area_ratio: float = 0.60
    max_total_coverage: float = 0.85

    max_iou_overlap: float = 0.30

    blending_probs: dict[str, float] = field(
        default_factory=lambda: {
            "hard": 0.40,
            "feather": 0.60,
        },
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate k_probs sum to 1
        prob_sum = sum(self.k_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            msg = f"k_probs must sum to 1.0, got {prob_sum}"
            raise ValueError(msg)

        # Validate blending_probs sum to 1
        blend_sum = sum(self.blending_probs.values())
        if abs(blend_sum - 1.0) > 1e-6:
            msg = f"blending_probs must sum to 1.0, got {blend_sum}"
            raise ValueError(msg)

        # Validate ranges
        if not 0 < self.min_area_ratio < self.max_area_ratio <= 1:
            msg = "Invalid area ratio range"
            raise ValueError(msg)
        if not 0 < self.max_total_coverage <= 1:
            msg = "Invalid max_total_coverage"
            raise ValueError(msg)
        if not 0 <= self.max_iou_overlap <= 1:
            msg = "Invalid max_iou_overlap"
            raise ValueError(msg)


class CopyPasteCompositor:
    """Multi-character Copy-Paste compositor.

    Generates synthetic training images by compositing multiple foreground
    characters onto background images with configurable placement and
    blending strategies.
    """

    def __init__(
        self,
        fg_pool: ForegroundPool,
        bg_pool: BackgroundPool,
        config: CompositorConfig | None = None,
        instance_transform: InstanceTransform | None = None,
    ) -> None:
        """Initialize compositor.

        Args:
            fg_pool: Pool of foreground character images.
            bg_pool: Pool of background images.
            config: Compositor configuration. Defaults to CompositorConfig().
            instance_transform: Optional transform applied to each foreground.

        """
        self.fg_pool = fg_pool
        self.bg_pool = bg_pool
        self.config = config or CompositorConfig()
        self.instance_transform = instance_transform

        # Initialize blending strategies
        self._blenders: dict[str, BlendingStrategy] = {
            "hard": HardPasteBlending(),
            "feather": FeatherBlending(),
        }

    def _sample_k(self, rng: Generator | None = None) -> int:
        """Sample number of characters from k_probs distribution.

        Args:
            rng: Optional random generator.

        Returns:
            Number of characters to place (can be 0 for negative examples).

        """
        ks = list(self.config.k_probs.keys())
        probs = list(self.config.k_probs.values())
        probs_tensor = torch.tensor(probs)

        if rng is not None:
            idx = torch.multinomial(probs_tensor, 1, generator=rng).item()
        else:
            idx = torch.multinomial(probs_tensor, 1).item()

        return ks[int(idx)]

    def _sample_blending_strategy(self, rng: Generator | None = None) -> BlendingStrategy:
        """Sample blending strategy from probability distribution.

        Args:
            rng: Optional random generator.

        Returns:
            Selected BlendingStrategy instance.

        """
        names = list(self.config.blending_probs.keys())
        probs = list(self.config.blending_probs.values())
        probs_tensor = torch.tensor(probs)

        if rng is not None:
            idx = torch.multinomial(probs_tensor, 1, generator=rng).item()
        else:
            idx = torch.multinomial(probs_tensor, 1).item()

        name = names[int(idx)]
        return self._blenders[name]

    def _scale_foreground_to_target_area(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        target_area_ratio: float,
        canvas_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Scale foreground to achieve target area ratio on canvas.

        Args:
            fg_rgb: Foreground RGB [3, H, W].
            fg_mask: Foreground mask [1, H, W].
            target_area_ratio: Desired mask area / canvas area.
            canvas_size: (H, W) of target canvas.

        Returns:
            Scaled (fg_rgb, fg_mask).

        """
        canvas_h, canvas_w = canvas_size
        canvas_area = canvas_h * canvas_w

        # Current mask area
        current_area = (fg_mask > 0.5).sum().item()
        if current_area == 0:
            return fg_rgb, fg_mask

        # Target area
        target_area = target_area_ratio * canvas_area

        # Scale factor
        scale = (target_area / current_area) ** 0.5

        # Apply scale
        _, fh, fw = fg_rgb.shape
        new_h = max(1, int(fh * scale))
        new_w = max(1, int(fw * scale))

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

    def _find_valid_position(
        self,
        fg_mask: Tensor,
        current_composite_mask: Tensor,
        canvas_size: tuple[int, int],
        max_attempts: int = 50,
        rng: Generator | None = None,
    ) -> tuple[int, int] | None:
        """Find valid position for foreground that respects overlap constraints.

        Args:
            fg_mask: Foreground mask [1, H, W].
            current_composite_mask: Current composite mask [1, H, W].
            canvas_size: (H, W) of canvas.
            max_attempts: Maximum random attempts.
            rng: Optional random generator.

        Returns:
            (y, x) position or None if no valid position found.

        """
        canvas_h, canvas_w = canvas_size
        _, fh, fw = fg_mask.shape

        # Range for top-left corner (allow partial off-screen)
        y_min = -fh // 4
        y_max = canvas_h - fh // 2
        x_min = -fw // 4
        x_max = canvas_w - fw // 2

        if y_max <= y_min or x_max <= x_min:
            return None

        for _ in range(max_attempts):
            # Random position
            if rng is not None:
                y = int(torch.randint(y_min, y_max, (1,), generator=rng).item())
                x = int(torch.randint(x_min, x_max, (1,), generator=rng).item())
            else:
                y = int(torch.randint(y_min, y_max, (1,)).item())
                x = int(torch.randint(x_min, x_max, (1,)).item())

            # Create temporary mask at this position
            temp_mask = torch.zeros(1, canvas_h, canvas_w, device=fg_mask.device)
            y1 = max(0, y)
            x1 = max(0, x)
            y2 = min(canvas_h, y + fh)
            x2 = min(canvas_w, x + fw)
            fy1 = y1 - y
            fx1 = x1 - x
            fy2 = fy1 + (y2 - y1)
            fx2 = fx1 + (x2 - x1)

            if y2 > y1 and x2 > x1:
                temp_mask[:, y1:y2, x1:x2] = fg_mask[:, fy1:fy2, fx1:fx2]

            # Check overlap with existing characters
            if current_composite_mask.sum() > 0:
                iou = compute_mask_iou(temp_mask, current_composite_mask)
                if iou > self.config.max_iou_overlap:
                    continue

            return y, x

        return None

    def synthesize(
        self,
        target_size: tuple[int, int],
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor, int]:
        """Synthesize a training image with ground truth mask.

        Args:
            target_size: (H, W) of output image.
            rng: Optional random generator for reproducibility.

        Returns:
            Tuple of (image [3, H, W], mask [1, H, W], k) where:
                - image: Composited RGB in [0, 1] range
                - mask: Binary GT mask (union of all foregrounds)
                - k: Number of characters placed

        """
        h, w = target_size
        canvas_area = h * w

        # Sample number of characters
        k = self._sample_k(rng)

        # Get background
        canvas = self.bg_pool.sample(target_size, rng)
        device = canvas.device

        # Initialize composite mask
        composite_mask = torch.zeros(1, h, w, device=device)

        # Handle negative examples (k=0)
        if k == 0:
            return canvas, composite_mask, k

        # Place characters
        placed_count = 0
        current_coverage = 0.0

        for _ in range(k):
            # Check total coverage limit
            if current_coverage >= self.config.max_total_coverage:
                break

            # Sample foreground
            fg_rgb, fg_mask = self.fg_pool.sample(rng)
            fg_rgb = fg_rgb.to(device)
            fg_mask = fg_mask.to(device)

            # Crop to content
            fg_rgb, fg_mask = crop_to_content(fg_rgb, fg_mask, padding=5)

            # Apply instance transform
            if self.instance_transform is not None:
                fg_rgb, fg_mask = self.instance_transform(fg_rgb, fg_mask, rng)

            # Sample target area ratio
            remaining_coverage = self.config.max_total_coverage - current_coverage
            max_area = min(self.config.max_area_ratio, remaining_coverage)

            if max_area < self.config.min_area_ratio:
                break

            if rng is not None:
                target_area = (
                    self.config.min_area_ratio
                    + (max_area - self.config.min_area_ratio) * torch.rand(1, generator=rng).item()
                )
            else:
                target_area = (
                    self.config.min_area_ratio
                    + (max_area - self.config.min_area_ratio) * torch.rand(1).item()
                )

            # Scale foreground
            fg_rgb, fg_mask = self._scale_foreground_to_target_area(
                fg_rgb,
                fg_mask,
                target_area,
                target_size,
            )

            # Find valid position
            position = self._find_valid_position(
                fg_mask,
                composite_mask,
                target_size,
                rng=rng,
            )

            if position is None:
                continue  # Skip if no valid position found

            # Select blending strategy
            blender = self._sample_blending_strategy(rng)

            # Blend foreground onto canvas
            canvas = blender.blend(fg_rgb, fg_mask, canvas, position, rng)

            # Update composite mask
            y, x = position
            _, fh, fw = fg_mask.shape
            y1 = max(0, y)
            x1 = max(0, x)
            y2 = min(h, y + fh)
            x2 = min(w, x + fw)
            fy1 = y1 - y
            fx1 = x1 - x
            fy2 = fy1 + (y2 - y1)
            fx2 = fx1 + (x2 - x1)

            if y2 > y1 and x2 > x1:
                # Union of masks
                composite_mask[:, y1:y2, x1:x2] = torch.maximum(
                    composite_mask[:, y1:y2, x1:x2],
                    fg_mask[:, fy1:fy2, fx1:fx2],
                )

            # Update coverage
            current_coverage = compute_mask_area_ratio(composite_mask, canvas_area)
            placed_count += 1

        # Binarize final mask
        composite_mask = (composite_mask > 0.5).float()

        return canvas, composite_mask, placed_count
