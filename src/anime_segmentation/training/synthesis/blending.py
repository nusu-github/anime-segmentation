"""Blending strategies for Copy-Paste synthesis.

Provides different methods for blending foreground characters onto
backgrounds, including hard paste and feathered edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import kornia.filters as KF
import kornia.morphology as KM
import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Generator, Tensor


@runtime_checkable
class BlendingStrategy(Protocol):
    """Protocol for blending strategies."""

    def blend(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        bg_rgb: Tensor,
        position: tuple[int, int],
        rng: Generator | None = None,
    ) -> Tensor:
        """Blend foreground onto background at specified position.

        Args:
            fg_rgb: Foreground RGB tensor [3, H_fg, W_fg] in [0, 1].
            fg_mask: Foreground mask tensor [1, H_fg, W_fg] in [0, 1].
            bg_rgb: Background RGB tensor [3, H_bg, W_bg] in [0, 1].
            position: (y, x) position for top-left corner of foreground.
            rng: Optional random generator.

        Returns:
            Composited image [3, H_bg, W_bg] in [0, 1].

        """
        ...


class HardPasteBlending:
    """Simple alpha compositing without edge treatment.

    Uses standard alpha blending: out = fg * mask + bg * (1 - mask)
    """

    def blend(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        bg_rgb: Tensor,
        position: tuple[int, int],
        rng: Generator | None = None,
    ) -> Tensor:
        """Blend with hard edges."""
        result = bg_rgb.clone()
        y, x = position
        _, fh, fw = fg_rgb.shape
        _, bh, bw = bg_rgb.shape

        # Compute valid region (handle out-of-bounds)
        y1 = max(0, y)
        x1 = max(0, x)
        y2 = min(bh, y + fh)
        x2 = min(bw, x + fw)

        # Corresponding region in foreground
        fy1 = y1 - y
        fx1 = x1 - x
        fy2 = fy1 + (y2 - y1)
        fx2 = fx1 + (x2 - x1)

        if y2 > y1 and x2 > x1:
            fg_region = fg_rgb[:, fy1:fy2, fx1:fx2]
            mask_region = fg_mask[:, fy1:fy2, fx1:fx2]
            bg_region = result[:, y1:y2, x1:x2]

            # Alpha compositing
            result[:, y1:y2, x1:x2] = fg_region * mask_region + bg_region * (1 - mask_region)

        return result


class FeatherBlending:
    """Blending with feathered (soft) edges.

    Applies Gaussian blur to the mask edges to create a smooth transition
    between foreground and background.
    """

    def __init__(
        self,
        feather_radius_range: tuple[int, int] = (1, 5),
    ) -> None:
        """Initialize feather blending.

        Args:
            feather_radius_range: (min, max) radius for Gaussian blur kernel.

        """
        if feather_radius_range[0] < 1:
            msg = "Minimum feather radius must be >= 1"
            raise ValueError(msg)
        if feather_radius_range[0] > feather_radius_range[1]:
            msg = f"feather_radius_range min > max: {feather_radius_range}"
            raise ValueError(msg)

        self.feather_radius_range = feather_radius_range

    def _create_feathered_mask(
        self,
        mask: Tensor,
        radius: int,
    ) -> Tensor:
        """Apply Gaussian blur only to mask edges, preserving interior.

        Args:
            mask: Binary mask [1, H, W].
            radius: Blur kernel radius.

        Returns:
            Feathered mask [1, H, W] with soft edges but solid interior.

        """
        kernel_size = 2 * radius + 1
        sigma = radius / 2.0

        blurred = KF.gaussian_blur2d(
            mask.unsqueeze(0),
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            border_type="replicate",
        ).squeeze(0)

        # Preserve interior: use max of original and blurred
        # This ensures interior stays at 1.0 while edges are softened
        return torch.maximum(mask, blurred)

    def blend(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        bg_rgb: Tensor,
        position: tuple[int, int],
        rng: Generator | None = None,
    ) -> Tensor:
        """Blend with feathered edges."""
        # Sample random feather radius
        min_r, max_r = self.feather_radius_range
        if rng is not None:
            radius = int(torch.randint(min_r, max_r + 1, (1,), generator=rng).item())
        else:
            radius = int(torch.randint(min_r, max_r + 1, (1,)).item())

        # Create feathered mask
        feathered_mask = self._create_feathered_mask(fg_mask, radius)

        # Use hard paste with feathered mask
        result = bg_rgb.clone()
        y, x = position
        _, fh, fw = fg_rgb.shape
        _, bh, bw = bg_rgb.shape

        y1 = max(0, y)
        x1 = max(0, x)
        y2 = min(bh, y + fh)
        x2 = min(bw, x + fw)

        fy1 = y1 - y
        fx1 = x1 - x
        fy2 = fy1 + (y2 - y1)
        fx2 = fx1 + (x2 - x1)

        if y2 > y1 and x2 > x1:
            fg_region = fg_rgb[:, fy1:fy2, fx1:fx2]
            mask_region = feathered_mask[:, fy1:fy2, fx1:fx2]
            bg_region = result[:, y1:y2, x1:x2]

            result[:, y1:y2, x1:x2] = fg_region * mask_region + bg_region * (1 - mask_region)

        return result


class BoundaryRGBRandomizer:
    """Randomize RGB values in mask boundary zone.

    Adds noise to the RGB values in the transition zone between
    foreground and background to reduce boundary artifacts.
    """

    def __init__(
        self,
        boundary_width: int = 3,
        noise_std: float = 0.05,
    ) -> None:
        """Initialize boundary randomizer.

        Args:
            boundary_width: Width of boundary zone in pixels.
            noise_std: Standard deviation of Gaussian noise to add.

        """
        if boundary_width < 1:
            msg = "boundary_width must be >= 1"
            raise ValueError(msg)
        if noise_std < 0:
            msg = "noise_std must be >= 0"
            raise ValueError(msg)

        self.boundary_width = boundary_width
        self.noise_std = noise_std

    def _extract_boundary_zone(self, mask: Tensor) -> Tensor:
        """Extract boundary zone from mask.

        Args:
            mask: Binary mask [1, H, W].

        Returns:
            Boundary zone mask [1, H, W] where boundary pixels are 1.

        """
        binary_mask = (mask > 0.5).float()
        kernel_size = 2 * self.boundary_width + 1
        kernel = torch.ones(
            kernel_size,
            kernel_size,
            device=mask.device,
            dtype=mask.dtype,
        )

        dilated = KM.dilation(binary_mask.unsqueeze(0), kernel).squeeze(0)
        eroded = KM.erosion(binary_mask.unsqueeze(0), kernel).squeeze(0)

        return torch.clamp(dilated - eroded, 0, 1)

    def apply(
        self,
        image: Tensor,
        mask: Tensor,
        rng: Generator | None = None,
    ) -> Tensor:
        """Apply boundary RGB randomization.

        Args:
            image: Composited image [3, H, W] in [0, 1].
            mask: Composite mask [1, H, W] in [0, 1].
            rng: Optional random generator.

        Returns:
            Image with randomized boundary RGB [3, H, W].

        """
        boundary_zone = self._extract_boundary_zone(mask)

        # Generate noise
        if rng is not None:
            noise = torch.randn(image.shape, generator=rng, device=image.device) * self.noise_std
        else:
            noise = torch.randn_like(image) * self.noise_std

        # Apply noise only in boundary zone
        result = image + noise * boundary_zone

        # Clamp to valid range
        return torch.clamp(result, 0, 1)


class SeamlessCloneBlending:
    """Blend using OpenCV seamlessClone for high-quality composites.

    Falls back to HardPasteBlending if OpenCV is unavailable or if cloning fails.
    """

    def __init__(self, mode: str = "normal") -> None:
        """Initialize seamless clone blending.

        Args:
            mode: "normal" or "mixed" clone mode.

        """
        if mode not in {"normal", "mixed"}:
            msg = f"Unknown seamless clone mode: {mode}"
            raise ValueError(msg)
        self.mode = mode
        self._fallback = HardPasteBlending()

    def blend(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        bg_rgb: Tensor,
        position: tuple[int, int],
        rng: Generator | None = None,
    ) -> Tensor:
        """Blend using Poisson-based seamless cloning."""
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return self._fallback.blend(fg_rgb, fg_mask, bg_rgb, position, rng)

        y, x = position
        _, fh, fw = fg_rgb.shape
        _, bh, bw = bg_rgb.shape

        # Compute valid region (handle out-of-bounds)
        y1 = max(0, y)
        x1 = max(0, x)
        y2 = min(bh, y + fh)
        x2 = min(bw, x + fw)

        fy1 = y1 - y
        fx1 = x1 - x
        fy2 = fy1 + (y2 - y1)
        fx2 = fx1 + (x2 - x1)

        if y2 <= y1 or x2 <= x1:
            return bg_rgb

        # Prepare full-size source and mask
        bg_np = (bg_rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        src_np = bg_np.copy()
        mask_np = (fg_mask.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")

        # Insert foreground into source and mask
        fg_np = (fg_rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        src_np[y1:y2, x1:x2] = fg_np[fy1:fy2, fx1:fx2]

        full_mask = np.zeros((bh, bw), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_np[fy1:fy2, fx1:fx2]

        if full_mask.sum() == 0:
            return bg_rgb

        # OpenCV uses BGR
        src_bgr = src_np[:, :, ::-1]
        bg_bgr = bg_np[:, :, ::-1]

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        clone_flag = cv2.NORMAL_CLONE if self.mode == "normal" else cv2.MIXED_CLONE

        try:
            blended_bgr = cv2.seamlessClone(src_bgr, bg_bgr, full_mask, center, clone_flag)
        except Exception:
            return self._fallback.blend(fg_rgb, fg_mask, bg_rgb, position, rng)

        blended = np.ascontiguousarray(blended_bgr[:, :, ::-1])
        return torch.from_numpy(blended).permute(2, 0, 1).float().to(bg_rgb.device) / 255.0
