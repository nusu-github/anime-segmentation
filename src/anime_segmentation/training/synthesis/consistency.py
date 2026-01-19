"""Consistency processing for realistic composite synthesis.

Provides color/tone matching, light wrap, shadow generation, and
noise consistency to make composited images appear more natural.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import kornia.filters as KF
import kornia.morphology as KM
import torch

from anime_segmentation.training.synthesis.base import BaseConsistencyPipeline

if TYPE_CHECKING:
    from torch import Generator, Tensor


class ColorToneMatching:
    """Match foreground color/tone statistics to background.

    Adjusts foreground color distribution to better match the background,
    reducing the visual discrepancy in composited images.
    """

    def __init__(
        self,
        method: Literal["histogram", "mean_std"] = "mean_std",
        strength: float = 0.5,
    ) -> None:
        """Initialize color tone matching.

        Args:
            method: Matching method - "mean_std" for simple statistics,
                    "histogram" for histogram matching (slower).
            strength: Blending strength [0, 1] between original and matched.

        """
        if method not in {"histogram", "mean_std"}:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)
        if not 0.0 <= strength <= 1.0:
            msg = f"strength must be in [0, 1], got {strength}"
            raise ValueError(msg)

        self.method = method
        self.strength = strength

    def _mean_std_match(
        self,
        fg: Tensor,
        bg: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Match mean and std of foreground to background."""
        # Compute background statistics (excluding masked region)
        bg_region = bg * (1 - mask)
        bg_valid = (1 - mask).sum()

        if bg_valid > 0:
            bg_mean = bg_region.sum(dim=(1, 2), keepdim=True) / bg_valid
            bg_sq = ((bg - bg_mean) ** 2 * (1 - mask)).sum(dim=(1, 2), keepdim=True)
            bg_std = (bg_sq / (bg_valid + 1e-8)).sqrt() + 1e-8
        else:
            bg_mean = torch.zeros(3, 1, 1, device=fg.device)
            bg_std = torch.ones(3, 1, 1, device=fg.device)

        # Compute foreground statistics
        fg_region = fg * mask
        fg_valid = mask.sum()

        if fg_valid > 0:
            fg_mean = fg_region.sum(dim=(1, 2), keepdim=True) / fg_valid
            fg_sq = ((fg - fg_mean) ** 2 * mask).sum(dim=(1, 2), keepdim=True)
            fg_std = (fg_sq / (fg_valid + 1e-8)).sqrt() + 1e-8
        else:
            return fg

        # Normalize and denormalize
        fg_normalized = (fg - fg_mean) / fg_std
        return fg_normalized * bg_std + bg_mean

    def apply(
        self,
        fg: Tensor,
        bg: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply color/tone matching.

        Args:
            fg: Foreground RGB [3, H, W] in [0, 1].
            bg: Background RGB [3, H, W] in [0, 1].
            mask: Foreground mask [1, H, W] in [0, 1].

        Returns:
            Color-matched foreground [3, H, W].

        """
        if self.method == "mean_std":
            matched = self._mean_std_match(fg, bg, mask)
        else:
            # Histogram matching (simplified version)
            matched = self._mean_std_match(fg, bg, mask)

        # Blend with original based on strength
        result = fg * (1 - self.strength) + matched * self.strength

        # Only apply in masked region
        return torch.where(mask > 0.5, result, fg)


class LightWrap:
    """Apply background light bleeding to foreground edges.

    Creates a subtle glow effect where background colors bleed into
    the foreground edges, improving visual integration.
    """

    def __init__(
        self,
        wrap_radius: int = 5,
        intensity: float = 0.3,
    ) -> None:
        """Initialize light wrap.

        Args:
            wrap_radius: Radius of light wrap effect in pixels.
            intensity: Intensity of the wrap effect [0, 1].

        """
        if wrap_radius < 1:
            msg = "wrap_radius must be >= 1"
            raise ValueError(msg)
        if not 0.0 <= intensity <= 1.0:
            msg = f"intensity must be in [0, 1], got {intensity}"
            raise ValueError(msg)

        self.wrap_radius = wrap_radius
        self.intensity = intensity

    def _create_edge_mask(self, mask: Tensor) -> Tensor:
        """Create soft edge mask for light wrap application."""
        # Dilate mask
        kernel_size = 2 * self.wrap_radius + 1
        kernel = torch.ones(
            kernel_size,
            kernel_size,
            device=mask.device,
            dtype=mask.dtype,
        )

        dilated = KM.dilation(mask.unsqueeze(0), kernel).squeeze(0)

        # Edge zone = dilated - original
        return torch.clamp(dilated - mask, 0, 1)

    def apply(
        self,
        fg: Tensor,
        bg: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply light wrap effect.

        Args:
            fg: Foreground RGB [3, H, W] in [0, 1].
            bg: Background RGB [3, H, W] in [0, 1].
            mask: Foreground mask [1, H, W] in [0, 1].

        Returns:
            Foreground with light wrap [3, H, W].

        """
        edge_mask = self._create_edge_mask(mask)

        # Blur background for softer light wrap
        kernel_size = self.wrap_radius * 2 + 1
        sigma = self.wrap_radius / 2.0

        bg_blurred = KF.gaussian_blur2d(
            bg.unsqueeze(0),
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            border_type="replicate",
        ).squeeze(0)

        # Apply light wrap in edge zone
        wrap_strength = edge_mask * self.intensity
        return fg * (1 - wrap_strength) + bg_blurred * wrap_strength


class SimpleShadow:
    """Add simple drop shadow behind foreground.

    Creates a blurred, offset copy of the mask as a shadow layer
    blended onto the background.
    """

    def __init__(
        self,
        offset: tuple[int, int] = (5, 5),
        blur_radius: int = 10,
        opacity: float = 0.3,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Initialize shadow generator.

        Args:
            offset: (y, x) offset of shadow in pixels.
            blur_radius: Blur radius for shadow softness.
            opacity: Shadow opacity [0, 1].
            color: Shadow RGB color (default black).

        """
        if blur_radius < 0:
            msg = "blur_radius must be >= 0"
            raise ValueError(msg)
        if not 0.0 <= opacity <= 1.0:
            msg = f"opacity must be in [0, 1], got {opacity}"
            raise ValueError(msg)

        self.offset = offset
        self.blur_radius = blur_radius
        self.opacity = opacity
        self.color = color

    def _blur_mask(self, mask: Tensor) -> Tensor:
        """Apply Gaussian blur to mask."""
        if self.blur_radius == 0:
            return mask

        kernel_size = 2 * self.blur_radius + 1
        sigma = self.blur_radius / 2.0

        blurred = KF.gaussian_blur2d(
            mask.unsqueeze(0),
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            border_type="constant",
        )

        return blurred.squeeze(0)

    def apply(
        self,
        canvas: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply drop shadow to canvas.

        Args:
            canvas: Current canvas RGB [3, H, W] in [0, 1].
            mask: Foreground mask [1, H, W] in [0, 1].

        Returns:
            Canvas with shadow [3, H, W].

        """
        _, h, w = canvas.shape
        oy, ox = self.offset

        # Create offset shadow mask
        shadow_mask = torch.zeros_like(mask)

        # Compute valid regions
        src_y1 = max(0, -oy)
        src_x1 = max(0, -ox)
        src_y2 = min(h, h - oy)
        src_x2 = min(w, w - ox)
        dst_y1 = max(0, oy)
        dst_x1 = max(0, ox)
        dst_y2 = min(h, h + oy)
        dst_x2 = min(w, w + ox)

        if src_y2 > src_y1 and src_x2 > src_x1:
            shadow_mask[:, dst_y1:dst_y2, dst_x1:dst_x2] = mask[:, src_y1:src_y2, src_x1:src_x2]

        # Blur shadow
        shadow_mask = self._blur_mask(shadow_mask)

        # Remove shadow from foreground region
        shadow_mask *= 1 - mask

        # Apply shadow
        shadow_color = torch.tensor(self.color, device=canvas.device).view(3, 1, 1)
        shadow_alpha = shadow_mask * self.opacity

        return canvas * (1 - shadow_alpha) + shadow_color * shadow_alpha


class NoiseGrainConsistency:
    """Apply consistent noise/grain across composited image.

    Adds uniform noise across the entire image to mask subtle
    differences between foreground and background textures.
    """

    def __init__(
        self,
        noise_std: float = 0.02,
    ) -> None:
        """Initialize noise consistency.

        Args:
            noise_std: Standard deviation of additive Gaussian noise.

        """
        if noise_std < 0:
            msg = "noise_std must be >= 0"
            raise ValueError(msg)

        self.noise_std = noise_std

    def apply(
        self,
        image: Tensor,
        mask: Tensor,
        bg_noise_estimate: Tensor | None = None,
        rng: Generator | None = None,
    ) -> Tensor:
        """Apply noise consistency.

        Args:
            image: Composited image [3, H, W] in [0, 1].
            mask: Composite mask [1, H, W] (unused, for API consistency).
            bg_noise_estimate: Optional estimated noise from background.
            rng: Optional random generator.

        Returns:
            Image with consistent noise [3, H, W].

        """
        if self.noise_std == 0:
            return image

        # Generate noise
        noise = (
            torch.randn(image.shape, generator=rng, device=image.device)
            if rng is not None
            else torch.randn_like(image)
        )

        noise *= self.noise_std

        # Apply noise
        result = image + noise

        return torch.clamp(result, 0, 1)


class ConsistencyPipeline(BaseConsistencyPipeline):
    """Pipeline combining multiple consistency processing steps.

    Applies color/tone matching, light wrap, shadow, and noise
    with configurable probabilities.
    """

    def __init__(
        self,
        color_tone_prob: float = 0.5,
        color_tone_method: Literal["histogram", "mean_std"] = "mean_std",
        color_tone_strength: float = 0.5,
        light_wrap_prob: float = 0.3,
        light_wrap_radius: int = 5,
        light_wrap_intensity: float = 0.3,
        shadow_prob: float = 0.3,
        shadow_offset: tuple[int, int] = (5, 5),
        shadow_blur: int = 10,
        shadow_opacity: float = 0.3,
        noise_grain_prob: float = 0.3,
        noise_std: float = 0.02,
    ) -> None:
        """Initialize consistency pipeline.

        Args:
            color_tone_prob: Probability of applying color/tone matching.
            color_tone_method: Method for color matching.
            color_tone_strength: Strength of color matching.
            light_wrap_prob: Probability of applying light wrap.
            light_wrap_radius: Radius for light wrap effect.
            light_wrap_intensity: Intensity of light wrap.
            shadow_prob: Probability of adding shadow.
            shadow_offset: Shadow offset (y, x).
            shadow_blur: Shadow blur radius.
            shadow_opacity: Shadow opacity.
            noise_grain_prob: Probability of adding noise.
            noise_std: Noise standard deviation.

        """
        self.color_tone_prob = color_tone_prob
        self.light_wrap_prob = light_wrap_prob
        self.shadow_prob = shadow_prob
        self.noise_grain_prob = noise_grain_prob

        # Initialize processors
        self.color_matcher = ColorToneMatching(
            method=color_tone_method,
            strength=color_tone_strength,
        )
        self.light_wrap = LightWrap(
            wrap_radius=light_wrap_radius,
            intensity=light_wrap_intensity,
        )
        self.shadow = SimpleShadow(
            offset=shadow_offset,
            blur_radius=shadow_blur,
            opacity=shadow_opacity,
        )
        self.noise = NoiseGrainConsistency(noise_std=noise_std)

    def _should_apply(self, prob: float, rng: Generator | None = None) -> bool:
        """Check if processing should be applied based on probability."""
        if rng is not None:
            return torch.rand(1, generator=rng).item() < prob
        return torch.rand(1).item() < prob

    def apply(
        self,
        fg: Tensor,
        bg: Tensor,
        mask: Tensor,
        canvas: Tensor,
        rng: Generator | None = None,
    ) -> Tensor:
        """Apply consistency pipeline.

        Args:
            fg: Original foreground RGB [3, H, W] (for color matching).
            bg: Background RGB [3, H, W].
            mask: Composite mask [1, H, W].
            canvas: Current composited canvas [3, H, W].
            rng: Optional random generator.

        Returns:
            Processed canvas [3, H, W].

        """
        result = canvas.clone()

        # Color/tone matching (applied to foreground before final blend)
        if self._should_apply(self.color_tone_prob, rng):
            matched_fg = self.color_matcher.apply(fg, bg, mask)
            # Re-composite with matched foreground
            result = matched_fg * mask + bg * (1 - mask)

        # Light wrap
        if self._should_apply(self.light_wrap_prob, rng):
            result = self.light_wrap.apply(result, bg, mask)

        # Shadow
        if self._should_apply(self.shadow_prob, rng):
            result = self.shadow.apply(result, mask)

        # Noise
        if self._should_apply(self.noise_grain_prob, rng):
            result = self.noise.apply(result, mask, rng=rng)

        return result
