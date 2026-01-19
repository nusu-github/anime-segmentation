"""Quality degradation for synthetic training data augmentation.

Applies realistic image degradations like JPEG compression artifacts,
blur, and noise to make synthetic data more representative of real-world
conditions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import kornia.augmentation as K
import kornia.filters as KF
import torch
from torch import nn

from anime_segmentation.training.synthesis.base import BaseDegradation

if TYPE_CHECKING:
    from torch import Generator, Tensor


class QualityDegradation(nn.Module, BaseDegradation):
    """Apply quality degradations to synthetic images.

    Randomly applies JPEG compression, blur, and noise to images
    to simulate real-world image quality variations.
    """

    def __init__(
        self,
        jpeg_prob: float = 0.3,
        jpeg_quality_range: tuple[int, int] = (70, 95),
        blur_prob: float = 0.1,
        blur_kernel_range: tuple[int, int] = (3, 7),
        noise_prob: float = 0.1,
        noise_std_range: tuple[float, float] = (0.01, 0.05),
    ) -> None:
        """Initialize quality degradation.

        Args:
            jpeg_prob: Probability of applying JPEG compression.
            jpeg_quality_range: (min, max) JPEG quality (1-100).
            blur_prob: Probability of applying Gaussian blur.
            blur_kernel_range: (min, max) blur kernel size (must be odd).
            noise_prob: Probability of applying Gaussian noise.
            noise_std_range: (min, max) noise standard deviation.

        Raises:
            ValueError: If parameters are invalid.

        """
        super().__init__()

        # Validate parameters
        if not 0.0 <= jpeg_prob <= 1.0:
            msg = f"jpeg_prob must be in [0, 1], got {jpeg_prob}"
            raise ValueError(msg)
        if not 0.0 <= blur_prob <= 1.0:
            msg = f"blur_prob must be in [0, 1], got {blur_prob}"
            raise ValueError(msg)
        if not 0.0 <= noise_prob <= 1.0:
            msg = f"noise_prob must be in [0, 1], got {noise_prob}"
            raise ValueError(msg)

        if not 1 <= jpeg_quality_range[0] <= jpeg_quality_range[1] <= 100:
            msg = f"Invalid jpeg_quality_range: {jpeg_quality_range}"
            raise ValueError(msg)
        if blur_kernel_range[0] < 1 or blur_kernel_range[0] > blur_kernel_range[1]:
            msg = f"Invalid blur_kernel_range: {blur_kernel_range}"
            raise ValueError(msg)
        if noise_std_range[0] < 0 or noise_std_range[0] > noise_std_range[1]:
            msg = f"Invalid noise_std_range: {noise_std_range}"
            raise ValueError(msg)

        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range

    def _apply_jpeg_compression(
        self,
        image: Tensor,
        quality: int,
    ) -> Tensor:
        """Apply JPEG compression and decompression.

        Args:
            image: Image tensor [3, H, W] in [0, 1].
            quality: JPEG quality (1-100).

        Returns:
            JPEG-compressed image [3, H, W].

        """
        jpeg = K.RandomJPEG(jpeg_quality=(quality, quality), p=1.0, same_on_batch=True)
        result = jpeg(image.unsqueeze(0))
        return result.squeeze(0)

    def _apply_gaussian_blur(
        self,
        image: Tensor,
        kernel_size: int,
    ) -> Tensor:
        """Apply Gaussian blur.

        Args:
            image: Image tensor [3, H, W] in [0, 1].
            kernel_size: Blur kernel size (must be odd).

        Returns:
            Blurred image [3, H, W].

        """
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = kernel_size / 6.0  # Reasonable default
        blurred = KF.gaussian_blur2d(
            image.unsqueeze(0),
            kernel_size=(kernel_size, kernel_size),
            sigma=(sigma, sigma),
            border_type="replicate",
        )
        return blurred.squeeze(0)

    def _apply_gaussian_noise(
        self,
        image: Tensor,
        noise_std: float,
        rng: Generator | None = None,
    ) -> Tensor:
        """Apply additive Gaussian noise.

        Args:
            image: Image tensor [3, H, W] in [0, 1].
            noise_std: Noise standard deviation.
            rng: Optional random generator.

        Returns:
            Noisy image [3, H, W].

        """
        noise = (
            torch.randn(image.shape, generator=rng, device=image.device) * noise_std
            if rng is not None
            else torch.randn_like(image) * noise_std
        )

        return torch.clamp(image + noise, 0, 1)

    def _sample_uniform(
        self,
        low: float,
        high: float,
        rng: Generator | None = None,
    ) -> float:
        """Sample uniform random value."""
        if rng is not None:
            return low + (high - low) * torch.rand(1, generator=rng).item()
        return low + (high - low) * torch.rand(1).item()

    def _sample_int(
        self,
        low: int,
        high: int,
        rng: Generator | None = None,
    ) -> int:
        """Sample uniform random integer."""
        if rng is not None:
            return int(torch.randint(low, high + 1, (1,), generator=rng).item())
        return int(torch.randint(low, high + 1, (1,)).item())

    def forward(
        self,
        image: Tensor,
        mask: Tensor,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply quality degradations to image.

        Args:
            image: Image tensor [3, H, W] in [0, 1].
            mask: Mask tensor [1, H, W] in [0, 1] (unchanged).
            rng: Optional random generator.

        Returns:
            Tuple of (degraded_image, mask). Mask is unchanged.

        """
        result = image

        # Apply JPEG compression
        if self.should_apply(self.jpeg_prob, rng):
            quality = self._sample_int(*self.jpeg_quality_range, rng)
            result = self._apply_jpeg_compression(result, quality)

        # Apply blur
        if self.should_apply(self.blur_prob, rng):
            kernel_size = self._sample_int(*self.blur_kernel_range, rng)
            result = self._apply_gaussian_blur(result, kernel_size)

        # Apply noise
        if self.should_apply(self.noise_prob, rng):
            noise_std = self._sample_uniform(*self.noise_std_range, rng)
            result = self._apply_gaussian_noise(result, noise_std, rng)

        return result, mask

    def __call__(
        self,
        image: Tensor,
        mask: Tensor,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        return super().__call__(image, mask, rng=rng)


class DegradationSequence(nn.Module):
    """Apply a sequence of degradations in order.

    This is useful for applying multiple degradations with
    controlled ordering.
    """

    def __init__(self, degradations: list[nn.Module]) -> None:
        """Initialize degradation sequence.

        Args:
            degradations: List of degradation modules.

        """
        super().__init__()
        self.degradations = nn.ModuleList(degradations)

    def forward(
        self,
        image: Tensor,
        mask: Tensor,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply all degradations in sequence.

        Args:
            image: Image tensor [3, H, W] in [0, 1].
            mask: Mask tensor [1, H, W] in [0, 1].
            rng: Optional random generator.

        Returns:
            Tuple of (degraded_image, mask).

        """
        result = image
        for degradation in self.degradations:
            if hasattr(degradation, "forward") and callable(degradation.forward):
                # Check if degradation accepts rng parameter
                import inspect

                sig = inspect.signature(degradation.forward)
                if "rng" in sig.parameters:
                    result, mask = degradation(result, mask, rng=rng)
                else:
                    result, mask = degradation(result, mask)

        return result, mask
