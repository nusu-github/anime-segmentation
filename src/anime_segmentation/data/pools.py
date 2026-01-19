"""Data pools for Copy-Paste synthesis.

Provides ForegroundPool and BackgroundPool classes for managing
character cutouts and background images used in synthetic data generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.io import ImageReadMode, read_image

from anime_segmentation.constants import VALID_IMAGE_EXTENSIONS
from anime_segmentation.exceptions import InvalidImageError

if TYPE_CHECKING:
    from torch import Generator, Tensor

logger = logging.getLogger(__name__)


def _load_rgba_image(path: str | Path) -> tuple[Tensor, Tensor]:
    """Load RGBA image and split into RGB and alpha mask.

    Args:
        path: Path to image file (PNG with alpha channel).

    Returns:
        Tuple of (rgb [3, H, W], mask [1, H, W]) tensors in [0, 1] range.

    Raises:
        FileNotFoundError: If the image file does not exist.
        InvalidImageError: If the image cannot be loaded by any method.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    torchvision_error = None
    try:
        # Try fast path with torchvision.io
        tensor = read_image(str(path), mode=ImageReadMode.RGB_ALPHA)
        # tensor shape: [4, H, W], uint8
        rgb = tensor[:3].float() / 255.0
        mask = tensor[3:4].float() / 255.0
        return rgb, mask
    except RuntimeError as e:
        # torchvision.io failed, will try PIL fallback
        torchvision_error = e
        logger.debug("torchvision.io failed for %s: %s, trying PIL fallback", path, e)

    try:
        # Fallback to PIL
        with Image.open(path) as img:
            img = img.convert("RGBA")
            tensor = TF.to_tensor(img)
        rgb = tensor[:3]
        mask = tensor[3:4]
        return rgb, mask
    except Exception as pil_error:
        # Both methods failed
        raise InvalidImageError(
            f"Cannot load RGBA image {path}: "
            f"torchvision error: {torchvision_error}, PIL error: {pil_error}"
        ) from pil_error


def _load_rgb_image(path: str | Path) -> Tensor:
    """Load RGB image.

    Args:
        path: Path to image file.

    Returns:
        RGB tensor [3, H, W] in [0, 1] range.

    Raises:
        FileNotFoundError: If the image file does not exist.
        InvalidImageError: If the image cannot be loaded by any method.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    torchvision_error = None
    try:
        tensor = read_image(str(path), mode=ImageReadMode.RGB)
        return tensor.float() / 255.0
    except RuntimeError as e:
        torchvision_error = e
        logger.debug("torchvision.io failed for %s: %s, trying PIL fallback", path, e)

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return TF.to_tensor(img)
    except Exception as pil_error:
        raise InvalidImageError(
            f"Cannot load RGB image {path}: "
            f"torchvision error: {torchvision_error}, PIL error: {pil_error}"
        ) from pil_error


class ForegroundPool:
    """Pool of foreground character cutouts for Copy-Paste synthesis.

    Loads PNG images with alpha channels from the fg/ directory.
    Each image provides both RGB content and binary mask from the alpha channel.
    """

    def __init__(self, fg_dir: str | Path) -> None:
        """Initialize foreground pool.

        Args:
            fg_dir: Directory containing foreground images (PNG with alpha).

        Raises:
            FileNotFoundError: If fg_dir does not exist.
            ValueError: If no valid images found.

        """
        self.fg_dir = Path(fg_dir)
        if not self.fg_dir.exists():
            msg = f"Foreground directory does not exist: {fg_dir}"
            raise FileNotFoundError(msg)

        # Collect all valid image paths
        self.paths: list[Path] = []
        for p in sorted(self.fg_dir.iterdir()):
            if p.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                self.paths.append(p)

        if not self.paths:
            msg = f"No valid images found in {fg_dir}"
            raise ValueError(msg)

        logger.info("ForegroundPool: loaded %d images from %s", len(self.paths), fg_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def sample(self, rng: Generator | None = None) -> tuple[Tensor, Tensor]:
        """Sample a random foreground image.

        Args:
            rng: Optional torch.Generator for reproducible sampling.

        Returns:
            Tuple of (rgb [3, H, W], mask [1, H, W]) tensors in [0, 1] range.
            Mask is binarized at threshold 0.5.

        """
        if rng is not None:
            idx = int(torch.randint(0, len(self.paths), (1,), generator=rng).item())
        else:
            idx = int(torch.randint(0, len(self.paths), (1,)).item())

        path = self.paths[idx]
        rgb, mask = _load_rgba_image(path)

        # Binarize mask at 0.5 threshold
        mask = (mask > 0.5).float()

        return rgb, mask


class BackgroundPool:
    """Pool of background images for Copy-Paste synthesis.

    Loads images from the bg/ directory for use as composition backgrounds.
    """

    def __init__(self, bg_dir: str | Path) -> None:
        """Initialize background pool.

        Args:
            bg_dir: Directory containing background images.

        Raises:
            FileNotFoundError: If bg_dir does not exist.
            ValueError: If no valid images found.

        """
        self.bg_dir = Path(bg_dir)
        if not self.bg_dir.exists():
            msg = f"Background directory does not exist: {bg_dir}"
            raise FileNotFoundError(msg)

        # Collect all valid image paths
        self.paths: list[Path] = []
        for p in sorted(self.bg_dir.iterdir()):
            if p.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                self.paths.append(p)

        if not self.paths:
            msg = f"No valid images found in {bg_dir}"
            raise ValueError(msg)

        logger.info("BackgroundPool: loaded %d images from %s", len(self.paths), bg_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def sample(
        self,
        target_size: tuple[int, int],
        rng: Generator | None = None,
    ) -> Tensor:
        """Sample a random background image and resize to target size.

        Args:
            target_size: Target (height, width) for the background.
            rng: Optional torch.Generator for reproducible sampling.

        Returns:
            Background tensor [3, H, W] in [0, 1] range.

        """
        if rng is not None:
            idx = int(torch.randint(0, len(self.paths), (1,), generator=rng).item())
        else:
            idx = int(torch.randint(0, len(self.paths), (1,)).item())

        path = self.paths[idx]
        rgb = _load_rgb_image(path)

        # Resize to target size using bilinear interpolation
        h, w = target_size
        if rgb.shape[1] != h or rgb.shape[2] != w:
            rgb = F.interpolate(
                rgb.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return rgb
