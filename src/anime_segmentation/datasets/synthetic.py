"""Dataset for synthetic anime images generated from fg/bg compositing."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class SyntheticDataset(Dataset):
    """Dataset that generates synthetic composites from fg/bg pairs.

    Composites foreground characters (with alpha) onto background images.
    Augmentations should be applied via external transforms.

    Args:
        fg_paths: List of paths to foreground images (RGBA with alpha as mask).
        bg_paths: List of paths to background images.
        output_size: Target output size (height, width).
        transform: Optional transform to apply after compositing.
        characters_range: Range for number of characters per sample. Default is (0, 3).
        seed: Random seed for reproducibility. Default is 1.
    """

    def __init__(
        self,
        fg_paths: list[Path],
        bg_paths: list[Path],
        output_size: tuple[int, int],
        transform: v2.Transform | None = None,
        characters_range: tuple[int, int] = (0, 3),
        seed: int = 1,
    ) -> None:
        self.fg_paths = fg_paths
        self.bg_paths = bg_paths
        self.output_size = output_size
        self.transform = transform
        self.rng = random.Random(seed)

        # Pre-compute character groupings (which foregrounds go together)
        self.characters_idx: list[list[int]] = []
        total = 0
        while total < len(fg_paths):
            num = self.rng.randint(*characters_range)
            group = [total + x for x in range(num) if total + x < len(fg_paths)]
            self.characters_idx.append(group)
            total += num

        # Track background rotation to ensure variety
        self._bg_offset = [0] * len(self.characters_idx)

    def __len__(self) -> int:
        return len(self.characters_idx)

    def __getitem__(self, idx: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        image, mask = self._composite(idx)

        # Wrap as tv_tensors
        image_tensor = tv_tensors.Image(image)
        mask_tensor = tv_tensors.Mask(mask)

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def _composite(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Core fg/bg compositing logic."""
        h, w = self.output_size

        # Load and prepare background
        bg_idx = (idx + self._bg_offset[idx]) % len(self.bg_paths)
        self._bg_offset[idx] += 1
        bg = self._load_bg(bg_idx, (h, w))

        # Load foregrounds for this sample
        fgs = [self._load_fg(i) for i in self.characters_idx[idx]]

        # Composite foregrounds onto background
        image = bg.clone()
        label = torch.zeros(1, h, w, dtype=torch.float32)

        for fg in fgs:
            fg_processed = self._process_fg(fg, (h, w))
            image, label = self._blend(image, label, fg_processed)

        # Binarize label
        label = (label > 0.5).float()

        return image, label

    def _load_bg(self, idx: int, output_size: tuple[int, int]) -> torch.Tensor:
        """Load and prepare background image."""
        bg = decode_image(str(self.bg_paths[idx]), mode=ImageReadMode.RGB)
        bg = F.to_dtype(bg, torch.float32, scale=True)

        # Random crop to output aspect ratio, then resize
        _, h, w = bg.shape  # CHW format
        out_h, out_w = output_size
        r = min(h / out_h, w / out_w)
        crop_h, crop_w = int(out_h * r), int(out_w * r)

        # Random crop position
        top = self.rng.randint(0, max(0, h - crop_h))
        left = self.rng.randint(0, max(0, w - crop_w))
        bg = F.crop(bg, top, left, crop_h, crop_w)

        # Resize to output size
        return F.resize(bg, list(output_size), antialias=True)

    def _load_fg(self, idx: int) -> torch.Tensor:
        """Load foreground image (RGBA)."""
        fg = decode_image(str(self.fg_paths[idx]), mode=ImageReadMode.RGB_ALPHA)
        return F.to_dtype(fg, torch.float32, scale=True)

    def _process_fg(self, fg: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Process foreground: resize, center, and apply random transform."""
        h, w = output_size

        # Resize to fit output (fg is CHW format)
        _, fg_h, fg_w = fg.shape
        r = min(h / fg_h, w / fg_w)
        new_h, new_w = int(fg_h * r), int(fg_w * r)
        fg = F.resize(fg, [new_h, new_w], antialias=True)

        # Pad to output size (centered)
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        fg = F.pad(fg, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

        # Center foreground based on alpha center of mass
        alpha = fg[3]  # Alpha channel (H, W)
        cy, cx = self._center_of_mass(alpha)

        # Translation to center the character
        dx = w / 2 - cx
        dy = h / 2 - cy
        fg = F.affine(fg, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0])

        # Random scale, rotation, and translation
        scale = self.rng.uniform(0.2, 0.8)
        angle = self.rng.randint(-90, 90)
        trans_dx = self.rng.randint(-w // 3, w // 3)
        trans_dy = self.rng.randint(-h // 3, h // 3)

        return F.affine(fg, angle=angle, translate=[trans_dx, trans_dy], scale=scale, shear=[0.0])

    def _center_of_mass(self, alpha: torch.Tensor) -> tuple[float, float]:
        """Compute center of mass for alpha channel."""
        h, w = alpha.shape
        total = alpha.sum()
        if total == 0:
            return h / 2, w / 2

        y_coords = torch.arange(h, dtype=alpha.dtype, device=alpha.device)
        x_coords = torch.arange(w, dtype=alpha.dtype, device=alpha.device)

        cy = (y_coords.unsqueeze(1) * alpha).sum() / total
        cx = (x_coords.unsqueeze(0) * alpha).sum() / total

        return cy.item(), cx.item()

    def _blend(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        fg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend foreground onto image and update label."""
        fg_rgb = fg[:3]  # (3, H, W)
        fg_alpha = fg[3:4]  # (1, H, W)

        # Soft edge blending with gaussian blur
        blurred_alpha = F.gaussian_blur(fg_alpha, kernel_size=[5, 5])
        mask = fg_alpha * blurred_alpha

        # Composite
        image = mask * fg_rgb + (1 - mask) * image
        label = torch.maximum(fg_alpha, label)

        return image, label
