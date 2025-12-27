"""Dataset for synthetic anime images generated from fg/bg compositing."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import center_of_mass
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


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

        # Convert to tv_tensors (HWC -> CHW)
        image_tensor = tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1))
        mask_tensor = tv_tensors.Mask(torch.from_numpy(mask).permute(2, 0, 1))

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def _composite(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Core fg/bg compositing logic."""
        h, w = self.output_size

        # Load and prepare background
        bg_idx = (idx + self._bg_offset[idx]) % len(self.bg_paths)
        self._bg_offset[idx] += 1
        bg = self._load_bg(bg_idx, (h, w))

        # Load foregrounds for this sample
        fgs = [self._load_fg(i) for i in self.characters_idx[idx]]

        # Composite foregrounds onto background
        image = bg.copy()
        label = np.zeros((h, w, 1), dtype=np.float32)

        for fg in fgs:
            fg_processed = self._process_fg(fg, (h, w))
            image, label = self._blend(image, label, fg_processed)

        # Binarize label
        label = (label > 0.5).astype(np.float32)

        return image, label

    def _load_bg(self, idx: int, output_size: tuple[int, int]) -> np.ndarray:
        """Load and prepare background image."""
        bg = cv2.imread(str(self.bg_paths[idx]), cv2.IMREAD_COLOR)
        if bg is None:
            msg = f"Failed to load background: {self.bg_paths[idx]}"
            raise FileNotFoundError(msg)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

        # Random crop to output aspect ratio, then resize
        h, w = bg.shape[:2]
        out_h, out_w = output_size
        r = min(h / out_h, w / out_w)
        crop_h, crop_w = int(out_h * r), int(out_w * r)

        # Random crop position
        top = self.rng.randint(0, max(0, h - crop_h))
        left = self.rng.randint(0, max(0, w - crop_w))
        bg = bg[top : top + crop_h, left : left + crop_w]

        # Resize to output size
        return cv2.resize(bg, (out_w, out_h))

    def _load_fg(self, idx: int) -> np.ndarray:
        """Load foreground image (RGBA)."""
        fg = cv2.imread(str(self.fg_paths[idx]), cv2.IMREAD_UNCHANGED)
        if fg is None:
            msg = f"Failed to load foreground: {self.fg_paths[idx]}"
            raise FileNotFoundError(msg)

        if fg.shape[2] != 4:
            msg = f"Foreground must have 4 channels (RGBA): {self.fg_paths[idx]}"
            raise ValueError(msg)

        return cv2.cvtColor(fg, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255

    def _process_fg(self, fg: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        """Process foreground: resize, center, and apply random transform."""
        h, w = output_size

        # Resize to fit output
        fg_h, fg_w = fg.shape[:2]
        r = min(h / fg_h, w / fg_w)
        new_h, new_w = int(fg_h * r), int(fg_w * r)
        fg = cv2.resize(fg, (new_w, new_h))

        # Center foreground based on alpha center of mass
        cy, cx = center_of_mass(fg[:, :, 3])
        if np.isnan(cy) or np.isnan(cx):
            cy, cx = new_h / 2, new_w / 2

        dx = w / 2 - cx
        dy = h / 2 - cy
        fg = cv2.warpAffine(
            fg,
            np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # Random scale, rotation, and translation
        scale = self.rng.uniform(0.2, 0.8)
        angle = self.rng.randint(-90, 90)
        trans_dx = self.rng.randint(-w // 3, w // 3)
        trans_dy = self.rng.randint(-h // 3, h // 3)

        trans_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        trans_mat[0][2] += trans_dx
        trans_mat[1][2] += trans_dy

        return cv2.warpAffine(
            fg,
            trans_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def _blend(
        self,
        image: np.ndarray,
        label: np.ndarray,
        fg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend foreground onto image and update label."""
        fg_rgb, fg_alpha = fg[:, :, :3], fg[:, :, 3:]

        # Soft edge blending
        mask = fg_alpha * cv2.blur(fg_alpha, (5, 5))[:, :, np.newaxis]

        # Composite
        image = mask * fg_rgb + (1 - mask) * image
        label = np.fmax(fg_alpha, label)

        return image, label
