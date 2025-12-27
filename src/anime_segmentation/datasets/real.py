"""Dataset for real (pre-composited) anime images with masks."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class RealImageDataset(Dataset):
    """Dataset for pre-composited real images with masks.

    Loads image/mask pairs from disk and applies transforms.
    Optionally crops edge pixels to handle dataset artifacts.

    Args:
        image_paths: List of paths to input images.
        mask_paths: List of paths to corresponding masks.
        transform: Optional transform to apply.
        edge_crop: Number of pixels to crop from each edge. Default is 10.
        mask_threshold: Threshold for binarizing mask. Default is 0.3.
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        transform: v2.Transform | None = None,
        edge_crop: int = 10,
        mask_threshold: float = 0.3,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            msg = f"Number of images ({len(image_paths)}) != number of masks ({len(mask_paths)})"
            raise ValueError(msg)

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.edge_crop = edge_crop
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        # Load image (BGR -> RGB)
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Failed to load image: {self.image_paths[idx]}"
            raise FileNotFoundError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"Failed to load mask: {self.mask_paths[idx]}"
            raise FileNotFoundError(msg)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255
        mask = mask.astype(np.float32) / 255

        # Binarize mask
        mask = (mask > self.mask_threshold).astype(np.float32)

        # Crop edges (handles dataset artifacts)
        if self.edge_crop > 0:
            c = self.edge_crop
            image = image[c:-c, c:-c]
            mask = mask[c:-c, c:-c]

        # Convert to tv_tensors (HWC -> CHW)
        image_tensor = tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1))
        mask_tensor = tv_tensors.Mask(torch.from_numpy(mask).unsqueeze(0))

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
