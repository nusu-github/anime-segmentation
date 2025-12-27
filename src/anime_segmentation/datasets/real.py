"""Dataset for real (pre-composited) anime images with masks."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_dtype


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
        # Load image (RGB, CHW, uint8)
        image = decode_image(str(self.image_paths[idx]), mode=ImageReadMode.RGB)

        # Load mask (grayscale, 1HW, uint8)
        mask = decode_image(str(self.mask_paths[idx]), mode=ImageReadMode.GRAY)

        # Normalize to [0, 1] float32
        image = to_dtype(image, torch.float32, scale=True)
        mask = to_dtype(mask, torch.float32, scale=True)

        # Binarize mask
        mask = (mask > self.mask_threshold).float()

        # Crop edges (handles dataset artifacts)
        if self.edge_crop > 0:
            c = self.edge_crop
            image = image[:, c:-c, c:-c]
            mask = mask[:, c:-c, c:-c]

        # Wrap in tv_tensors
        image_tensor = tv_tensors.Image(image)
        mask_tensor = tv_tensors.Mask(mask)

        if self.transform:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
