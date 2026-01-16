"""LightningDataModule for anime segmentation training."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import kornia.augmentation as K
import lightning as L
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from anime_segmentation.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # Remove DecompressionBombWarning


def _load_image_fast(path: str, mode: ImageReadMode = ImageReadMode.RGB) -> Image.Image:
    """Load image using fast torchvision.io decoder with PIL fallback.

    Uses torchvision.io.read_image for faster JPEG/PNG decoding (libjpeg-turbo),
    falling back to PIL for unsupported formats.

    Args:
        path: Path to image file.
        mode: ImageReadMode (RGB or GRAY).

    Returns:
        PIL Image in the requested mode.

    """
    try:
        # Fast path: use torchvision.io (faster for JPEG due to libjpeg-turbo)
        tensor = read_image(path, mode=mode)
        # Convert tensor [C, H, W] uint8 -> PIL Image
        if mode == ImageReadMode.GRAY:
            return TF.to_pil_image(tensor, mode="L")
        return TF.to_pil_image(tensor, mode="RGB")
    except Exception:
        # Fallback to PIL for unsupported formats or errors
        if mode == ImageReadMode.GRAY:
            return Image.open(path).convert("L")
        return Image.open(path).convert("RGB")


class GPUAugmentation(torch.nn.Module):
    """GPU-accelerated augmentation using kornia.

    Applies augmentations on GPU for better performance.
    Processes image and mask pairs with synchronized geometric transforms.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        color_jitter: bool = True,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
    ) -> None:
        """Initialize GPU augmentation.

        Args:
            hflip_prob: Probability of horizontal flip.
            rotation_degrees: Max rotation angle.
            color_jitter: Whether to apply color jitter to image.
            color_jitter_brightness: Brightness jitter range.
            color_jitter_contrast: Contrast jitter range.
            color_jitter_saturation: Saturation jitter range.
            color_jitter_hue: Hue jitter range.

        """
        super().__init__()

        # Geometric augmentations (applied to both image and mask)
        self.geometric = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=hflip_prob),
            K.RandomRotation(degrees=rotation_degrees, p=1.0 if rotation_degrees > 0 else 0.0),
            data_keys=["input", "mask"],
            same_on_batch=False,
        )

        # Color augmentations (applied only to image)
        if color_jitter:
            self.color = K.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
                p=1.0,
            )
        else:
            self.color = None

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to batch.

        Args:
            images: [B, 3, H, W] tensor, ImageNet normalized.
            masks: [B, 1, H, W] tensor, 0-1 range.

        Returns:
            Tuple of (augmented_images, augmented_masks).

        """
        # Apply geometric transforms to both
        augmented = self.geometric(images, masks)
        images = augmented[0]
        masks = augmented[1]

        # Apply color jitter to images only
        if self.color is not None:
            images = self.color(images)

        return images, masks


class PairedTransform:
    """Apply synchronized transforms to image and mask pairs."""

    def __init__(
        self,
        size: tuple[int, int] | None = None,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        color_jitter: bool = True,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
    ) -> None:
        """Initialize transforms.

        Args:
            size: Target size (width, height). None for original size.
            is_train: Whether this is for training (enables augmentation).
            hflip_prob: Probability of horizontal flip.
            rotation_degrees: Max rotation angle.
            color_jitter: Whether to apply color jitter to image.
            color_jitter_brightness: Brightness jitter range.
            color_jitter_contrast: Contrast jitter range.
            color_jitter_saturation: Saturation jitter range.
            color_jitter_hue: Hue jitter range.

        Raises:
            ValueError: If parameters are invalid.

        """
        # Input validation (fail-fast)
        if size is not None:
            if len(size) != 2:
                msg = f"size must be (width, height), got {size}"
                raise ValueError(msg)
            if size[0] <= 0 or size[1] <= 0:
                msg = f"size dimensions must be positive, got {size}"
                raise ValueError(msg)
        if not 0.0 <= hflip_prob <= 1.0:
            msg = f"hflip_prob must be in [0, 1], got {hflip_prob}"
            raise ValueError(msg)
        if rotation_degrees < 0:
            msg = f"rotation_degrees must be non-negative, got {rotation_degrees}"
            raise ValueError(msg)

        self.size = size
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter

        # Image normalization (applied after ToTensor)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

        # Color jitter for image only
        if color_jitter and is_train:
            self.jitter = transforms.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
            )
        else:
            self.jitter = None

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and mask.

        Args:
            image: PIL Image (RGB).
            mask: PIL Image (grayscale).

        Returns:
            Tuple of (image_tensor, mask_tensor).

        """
        # Resize if size is specified
        if self.size is not None:
            image = image.resize(self.size, resample=Image.Resampling.BILINEAR)
            mask = mask.resize(self.size, resample=Image.Resampling.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() < self.hflip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Random rotation
            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = image.rotate(angle)
                mask = mask.rotate(angle)

            # Color jitter (image only)
            if self.jitter is not None:
                image = self.jitter(image)

        # Convert to tensor
        image_tensor = TF.to_tensor(image)  # [3, H, W], 0-1
        mask_tensor = TF.to_tensor(mask)  # [1, H, W], 0-1

        # Normalize image
        image_tensor = self.normalize(image_tensor)

        return image_tensor, mask_tensor


class SegmentationDataset(Dataset):
    """Dataset for image segmentation with image-mask pairs."""

    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str],
        transform: PairedTransform | None = None,
        class_label: int = -1,
    ) -> None:
        """Initialize dataset.

        Args:
            image_paths: List of image file paths.
            label_paths: List of corresponding label file paths.
            transform: Transform to apply to image/mask pairs.
            class_label: Class label for all samples (-1 if not using classification).

        """
        if len(image_paths) != len(label_paths):
            msg = f"Number of images ({len(image_paths)}) != number of labels ({len(label_paths)})"
            raise ValueError(msg)

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.class_label = class_label

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get a sample.

        Returns:
            Tuple of (image, mask, class_label):
                - image: [3, H, W] tensor, ImageNet normalized
                - mask: [1, H, W] tensor, 0-1 range
                - class_label: int (-1 if not using classification)

        """
        # Load image and mask using fast decoder
        image = _load_image_fast(self.image_paths[index], ImageReadMode.RGB)
        mask = _load_image_fast(self.label_paths[index], ImageReadMode.GRAY)

        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # Default: just convert to tensor
            image = TF.to_tensor(image)
            image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
            mask = TF.to_tensor(mask)

        return image, mask, self.class_label


def _find_anime_segmentation_pairs(
    data_root: str | Path,
) -> tuple[list[str], list[str]]:
    """Find image-mask pairs for anime-segmentation dataset.

    Expects structure:
        {data_root}/
        ├── imgs/   # Images
        └── masks/  # Masks

    Args:
        data_root: Root directory containing imgs/ and masks/.

    Returns:
        Tuple of (image_paths, mask_paths).

    Raises:
        FileNotFoundError: If data_root or required directories don't exist.
        ValueError: If no image-mask pairs found.

    """
    data_root = Path(data_root)
    if not data_root.exists():
        msg = f"Data root does not exist: {data_root}"
        raise FileNotFoundError(msg)

    imgs_dir = data_root / "imgs"
    masks_dir = data_root / "masks"

    if not imgs_dir.exists() or not masks_dir.exists():
        msg = f"Expected 'imgs/' and 'masks/' directories in {data_root}"
        raise FileNotFoundError(msg)

    # Build mask lookup map for efficiency
    mask_map: dict[str, Path] = {}
    for mask_path in masks_dir.iterdir():
        if mask_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            mask_map[mask_path.stem] = mask_path

    image_paths: list[str] = []
    mask_paths: list[str] = []

    for img_path in sorted(imgs_dir.iterdir()):
        if img_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        stem = img_path.stem
        if stem in mask_map:
            image_paths.append(str(img_path))
            mask_paths.append(str(mask_map[stem]))

    if not image_paths:
        msg = f"No image-mask pairs found in {data_root}"
        raise ValueError(msg)

    return image_paths, mask_paths


class AnimeSegmentationDataModule(L.LightningDataModule):
    """LightningDataModule for anime-segmentation dataset.

    Loads dataset from local files. Use scripts/download_anime_segmentation.py
    to download the dataset from Hugging Face Hub first.

    Dataset structure:
        {data_root}/
        ├── bg/       # Background images (8,057 JPG) - for future Copy-Paste mode
        ├── fg/       # Character cutouts (11,802 PNG with alpha) - for future Copy-Paste mode
        ├── imgs/     # Pre-composed images (1,111 JPG)
        └── masks/    # Segmentation masks (1,111 JPG)

    Currently implements Path A (Pre-composed mode) using imgs/ + masks/.
    Path B (Copy-Paste mode using bg/ + fg/) is planned for future implementation.
    """

    def __init__(
        self,
        # Data source
        data_root: str = "datasets/anime-segmentation",
        # Split configuration
        val_ratio: float = 0.1,
        # DataLoader settings
        batch_size: int = 8,
        num_workers: int | None = None,
        size: tuple[int, int] = (1024, 1024),
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        gpu_augmentation: bool = False,
        # Augmentation settings
        hflip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        color_jitter: bool = True,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
    ) -> None:
        """Initialize AnimeSegmentationDataModule.

        Args:
            data_root: Path to local dataset directory containing imgs/ and masks/.
            val_ratio: Ratio for validation split when auto-splitting (0.0-1.0).
            batch_size: Batch size for all dataloaders.
            num_workers: Number of workers for dataloaders. Defaults to 4.
            size: Target image size (width, height).
            pin_memory: Whether to pin memory in dataloaders.
            gpu_augmentation: If True, apply augmentations on GPU (faster).
            hflip_prob: Probability of horizontal flip during training.
            rotation_degrees: Max rotation angle during training.
            color_jitter: Whether to apply color jitter during training.
            color_jitter_brightness: Brightness jitter range (0.0-1.0).
            color_jitter_contrast: Contrast jitter range (0.0-1.0).
            color_jitter_saturation: Saturation jitter range (0.0-1.0).
            color_jitter_hue: Hue jitter range (0.0-0.5).

        """
        super().__init__()
        self.save_hyperparameters()

        # Data source configuration
        self.data_root = data_root

        # Split configuration
        self.val_ratio = val_ratio

        # DataLoader settings
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else 4
        self.size = size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.gpu_augmentation = gpu_augmentation

        # Augmentation settings
        self.hflip_prob = hflip_prob
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue

        # GPU augmentation module (initialized lazily)
        self._gpu_augmentation: GPUAugmentation | None = None

        # Datasets (initialized in setup)
        self.train_dataset: SegmentationDataset | None = None
        self.val_dataset: SegmentationDataset | None = None
        self.test_dataset: SegmentationDataset | None = None

        # Raw paths (loaded once, then split)
        self._image_paths: list[str] | None = None
        self._mask_paths: list[str] | None = None
        self._train_indices: list[int] | None = None
        self._val_indices: list[int] | None = None

    def _load_and_split_paths(self) -> None:
        """Load image-mask paths and create train/val splits.

        Raises:
            FileNotFoundError: If data_root or required directories don't exist.
            ValueError: If no image-mask pairs found.

        """
        logger.info("Loading dataset from: %s", self.data_root)

        # Find image-mask pairs
        self._image_paths, self._mask_paths = _find_anime_segmentation_pairs(
            self.data_root,
        )
        logger.info("Found %d image-mask pairs", len(self._image_paths))

        # Create train/val split indices
        n_samples = len(self._image_paths)
        indices = list(range(n_samples))

        # Shuffle with fixed seed for reproducibility
        rng = random.Random(42)
        rng.shuffle(indices)

        # Split
        n_val = int(n_samples * self.val_ratio)
        self._val_indices = indices[:n_val]
        self._train_indices = indices[n_val:]

        logger.info(
            "Split: %d train, %d validation samples",
            len(self._train_indices),
            len(self._val_indices),
        )

    def _create_dataset(
        self,
        indices: list[int],
        *,
        is_train: bool,
    ) -> SegmentationDataset:
        """Create SegmentationDataset from indices.

        Args:
            indices: List of indices to include.
            is_train: Whether to apply training augmentations.

        Returns:
            Configured SegmentationDataset.

        """
        assert self._image_paths is not None
        assert self._mask_paths is not None

        image_paths = [self._image_paths[i] for i in indices]
        mask_paths = [self._mask_paths[i] for i in indices]

        # When gpu_augmentation is enabled, skip CPU augmentation for training
        # (augmentation will be applied in on_after_batch_transfer)
        if is_train and self.gpu_augmentation:
            # Only resize and normalize, no augmentation
            transform = PairedTransform(size=self.size, is_train=False)
        elif is_train:
            transform = PairedTransform(
                size=self.size,
                is_train=True,
                hflip_prob=self.hflip_prob,
                rotation_degrees=self.rotation_degrees,
                color_jitter=self.color_jitter,
                color_jitter_brightness=self.color_jitter_brightness,
                color_jitter_contrast=self.color_jitter_contrast,
                color_jitter_saturation=self.color_jitter_saturation,
                color_jitter_hue=self.color_jitter_hue,
            )
        else:
            transform = PairedTransform(size=self.size, is_train=False)

        return SegmentationDataset(image_paths, mask_paths, transform=transform)

    def on_after_batch_transfer(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataloader_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply GPU augmentation after batch transfer.

        This hook is called after the batch is moved to the device.
        When gpu_augmentation is enabled, applies kornia-based augmentations.

        Args:
            batch: Tuple of (images, masks, class_labels).
            dataloader_idx: Index of the dataloader.

        Returns:
            Augmented batch.

        """
        images, masks, class_labels = batch

        # Only apply GPU augmentation during training
        if self.trainer is not None and self.trainer.training and self.gpu_augmentation:
            # Lazy initialization of GPU augmentation module
            if self._gpu_augmentation is None:
                self._gpu_augmentation = GPUAugmentation(
                    hflip_prob=self.hflip_prob,
                    rotation_degrees=self.rotation_degrees,
                    color_jitter=self.color_jitter,
                    color_jitter_brightness=self.color_jitter_brightness,
                    color_jitter_contrast=self.color_jitter_contrast,
                    color_jitter_saturation=self.color_jitter_saturation,
                    color_jitter_hue=self.color_jitter_hue,
                )
                # Move to same device as batch
                self._gpu_augmentation = self._gpu_augmentation.to(images.device)

            images, masks = self._gpu_augmentation(images, masks)

        return images, masks, class_labels

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.

        """
        if self._image_paths is None:
            self._load_and_split_paths()

        assert self._train_indices is not None
        assert self._val_indices is not None

        match stage:
            case "fit" | None:
                self.train_dataset = self._create_dataset(
                    self._train_indices,
                    is_train=True,
                )
                self.val_dataset = self._create_dataset(
                    self._val_indices,
                    is_train=False,
                )
            case "validate":
                self.val_dataset = self._create_dataset(
                    self._val_indices,
                    is_train=False,
                )
            case "test" | "predict":
                # Use validation as test if no separate test set
                self.test_dataset = self._create_dataset(
                    self._val_indices,
                    is_train=False,
                )

    def _create_dataloader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """Create a dataloader with common settings.

        Args:
            dataset: Dataset to wrap.
            shuffle: Whether to shuffle the data.
            drop_last: Whether to drop the last incomplete batch.

        Returns:
            Configured DataLoader instance.

        """
        # prefetch_factor requires num_workers > 0
        prefetch = self.prefetch_factor if self.num_workers > 0 else None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=prefetch,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            msg = "Training dataset not initialized. Call setup('fit') first."
            raise RuntimeError(msg)

        return self._create_dataloader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader.

        Raises:
            RuntimeError: If validation dataset is not available.

        """
        if self.val_dataset is None:
            msg = "Validation dataset not available."
            raise RuntimeError(msg)

        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader.

        Raises:
            RuntimeError: If test dataset is not available.

        """
        if self.test_dataset is None:
            msg = "Test dataset not available. Call setup('test') first."
            raise RuntimeError(msg)

        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        """Return predict dataloader (same as test).

        Raises:
            RuntimeError: If test dataset is not available.

        """
        return self.test_dataloader()
