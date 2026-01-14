"""LightningDataModule for BiRefNet training.

Supports BiRefNet-style dataset structure:
    {data_root}/{dataset_name}/im/*.png  (images)
    {data_root}/{dataset_name}/gt/*.png  (ground truth masks)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import lightning as L
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # Remove DecompressionBombWarning

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def _find_image_label_pairs(
    data_root: str | Path,
    dataset_names: list[str],
) -> tuple[list[str], list[str]]:
    """Find matching image and label pairs from BiRefNet-style datasets.

    Args:
        data_root: Root directory containing datasets.
        dataset_names: List of dataset folder names to include.

    Returns:
        Tuple of (image_paths, label_paths).

    Raises:
        FileNotFoundError: If data_root does not exist.
        ValueError: If dataset_names is empty.

    """
    if not dataset_names:
        msg = "dataset_names cannot be empty"
        raise ValueError(msg)

    data_root = Path(data_root)
    if not data_root.exists():
        msg = f"Data root directory does not exist: {data_root}"
        raise FileNotFoundError(msg)

    image_paths: list[str] = []
    label_paths: list[str] = []

    for dataset_name in dataset_names:
        image_dir = data_root / dataset_name / "im"
        label_dir = data_root / dataset_name / "gt"

        if not image_dir.exists():
            continue

        # Optimize: Scan label directory once to build a lookup map
        # key: stem, value: full_path
        label_map = {}
        if label_dir.exists():
            for label_path in label_dir.iterdir():
                if label_path.suffix in VALID_EXTENSIONS:
                    label_map[label_path.stem] = str(label_path)

        for img_path in image_dir.iterdir():
            if img_path.suffix not in VALID_EXTENSIONS:
                continue

            stem = img_path.stem
            # Check if matching label exists in map
            if stem in label_map:
                image_paths.append(str(img_path))
                label_paths.append(label_map[stem])

    return image_paths, label_paths


class PairedTransform:
    """Apply synchronized transforms to image and mask pairs."""

    def __init__(
        self,
        size: tuple[int, int] | None = None,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        color_jitter: bool = True,
    ) -> None:
        """Initialize transforms.

        Args:
            size: Target size (width, height). None for original size.
            is_train: Whether this is for training (enables augmentation).
            hflip_prob: Probability of horizontal flip.
            rotation_degrees: Max rotation angle.
            color_jitter: Whether to apply color jitter to image.

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
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
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
    """Dataset for segmentation with BiRefNet-style structure."""

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
        # Load image and mask
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.label_paths[index]).convert("L")

        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # Default: just convert to tensor
            image = TF.to_tensor(image)
            image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
            mask = TF.to_tensor(mask)

        return image, mask, self.class_label


class BiRefNetDataModule(L.LightningDataModule):
    """LightningDataModule for BiRefNet training."""

    def __init__(
        self,
        data_root: str,
        training_sets: list[str],
        validation_sets: list[str] | None = None,
        test_sets: list[str] | None = None,
        batch_size: int = 8,
        num_workers: int | None = None,
        size: tuple[int, int] = (1024, 1024),
        pin_memory: bool = True,
        # Augmentation settings
        hflip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        color_jitter: bool = True,
    ) -> None:
        """Initialize DataModule.

        Args:
            data_root: Root directory containing dataset folders.
            training_sets: List of dataset names for training.
            validation_sets: List of dataset names for validation.
            test_sets: List of dataset names for testing.
            batch_size: Batch size for all dataloaders.
            num_workers: Number of workers for dataloaders. Defaults to 4.
            size: Target image size (width, height).
            pin_memory: Whether to pin memory in dataloaders.
            hflip_prob: Probability of horizontal flip during training.
            rotation_degrees: Max rotation angle during training.
            color_jitter: Whether to apply color jitter during training.

        """
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.training_sets = training_sets
        self.validation_sets = validation_sets or []
        self.test_sets = test_sets or []
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else 4
        self.size = size
        self.pin_memory = pin_memory

        # Augmentation settings
        self.hflip_prob = hflip_prob
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter

        # Will be set in setup()
        self.train_dataset: SegmentationDataset | None = None
        self.val_dataset: SegmentationDataset | None = None
        self.test_dataset: SegmentationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.

        Raises:
            ValueError: If no training samples found for 'fit' stage.

        """
        match stage:
            case "fit" | None:
                self.train_dataset = self._setup_dataset(
                    "train",
                    self.training_sets,
                    is_train=True,
                    require_data=True,
                )
                self.val_dataset = self._setup_dataset("val", self.validation_sets, is_train=False)
            case "validate":
                self.val_dataset = self._setup_dataset("val", self.validation_sets, is_train=False)
            case "test" | "predict":
                self.test_dataset = self._setup_dataset("test", self.test_sets, is_train=False)

    def _setup_dataset(
        self,
        dataset_type: Literal["train", "val", "test"],
        dataset_sets: list[str],
        *,
        is_train: bool,
        require_data: bool = False,
    ) -> SegmentationDataset | None:
        """Set up a dataset of the specified type.

        Args:
            dataset_type: Type of dataset ('train', 'val', or 'test').
            dataset_sets: List of dataset folder names.
            is_train: Whether to apply training augmentations.
            require_data: If True, raise ValueError when no samples found.

        Returns:
            Configured SegmentationDataset or None if no samples found.

        Raises:
            ValueError: If require_data=True and no samples found.

        """
        if not dataset_sets:
            return None

        images, labels = _find_image_label_pairs(self.data_root, dataset_sets)

        if not images:
            if require_data:
                msg = f"No {dataset_type} samples found in {self.data_root} for datasets: {dataset_sets}"
                raise ValueError(msg)
            logger.warning("No %s samples found for datasets: %s", dataset_type, dataset_sets)
            return None

        logger.info("Found %d %s samples", len(images), dataset_type)

        if is_train:
            transform = PairedTransform(
                size=self.size,
                is_train=True,
                hflip_prob=self.hflip_prob,
                rotation_degrees=self.rotation_degrees,
                color_jitter=self.color_jitter,
            )
        else:
            transform = PairedTransform(size=self.size, is_train=False)

        return SegmentationDataset(images, labels, transform=transform)

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
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
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
            if not self.validation_sets:
                msg = "No validation_sets configured in DataModule."
            else:
                msg = "Validation dataset not initialized. Call setup('fit') first."
            raise RuntimeError(msg)

        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader.

        Raises:
            RuntimeError: If test dataset is not available.

        """
        if self.test_dataset is None:
            if not self.test_sets:
                msg = "No test_sets configured in DataModule."
            else:
                msg = "Test dataset not initialized. Call setup('test') first."
            raise RuntimeError(msg)

        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        """Return predict dataloader (same as test).

        Raises:
            RuntimeError: If test dataset is not available.

        """
        return self.test_dataloader()
