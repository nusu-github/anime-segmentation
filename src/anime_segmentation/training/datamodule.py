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

import datasets as hf_datasets

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
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
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
            color_jitter_brightness: Brightness jitter range (0.0-1.0).
            color_jitter_contrast: Contrast jitter range (0.0-1.0).
            color_jitter_saturation: Saturation jitter range (0.0-1.0).
            color_jitter_hue: Hue jitter range (0.0-0.5).

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
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue

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
                color_jitter_brightness=self.color_jitter_brightness,
                color_jitter_contrast=self.color_jitter_contrast,
                color_jitter_saturation=self.color_jitter_saturation,
                color_jitter_hue=self.color_jitter_hue,
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


class HuggingFaceSegmentationDataset(Dataset):
    """PyTorch Dataset wrapping a Hugging Face datasets.Dataset for segmentation.

    This class bridges Hugging Face datasets with the existing training pipeline,
    providing the same interface as SegmentationDataset.
    """

    def __init__(
        self,
        hf_dataset: hf_datasets.Dataset,
        image_column: str = "image",
        mask_column: str = "mask",
        transform: PairedTransform | None = None,
        class_label: int = -1,
    ) -> None:
        """Initialize dataset.

        Args:
            hf_dataset: Hugging Face Dataset object.
            image_column: Column name for images.
            mask_column: Column name for masks.
            transform: Transform to apply to image/mask pairs.
            class_label: Class label for all samples (-1 if not using classification).

        """
        self.dataset = hf_dataset
        self.image_column = image_column
        self.mask_column = mask_column
        self.transform = transform
        self.class_label = class_label

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get a sample.

        Returns:
            Tuple of (image, mask, class_label):
                - image: [3, H, W] tensor, ImageNet normalized
                - mask: [1, H, W] tensor, 0-1 range
                - class_label: int (-1 if not using classification)

        """
        sample = self.dataset[index]

        # HF datasets returns PIL Images when using Image feature
        image = sample[self.image_column]
        mask = sample[self.mask_column]

        # Ensure correct format (handle both PIL Images and file paths)
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        if not isinstance(mask, Image.Image):
            mask = Image.open(mask).convert("L")
        else:
            mask = mask.convert("L")

        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # Default: just convert to tensor
            image = TF.to_tensor(image)
            image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
            mask = TF.to_tensor(mask)

        return image, mask, self.class_label


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

        # Augmentation settings
        self.hflip_prob = hflip_prob
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue

        # Datasets (initialized in setup)
        self.train_dataset: HuggingFaceSegmentationDataset | None = None
        self.val_dataset: HuggingFaceSegmentationDataset | None = None
        self.test_dataset: HuggingFaceSegmentationDataset | None = None

        # Raw HF datasets (for internal use)
        self._hf_dataset: hf_datasets.DatasetDict | None = None
        self._train_hf: hf_datasets.Dataset | None = None
        self._val_hf: hf_datasets.Dataset | None = None

    def _load_dataset(self) -> hf_datasets.DatasetDict:
        """Load dataset from local directory.

        Expects structure:
            {data_root}/
            ├── imgs/   # Images
            └── masks/  # Masks

        Returns:
            DatasetDict with train split.

        Raises:
            FileNotFoundError: If data_root or required directories don't exist.

        """
        data_root = Path(self.data_root)
        if not data_root.exists():
            msg = f"Data root does not exist: {data_root}"
            raise FileNotFoundError(msg)

        logger.info("Loading dataset from: %s", data_root)

        # Find image-mask pairs
        imgs_dir = data_root / "imgs"
        masks_dir = data_root / "masks"

        if not imgs_dir.exists() or not masks_dir.exists():
            msg = f"Expected 'imgs/' and 'masks/' directories in {data_root}"
            raise FileNotFoundError(msg)

        # Build pairs
        image_paths: list[str] = []
        mask_paths: list[str] = []

        # Build mask lookup map for efficiency
        mask_map: dict[str, Path] = {}
        for mask_path in masks_dir.iterdir():
            if mask_path.suffix.lower() in VALID_EXTENSIONS:
                mask_map[mask_path.stem] = mask_path

        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            # Find matching mask
            stem = img_path.stem
            if stem in mask_map:
                image_paths.append(str(img_path))
                mask_paths.append(str(mask_map[stem]))

        if not image_paths:
            msg = f"No image-mask pairs found in {data_root}"
            raise ValueError(msg)

        logger.info("Found %d image-mask pairs", len(image_paths))

        # Create HF Dataset with Image feature for automatic loading
        dataset = hf_datasets.Dataset.from_dict(
            {
                "image": image_paths,
                "mask": mask_paths,
            },
        )

        # Cast to Image type for automatic PIL Image loading
        dataset = dataset.cast_column("image", hf_datasets.Image())
        dataset = dataset.cast_column("mask", hf_datasets.Image())

        return hf_datasets.DatasetDict({"train": dataset})

    def _create_splits(
        self,
        dataset: hf_datasets.DatasetDict,
    ) -> tuple[hf_datasets.Dataset | None, hf_datasets.Dataset | None]:
        """Create train/val splits from loaded dataset.

        Args:
            dataset: Loaded DatasetDict.

        Returns:
            Tuple of (train_dataset, val_dataset).

        """
        train_ds: hf_datasets.Dataset | None = None
        val_ds: hf_datasets.Dataset | None = None

        # Get train split
        if "train" in dataset:
            train_ds = dataset["train"]
        else:
            # Use first available split
            first_split = next(iter(dataset.keys()))
            train_ds = dataset[first_split]
            logger.warning("Using '%s' as training split", first_split)

        # Handle validation split
        if "validation" in dataset:
            val_ds = dataset["validation"]
        elif "test" in dataset:
            # Use test as validation if no validation split
            val_ds = dataset["test"]
            logger.info("Using 'test' split as validation")
        elif self.val_ratio > 0 and train_ds is not None:
            # Auto-split from training data
            split = train_ds.train_test_split(
                test_size=self.val_ratio,
                seed=42,
            )
            train_ds = split["train"]
            val_ds = split["test"]
            logger.info(
                "Auto-split: %d train, %d validation samples",
                len(train_ds),
                len(val_ds),
            )

        return train_ds, val_ds

    def _create_torch_dataset(
        self,
        hf_dataset: hf_datasets.Dataset | None,
        *,
        is_train: bool,
    ) -> HuggingFaceSegmentationDataset | None:
        """Create PyTorch dataset from HF dataset.

        Args:
            hf_dataset: Hugging Face dataset.
            is_train: Whether to apply training augmentations.

        Returns:
            Wrapped PyTorch dataset or None.

        """
        if hf_dataset is None:
            return None

        if is_train:
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

        return HuggingFaceSegmentationDataset(
            hf_dataset=hf_dataset,
            image_column="image",
            mask_column="mask",
            transform=transform,
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.

        """
        if self._hf_dataset is None:
            # Load dataset from local directory
            self._hf_dataset = self._load_dataset()

            # Create splits
            self._train_hf, self._val_hf = self._create_splits(self._hf_dataset)

        match stage:
            case "fit" | None:
                self.train_dataset = self._create_torch_dataset(
                    self._train_hf,
                    is_train=True,
                )
                self.val_dataset = self._create_torch_dataset(
                    self._val_hf,
                    is_train=False,
                )
            case "validate":
                self.val_dataset = self._create_torch_dataset(
                    self._val_hf,
                    is_train=False,
                )
            case "test" | "predict":
                # Use validation as test if no separate test set
                self.test_dataset = self._create_torch_dataset(
                    self._val_hf,
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
