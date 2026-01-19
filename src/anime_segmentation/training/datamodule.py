"""LightningDataModule for anime segmentation training."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import kornia.augmentation as K
import lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from anime_segmentation.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_IMAGE_EXTENSIONS,
)
from anime_segmentation.exceptions import SynthesisValidationError
from anime_segmentation.training.config import (
    AugmentationConfig,
    DataLoaderConfig,
    SynthesisConfig,
)

if TYPE_CHECKING:
    from anime_segmentation.training.synthesis.base import (
        BaseBackgroundPool,
        BaseCompositor,
        BaseConsistencyPipeline,
        BaseDegradation,
        BaseForegroundPool,
        BaseInstanceTransform,
        BaseValidator,
    )

logger = logging.getLogger(__name__)


def _load_image_fast(path: str, mode: ImageReadMode = ImageReadMode.RGB) -> torch.Tensor:
    """Load image using fast torchvision.io decoder with PIL fallback.

    Uses torchvision.io.read_image for faster JPEG/PNG decoding (libjpeg-turbo).

    Args:
        path: Path to image file.
        mode: ImageReadMode (RGB or GRAY).

    Returns:
        Tensor [C, H, W] in uint8.

    """
    # Fast path: use torchvision.io (faster for JPEG due to libjpeg-turbo)
    return read_image(path, mode=mode)


class GPUAugmentation(torch.nn.Module):
    """GPU-accelerated augmentation using kornia.

    Applies augmentations on GPU for better performance.
    Processes image and mask pairs with synchronized geometric transforms.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        """Initialize GPU augmentation.

        Args:
            config: AugmentationConfig

        """
        super().__init__()
        self._config = config

        # Geometric augmentations (applied to both image and mask)
        self.geometric = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=config.hflip_prob),
            K.RandomRotation(
                degrees=config.rotation_degrees,
                p=1.0 if config.rotation_degrees > 0 else 0.0,
                align_corners=False,
            ),
            data_keys=["input", "mask"],
            same_on_batch=False,
        )

        if config.color_jitter:
            self.color = K.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
                hue=config.hue,
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
        normalize: bool = True,
    ) -> None:
        """Initialize transforms.

        Args:
            size: Target size (width, height). None for original size.
            normalize: Whether to apply ImageNet normalization.

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

        self.size = size
        self.normalize = normalize

        # Image normalization (applied after ToTensor)
        self._normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and mask.

        Args:
            image: PIL Image (RGB).
            mask: PIL Image (grayscale).

        Returns:
            Tuple of (image_tensor, mask_tensor).
            image: Tensor [3, H, W] float32.
            mask: Tensor [1, H, W] float32.
        """
        image_tensor = image.float() / 255.0
        mask_tensor = mask.float() / 255.0

        # Resize if size is specified (size is width, height)
        if self.size is not None:
            target_h, target_w = self.size[1], self.size[0]
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode="nearest",
            ).squeeze(0)

        # Ensure masks are strictly binary once at load time.
        mask_tensor = (mask_tensor > 0.5).float()

        # Normalize image
        if self.normalize:
            image_tensor = self._normalize(image_tensor)

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
            image = image.float() / 255.0
            image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
            mask = mask.float() / 255.0
            mask = (mask > 0.5).float()

        return image, mask, self.class_label


class SynthesisDataset(Dataset):
    """Dataset generating synthetic images via Copy-Paste composition.

    Each sample is generated on-the-fly using the compositor and optional
    consistency/degradation processing.
    """

    def __init__(
        self,
        compositor: BaseCompositor,
        size: tuple[int, int],
        length: int,
        consistency_pipeline: BaseConsistencyPipeline | None = None,
        degradation: BaseDegradation | None = None,
        validator: BaseValidator | None = None,
        normalize: bool = True,
        strict_validation: bool = False,
    ) -> None:
        """Initialize synthesis dataset.

        Args:
            compositor: CopyPasteCompositor instance.
            size: Target (height, width) for synthesized images.
            length: Number of samples per epoch.
            consistency_pipeline: Optional ConsistencyPipeline.
            degradation: Optional QualityDegradation module.
            validator: Optional DataValidator.
            normalize: Whether to apply ImageNet normalization.
            strict_validation: If True, raise exception on validation failure
                instead of just logging a warning.

        """
        self.compositor = compositor
        self.size = size
        self.length = length
        self.consistency_pipeline = consistency_pipeline
        self.degradation = degradation
        self.validator = validator
        self.normalize = normalize
        self.strict_validation = strict_validation

        # Normalization transform
        if normalize:
            self._normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        else:
            self._normalize = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Generate a synthetic sample.

        Returns:
            Tuple of (image, mask, class_label):
                - image: [3, H, W] tensor, optionally ImageNet normalized
                - mask: [1, H, W] tensor, 0-1 range
                - class_label: Number of characters (k)

        """
        # Generate synthetic image
        if self.consistency_pipeline is not None:
            image, mask, k, bg = self.compositor.synthesize(self.size, return_background=True)
        else:
            image, mask, k = self.compositor.synthesize(self.size)
            bg = None

        # Apply consistency processing
        if self.consistency_pipeline is not None and k > 0 and bg is not None:
            image = self.consistency_pipeline.apply(
                fg=image,
                bg=bg,
                mask=mask,
                canvas=image,
            )

        # Apply degradation
        if self.degradation is not None:
            image, mask = self.degradation(image, mask)

        # Ensure value range is valid before validation/normalization
        image = torch.clamp(image, 0.0, 1.0)
        mask = torch.clamp(mask, 0.0, 1.0)

        # Validate if validator is provided
        if self.validator is not None:
            result = self.validator.validate(image, mask)
            if not result.is_valid:
                if self.strict_validation:
                    msg = f"Synthesis validation failed: {result.errors}"
                    raise SynthesisValidationError(msg)
                logger.warning("Synthesis validation failed: %s", result.errors)

        # Normalize image
        if self._normalize is not None:
            image = self._normalize(image)

        return image, mask, k


class MixedDataset(Dataset):
    """Dataset mixing real and synthetic samples.

    Randomly selects from real or synthetic dataset based on synthesis_ratio.
    """

    def __init__(
        self,
        real_dataset: SegmentationDataset,
        synth_dataset: SynthesisDataset,
        synthesis_ratio: float = 0.5,
    ) -> None:
        """Initialize mixed dataset.

        Args:
            real_dataset: Dataset of real image-mask pairs.
            synth_dataset: Dataset generating synthetic samples.
            synthesis_ratio: Probability of returning synthetic sample.

        """
        if not 0.0 <= synthesis_ratio <= 1.0:
            msg = f"synthesis_ratio must be in [0, 1], got {synthesis_ratio}"
            raise ValueError(msg)

        self.real_dataset = real_dataset
        self.synth_dataset = synth_dataset
        self.synthesis_ratio = synthesis_ratio

    def __len__(self) -> int:
        if self.synthesis_ratio <= 0.0:
            return len(self.real_dataset)
        if self.synthesis_ratio >= 1.0:
            return len(self.synth_dataset)
        # Keep expected real samples per epoch roughly constant
        return math.ceil(len(self.real_dataset) / (1.0 - self.synthesis_ratio))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get a sample from either real or synthetic dataset.

        Returns:
            Tuple of (image, mask, class_label).

        """
        if torch.rand(1).item() < self.synthesis_ratio:
            # Synthetic sample (random index)
            synth_index = int(torch.randint(0, len(self.synth_dataset), (1,)).item())
            return self.synth_dataset[synth_index]
        # Real sample
        real_index = index % len(self.real_dataset)
        return self.real_dataset[real_index]


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
    """LightningDataModule for anime-segmentation training.

    Loads the local dataset layout (imgs/, masks/, fg/, bg/) and resolves
    synthesis components via dependency injection for testability and reuse.
    """

    def __init__(
        self,
        data_root: str = "datasets/anime-segmentation",
        size: tuple[int, int] = (1024, 1024),
        val_ratio: float = 0.1,
        loader: DataLoaderConfig | None = None,
        augmentation: AugmentationConfig | None = None,
        synthesis: SynthesisConfig | None = None,
        fg_pool: BaseForegroundPool | None = None,
        bg_pool: BaseBackgroundPool | None = None,
        compositor: BaseCompositor | None = None,
        instance_transform: BaseInstanceTransform | None = None,
        consistency_pipeline: BaseConsistencyPipeline | None = None,
        degradation: BaseDegradation | None = None,
        validator: BaseValidator | None = None,
    ) -> None:
        """Initialize the DataModule."""
        super().__init__()
        self.data_root = data_root
        self.size = size
        self.val_ratio = val_ratio
        self.loader = loader if loader is not None else DataLoaderConfig()
        self.augmentation = augmentation if augmentation is not None else AugmentationConfig()
        self.synthesis = synthesis if synthesis is not None else SynthesisConfig()

        self._injected_fg_pool = fg_pool
        self._injected_bg_pool = bg_pool
        self._injected_compositor = compositor
        self._injected_instance_transform = instance_transform
        self._injected_consistency = consistency_pipeline
        self._injected_degradation = degradation
        self._injected_validator = validator

        self._fg_pool: BaseForegroundPool | None = None
        self._bg_pool: BaseBackgroundPool | None = None
        self._compositor: BaseCompositor | None = None
        self._instance_transform: BaseInstanceTransform | None = None
        self._consistency_pipeline: BaseConsistencyPipeline | None = None
        self._degradation: BaseDegradation | None = None
        self._validator: BaseValidator | None = None

        self._gpu_augmentation: GPUAugmentation | None = None

        self.train_dataset: Dataset | MixedDataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self._image_paths: list[str] | None = None
        self._mask_paths: list[str] | None = None
        self._train_indices: list[int] | None = None
        self._val_indices: list[int] | None = None

        self.save_hyperparameters(
            ignore=[
                "fg_pool",
                "bg_pool",
                "compositor",
                "instance_transform",
                "consistency_pipeline",
                "degradation",
                "validator",
            ],
        )

    def _load_and_split_paths(self) -> None:
        """Load image-mask pairs and split them deterministically."""
        logger.info("Loading dataset from: %s", self.data_root)
        self._image_paths, self._mask_paths = _find_anime_segmentation_pairs(
            self.data_root,
        )
        logger.info("Found %d image-mask pairs", len(self._image_paths))

        n_samples = len(self._image_paths)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_samples, generator=generator).tolist()

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
        assert self._image_paths is not None
        assert self._mask_paths is not None

        image_paths = [self._image_paths[i] for i in indices]
        mask_paths = [self._mask_paths[i] for i in indices]

        normalize = not (is_train and self.augmentation.enabled)
        transform = PairedTransform(size=self.size, normalize=normalize)

        return SegmentationDataset(image_paths, mask_paths, transform=transform)

    def on_after_batch_transfer(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
        dataloader_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            return batch

        images, masks, class_labels = batch

        if self.trainer is not None and self.trainer.training and self.augmentation.enabled:
            augmentor = self._ensure_gpu_augmentation(images.device)
            images, masks = augmentor(images, masks)
            images = TF.normalize(images, IMAGENET_MEAN, IMAGENET_STD)

        return images, masks, class_labels

    def _ensure_gpu_augmentation(self, device: torch.device) -> GPUAugmentation:
        if self._gpu_augmentation is None:
            self._gpu_augmentation = GPUAugmentation(self.augmentation)
        self._gpu_augmentation = self._gpu_augmentation.to(device)
        return self._gpu_augmentation

    def _resolve_pools(self, fg_root: Path, bg_root: Path) -> None:
        if self._fg_pool is None:
            if self._injected_fg_pool is not None:
                self._fg_pool = self._injected_fg_pool
            else:
                self._fg_pool = self._build_default_fg_pool(fg_root)
        if self._bg_pool is None:
            if self._injected_bg_pool is not None:
                self._bg_pool = self._injected_bg_pool
            else:
                self._bg_pool = self._build_default_bg_pool(bg_root)

    def _resolve_instance_transform(self) -> None:
        if self._instance_transform is None:
            if self._injected_instance_transform is not None:
                self._instance_transform = self._injected_instance_transform
            else:
                self._instance_transform = self._build_default_instance_transform()

    def _resolve_compositor(self) -> None:
        if self._compositor is None:
            if self._injected_compositor is not None:
                self._compositor = self._injected_compositor
            else:
                self._compositor = self._build_default_compositor()

    def _resolve_consistency(self) -> None:
        if self._consistency_pipeline is None and self.synthesis.consistency.enabled:
            if self._injected_consistency is not None:
                self._consistency_pipeline = self._injected_consistency
            else:
                self._consistency_pipeline = self._build_default_consistency()

    def _resolve_degradation(self) -> None:
        if self._degradation is None and self.synthesis.degradation.enabled:
            if self._injected_degradation is not None:
                self._degradation = self._injected_degradation
            else:
                self._degradation = self._build_default_degradation()

    def _resolve_validator(self) -> None:
        if self._validator is None:
            if self._injected_validator is not None:
                self._validator = self._injected_validator
            else:
                self._validator = self._build_default_validator()

    def _build_default_fg_pool(self, fg_root: Path) -> BaseForegroundPool:
        from anime_segmentation.data.pools import ForegroundPool

        return ForegroundPool(fg_root)

    def _build_default_bg_pool(self, bg_root: Path) -> BaseBackgroundPool:
        from anime_segmentation.data.pools import BackgroundPool

        return BackgroundPool(bg_root)

    def _build_default_compositor(self) -> BaseCompositor:
        from .synthesis.compositor import CopyPasteCompositor

        if self._fg_pool is None or self._bg_pool is None:
            msg = "Pools must be resolved before building compositor"
            raise RuntimeError(msg)

        return CopyPasteCompositor(
            fg_pool=self._fg_pool,
            bg_pool=self._bg_pool,
            config=self.synthesis.compositor,
            instance_transform=self._instance_transform,
        )

    def _build_default_instance_transform(self) -> BaseInstanceTransform:
        from .synthesis.transforms import InstanceTransform

        cfg = self.augmentation
        return InstanceTransform(
            hflip_prob=cfg.hflip_prob,
            rotation_range=(-cfg.rotation_degrees, cfg.rotation_degrees),
            scale_range=(0.5, 1.5),
        )

    def _build_default_consistency(self) -> BaseConsistencyPipeline:
        from .synthesis.consistency import ConsistencyPipeline

        cfg = self.synthesis.consistency
        return ConsistencyPipeline(
            color_tone_prob=cfg.color_prob,
            light_wrap_prob=cfg.light_wrap_prob,
            shadow_prob=cfg.shadow_prob,
            noise_grain_prob=cfg.noise_prob,
        )

    def _build_default_degradation(self) -> BaseDegradation:
        from .synthesis.degradation import QualityDegradation

        cfg = self.synthesis.degradation
        return QualityDegradation(
            jpeg_prob=cfg.jpeg_prob,
            blur_prob=cfg.blur_prob,
            noise_prob=cfg.noise_prob,
        )

    def _build_default_validator(self) -> BaseValidator:
        from .synthesis.validation import DataValidator

        return DataValidator()

    def _setup_synthesis_components(self) -> None:
        if not self.synthesis.enabled:
            return

        fg_root = Path(self.data_root) / "fg"
        bg_root = Path(self.data_root) / "bg"

        if not fg_root.exists() or not bg_root.exists():
            logger.warning(
                "Synthesis enabled but fg/ or bg/ directories missing in %s. Disabling synthesis.",
                self.data_root,
            )
            self.synthesis.enabled = False
            return

        self._resolve_pools(fg_root, bg_root)
        self._resolve_instance_transform()
        self._resolve_compositor()
        self._resolve_consistency()
        self._resolve_degradation()
        self._resolve_validator()

        logger.info(
            "Synthesis pipeline initialized: fg=%d, bg=%d",
            len(self._fg_pool),
            len(self._bg_pool),
        )

    def setup(self, stage: str | None = None) -> None:
        if self._image_paths is None:
            self._load_and_split_paths()

        assert self._train_indices is not None
        assert self._val_indices is not None

        if self.synthesis.enabled and self._compositor is None:
            self._setup_synthesis_components()

        match stage:
            case "fit" | None:
                real_train_dataset = self._create_dataset(
                    self._train_indices,
                    is_train=True,
                )

                if self.synthesis.enabled and self._compositor is not None:
                    synth_dataset = SynthesisDataset(
                        compositor=self._compositor,
                        size=self.size,
                        length=self.synthesis.length,
                        consistency_pipeline=self._consistency_pipeline,
                        degradation=self._degradation,
                        validator=self._validator,
                        normalize=not self.augmentation.enabled,
                        strict_validation=self.synthesis.strict_validation,
                    )
                    self.train_dataset = MixedDataset(
                        real_dataset=real_train_dataset,
                        synth_dataset=synth_dataset,
                        synthesis_ratio=self.synthesis.ratio,
                    )
                    logger.info(
                        "Mixed dataset created: real=%d, synth=%d, ratio=%.2f",
                        len(real_train_dataset),
                        len(synth_dataset),
                        self.synthesis.ratio,
                    )
                else:
                    self.train_dataset = real_train_dataset

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
        num_workers = self.loader.num_workers
        prefetch = self.loader.prefetch_factor if num_workers > 0 else None

        return DataLoader(
            dataset,
            batch_size=self.loader.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.loader.pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            msg = "Training dataset not initialized. Call setup('fit') first."
            raise RuntimeError(msg)
        return self._create_dataloader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            msg = "Validation dataset not available."
            raise RuntimeError(msg)
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            msg = "Test dataset not available. Call setup('test') first."
            raise RuntimeError(msg)
        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
