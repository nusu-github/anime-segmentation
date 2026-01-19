"""LightningDataModule for anime segmentation training."""

from __future__ import annotations

import logging
import math
from pathlib import Path

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
            K.RandomRotation(
                degrees=rotation_degrees,
                p=1.0 if rotation_degrees > 0 else 0.0,
                align_corners=False,
            ),
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
        compositor,  # CopyPasteCompositor
        size: tuple[int, int],
        length: int,
        consistency_pipeline=None,  # ConsistencyPipeline
        degradation=None,  # QualityDegradation
        validator=None,  # DataValidator
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
                    raise SynthesisValidationError(
                        f"Synthesis validation failed: {result.errors}"
                    )
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
        gpu_augmentation: bool = True,
        # Augmentation settings
        hflip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        color_jitter: bool = True,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
        # Route B (Synthesis) settings
        enable_synthesis: bool = False,
        synthesis_ratio: float = 0.5,
        synthesis_length: int = 1000,
        # Synthesis compositor settings
        synthesis_k_probs: dict[int, float] | None = None,
        synthesis_min_area: float = 0.02,
        synthesis_max_area: float = 0.60,
        synthesis_max_coverage: float = 0.85,
        synthesis_max_overlap: float = 0.30,
        synthesis_blending_probs: dict[str, float] | None = None,
        synthesis_boundary_randomize_prob: float = 0.3,
        synthesis_boundary_randomize_width: int = 3,
        synthesis_boundary_randomize_noise_std: float = 0.05,
        # Consistency settings
        enable_consistency: bool = True,
        consistency_color_prob: float = 0.5,
        consistency_light_wrap_prob: float = 0.3,
        consistency_shadow_prob: float = 0.3,
        consistency_noise_prob: float = 0.3,
        # Degradation settings
        enable_degradation: bool = True,
        degradation_jpeg_prob: float = 0.3,
        degradation_blur_prob: float = 0.1,
        degradation_noise_prob: float = 0.1,
        # Validation settings
        enable_validation: bool = True,
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
            enable_synthesis: Enable Route B (Copy-Paste synthesis).
            synthesis_ratio: Ratio of synthetic samples in training.
            synthesis_length: Number of synthetic samples per epoch.
            synthesis_k_probs: Probability distribution for character count.
            synthesis_min_area: Minimum character area ratio.
            synthesis_max_area: Maximum character area ratio.
            synthesis_max_coverage: Maximum total coverage.
            synthesis_max_overlap: Maximum IoU overlap between characters.
            synthesis_blending_probs: Blending strategy probabilities.
            synthesis_boundary_randomize_prob: Probability of boundary RGB randomization.
            synthesis_boundary_randomize_width: Boundary width for randomization.
            synthesis_boundary_randomize_noise_std: Noise std for boundary randomization.
            enable_consistency: Enable consistency processing.
            consistency_color_prob: Color matching probability.
            consistency_light_wrap_prob: Light wrap probability.
            consistency_shadow_prob: Shadow probability.
            consistency_noise_prob: Noise consistency probability.
            enable_degradation: Enable quality degradation.
            degradation_jpeg_prob: JPEG compression probability.
            degradation_blur_prob: Blur probability.
            degradation_noise_prob: Noise probability.
            enable_validation: Enable data validation.

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

        # Synthesis settings
        self.enable_synthesis = enable_synthesis
        self.synthesis_ratio = synthesis_ratio
        self.synthesis_length = synthesis_length
        self.synthesis_k_probs = synthesis_k_probs
        self.synthesis_min_area = synthesis_min_area
        self.synthesis_max_area = synthesis_max_area
        self.synthesis_max_coverage = synthesis_max_coverage
        self.synthesis_max_overlap = synthesis_max_overlap
        self.synthesis_blending_probs = synthesis_blending_probs
        self.synthesis_boundary_randomize_prob = synthesis_boundary_randomize_prob
        self.synthesis_boundary_randomize_width = synthesis_boundary_randomize_width
        self.synthesis_boundary_randomize_noise_std = synthesis_boundary_randomize_noise_std

        # Consistency settings
        self.enable_consistency = enable_consistency
        self.consistency_color_prob = consistency_color_prob
        self.consistency_light_wrap_prob = consistency_light_wrap_prob
        self.consistency_shadow_prob = consistency_shadow_prob
        self.consistency_noise_prob = consistency_noise_prob

        # Degradation settings
        self.enable_degradation = enable_degradation
        self.degradation_jpeg_prob = degradation_jpeg_prob
        self.degradation_blur_prob = degradation_blur_prob
        self.degradation_noise_prob = degradation_noise_prob

        # Validation settings
        self.enable_validation = enable_validation

        # GPU augmentation module (initialized lazily)
        self._gpu_augmentation: GPUAugmentation | None = None

        # Synthesis components (initialized in setup)
        self._compositor = None
        self._consistency_pipeline = None
        self._degradation = None
        self._validator = None

        # Datasets (initialized in setup)
        self.train_dataset: SegmentationDataset | MixedDataset | None = None
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
        # Shuffle with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_samples, generator=generator).tolist()

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

        # CPU augmentations are disabled; GPU augmentations are applied
        # in on_after_batch_transfer when enabled.
        normalize = not (is_train and self.gpu_augmentation)
        transform = PairedTransform(size=self.size, normalize=normalize)

        return SegmentationDataset(image_paths, mask_paths, transform=transform)

    def on_after_batch_transfer(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
        dataloader_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """Apply GPU augmentation after batch transfer.

        This hook is called after the batch is moved to the device.
        When gpu_augmentation is enabled, applies kornia-based augmentations.

        Args:
            batch: Tuple of (images, masks, class_labels).
            dataloader_idx: Index of the dataloader.

        Returns:
            Augmented batch.

        """
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            return batch

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
            images = TF.normalize(images, IMAGENET_MEAN, IMAGENET_STD)

        return images, masks, class_labels

    def _setup_synthesis_components(self) -> None:
        """Initialize synthesis pipeline components."""
        if not self.enable_synthesis:
            return

        data_root = Path(self.data_root)
        fg_dir = data_root / "fg"
        bg_dir = data_root / "bg"

        # Check if fg/ and bg/ directories exist
        if not fg_dir.exists() or not bg_dir.exists():
            logger.warning(
                "Synthesis enabled but fg/ or bg/ directories not found in %s. "
                "Disabling synthesis.",
                self.data_root,
            )
            self.enable_synthesis = False
            return

        # Import synthesis modules
        from anime_segmentation.data.pools import BackgroundPool, ForegroundPool
        from anime_segmentation.training.synthesis.compositor import (
            CompositorConfig,
            CopyPasteCompositor,
        )
        from anime_segmentation.training.synthesis.consistency import ConsistencyPipeline
        from anime_segmentation.training.synthesis.degradation import QualityDegradation
        from anime_segmentation.training.synthesis.transforms import InstanceTransform
        from anime_segmentation.training.synthesis.validation import DataValidator

        # Initialize pools
        fg_pool = ForegroundPool(fg_dir)
        bg_pool = BackgroundPool(bg_dir)

        # Build compositor config
        config_kwargs: dict = {
            "min_area_ratio": self.synthesis_min_area,
            "max_area_ratio": self.synthesis_max_area,
            "max_total_coverage": self.synthesis_max_coverage,
            "max_iou_overlap": self.synthesis_max_overlap,
            "boundary_randomize_prob": self.synthesis_boundary_randomize_prob,
            "boundary_randomize_width": self.synthesis_boundary_randomize_width,
            "boundary_randomize_noise_std": self.synthesis_boundary_randomize_noise_std,
        }
        if self.synthesis_k_probs is not None:
            config_kwargs["k_probs"] = self.synthesis_k_probs
        if self.synthesis_blending_probs is not None:
            config_kwargs["blending_probs"] = self.synthesis_blending_probs

        compositor_config = CompositorConfig(**config_kwargs)

        # Instance transform
        instance_transform = InstanceTransform(
            hflip_prob=self.hflip_prob,
            rotation_range=(-self.rotation_degrees, self.rotation_degrees),
            scale_range=(0.5, 1.5),
        )

        # Create compositor
        self._compositor = CopyPasteCompositor(
            fg_pool=fg_pool,
            bg_pool=bg_pool,
            config=compositor_config,
            instance_transform=instance_transform,
        )

        # Consistency pipeline
        if self.enable_consistency:
            self._consistency_pipeline = ConsistencyPipeline(
                color_tone_prob=self.consistency_color_prob,
                light_wrap_prob=self.consistency_light_wrap_prob,
                shadow_prob=self.consistency_shadow_prob,
                noise_grain_prob=self.consistency_noise_prob,
            )

        # Degradation
        if self.enable_degradation:
            self._degradation = QualityDegradation(
                jpeg_prob=self.degradation_jpeg_prob,
                blur_prob=self.degradation_blur_prob,
                noise_prob=self.degradation_noise_prob,
            )

        # Validator
        if self.enable_validation:
            self._validator = DataValidator()

        logger.info(
            "Synthesis pipeline initialized: %d fg, %d bg images",
            len(fg_pool),
            len(bg_pool),
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.

        """
        if self._image_paths is None:
            self._load_and_split_paths()

        assert self._train_indices is not None
        assert self._val_indices is not None

        # Initialize synthesis components if needed
        if self.enable_synthesis and self._compositor is None:
            self._setup_synthesis_components()

        match stage:
            case "fit" | None:
                real_train_dataset = self._create_dataset(
                    self._train_indices,
                    is_train=True,
                )

                # Create mixed dataset if synthesis is enabled
                if self.enable_synthesis and self._compositor is not None:
                    synth_dataset = SynthesisDataset(
                        compositor=self._compositor,
                        size=self.size,
                        length=self.synthesis_length,
                        consistency_pipeline=self._consistency_pipeline,
                        degradation=self._degradation,
                        validator=self._validator,
                        normalize=not self.gpu_augmentation,
                    )
                    self.train_dataset = MixedDataset(
                        real_dataset=real_train_dataset,
                        synth_dataset=synth_dataset,
                        synthesis_ratio=self.synthesis_ratio,
                    )
                    logger.info(
                        "Mixed dataset created: real=%d, synth=%d, ratio=%.2f",
                        len(real_train_dataset),
                        len(synth_dataset),
                        self.synthesis_ratio,
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
