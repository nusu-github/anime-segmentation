"""LightningDataModule for anime segmentation training with HuggingFace Datasets."""

# ruff: noqa: ARG002

from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from datasets import Dataset, DatasetDict, IterableDataset
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from .hf_dataset import (
    RealImageTransform,
    SyntheticCompositor,
    create_interleaved_dataset,
    create_synthetic_index_dataset,
    load_anime_seg_dataset,
)
from .transforms import (
    JPEGCompression,
    RandomColor,
    RandomColorBlocks,
    RandomTextOverlay,
    RescalePad,
    ResizeBlur,
    SharpBackground,
    SimulateLight,
    SketchConvert,
)


def segmentation_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Collate HF Datasets dicts to tensor dict format.

    Args:
        batch: List of dicts with 'image' and 'mask' tensors.

    Returns:
        Dict with batched 'image' and 'label' tensors.
    """
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["mask"] for b in batch])
    return {"image": images, "label": labels}


class AnimeSegDataModule(L.LightningDataModule):
    """LightningDataModule for anime segmentation training.

    Supports loading from local directories or HuggingFace Hub.
    Uses HuggingFace Datasets for efficient data loading and streaming.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        dataset_name: str | None = None,
        fg_dir: str = "fg",
        bg_dir: str = "bg",
        img_dir: str = "imgs",
        mask_dir: str = "masks",
        fg_ext: str = ".png",
        bg_ext: str = ".jpg",
        img_ext: str = ".jpg",
        mask_ext: str = ".jpg",
        data_split: float = 0.95,
        img_size: int = 1024,
        batch_size_train: int = 2,
        batch_size_val: int = 2,
        num_workers_train: int = 4,
        num_workers_val: int = 4,
        characters_range: tuple[int, int] = (0, 3),
        *,
        streaming: bool = False,
        # Augmentation probabilities for synthetic images (set to 0.0 to disable)
        aug_sharp_background_p: float = 0.0,
        aug_sketch_p: float = 0.0,
        aug_grayscale_p: float = 0.25,
        aug_color_blocks_p: float = 0.0,
        aug_text_overlay_p: float = 0.0,
        aug_light_p: float = 0.0,
        aug_resize_blur_p: float = 0.15,
        aug_jpeg_p: float = 0.15,
        aug_color_p: float = 0.3,
        aug_noise_p: float = 0.15,
        # Augmentation parameters
        aug_rotation_degrees_synth: float = 25.0,
        aug_rotation_degrees_real: float = 15.0,
        aug_jpeg_quality_range: tuple[int, int] = (50, 95),
        aug_resize_blur_scale_range: tuple[float, float] = (0.6, 0.9),
        aug_edge_blur_p: float = 0.2,
        aug_edge_blur_kernel_size: int = 5,
        aug_noise_sigma: float = 0.05,
        aug_rescale_pad_ratio: float = 1.25,
        aug_text_font_size_ratio: float = 0.05,
    ) -> None:
        """Initialize the data module.

        Args:
            data_dir: Local data directory path.
            dataset_name: HuggingFace Hub dataset name (e.g., 'user/anime-seg').
            fg_dir: Foreground subdirectory name.
            bg_dir: Background subdirectory name.
            img_dir: Image subdirectory name.
            mask_dir: Mask subdirectory name.
            fg_ext: Foreground file extension.
            bg_ext: Background file extension.
            img_ext: Image file extension.
            mask_ext: Mask file extension.
            data_split: Train/validation split ratio.
            img_size: Target image size.
            batch_size_train: Training batch size.
            batch_size_val: Validation batch size.
            num_workers_train: Number of training data loader workers.
            num_workers_val: Number of validation data loader workers.
            characters_range: Range for number of characters per synthetic sample.
            streaming: Whether to use streaming mode (Hub only).
            aug_sharp_background_p: Probability for sharp background augmentation.
            aug_sketch_p: Probability for sketch conversion.
            aug_grayscale_p: Probability for grayscale conversion.
            aug_color_blocks_p: Probability for random color blocks overlay.
            aug_text_overlay_p: Probability for random text overlay.
            aug_light_p: Probability for light simulation.
            aug_resize_blur_p: Probability for resize blur.
            aug_jpeg_p: Probability for JPEG compression.
            aug_color_p: Probability for color augmentation.
            aug_noise_p: Probability for Gaussian noise.
            aug_rotation_degrees_synth: Max rotation degrees for synthetic images.
            aug_rotation_degrees_real: Max rotation degrees for real images.
            aug_jpeg_quality_range: JPEG quality range (min, max).
            aug_resize_blur_scale_range: Resize blur scale range (min, max).
            aug_edge_blur_p: Probability for edge blur in compositing.
            aug_edge_blur_kernel_size: Kernel size for edge blur.
            aug_noise_sigma: Sigma for Gaussian noise.
            aug_rescale_pad_ratio: Ratio for rescale padding (1.25 = img_size * 1.25).
            aug_text_font_size_ratio: Font size ratio relative to image size.
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Dataset | IterableDataset | None = None
        self.val_dataset: Dataset | IterableDataset | None = None
        self._use_iterable = streaming

        # Store FG/BG datasets for synthetic compositing (lazy loading)
        self._train_fg_dataset: Dataset | None = None
        self._train_bg_dataset: Dataset | None = None
        self._val_fg_dataset: Dataset | None = None
        self._val_bg_dataset: Dataset | None = None

    def prepare_data(self) -> None:
        """Verify data directories exist. Called only on rank 0 in distributed training."""
        if self.hparams["dataset_name"] is not None:
            # Hub dataset - no local verification needed
            return

        data_dir = self.hparams["data_dir"]
        if data_dir is None:
            msg = "Either data_dir or dataset_name must be provided"
            raise ValueError(msg)

        data_root = Path(data_dir)
        required_dirs = [
            data_root / self.hparams["img_dir"],
            data_root / self.hparams["mask_dir"],
            data_root / self.hparams["fg_dir"],
            data_root / self.hparams["bg_dir"],
        ]
        for d in required_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Required data directory not found: {d}")

    def setup(self, stage: str | None = None) -> None:
        """Create datasets. Called on every process in distributed training."""
        if stage == "fit" or stage is None:
            self._create_datasets()

        if stage == "validate" and self.val_dataset is None:
            self._create_datasets()

    def _create_datasets(self) -> None:
        """Create train and validation datasets using HuggingFace Datasets."""
        img_size = self.hparams["img_size"]
        split_ratio = self.hparams["data_split"]
        streaming = self.hparams["streaming"]

        # Load datasets
        datasets = load_anime_seg_dataset(
            data_dir=self.hparams["data_dir"],
            dataset_name=self.hparams["dataset_name"],
            fg_dir=self.hparams["fg_dir"],
            bg_dir=self.hparams["bg_dir"],
            img_dir=self.hparams["img_dir"],
            mask_dir=self.hparams["mask_dir"],
            fg_ext=self.hparams["fg_ext"],
            bg_ext=self.hparams["bg_ext"],
            img_ext=self.hparams["img_ext"],
            mask_ext=self.hparams["mask_ext"],
            split_ratio=split_ratio,
            streaming=streaming,
        )

        # Cast to DatasetDict (streaming mode not fully supported yet)
        real_ds = cast("DatasetDict", datasets["real"])
        fg_ds = cast("DatasetDict", datasets["foreground"])
        bg_ds = cast("DatasetDict", datasets["background"])

        # Get splits as Dataset
        train_real_raw = cast("Dataset", real_ds["train"])
        val_real_raw = cast("Dataset", real_ds["validation"])
        train_fg_raw = cast("Dataset", fg_ds["train"])
        val_fg_raw = cast("Dataset", fg_ds["validation"])
        train_bg_raw = cast("Dataset", bg_ds["train"])
        val_bg_raw = cast("Dataset", bg_ds["validation"])

        # Log dataset sizes
        print("---")
        print(f"train real images: {len(train_real_raw)}")
        print(f"train foregrounds: {len(train_fg_raw)}")
        print(f"train backgrounds: {len(train_bg_raw)}")
        print(f"val real images: {len(val_real_raw)}")
        print(f"val foregrounds: {len(val_fg_raw)}")
        print(f"val backgrounds: {len(val_bg_raw)}")
        print("---")

        # Store FG/BG datasets for lazy loading in compositor (avoid OOM)
        self._train_fg_dataset = train_fg_raw
        self._train_bg_dataset = train_bg_raw
        self._val_fg_dataset = val_fg_raw
        self._val_bg_dataset = val_bg_raw

        # Create synthetic index datasets
        train_synthetic_idx = create_synthetic_index_dataset(
            num_foregrounds=len(self._train_fg_dataset),
            characters_range=self.hparams["characters_range"],
            seed=42,
        )
        val_synthetic_idx = create_synthetic_index_dataset(
            num_foregrounds=len(self._val_fg_dataset),
            characters_range=self.hparams["characters_range"],
            seed=43,
        )

        # Build transforms
        real_transform = self._build_real_transform(img_size)
        synthetic_transform = self._build_synthetic_transform(img_size)
        val_transform = self._build_val_transform(img_size)

        # Create compositors for synthetic data
        train_compositor = SyntheticCompositor(
            foreground_dataset=self._train_fg_dataset,
            background_dataset=self._train_bg_dataset,
            output_size=(img_size, img_size),
            edge_blur_p=self.hparams["aug_edge_blur_p"],
            edge_blur_kernel_size=self.hparams["aug_edge_blur_kernel_size"],
        )
        val_compositor = SyntheticCompositor(
            foreground_dataset=self._val_fg_dataset,
            background_dataset=self._val_bg_dataset,
            output_size=(img_size, img_size),
            seed=43,
            edge_blur_p=0.0,  # No augmentation for validation
        )

        # Combine datasets using HuggingFace's native interleave_datasets
        # Note: interleave_datasets does not preserve set_transform, so we apply
        # transforms after interleaving using a unified transform function
        num_shards = max(self.hparams["num_workers_train"] * 4, 16)
        train_interleaved = create_interleaved_dataset(
            datasets=[train_real_raw, train_synthetic_idx],
            seed=42,
            num_shards=num_shards,
            use_iterable=self._use_iterable,
        )
        val_interleaved = create_interleaved_dataset(
            datasets=[val_real_raw, val_synthetic_idx],
            seed=43,
            num_shards=num_shards,
            use_iterable=self._use_iterable,
        )

        # Apply unified transform that dispatches based on item type
        # Note: set_transform is only available on Dataset, not IterableDataset
        if hasattr(train_interleaved, "set_transform"):
            cast("Dataset", train_interleaved).set_transform(
                self._make_unified_transform_fn(
                    real_base=RealImageTransform(),
                    real_aug=real_transform,
                    compositor=train_compositor,
                    synthetic_aug=synthetic_transform,
                )
            )
        if hasattr(val_interleaved, "set_transform"):
            cast("Dataset", val_interleaved).set_transform(
                self._make_unified_transform_fn(
                    real_base=RealImageTransform(),
                    real_aug=val_transform,
                    compositor=val_compositor,
                    synthetic_aug=val_transform,
                )
            )

        self.train_dataset = train_interleaved
        self.val_dataset = val_interleaved

    def _make_real_transform_fn(
        self,
        base_transform: RealImageTransform,
        augmentation: v2.Compose,
    ):
        """Create transform function for real images."""

        def transform_fn(examples: dict) -> dict:
            # Apply base transform (PIL -> tensor)
            result = base_transform(examples)

            # Apply augmentation
            augmented_images = []
            augmented_masks = []
            for img, mask in zip(result["image"], result["mask"], strict=True):
                img_tv = tv_tensors.Image(img)
                mask_tv = tv_tensors.Mask(mask)
                img_aug, mask_aug = augmentation(img_tv, mask_tv)
                augmented_images.append(img_aug)
                augmented_masks.append(mask_aug)

            return {"image": augmented_images, "mask": augmented_masks}

        return transform_fn

    def _make_synthetic_transform_fn(
        self,
        compositor: SyntheticCompositor,
        augmentation: v2.Compose,
    ):
        """Create transform function for synthetic images."""

        def transform_fn(examples: dict) -> dict:
            # Apply compositor
            result = compositor(examples)

            # Apply augmentation
            augmented_images = []
            augmented_masks = []
            for img, mask in zip(result["image"], result["mask"], strict=True):
                img_tv = tv_tensors.Image(img)
                mask_tv = tv_tensors.Mask(mask)
                img_aug, mask_aug = augmentation(img_tv, mask_tv)
                augmented_images.append(img_aug)
                augmented_masks.append(mask_aug)

            return {"image": augmented_images, "mask": augmented_masks}

        return transform_fn

    def _make_unified_transform_fn(
        self,
        real_base: RealImageTransform,
        real_aug: v2.Compose,
        compositor: SyntheticCompositor,
        synthetic_aug: v2.Compose,
    ):
        """Create unified transform function that dispatches based on item type.

        This is needed because interleave_datasets does not preserve set_transform.
        We detect item type by checking for 'fg_indices' key (synthetic) vs 'image' key (real).
        """

        def transform_fn(examples: dict) -> dict:
            # Handle mixed batches: process each item individually based on its type
            # After interleave, both 'fg_indices' and 'image'/'mask' keys exist,
            # but one set will have None values for each item
            fg_indices_batch = examples.get("fg_indices", [])
            images_batch = examples.get("image", [])

            # Determine batch size
            batch_size = len(fg_indices_batch) if fg_indices_batch else len(images_batch)

            augmented_images = []
            augmented_masks = []

            for i in range(batch_size):
                fg_idx = fg_indices_batch[i] if i < len(fg_indices_batch) else None
                is_synthetic = fg_idx is not None

                if is_synthetic:
                    # Synthetic: apply compositor then augmentation
                    single_example = {"fg_indices": [fg_idx]}
                    result = compositor(single_example)
                    img = result["image"][0]
                    mask = result["mask"][0]
                    img_tv = tv_tensors.Image(img)
                    mask_tv = tv_tensors.Mask(mask)
                    img_aug, mask_aug = synthetic_aug(img_tv, mask_tv)
                else:
                    # Real: apply base transform then augmentation
                    single_example = {
                        "image": [images_batch[i]],
                        "mask": [examples["mask"][i]],
                    }
                    result = real_base(single_example)
                    img = result["image"][0]
                    mask = result["mask"][0]
                    img_tv = tv_tensors.Image(img)
                    mask_tv = tv_tensors.Mask(mask)
                    img_aug, mask_aug = real_aug(img_tv, mask_tv)

                augmented_images.append(img_aug)
                augmented_masks.append(mask_aug)

            return {"image": augmented_images, "mask": augmented_masks}

        return transform_fn

    def _build_real_transform(self, img_size: int) -> v2.Compose:
        """Build transform pipeline for real images."""
        rot_deg = self.hparams["aug_rotation_degrees_real"]
        pad_size = int(img_size * self.hparams["aug_rescale_pad_ratio"])
        transforms: list[v2.Transform] = [
            RescalePad(pad_size),
            v2.RandomRotation(degrees=(-rot_deg, rot_deg), fill=[0.0]),
            v2.RandomCrop(img_size),
        ]

        if self.hparams["aug_color_p"] > 0:
            transforms.append(RandomColor(p=self.hparams["aug_color_p"]))
        if self.hparams["aug_noise_p"] > 0:
            sigma = self.hparams["aug_noise_sigma"]
            transforms.append(
                v2.RandomApply(
                    [v2.GaussianNoise(mean=0.0, sigma=sigma)], p=self.hparams["aug_noise_p"]
                )
            )

        return v2.Compose(transforms)

    def _build_synthetic_transform(self, img_size: int) -> v2.Compose:
        """Build transform pipeline for synthetic images.

        Augmentations are conditionally added based on their probability parameters.
        Set a probability to 0.0 to disable that augmentation.
        """
        transforms: list[v2.Transform] = []

        # Optional augmentations (controlled by hparams)
        if self.hparams["aug_sharp_background_p"] > 0:
            transforms.append(SharpBackground(p=self.hparams["aug_sharp_background_p"]))
        if self.hparams["aug_sketch_p"] > 0:
            transforms.append(SketchConvert(p=self.hparams["aug_sketch_p"]))
        if self.hparams["aug_grayscale_p"] > 0:
            transforms.append(v2.RandomGrayscale(p=self.hparams["aug_grayscale_p"]))
        if self.hparams["aug_color_blocks_p"] > 0:
            transforms.append(RandomColorBlocks(p=self.hparams["aug_color_blocks_p"]))
        if self.hparams["aug_text_overlay_p"] > 0:
            transforms.append(
                RandomTextOverlay(
                    p=self.hparams["aug_text_overlay_p"],
                    font_size_ratio=self.hparams["aug_text_font_size_ratio"],
                )
            )
        if self.hparams["aug_light_p"] > 0:
            transforms.append(SimulateLight(p=self.hparams["aug_light_p"]))

        # Rotation (geometric consistency)
        rot_deg = self.hparams["aug_rotation_degrees_synth"]
        transforms.append(
            v2.RandomRotation(
                degrees=(-rot_deg, rot_deg),
                interpolation=v2.InterpolationMode.BILINEAR,
                fill=0.0,
            )
        )

        # Quality degradation augmentations
        if self.hparams["aug_resize_blur_p"] > 0:
            transforms.append(
                ResizeBlur(
                    p=self.hparams["aug_resize_blur_p"],
                    scale_range=self.hparams["aug_resize_blur_scale_range"],
                )
            )
        if self.hparams["aug_jpeg_p"] > 0:
            transforms.append(
                JPEGCompression(
                    p=self.hparams["aug_jpeg_p"],
                    quality_range=self.hparams["aug_jpeg_quality_range"],
                )
            )

        # Color augmentations (same as real images)
        if self.hparams["aug_color_p"] > 0:
            transforms.append(RandomColor(p=self.hparams["aug_color_p"]))
        if self.hparams["aug_noise_p"] > 0:
            sigma = self.hparams["aug_noise_sigma"]
            transforms.append(
                v2.RandomApply(
                    [v2.GaussianNoise(mean=0.0, sigma=sigma)], p=self.hparams["aug_noise_p"]
                )
            )

        return v2.Compose(transforms)

    def _build_val_transform(self, img_size: int) -> v2.Compose:
        """Build transform pipeline for validation (minimal augmentation)."""
        return v2.Compose(
            [
                RescalePad(img_size),
            ]
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            msg = "setup() must be called before train_dataloader()"
            raise RuntimeError(msg)

        # IterableDataset doesn't support shuffle in DataLoader
        # (shuffling is handled by the dataset itself via shuffle buffer)
        is_iterable = isinstance(self.train_dataset, IterableDataset)

        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.hparams["batch_size_train"],
            shuffle=not is_iterable,
            drop_last=True,
            persistent_workers=self.hparams["num_workers_train"] > 0,
            num_workers=self.hparams["num_workers_train"],
            pin_memory=True,
            collate_fn=segmentation_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            msg = "setup() must be called before val_dataloader()"
            raise RuntimeError(msg)

        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.hparams["batch_size_val"],
            shuffle=False,
            persistent_workers=self.hparams["num_workers_val"] > 0,
            num_workers=self.hparams["num_workers_val"],
            pin_memory=True,
            collate_fn=segmentation_collate_fn,
        )

    def state_dict(self) -> dict:
        """Save DataModule state for checkpointing."""
        train_len = 0
        val_len = 0

        # IterableDataset doesn't have len(), use -1 to indicate streaming mode
        if self.train_dataset is not None:
            train_len = (
                -1 if isinstance(self.train_dataset, IterableDataset) else len(self.train_dataset)
            )

        if self.val_dataset is not None:
            val_len = -1 if isinstance(self.val_dataset, IterableDataset) else len(self.val_dataset)

        return {
            "train_dataset_len": train_len,
            "val_dataset_len": val_len,
            "use_iterable": self._use_iterable,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore DataModule state from checkpoint."""
        # Dataset is recreated on setup(), no state to restore
