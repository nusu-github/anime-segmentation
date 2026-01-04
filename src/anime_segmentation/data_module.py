"""LightningDataModule for anime segmentation training with HuggingFace Datasets."""

# ruff: noqa: ARG002

from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from datasets import Dataset, DatasetDict, IterableDataset
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2

from .hf_dataset import (
    RealImageTransform,
    SyntheticCompositor,
    create_interleaved_dataset,
    create_synthetic_index_dataset,
    load_anime_seg_dataset,
)
from .transforms import RescalePad


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

    Augmentation philosophy (based on anime domain research):
    - Background synthesis is the most critical augmentation
    - Light geometric transforms for pose variation
    - Minimal intensity augmentation to preserve line art quality
    - Avoid heavy noise/blur that destroys high-frequency details
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
        # Simplified augmentation parameters
        aug_rotation_degrees: float = 15.0,
        aug_scale_range: tuple[float, float] = (0.8, 1.2),
        aug_translate_ratio: float = 0.1,
        aug_hflip_p: float = 0.5,
        aug_grayscale_p: float = 0.1,
        aug_color_jitter_p: float = 0.2,
        aug_brightness: float = 0.1,
        aug_contrast: float = 0.1,
        aug_saturation: float = 0.1,
        aug_hue: float = 0.02,
        # Compositor parameters
        aug_edge_blur_p: float = 0.2,
        aug_edge_blur_kernel_size: int = 5,
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
            aug_rotation_degrees: Max rotation degrees.
            aug_scale_range: Scale range (min, max).
            aug_translate_ratio: Translation ratio relative to image size.
            aug_hflip_p: Horizontal flip probability.
            aug_grayscale_p: Grayscale conversion probability.
            aug_color_jitter_p: Color jitter probability.
            aug_brightness: Brightness jitter range.
            aug_contrast: Contrast jitter range.
            aug_saturation: Saturation jitter range.
            aug_hue: Hue jitter range.
            aug_edge_blur_p: Edge blur probability in compositing.
            aug_edge_blur_kernel_size: Kernel size for edge blur.
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
                msg = f"Required data directory not found: {d}"
                raise FileNotFoundError(msg)

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

        real_ds = cast("DatasetDict", datasets["real"])
        fg_ds = cast("DatasetDict", datasets["foreground"])
        bg_ds = cast("DatasetDict", datasets["background"])

        train_real_raw = cast("Dataset", real_ds["train"])
        val_real_raw = cast("Dataset", real_ds["validation"])
        train_fg_raw = cast("Dataset", fg_ds["train"])
        val_fg_raw = cast("Dataset", fg_ds["validation"])
        train_bg_raw = cast("Dataset", bg_ds["train"])
        val_bg_raw = cast("Dataset", bg_ds["validation"])

        print("---")
        print(f"train real images: {len(train_real_raw)}")
        print(f"train foregrounds: {len(train_fg_raw)}")
        print(f"train backgrounds: {len(train_bg_raw)}")
        print(f"val real images: {len(val_real_raw)}")
        print(f"val foregrounds: {len(val_fg_raw)}")
        print(f"val backgrounds: {len(val_bg_raw)}")
        print("---")

        self._train_fg_dataset = train_fg_raw
        self._train_bg_dataset = train_bg_raw
        self._val_fg_dataset = val_fg_raw
        self._val_bg_dataset = val_bg_raw

        train_synthetic_idx = create_synthetic_index_dataset(
            num_foregrounds=len(self._train_fg_dataset),
            characters_range=self.hparams["characters_range"],
            seed=42,
        )
        val_synthetic_idx = create_synthetic_index_dataset(
            num_foregrounds=len(self._val_fg_dataset),
            characters_range=self.hparams["characters_range"],
            seed=42,
        )

        # Build transforms
        train_transform = self._build_train_transform(img_size)
        val_transform = self._build_val_transform(img_size)

        # Create compositors
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
            seed=42,
            edge_blur_p=0.0,
        )

        train_interleaved = create_interleaved_dataset(
            datasets=[train_real_raw, train_synthetic_idx],
            seed=42,
            use_iterable=self._use_iterable,
        )
        val_interleaved = create_interleaved_dataset(
            datasets=[val_real_raw, val_synthetic_idx],
            seed=42,
            use_iterable=self._use_iterable,
        )

        if hasattr(train_interleaved, "set_transform"):
            cast("Dataset", train_interleaved).set_transform(
                self._make_unified_transform_fn(
                    real_base=RealImageTransform(),
                    real_aug=train_transform,
                    compositor=train_compositor,
                    synthetic_aug=train_transform,
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

    def _make_unified_transform_fn(
        self,
        real_base: RealImageTransform,
        real_aug: v2.Compose,
        compositor: SyntheticCompositor,
        synthetic_aug: v2.Compose,
    ):
        """Create unified transform function that dispatches based on item type."""

        def transform_fn(examples: dict) -> dict:
            fg_indices_batch = examples.get("fg_indices", [])
            images_batch = examples.get("image", [])
            batch_size = len(fg_indices_batch) if fg_indices_batch else len(images_batch)

            augmented_images = []
            augmented_masks = []

            for i in range(batch_size):
                fg_idx = fg_indices_batch[i] if i < len(fg_indices_batch) else None
                is_synthetic = fg_idx is not None

                if is_synthetic:
                    single_example = {"fg_indices": [fg_idx]}
                    result = compositor(single_example)
                    img = result["image"][0]
                    mask = result["mask"][0]
                    img_tv = tv_tensors.Image(img)
                    mask_tv = tv_tensors.Mask(mask)
                    img_aug, mask_aug = synthetic_aug(img_tv, mask_tv)
                else:
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

    def _build_train_transform(self, img_size: int) -> v2.Compose:
        """Build simplified training transform pipeline.

        Focuses on:
        - Geometric augmentation (rotation, scale, flip)
        - Light color jitter (preserves line art)
        - NO heavy noise/blur that destroys anime line details
        """
        transforms: list[v2.Transform] = [
            # Resize and pad to square
            RescalePad(img_size),
        ]

        # Geometric augmentation
        rot_deg = self.hparams["aug_rotation_degrees"]
        if rot_deg > 0:
            transforms.append(
                v2.RandomRotation(
                    degrees=(-rot_deg, rot_deg),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            )

        scale_min, scale_max = self.hparams["aug_scale_range"]
        translate = self.hparams["aug_translate_ratio"]
        if scale_min != 1.0 or scale_max != 1.0 or translate > 0:
            transforms.append(
                v2.RandomAffine(
                    degrees=0,
                    translate=(translate, translate) if translate > 0 else None,
                    scale=(scale_min, scale_max),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            )

        if self.hparams["aug_hflip_p"] > 0:
            transforms.append(v2.RandomHorizontalFlip(p=self.hparams["aug_hflip_p"]))

        # Light color augmentation (only on image, not mask)
        if self.hparams["aug_grayscale_p"] > 0:
            transforms.append(v2.RandomGrayscale(p=self.hparams["aug_grayscale_p"]))

        if self.hparams["aug_color_jitter_p"] > 0:
            transforms.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=self.hparams["aug_brightness"],
                            contrast=self.hparams["aug_contrast"],
                            saturation=self.hparams["aug_saturation"],
                            hue=self.hparams["aug_hue"],
                        )
                    ],
                    p=self.hparams["aug_color_jitter_p"],
                )
            )

        return v2.Compose(transforms)

    def _build_val_transform(self, img_size: int) -> v2.Compose:
        """Build validation transform (minimal, deterministic)."""
        return v2.Compose([RescalePad(img_size)])

    def train_dataloader(self) -> "torch.utils.data.DataLoader":
        if self.train_dataset is None:
            msg = "setup() must be called before train_dataloader()"
            raise RuntimeError(msg)

        is_iterable = isinstance(self.train_dataset, IterableDataset)

        return torch.utils.data.DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.hparams["batch_size_train"],
            shuffle=not is_iterable,
            drop_last=True,
            persistent_workers=self.hparams["num_workers_train"] > 0,
            num_workers=self.hparams["num_workers_train"],
            pin_memory=True,
            collate_fn=segmentation_collate_fn,
        )

    def val_dataloader(self) -> "torch.utils.data.DataLoader":
        if self.val_dataset is None:
            msg = "setup() must be called before val_dataloader()"
            raise RuntimeError(msg)

        return torch.utils.data.DataLoader(
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
