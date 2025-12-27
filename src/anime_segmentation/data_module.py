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
    get_image_paths_from_column,
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
    WithTrimap,
)


def segmentation_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Collate HF Datasets dicts to tensor dict format.

    Args:
        batch: List of dicts with 'image' and 'mask' tensors.

    Returns:
        Dict with batched 'image', 'label', and optionally 'trimap' tensors.
    """
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["mask"] for b in batch])

    result: dict[str, Tensor] = {"image": images, "label": labels}

    if "trimap" in batch[0]:
        trimaps = torch.stack([b["trimap"] for b in batch])
        result["trimap"] = trimaps

    return result


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
        with_trimap: bool = False,
        streaming: bool = False,
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
            with_trimap: Whether to generate trimap for MODNet.
            streaming: Whether to use streaming mode (Hub only).
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Dataset | IterableDataset | None = None
        self.val_dataset: Dataset | IterableDataset | None = None
        self._with_trimap_transform: WithTrimap | None = WithTrimap() if with_trimap else None
        self._use_iterable = streaming

        # Store image paths for synthetic compositing (lazy loading)
        self._train_fg_paths: list[str] | None = None
        self._train_bg_paths: list[str] | None = None
        self._val_fg_paths: list[str] | None = None
        self._val_bg_paths: list[str] | None = None

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
                raise FileNotFoundError(f"Required data directory not found: {d}")  # noqa: TRY003

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

        # Get FG/BG image paths for lazy loading (avoid OOM)
        self._train_fg_paths = get_image_paths_from_column(train_fg_raw, "image")
        self._train_bg_paths = get_image_paths_from_column(train_bg_raw, "image")
        self._val_fg_paths = get_image_paths_from_column(val_fg_raw, "image")
        self._val_bg_paths = get_image_paths_from_column(val_bg_raw, "image")

        # Create synthetic index datasets
        train_synthetic_idx = create_synthetic_index_dataset(
            num_foregrounds=len(self._train_fg_paths),
            characters_range=self.hparams["characters_range"],
            seed=42,
        )
        val_synthetic_idx = create_synthetic_index_dataset(
            num_foregrounds=len(self._val_fg_paths),
            characters_range=self.hparams["characters_range"],
            seed=43,
        )

        # Build transforms
        real_transform = self._build_real_transform(img_size)
        synthetic_transform = self._build_synthetic_transform(img_size)
        val_transform = self._build_val_transform(img_size)

        # Create train real dataset with transform
        train_real_raw.set_transform(
            self._make_real_transform_fn(RealImageTransform(), real_transform)
        )

        # Create train synthetic dataset with compositor
        train_compositor = SyntheticCompositor(
            foreground_paths=self._train_fg_paths,
            background_paths=self._train_bg_paths,
            output_size=(img_size, img_size),
        )
        train_synthetic_idx.set_transform(
            self._make_synthetic_transform_fn(train_compositor, synthetic_transform)
        )

        # Create val real dataset
        val_real_raw.set_transform(
            self._make_real_transform_fn(RealImageTransform(), val_transform)
        )

        # Create val synthetic dataset
        val_compositor = SyntheticCompositor(
            foreground_paths=self._val_fg_paths,
            background_paths=self._val_bg_paths,
            output_size=(img_size, img_size),
            seed=43,
        )
        val_synthetic_idx.set_transform(
            self._make_synthetic_transform_fn(val_compositor, val_transform)
        )

        # Combine datasets using HuggingFace's native interleave_datasets
        # This provides better performance and compatibility with DataLoader
        num_shards = max(self.hparams["num_workers_train"] * 4, 16)
        self.train_dataset = create_interleaved_dataset(
            datasets=[train_real_raw, train_synthetic_idx],
            seed=42,
            num_shards=num_shards,
            use_iterable=self._use_iterable,
        )
        self.val_dataset = create_interleaved_dataset(
            datasets=[val_real_raw, val_synthetic_idx],
            seed=43,
            num_shards=num_shards,
            use_iterable=self._use_iterable,
        )

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

    def _build_real_transform(self, img_size: int) -> v2.Compose:
        """Build transform pipeline for real images."""
        return v2.Compose(
            [
                RescalePad(img_size + img_size // 4),
                v2.RandomRotation(degrees=(-90, 90), fill=[0.0]),
                v2.RandomCrop(img_size),
                RandomColor(p=0.5),
                v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.05)], p=0.5),
            ]
        )

    def _build_synthetic_transform(self, img_size: int) -> v2.Compose:
        """Build transform pipeline for synthetic images (includes heavy augmentation)."""
        return v2.Compose(
            [
                # DatasetGenerator augmentations
                SharpBackground(p=0.5),
                SketchConvert(p=0.25),
                v2.RandomGrayscale(p=0.5),
                RandomColorBlocks(p=0.5),
                RandomTextOverlay(p=0.5),
                SimulateLight(p=0.5),
                v2.RandomRotation(
                    degrees=(-180, 180),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=0.0,
                ),
                ResizeBlur(p=0.5),
                JPEGCompression(p=0.5),
                # Final color augmentations (same as real)
                RandomColor(p=0.5),
                v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.05)], p=0.5),
            ]
        )

    def _build_val_transform(self, img_size: int) -> v2.Compose:
        """Build transform pipeline for validation (minimal augmentation)."""
        return v2.Compose(
            [
                RescalePad(img_size),
            ]
        )

    def _apply_trimap_if_needed(
        self,
        batch: list[dict[str, Tensor]],
    ) -> list[dict[str, Tensor]]:
        """Apply trimap transform if enabled."""
        if self._with_trimap_transform is None:
            return batch

        result = []
        for item in batch:
            img_tv = tv_tensors.Image(item["image"])
            mask_tv = tv_tensors.Mask(item["mask"])
            img, mask, trimap = self._with_trimap_transform((img_tv, mask_tv))
            result.append({"image": img, "mask": mask, "trimap": trimap})
        return result

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            msg = "setup() must be called before train_dataloader()"
            raise RuntimeError(msg)

        def collate_with_trimap(batch):
            batch = self._apply_trimap_if_needed(batch)
            return segmentation_collate_fn(batch)

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
            collate_fn=collate_with_trimap,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            msg = "setup() must be called before val_dataloader()"
            raise RuntimeError(msg)

        def collate_with_trimap(batch):
            batch = self._apply_trimap_if_needed(batch)
            return segmentation_collate_fn(batch)

        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.hparams["batch_size_val"],
            shuffle=False,
            persistent_workers=self.hparams["num_workers_val"] > 0,
            num_workers=self.hparams["num_workers_val"],
            pin_memory=True,
            collate_fn=collate_with_trimap,
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
