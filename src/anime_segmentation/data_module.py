"""LightningDataModule for anime segmentation training."""

# ruff: noqa: ARG002

import random
from pathlib import Path

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from .datasets import CombinedDataset, RealImageDataset, SyntheticDataset
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


def segmentation_collate_fn(batch: list[tuple]) -> dict[str, Tensor]:
    """Collate tv_tensor tuples to dict format for backward compatibility.

    Args:
        batch: List of (image, mask) or (image, mask, trimap) tuples.

    Returns:
        Dict with keys "image", "label", and optionally "trimap".
    """
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])

    result: dict[str, Tensor] = {"image": images, "label": labels}

    if len(batch[0]) >= 3:
        trimaps = torch.stack([b[2] for b in batch])
        result["trimap"] = trimaps

    return result


class AnimeSegDataModule(L.LightningDataModule):
    """LightningDataModule for anime segmentation training.

    Uses torchvision.transforms.v2 and tv_tensors for modern data pipeline.
    Supports both real image datasets and synthetic fg/bg compositing.
    """

    def __init__(
        self,
        data_dir: str = "../../dataset/anime-seg",
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
        *,
        with_trimap: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: CombinedDataset | None = None
        self.val_dataset: CombinedDataset | None = None
        self._with_trimap_transform: WithTrimap | None = WithTrimap() if with_trimap else None

    def prepare_data(self) -> None:
        """Verify data directories exist. Called only on rank 0 in distributed training."""
        data_root = Path(self.hparams["data_dir"])
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
        """Create train and validation datasets."""
        data_root = Path(self.hparams["data_dir"])
        img_size = self.hparams["img_size"]
        split_rate = self.hparams["data_split"]

        # Collect file paths
        imgs_dir = data_root / self.hparams["img_dir"]
        masks_dir = data_root / self.hparams["mask_dir"]
        fgs_dir = data_root / self.hparams["fg_dir"]
        bgs_dir = data_root / self.hparams["bg_dir"]

        img_list = sorted(imgs_dir.glob(f"*{self.hparams['img_ext']}"))
        mask_list = [
            masks_dir / p.name.replace(self.hparams["img_ext"], self.hparams["mask_ext"])
            for p in img_list
        ]
        fg_list = sorted(fgs_dir.glob(f"*{self.hparams['fg_ext']}"))
        bg_list = sorted(bgs_dir.glob(f"*{self.hparams['bg_ext']}"))

        # Shuffle with fixed seed for reproducibility
        rng = random.Random(1)
        rng.shuffle(fg_list)
        rng.shuffle(bg_list)

        # Shuffle image/mask pairs together
        paired = list(zip(img_list, mask_list, strict=True))
        rng.shuffle(paired)
        img_list, mask_list = zip(*paired, strict=True) if paired else ([], [])
        img_list, mask_list = list(img_list), list(mask_list)

        # Split into train/val
        def split_list(lst: list, rate: float) -> tuple[list, list]:
            n = int(len(lst) * rate)
            return lst[:n], lst[n:]

        train_fg, val_fg = split_list(fg_list, split_rate)
        train_bg, val_bg = split_list(bg_list, split_rate)
        train_img, val_img = split_list(img_list, split_rate)
        train_mask, val_mask = split_list(mask_list, split_rate)

        # Log dataset sizes
        print("---")
        print(f"train fgs: {len(train_fg)}")
        print(f"train bgs: {len(train_bg)}")
        print(f"train imgs: {len(train_img)}")
        print(f"train masks: {len(train_mask)}")
        print(f"val fgs: {len(val_fg)}")
        print(f"val bgs: {len(val_bg)}")
        print(f"val imgs: {len(val_img)}")
        print(f"val masks: {len(val_mask)}")
        print("---")

        # Build transforms
        real_transform = self._build_real_transform(img_size)
        synthetic_transform = self._build_synthetic_transform(img_size)
        val_transform = self._build_val_transform(img_size)

        # Create train datasets
        train_real = RealImageDataset(train_img, train_mask, transform=real_transform)
        train_synthetic = SyntheticDataset(
            train_fg, train_bg, (img_size, img_size), transform=synthetic_transform
        )
        self.train_dataset = CombinedDataset([train_real, train_synthetic])

        # Create val datasets
        val_real = RealImageDataset(val_img, val_mask, transform=val_transform)
        val_synthetic = SyntheticDataset(
            val_fg, val_bg, (img_size, img_size), transform=val_transform
        )
        self.val_dataset = CombinedDataset([val_real, val_synthetic])

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
        batch: list[tuple[tv_tensors.Image, tv_tensors.Mask]],
    ) -> list[tuple]:
        """Apply trimap transform if enabled."""
        if self._with_trimap_transform is None:
            return batch  # type: ignore[return-value]
        return [self._with_trimap_transform(item) for item in batch]

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            msg = "setup() must be called before train_dataloader()"
            raise RuntimeError(msg)

        def collate_with_trimap(batch):
            batch = self._apply_trimap_if_needed(batch)
            return segmentation_collate_fn(batch)

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size_train"],
            shuffle=True,
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
            self.val_dataset,
            batch_size=self.hparams["batch_size_val"],
            shuffle=False,
            persistent_workers=self.hparams["num_workers_val"] > 0,
            num_workers=self.hparams["num_workers_val"],
            pin_memory=True,
            collate_fn=collate_with_trimap,
        )

    def state_dict(self) -> dict:
        """Save DataModule state for checkpointing."""
        return {
            "train_dataset_len": len(self.train_dataset) if self.train_dataset else 0,
            "val_dataset_len": len(self.val_dataset) if self.val_dataset else 0,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore DataModule state from checkpoint."""
        # Dataset is recreated on setup(), no state to restore
