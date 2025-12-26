"""LightningDataModule for anime segmentation training."""
# ruff: noqa: ARG002

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from .data_loader import create_training_datasets


class AnimeSegDataModule(L.LightningDataModule):
    """LightningDataModule that encapsulates all data loading logic for anime segmentation."""

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
        with_trimap: bool = False,
        cache_ratio: float = 0.0,
        cache_update_epoch: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None

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
                raise FileNotFoundError(f"Required data directory not found: {d}")

    def setup(self, stage: str | None = None) -> None:
        """Create datasets. Called on every process in distributed training."""
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = create_training_datasets(
                self.hparams["data_dir"],
                self.hparams["fg_dir"],
                self.hparams["bg_dir"],
                self.hparams["img_dir"],
                self.hparams["mask_dir"],
                self.hparams["fg_ext"],
                self.hparams["bg_ext"],
                self.hparams["img_ext"],
                self.hparams["mask_ext"],
                self.hparams["data_split"],
                self.hparams["img_size"],
                with_trimap=self.hparams["with_trimap"],
                cache_ratio=self.hparams["cache_ratio"],
                cache_update_epoch=self.hparams["cache_update_epoch"],
            )

        if stage == "validate" and self.val_dataset is None:
            _, self.val_dataset = create_training_datasets(
                self.hparams["data_dir"],
                self.hparams["fg_dir"],
                self.hparams["bg_dir"],
                self.hparams["img_dir"],
                self.hparams["mask_dir"],
                self.hparams["fg_ext"],
                self.hparams["bg_ext"],
                self.hparams["img_ext"],
                self.hparams["mask_ext"],
                self.hparams["data_split"],
                self.hparams["img_size"],
                with_trimap=self.hparams["with_trimap"],
                cache_ratio=0,
                cache_update_epoch=self.hparams["cache_update_epoch"],
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "setup() must be called before train_dataloader()"
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size_train"],
            shuffle=True,
            drop_last=True,  # Important for DDP consistency
            persistent_workers=True,
            num_workers=self.hparams["num_workers_train"],
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "setup() must be called before val_dataloader()"
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams["batch_size_val"],
            shuffle=False,
            persistent_workers=True,
            num_workers=self.hparams["num_workers_val"],
            pin_memory=True,
        )

    def state_dict(self) -> dict:
        """Save DataModule state for checkpointing."""
        return {
            "train_dataset_len": len(self.train_dataset) if self.train_dataset else 0,
            "val_dataset_len": len(self.val_dataset) if self.val_dataset else 0,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore DataModule state from checkpoint."""
