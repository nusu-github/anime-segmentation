from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .data_loader import create_training_datasets


class AnimeSegDataModule(pl.LightningDataModule):
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
        workers_train: int = 4,
        workers_val: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.fg_ext = fg_ext
        self.bg_ext = bg_ext
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.data_split = float(data_split)
        self.img_size = int(img_size)

        self.batch_size_train = int(batch_size_train)
        self.batch_size_val = int(batch_size_val)
        self.workers_train = int(workers_train)
        self.workers_val = int(workers_val)

        self.train_dataset: Any | None = None
        self.val_dataset: Any | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            return

        train_dataset, val_dataset = create_training_datasets(
            self.data_dir,
            self.fg_dir,
            self.bg_dir,
            self.img_dir,
            self.mask_dir,
            self.fg_ext,
            self.bg_ext,
            self.img_ext,
            self.mask_ext,
            self.data_split,
            self.img_size,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            msg = "DataModule is not set up (train_dataset is None)"
            raise RuntimeError(msg)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            persistent_workers=self.workers_train > 0,
            num_workers=self.workers_train,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            msg = "DataModule is not set up (val_dataset is None)"
            raise RuntimeError(msg)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            persistent_workers=self.workers_val > 0,
            num_workers=self.workers_val,
            pin_memory=True,
        )
