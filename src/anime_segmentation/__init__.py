"""Anime Segmentation: IS-Net focused anime character segmentation.

This package provides a PyTorch Lightning-based implementation for training
and inference of IS-Net models for anime character segmentation and
background removal tasks.

Modules:
    data_loader: Dataset classes and data loading utilities.
    lightning: PyTorch Lightning modules for training.
    train_cli: Command-line interface for training and inference.
"""

from . import train_cli
from .data_loader import (
    AugmentationConfig,
    GOSDataset,
    GOSNormalize,
    create_dataloaders,
    get_im_gt_name_dict,
)
from .lightning import (
    AnimeSegDataModule,
    GTEncoderLightningModule,
    ISNetLightningModule,
)

__all__ = [
    "AnimeSegDataModule",
    "AugmentationConfig",
    "GOSDataset",
    "GOSNormalize",
    "GTEncoderLightningModule",
    "ISNetLightningModule",
    "create_dataloaders",
    "get_im_gt_name_dict",
    "train_cli",
]
