"""PyTorch Lightning integration for IS-Net training and inference.

This module provides Lightning-compatible wrappers for IS-Net models and data handling:

- :class:`AnimeSegDataModule`: Data pipeline with configurable augmentation presets
- :class:`ISNetLightningModule`: Training wrapper for ISNet segmentation models
- :class:`GTEncoderLightningModule`: Training wrapper for ground truth encoder models
"""

from .data import AnimeSegDataModule
from .modules import CompileMode, GTEncoderLightningModule, ISNetLightningModule

__all__ = [
    "AnimeSegDataModule",
    "CompileMode",
    "GTEncoderLightningModule",
    "ISNetLightningModule",
]
