"""Anime segmentation datasets."""

from .combined import CombinedDataset
from .real import RealImageDataset
from .synthetic import SyntheticDataset

__all__ = [
    "CombinedDataset",
    "RealImageDataset",
    "SyntheticDataset",
]
