"""Anime segmentation datasets."""

from .real import RealImageDataset
from .synthetic import SyntheticDataset

__all__ = [
    "RealImageDataset",
    "SyntheticDataset",
]
