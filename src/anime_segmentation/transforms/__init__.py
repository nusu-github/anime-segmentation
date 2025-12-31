"""Anime segmentation transforms based on torchvision.transforms.v2."""

from .augmentation import (
    JPEGCompression,
    RandomColorBlocks,
    RandomTextOverlay,
    ResizeBlur,
    SharpBackground,
    SimulateLight,
    SketchConvert,
)
from .color import RandomColor
from .geometric import RescalePad

__all__ = [
    "JPEGCompression",
    "RandomColor",
    "RandomColorBlocks",
    "RandomTextOverlay",
    "RescalePad",
    "ResizeBlur",
    "SharpBackground",
    "SimulateLight",
    "SketchConvert",
]
