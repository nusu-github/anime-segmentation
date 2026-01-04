"""TorchMetrics-based evaluation metrics for binary segmentation.

This module provides a comprehensive set of metrics for evaluating binary
segmentation models, particularly for salient object detection tasks.
All metrics are implemented as TorchMetrics classes supporting distributed
training and automatic state synchronization.

Metrics:
    MeanAbsoluteError (MAE): Pixel-wise absolute error between prediction and ground truth.
    StructuralMeasure (Sm): Structure-aware similarity combining object and region scores.
    FMeasure (Fm): Adaptive and threshold-based F-measure with PR curves.
    WeightedFMeasure (wFm): Distance-weighted F-measure penalizing errors far from boundaries.
    BoundaryIoU (BIoU): Intersection over union computed on boundary regions.
    MeanBoundaryAccuracy (MBA): Multi-scale boundary region accuracy.
    HumanCorrectionEffort (HCE): Estimated control points needed to correct errors.

Example:
    >>> from anime_segmentation.metrics import SegmentationMetrics
    >>> metrics = SegmentationMetrics(metrics=("F", "MAE", "S"))
    >>> metrics.update(pred, gt)
    >>> results = metrics.compute()
"""

from .biou import BoundaryIoU
from .collection import SegmentationMetrics, build_metrics
from .fmeasure import FMeasure
from .hce import HumanCorrectionEffort
from .mae import MeanAbsoluteError
from .mba import MeanBoundaryAccuracy
from .smeasure import StructuralMeasure
from .weighted_fmeasure import WeightedFMeasure

# Short aliases for backward compatibility with legacy codebases
MAE = MeanAbsoluteError
HCE = HumanCorrectionEffort
MBA = MeanBoundaryAccuracy
BIoU = BoundaryIoU
Sm = StructuralMeasure
Fm = FMeasure
wFm = WeightedFMeasure

__all__ = [
    "HCE",
    "MAE",
    "MBA",
    # Short aliases (legacy compatibility)
    "BIoU",
    # Primary TorchMetrics classes
    "BoundaryIoU",
    "FMeasure",
    "Fm",
    "HumanCorrectionEffort",
    "MeanAbsoluteError",
    "MeanBoundaryAccuracy",
    # Collection and factory utilities
    "SegmentationMetrics",
    "Sm",
    "StructuralMeasure",
    "WeightedFMeasure",
    "build_metrics",
    "wFm",
]
