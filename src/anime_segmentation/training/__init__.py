"""BiRefNet training module with PyTorch Lightning."""

from anime_segmentation.constants import IMAGENET_MEAN, IMAGENET_STD

from .callbacks import (
    BackboneFreezeCallback,
    FinetuneCallback,
    HubUploadCallback,
    ScheduleFreeCallback,
    VisualizationCallback,
)
from .datamodule import (
    AnimeSegmentationDataModule,
    PairedTransform,
    SegmentationDataset,
)
from .lightning_module import BiRefNetLightning
from .loss import BaseLoss, ClsLoss, PixLoss, ThresholdRegularizationLoss
from .metrics import (
    EMeasureMetric,
    FMeasureMetric,
    IoUMetric,
    MAEMetric,
    SegmentationMetrics,
    SMeasureMetric,
)
from .model_card_template import MODEL_CARD_TEMPLATE
from .protocols import Finetunable, HasBackbone

__all__ = [
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MODEL_CARD_TEMPLATE",
    # Data
    "AnimeSegmentationDataModule",
    # Callbacks
    "BackboneFreezeCallback",
    # Loss
    "BaseLoss",
    # Core Module
    "BiRefNetLightning",
    "ClsLoss",
    # Metrics
    "EMeasureMetric",
    "FMeasureMetric",
    # Protocols
    "Finetunable",
    "FinetuneCallback",
    "HasBackbone",
    "HubUploadCallback",
    "IoUMetric",
    "MAEMetric",
    "PairedTransform",
    "PixLoss",
    "SMeasureMetric",
    "ScheduleFreeCallback",
    "SegmentationDataset",
    "SegmentationMetrics",
    "ThresholdRegularizationLoss",
    "VisualizationCallback",
]
