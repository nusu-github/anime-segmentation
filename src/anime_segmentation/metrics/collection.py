"""Metric collection and factory utilities for binary segmentation evaluation.

This module provides a unified interface for computing multiple segmentation
metrics simultaneously. The SegmentationMetrics class wraps individual metrics
into a TorchMetrics MetricCollection, enabling synchronized updates and
distributed training support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from torch import Tensor
from torchmetrics import MetricCollection

from .biou import BoundaryIoU
from .fmeasure import FMeasure
from .hce import HumanCorrectionEffort
from .mae import MeanAbsoluteError
from .mba import MeanBoundaryAccuracy
from .smeasure import StructuralMeasure
from .weighted_fmeasure import WeightedFMeasure

if TYPE_CHECKING:
    from collections.abc import Iterable

METRIC_ALIASES: dict[str, str] = {
    "F": "FMeasure",
    "WF": "WeightedFMeasure",
    "MAE": "MeanAbsoluteError",
    "S": "StructuralMeasure",
    "HCE": "HumanCorrectionEffort",
    "MBA": "MeanBoundaryAccuracy",
    "BIoU": "BoundaryIoU",
}
"""Mapping from short metric codes to full class names."""

MetricName = Literal["F", "WF", "MAE", "S", "HCE", "MBA", "BIoU"]
"""Valid metric name codes for SegmentationMetrics configuration."""


class SegmentationMetrics(MetricCollection):
    """Collection of segmentation metrics based on TorchMetrics.

    A convenience wrapper around MetricCollection that provides all standard
    binary segmentation evaluation metrics. Supports distributed training
    through automatic state synchronization.

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W)
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W)
        - ``gt_ske``: Optional skeleton tensor for HCE metric

    As output of ``forward`` and ``compute`` the metric returns a dict
    with all computed metric values.

    Args:
        metrics: Tuple of metric codes to include.
            Valid options: "F", "WF", "MAE", "S", "HCE", "MBA", "BIoU"
        prefix: Optional prefix for metric names in output dict.
        postfix: Optional postfix for metric names in output dict.
        **kwargs: Additional arguments passed to MetricCollection.

    Attributes:
        _metric_names: Tuple of metric codes configured for this collection.

    Example:
        >>> metrics = SegmentationMetrics(metrics=("F", "MAE", "S"))
        >>> pred = torch.rand(2, 1, 256, 256)
        >>> gt = (torch.rand(2, 1, 256, 256) > 0.5).float()
        >>> metrics.update(pred, gt)
        >>> results = metrics.compute()
    """

    def __init__(
        self,
        metrics: tuple[MetricName, ...] = ("F", "WF", "MAE", "S", "HCE", "MBA", "BIoU"),
        prefix: str | None = None,
        postfix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the metric collection with specified metrics.

        Args:
            metrics: Tuple of metric codes to include in the collection.
            prefix: Optional prefix for metric names in output.
            postfix: Optional postfix for metric names in output.
            **kwargs: Additional arguments passed to MetricCollection.
        """
        metric_dict = {}

        if "F" in metrics:
            metric_dict["FMeasure"] = FMeasure()
        if "WF" in metrics:
            metric_dict["WeightedFMeasure"] = WeightedFMeasure()
        if "MAE" in metrics:
            metric_dict["MeanAbsoluteError"] = MeanAbsoluteError()
        if "S" in metrics:
            metric_dict["StructuralMeasure"] = StructuralMeasure()
        if "HCE" in metrics:
            metric_dict["HumanCorrectionEffort"] = HumanCorrectionEffort()
        if "MBA" in metrics:
            metric_dict["MeanBoundaryAccuracy"] = MeanBoundaryAccuracy()
        if "BIoU" in metrics:
            metric_dict["BoundaryIoU"] = BoundaryIoU()

        super().__init__(metric_dict, prefix=prefix, postfix=postfix, **kwargs)
        self._metric_names = metrics

    def update(
        self,
        pred: Tensor,
        gt: Tensor,
        gt_ske: Tensor | None = None,
    ) -> None:
        """Update all metrics with predictions and targets.

        HCE metric receives the optional skeleton tensor; other metrics
        receive only predictions and ground truth.

        Args:
            pred: Predicted mask tensor.
            gt: Ground truth mask tensor.
            gt_ske: Optional pre-computed skeleton for HCE metric.
        """
        for name, metric in self.items():
            is_hce = "HumanCorrectionEffort" in name
            if is_hce:
                metric.update(pred, gt, gt_ske)
            else:
                metric.update(pred, gt)

    def compute(self) -> dict[str, Tensor]:
        """Compute all metrics and return flattened results.

        Metrics returning dictionaries (FMeasure, BIoU) have their scalar
        values flattened with "/" separator (e.g., "FMeasure/maxF").

        Returns:
            Dictionary mapping metric names to computed tensor values.
        """
        results = {}

        for name, metric in self.items():
            value = metric.compute()

            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, Tensor) and v.ndim == 0:
                        results[f"{name}/{k}"] = v
            else:
                results[name] = value

        return results

    def get_results(self) -> dict[str, float]:
        """Get results in legacy format for backward compatibility.

        Converts tensor results to Python floats with standardized keys
        matching the original BiRefNetMetrics format.

        Returns:
            Dictionary with legacy metric keys:
                - maxF, meanF, adpF (from FMeasure)
                - wFm (from WeightedFMeasure)
                - MAE (from MeanAbsoluteError)
                - Sm (from StructuralMeasure)
                - HCE (from HumanCorrectionEffort)
                - MBA (from MeanBoundaryAccuracy)
                - maxBIoU, meanBIoU (from BoundaryIoU)
        """
        results: dict[str, float] = {}

        for name, metric in self.items():
            prefix = self.prefix if hasattr(self, "prefix") and self.prefix else ""
            postfix = self.postfix if hasattr(self, "postfix") and self.postfix else ""
            base_name = name.replace(prefix, "").replace(postfix, "")
            value = metric.compute()

            if base_name == "FMeasure" and isinstance(value, dict):
                results["maxF"] = float(value["maxF"])
                results["meanF"] = float(value["meanF"])
                results["adpF"] = float(value["adpF"])
            elif base_name == "WeightedFMeasure":
                results["wFm"] = float(value)
            elif base_name == "MeanAbsoluteError":
                results["MAE"] = float(value)
            elif base_name == "StructuralMeasure":
                results["Sm"] = float(value)
            elif base_name == "HumanCorrectionEffort":
                results["HCE"] = float(value)
            elif base_name == "MeanBoundaryAccuracy":
                results["MBA"] = float(value)
            elif base_name == "BoundaryIoU" and isinstance(value, dict):
                results["maxBIoU"] = float(value["maxBIoU"])
                results["meanBIoU"] = float(value["meanBIoU"])

        return results


def build_metrics(metric_names: Iterable[str]) -> SegmentationMetrics:
    """Factory function to build a SegmentationMetrics collection.

    Args:
        metric_names: Iterable of metric codes (e.g., ["F", "MAE", "S"]).

    Returns:
        Configured SegmentationMetrics collection.

    Example:
        >>> metrics = build_metrics(["F", "MAE", "S"])
        >>> metrics.update(pred, gt)
        >>> results = metrics.compute()
    """
    return SegmentationMetrics(metrics=tuple(metric_names))  # type: ignore[arg-type]
