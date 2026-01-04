"""Mean Absolute Error (MAE) metric for binary segmentation evaluation.

This module implements the Mean Absolute Error metric, which measures
the average pixel-wise absolute difference between predicted saliency
maps and binary ground truth masks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) for binary segmentation masks.

    Computes the mean absolute difference between predicted and ground truth masks.
    Predictions are normalized to [0, 1] range before computation.

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W) with values in [0, 1] or [0, 255]
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W) with binary values

    As output of ``forward`` and ``compute`` the metric returns:
        - ``mae``: Scalar tensor with the mean absolute error

    Attributes:
        is_differentiable: Whether the metric supports gradient computation.
        higher_is_better: Whether higher values indicate better performance.
        full_state_update: Whether update requires full state recomputation.
        sum_abs_error: Accumulated sum of absolute errors across all pixels.
        total_pixels: Total number of pixels processed.

    Example:
        >>> metric = MeanAbsoluteError()
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> gt = (torch.rand(4, 1, 256, 256) > 0.5).float()
        >>> metric.update(pred, gt)
        >>> mae = metric.compute()
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    sum_abs_error: Tensor
    total_pixels: Tensor

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the MAE metric.

        Args:
            **kwargs: Additional arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred: Predicted mask tensor with values in [0, 1] or [0, 255].
            gt: Ground truth mask tensor, binarized at threshold 0.5.
        """
        pred, gt = self._prepare_inputs(pred, gt)

        self.sum_abs_error += torch.abs(pred - gt).sum()
        self.total_pixels += pred.numel()

    def compute(self) -> Tensor:
        """Compute the mean absolute error over all accumulated samples.

        Returns:
            Scalar tensor containing the MAE value in range [0, 1].
        """
        return self.sum_abs_error / self.total_pixels

    def _prepare_inputs(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        """Normalize inputs to [0, 1] range and apply standard preprocessing.

        Args:
            pred: Raw prediction tensor.
            gt: Raw ground truth tensor.

        Returns:
            Tuple of normalized prediction and binarized ground truth tensors.
        """
        pred = pred.float()
        gt = gt.float()

        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt.squeeze(1)

        if pred.max() > 1.0:
            pred /= 255.0

        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())

        gt = (gt > 0.5).float()

        return pred, gt
