"""F-Measure metric with adaptive thresholding and precision-recall curve analysis.

This module implements the F-Measure metric for binary segmentation evaluation,
providing both adaptive threshold-based F-measure and comprehensive PR curve
analysis across multiple thresholds. The histogram-based implementation enables
efficient computation of precision-recall curves.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

_EPS = torch.finfo(torch.float64).eps


class FMeasure(Metric):
    """F-Measure for binary segmentation with adaptive threshold and PR curves.

    Computes multiple F-measure variants:
        - Adaptive F-measure using 2*mean(pred) as a data-driven threshold
        - Precision-Recall curves computed efficiently via histogram accumulation
        - Max F-measure (best threshold) and Mean F-measure (average across thresholds)

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W) with values in [0, 1] or [0, 255]
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W) with binary values

    As output of ``forward`` and ``compute`` the metric returns a dict:
        - ``maxF``: Maximum F-measure across all thresholds
        - ``meanF``: Mean F-measure across all thresholds
        - ``adpF``: Adaptive F-measure using 2*mean threshold
        - ``precision``: Precision curve (num_thresholds+1 points)
        - ``recall``: Recall curve (num_thresholds+1 points)
        - ``fm_curve``: F-measure curve (num_thresholds+1 points)

    Args:
        beta: Weight of recall vs precision in F-measure formula.
              Default is 0.3 (emphasizes precision over recall).
        num_thresholds: Number of thresholds for PR curve. Default is 256.
        **kwargs: Additional arguments passed to the base Metric class.

    Attributes:
        adaptive_fms: List of per-sample adaptive F-measure scores.
        fg_hist: Histogram of prediction values in foreground regions.
        bg_hist: Histogram of prediction values in background regions.
        gt_count: Total count of ground truth foreground pixels.
        num_samples: Number of samples processed.

    Example:
        >>> metric = FMeasure(beta=0.3, num_thresholds=256)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> gt = (torch.rand(4, 1, 256, 256) > 0.5).float()
        >>> metric.update(pred, gt)
        >>> results = metric.compute()
        >>> print(f"Max F: {results['maxF']:.4f}, Adaptive F: {results['adpF']:.4f}")
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    adaptive_fms: list[Tensor]
    fg_hist: Tensor
    bg_hist: Tensor
    gt_count: Tensor
    num_samples: Tensor

    def __init__(
        self,
        beta: float = 0.3,
        num_thresholds: int = 256,
        **kwargs: Any,
    ) -> None:
        """Initialize the F-Measure metric.

        Args:
            beta: Weight parameter for F-measure. Lower values emphasize precision.
            num_thresholds: Number of threshold levels for PR curve computation.
            **kwargs: Additional arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.beta_sq = beta**2
        self.num_thresholds = num_thresholds

        self.add_state("adaptive_fms", default=[], dist_reduce_fx="cat")
        self.add_state("fg_hist", default=torch.zeros(num_thresholds + 1), dist_reduce_fx="sum")
        self.add_state("bg_hist", default=torch.zeros(num_thresholds + 1), dist_reduce_fx="sum")
        self.add_state("gt_count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred: Predicted saliency map tensor.
            gt: Binary ground truth mask tensor.
        """
        pred, gt = self._prepare_inputs(pred, gt)

        batch_size = pred.shape[0] if pred.ndim > 2 else 1
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        for i in range(batch_size):
            p, g = pred[i], gt[i]
            self._update_single(p, g)

    def _update_single(self, pred: Tensor, gt: Tensor) -> None:
        """Update state for a single sample.

        Computes adaptive F-measure and accumulates histograms for PR curve.

        Args:
            pred: Single prediction tensor of shape (H, W).
            gt: Single ground truth tensor of shape (H, W).
        """
        gt_bool = gt.bool()

        adaptive_fm = self._compute_adaptive_fm(pred, gt_bool)
        self.adaptive_fms.append(adaptive_fm.unsqueeze(0))

        pred_uint8 = (pred * 255).to(torch.uint8)

        fg_hist = torch.histc(
            pred_uint8[gt_bool].float(),
            bins=self.num_thresholds + 1,
            min=0,
            max=self.num_thresholds,
        )
        bg_hist = torch.histc(
            pred_uint8[~gt_bool].float(),
            bins=self.num_thresholds + 1,
            min=0,
            max=self.num_thresholds,
        )

        self.fg_hist += fg_hist
        self.bg_hist += bg_hist
        self.gt_count += gt_bool.sum().float()
        self.num_samples += 1

    def _compute_adaptive_fm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute adaptive F-measure using a data-driven threshold.

        The threshold is set to 2*mean(pred), capped at 1.0, which adapts
        to the overall intensity distribution of the prediction.

        Args:
            pred: Prediction tensor of shape (H, W).
            gt: Boolean ground truth tensor of shape (H, W).

        Returns:
            Scalar tensor with the adaptive F-measure score.
        """
        adaptive_threshold = min(2 * pred.mean().item(), 1.0)
        binary_pred = pred >= adaptive_threshold

        tp = (binary_pred & gt).sum().float()
        pred_positive = binary_pred.sum().float()
        gt_positive = gt.sum().float()

        if tp == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        precision = tp / (pred_positive + _EPS)
        recall = tp / (gt_positive + _EPS)

        return (1 + self.beta_sq) * precision * recall / (self.beta_sq * precision + recall + _EPS)

    def compute(self) -> dict[str, Tensor]:
        """Compute all F-measure metrics from accumulated state.

        Derives PR curves from histograms using cumulative sums, avoiding
        the need to store per-sample predictions.

        Returns:
            Dictionary containing maxF, meanF, adpF, precision, recall, and fm_curve.
        """
        if len(self.adaptive_fms) > 0:
            adaptive_fm = torch.cat(self.adaptive_fms).mean()
        else:
            adaptive_fm = torch.tensor(0.0, device=self.device)

        # Cumulative sums from high to low threshold for efficient TP/FP computation
        fg_cumsum = torch.flip(torch.cumsum(torch.flip(self.fg_hist, [0]), 0), [0])
        bg_cumsum = torch.flip(torch.cumsum(torch.flip(self.bg_hist, [0]), 0), [0])

        tp = fg_cumsum
        fp = bg_cumsum

        precision = tp / (tp + fp + _EPS)
        recall = tp / (self.gt_count + _EPS)

        numerator = (1 + self.beta_sq) * precision * recall
        denominator = torch.where(
            numerator == 0,
            torch.ones_like(numerator),
            self.beta_sq * precision + recall,
        )
        fm_curve = numerator / (denominator + _EPS)

        return {
            "maxF": fm_curve.max(),
            "meanF": fm_curve.mean(),
            "adpF": adaptive_fm,
            "precision": precision,
            "recall": recall,
            "fm_curve": fm_curve,
        }

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

        if pred.ndim == 3:
            for i in range(pred.shape[0]):
                p = pred[i]
                if p.max() != p.min():
                    pred[i] = (p - p.min()) / (p.max() - p.min())
        elif pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())

        gt = (gt > 0.5).float()

        return pred, gt
