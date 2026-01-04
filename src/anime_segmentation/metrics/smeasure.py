"""Structural Measure (S-Measure) metric for salient object detection.

This module implements the S-Measure metric from Fan et al. (ICCV 2017),
which evaluates structural similarity by combining object-aware and
region-aware components. Unlike pixel-wise metrics, S-Measure captures
the structural integrity of predicted saliency maps.

Reference:
    Fan, D.-P., Cheng, M.-M., Liu, Y., Li, T., & Borji, A. (2017).
    Structure-measure: A new way to evaluate foreground maps. ICCV.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric

_EPS = torch.finfo(torch.float64).eps


class StructuralMeasure(Metric):
    """Structural Measure (S-Measure) for salient object detection.

    Combines object-aware and region-aware structural similarity to evaluate
    the structural integrity of predicted saliency maps. The object component
    measures foreground/background separation quality, while the region
    component uses quadrant-based SSIM-like comparisons.

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W) with values in [0, 1] or [0, 255]
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W) with binary values

    As output of ``forward`` and ``compute`` the metric returns:
        - ``Sm``: Structural measure scalar in range [0, 1]

    Args:
        alpha: Balance between object and region similarity. Default is 0.5.
        **kwargs: Additional arguments passed to the base Metric class.

    Attributes:
        sm_scores: List of per-sample structural measure scores.

    Example:
        >>> metric = StructuralMeasure(alpha=0.5)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> gt = (torch.rand(4, 1, 256, 256) > 0.5).float()
        >>> metric.update(pred, gt)
        >>> sm = metric.compute()
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    sm_scores: list[Tensor]

    def __init__(self, alpha: float = 0.5, **kwargs: Any) -> None:
        """Initialize the S-Measure metric.

        Args:
            alpha: Weight for combining object (So) and region (Sr) scores.
                   Final score = alpha * So + (1 - alpha) * Sr.
            **kwargs: Additional arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.add_state("sm_scores", default=[], dist_reduce_fx="cat")

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
            sm = self._compute_sm(p, g)
            self.sm_scores.append(sm.unsqueeze(0))

    def _compute_sm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute structural measure for a single sample.

        Handles edge cases where ground truth is entirely foreground or background,
        then combines object and region scores for mixed cases.

        Args:
            pred: Single prediction tensor of shape (H, W).
            gt: Single ground truth tensor of shape (H, W).

        Returns:
            Scalar tensor with the structural measure score.
        """
        gt_bool = gt.bool()
        y = gt.float().mean()

        if y == 0:
            # Empty ground truth: penalize false positives
            sm = 1.0 - pred.mean()
        elif y == 1:
            # Full foreground: reward coverage
            sm = pred.mean()
        else:
            so = self._object_score(pred, gt_bool)
            sr = self._region_score(pred, gt_bool)
            sm = self.alpha * so + (1 - self.alpha) * sr
            sm = max(0.0, sm)

        return torch.tensor(sm, device=pred.device, dtype=pred.dtype)

    def _object_score(self, pred: Tensor, gt: Tensor) -> float:
        """Compute object-aware structural similarity.

        Evaluates how well the prediction separates foreground from background
        by computing weighted scores for both regions.

        Args:
            pred: Prediction tensor of shape (H, W).
            gt: Boolean ground truth tensor of shape (H, W).

        Returns:
            Object-aware structural similarity score.
        """
        fg = pred * gt.float()
        bg = (1 - pred) * (1 - gt.float())

        u = gt.float().mean().item()

        fg_score = self._s_object(fg, gt)
        bg_score = self._s_object(bg, ~gt)

        return u * fg_score + (1 - u) * bg_score

    def _s_object(self, pred: Tensor, gt: Tensor) -> float:
        """Compute similarity score for a single region (foreground or background).

        Args:
            pred: Masked prediction values for the region.
            gt: Boolean mask indicating the region.

        Returns:
            Similarity score based on mean and variance of predictions in the region.
        """
        gt_float = gt.float()

        if gt_float.sum() == 0:
            return 0.0

        x = pred[gt].mean().item()
        sigma_x = pred[gt].std(unbiased=True).item() if gt.sum() > 1 else 0.0

        score = 2 * x / (x**2 + 1 + sigma_x + _EPS)
        return float(score)

    def _region_score(self, pred: Tensor, gt: Tensor) -> float:
        """Compute region-aware structural similarity using quadrant decomposition.

        Divides the image into four quadrants centered at the ground truth centroid
        and computes SSIM-like scores for each, weighted by quadrant area.

        Args:
            pred: Prediction tensor of shape (H, W).
            gt: Boolean ground truth tensor of shape (H, W).

        Returns:
            Area-weighted average of quadrant similarity scores.
        """
        x, y = self._centroid(gt)
        parts = self._divide_with_xy(pred, gt, x, y)

        w1, w2, w3, w4 = parts["weight"]
        pred1, pred2, pred3, pred4 = parts["pred"]
        gt1, gt2, gt3, gt4 = parts["gt"]

        score1 = self._ssim(pred1, gt1)
        score2 = self._ssim(pred2, gt2)
        score3 = self._ssim(pred3, gt3)
        score4 = self._ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def _centroid(self, gt: Tensor) -> tuple[int, int]:
        """Compute the centroid of foreground pixels in ground truth.

        Args:
            gt: Boolean ground truth tensor of shape (H, W).

        Returns:
            Tuple (x, y) of centroid coordinates (1-indexed for slicing).
        """
        h, w = gt.shape
        gt_float = gt.float()

        if gt_float.sum() == 0:
            return round(w / 2), round(h / 2)

        indices = torch.nonzero(gt_float, as_tuple=True)
        y_mean = indices[0].float().mean().round().int().item()
        x_mean = indices[1].float().mean().round().int().item()

        return x_mean + 1, y_mean + 1

    def _divide_with_xy(self, pred: Tensor, gt: Tensor, x: int, y: int) -> dict[str, Any]:
        """Divide prediction and ground truth into four quadrants around centroid.

        Args:
            pred: Prediction tensor of shape (H, W).
            gt: Ground truth tensor of shape (H, W).
            x: Horizontal split coordinate.
            y: Vertical split coordinate.

        Returns:
            Dictionary with 'gt', 'pred', and 'weight' keys containing
            tuples of (top-left, top-right, bottom-left, bottom-right) values.
        """
        h, w = gt.shape
        area = h * w

        gt_lt = gt[0:y, 0:x].float()
        gt_rt = gt[0:y, x:w].float()
        gt_lb = gt[y:h, 0:x].float()
        gt_rb = gt[y:h, x:w].float()

        pred_lt = pred[0:y, 0:x]
        pred_rt = pred[0:y, x:w]
        pred_lb = pred[y:h, 0:x]
        pred_rb = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return {
            "gt": (gt_lt, gt_rt, gt_lb, gt_rb),
            "pred": (pred_lt, pred_rt, pred_lb, pred_rb),
            "weight": (w1, w2, w3, w4),
        }

    def _ssim(self, pred: Tensor, gt: Tensor) -> float:
        """Compute simplified SSIM-like score for a quadrant region.

        Uses mean, variance, and covariance to measure structural similarity
        between prediction and ground truth in the given region.

        Args:
            pred: Prediction tensor for the quadrant.
            gt: Ground truth tensor for the quadrant.

        Returns:
            SSIM-like similarity score for the region.
        """
        if pred.numel() == 0:
            return 1.0

        h, w = pred.shape
        n = h * w

        x = pred.mean().item()
        y = gt.mean().item()

        sigma_x = ((pred - x) ** 2).sum().item() / max(n - 1, 1)
        sigma_y = ((gt - y) ** 2).sum().item() / max(n - 1, 1)
        sigma_xy = ((pred - x) * (gt - y)).sum().item() / max(n - 1, 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif beta == 0:
            score = 1.0
        else:
            score = 0.0

        return float(score)

    def compute(self) -> Tensor:
        """Compute mean structural measure over all accumulated samples.

        Returns:
            Scalar tensor with the mean S-Measure score.
        """
        if len(self.sm_scores) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.cat(self.sm_scores).mean()

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
