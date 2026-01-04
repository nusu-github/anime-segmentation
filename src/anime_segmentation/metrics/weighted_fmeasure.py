"""Weighted F-Measure (wFm) metric with distance-based error weighting.

This module implements the Weighted F-Measure, which applies distance-based
weighting to errors. Errors occurring far from ground truth boundaries are
penalized more heavily than errors near boundaries, reflecting that boundary
localization is inherently more ambiguous.

The implementation uses scipy for distance transform and convolution operations,
requiring CPU computation for these steps.
"""

from __future__ import annotations

from typing import Any

import torch
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist
from torch import Tensor
from torchmetrics import Metric

_EPS = torch.finfo(torch.float64).eps


class WeightedFMeasure(Metric):
    """Weighted F-Measure for binary segmentation evaluation.

    Uses distance transform to weight errors based on proximity to ground truth
    boundaries. Errors near boundaries are weighted less heavily than errors
    far from boundaries, acknowledging boundary localization ambiguity.

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W) with values in [0, 1] or [0, 255]
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W) with binary values

    As output of ``forward`` and ``compute`` the metric returns:
        - ``wFm``: Weighted F-measure scalar in range [0, 1]

    Args:
        beta: Weight parameter for precision-recall balance.
              Default is 1.0 (equal weight for precision and recall).
        **kwargs: Additional arguments passed to the base Metric class.

    Attributes:
        weighted_fms: List of per-sample weighted F-measure scores.

    Note:
        This metric uses scipy for distance transform and convolution operations,
        so computations happen on CPU. Consider batching for efficiency.

    Example:
        >>> metric = WeightedFMeasure(beta=1.0)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> gt = (torch.rand(4, 1, 256, 256) > 0.5).float()
        >>> metric.update(pred, gt)
        >>> wfm = metric.compute()
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    weighted_fms: list[Tensor]

    def __init__(self, beta: float = 1.0, **kwargs: Any) -> None:
        """Initialize the Weighted F-Measure metric.

        Args:
            beta: Weight parameter balancing precision and recall contributions.
            **kwargs: Additional arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.add_state("weighted_fms", default=[], dist_reduce_fx="cat")

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
            wfm = self._compute_wfm(p, g)
            self.weighted_fms.append(wfm.unsqueeze(0))

    def _compute_wfm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute weighted F-measure for a single sample.

        The algorithm:
        1. Compute distance transform from background pixels
        2. Propagate errors from foreground to nearest background pixels
        3. Apply Gaussian smoothing to error map
        4. Weight errors inversely with distance from boundaries
        5. Compute weighted precision and recall

        Args:
            pred: Single prediction tensor of shape (H, W).
            gt: Single ground truth tensor of shape (H, W).

        Returns:
            Scalar tensor with the weighted F-measure score.
        """
        gt_bool = gt.bool()

        if not gt_bool.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        pred_np = pred.detach().cpu().numpy()
        gt_np = gt_bool.cpu().numpy()

        # Distance transform from background to foreground
        dst, idxt = bwdist(gt_np == 0, return_indices=True)

        e = abs(pred_np - gt_np.astype(pred_np.dtype))
        et = e.copy()

        # Propagate errors to nearest foreground pixel locations
        et[gt_np == 0] = et[idxt[0][gt_np == 0], idxt[1][gt_np == 0]]

        k = self._matlab_style_gauss2d((7, 7), sigma=5)
        ea = convolve(et, weights=k, mode="constant", cval=0)

        # Use smoothed error where it reduces foreground error
        min_e_ea = (gt_np & (ea < e)) * ea + (~(gt_np & (ea < e))) * e

        # Distance-based weights: higher weight for errors far from boundaries
        b = (gt_np == 0) * (2 - (-dst / 5.0).clip(max=0).clip(min=-1) - 1) + (gt_np != 0) * 1.0

        ew = min_e_ea * b

        tpw = gt_np.sum() - ew[gt_np == 1].sum()
        fpw = ew[gt_np == 0].sum()

        r = 1 - ew[gt_np == 1].mean() if gt_np.sum() > 0 else 0.0
        p = tpw / (tpw + fpw + _EPS)

        q = (1 + self.beta) * r * p / (r + self.beta * p + _EPS)

        return torch.tensor(q, device=pred.device, dtype=pred.dtype)

    @staticmethod
    def _matlab_style_gauss2d(shape: tuple[int, int] = (7, 7), sigma: float = 5.0) -> Any:
        """Create a 2D Gaussian kernel matching MATLAB's fspecial('gaussian').

        Args:
            shape: Kernel dimensions (height, width).
            sigma: Standard deviation of the Gaussian.

        Returns:
            Normalized 2D Gaussian kernel as numpy array.
        """
        import numpy as np

        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def compute(self) -> Tensor:
        """Compute mean weighted F-measure over all accumulated samples.

        Returns:
            Scalar tensor with the mean weighted F-measure score.
        """
        if len(self.weighted_fms) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.cat(self.weighted_fms).mean()

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
