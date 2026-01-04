"""Boundary IoU (BIoU) metric for segmentation evaluation.

This module implements Boundary IoU, which computes intersection over union
specifically for boundary regions of segmentation masks. Boundaries are
extracted using morphological erosion, and the metric evaluates how well
predicted boundaries align with ground truth boundaries.

The implementation uses Kornia for GPU-accelerated morphological operations.
"""

from __future__ import annotations

from typing import Any

import kornia.morphology as km
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric


class BoundaryIoU(Metric):
    """Boundary IoU (BIoU) for segmentation boundary evaluation.

    Computes IoU specifically for boundary regions of the masks, providing
    a focused evaluation of boundary localization accuracy. Boundaries are
    extracted by morphological erosion, with width proportional to image size.

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W) with values in [0, 255] or [0, 1]
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W) with binary values

    As output of ``forward`` and ``compute`` the metric returns a dict:
        - ``maxBIoU``: Maximum boundary IoU across all thresholds
        - ``meanBIoU``: Mean boundary IoU across all thresholds
        - ``curve``: BIoU curve (num_thresholds points)

    Args:
        dilation_ratio: Ratio of image diagonal for boundary width.
                        Default is 0.02 (2% of diagonal).
        num_thresholds: Number of thresholds for BIoU curve. Default is 256.
        **kwargs: Additional arguments passed to the base Metric class.

    Attributes:
        biou_curves: List of per-sample BIoU curves across thresholds.

    Example:
        >>> metric = BoundaryIoU(dilation_ratio=0.02, num_thresholds=256)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> gt = (torch.rand(4, 1, 256, 256) > 0.5).float()
        >>> metric.update(pred, gt)
        >>> results = metric.compute()
        >>> print(f"Max BIoU: {results['maxBIoU']:.4f}")
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    biou_curves: list[Tensor]

    def __init__(
        self,
        dilation_ratio: float = 0.02,
        num_thresholds: int = 256,
        **kwargs: Any,
    ) -> None:
        """Initialize the Boundary IoU metric.

        Args:
            dilation_ratio: Boundary width as fraction of image diagonal.
            num_thresholds: Number of threshold levels for curve computation.
            **kwargs: Additional arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)
        self.dilation_ratio = dilation_ratio
        self.num_thresholds = num_thresholds
        self.add_state("biou_curves", default=[], dist_reduce_fx="cat")

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
            biou = self._compute_biou(p, g)
            self.biou_curves.append(biou.unsqueeze(0))

    def _compute_biou(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute Boundary IoU curve for a single sample.

        Extracts boundaries from both prediction and ground truth,
        then computes IoU across multiple binarization thresholds.

        Args:
            pred: Single prediction tensor of shape (H, W).
            gt: Single ground truth tensor of shape (H, W).

        Returns:
            Tensor of shape (num_thresholds,) with BIoU values per threshold.
        """
        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()

        pred_boundary = self._mask_to_boundary(
            (pred_np * 255).astype(np.uint8),
            gt_np.shape,
        )
        gt_boundary = self._mask_to_boundary(
            (gt_np * 255).astype(np.uint8),
            gt_np.shape,
        )

        gt_boundary_bool = gt_boundary > 128
        gt_count = max(np.count_nonzero(gt_boundary_bool), 1)

        bins = np.linspace(0, 256, self.num_thresholds + 1)
        fg_hist, _ = np.histogram(pred_boundary[gt_boundary_bool], bins=bins)
        bg_hist, _ = np.histogram(pred_boundary[~gt_boundary_bool], bins=bins)

        fg_cumsum = np.cumsum(np.flip(fg_hist), axis=0)
        bg_cumsum = np.cumsum(np.flip(bg_hist), axis=0)

        tp = fg_cumsum
        fp = bg_cumsum

        biou = tp / (gt_count + fp + 1e-8)

        return torch.tensor(biou, device=pred.device, dtype=torch.float32)

    def _mask_to_boundary(self, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Extract boundary region from a mask using morphological erosion.

        The boundary width is computed as a fraction of the image diagonal,
        ensuring consistent relative boundary thickness across image sizes.

        Args:
            mask: Input mask as uint8 array with values in [0, 255].
            shape: Original image shape (H, W) for diagonal computation.

        Returns:
            Boundary mask as float array with values in [0, 255].
        """
        h, w = shape
        img_diag = np.sqrt(h**2 + w**2)
        dilation_px = max(1, round(self.dilation_ratio * img_diag))

        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        padded = torch.nn.functional.pad(mask_tensor, (1, 1, 1, 1), mode="constant", value=0.0)
        kernel = torch.ones((3, 3), dtype=padded.dtype, device=padded.device)

        mask_erode = padded
        for _ in range(dilation_px):
            mask_erode = km.erosion(mask_erode, kernel, border_type="constant", border_value=0.0)

        mask_erode = mask_erode[:, :, 1 : h + 1, 1 : w + 1]

        boundary = mask_tensor - mask_erode
        return boundary.squeeze(0).squeeze(0).cpu().numpy()

    def compute(self) -> dict[str, Tensor]:
        """Compute Boundary IoU metrics from accumulated curves.

        Returns:
            Dictionary containing:
                - maxBIoU: Best BIoU across all thresholds
                - meanBIoU: Average BIoU across all thresholds
                - curve: Full BIoU curve for analysis
        """
        if len(self.biou_curves) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "maxBIoU": zero,
                "meanBIoU": zero,
                "curve": torch.zeros(self.num_thresholds, device=self.device),
            }

        curves = torch.cat(self.biou_curves, dim=0)
        mean_curve = curves.mean(dim=0)

        return {
            "maxBIoU": mean_curve.max(),
            "meanBIoU": mean_curve.mean(),
            "curve": mean_curve,
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
