"""Mean Boundary Accuracy (MBA) metric for segmentation evaluation.

This module implements Mean Boundary Accuracy, which evaluates prediction
accuracy specifically within boundary regions at multiple scales. The
multi-scale evaluation captures both fine and coarse boundary alignment.

The implementation uses Kornia for GPU-accelerated morphological operations.
"""

from __future__ import annotations

from typing import Any

import kornia
import kornia.morphology as km
import torch
from torch import Tensor
from torchmetrics import Metric


class MeanBoundaryAccuracy(Metric):
    """Mean Boundary Accuracy (MBA) for multi-scale boundary evaluation.

    Evaluates prediction accuracy along object boundaries at multiple scales,
    from fine to coarse. Uses morphological gradient (dilation - erosion) to
    extract boundary regions and measures pixel-wise accuracy within those regions.

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W) with values in [0, 255] or [0, 1]
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W) with binary values

    As output of ``forward`` and ``compute`` the metric returns:
        - ``MBA``: Mean boundary accuracy score in range [0, 1]

    Args:
        num_steps: Number of boundary thickness scales to evaluate. Default is 5.
        **kwargs: Additional arguments passed to the base Metric class.

    Attributes:
        mba_scores: List of per-sample MBA scores.

    Note:
        This metric uses Kornia for morphological operations. Samples with
        very small ground truth regions (< 32x32 pixels) return a neutral
        score of 0.5 to avoid unstable measurements.

    Example:
        >>> metric = MeanBoundaryAccuracy(num_steps=5)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> gt = (torch.rand(4, 1, 256, 256) > 0.5).float()
        >>> metric.update(pred, gt)
        >>> mba = metric.compute()
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    mba_scores: list[Tensor]

    def __init__(self, num_steps: int = 5, **kwargs: Any) -> None:
        """Initialize the Mean Boundary Accuracy metric.

        Args:
            num_steps: Number of boundary thickness scales to average over.
            **kwargs: Additional arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.add_state("mba_scores", default=[], dist_reduce_fx="cat")

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
            mba = self._compute_mba(p, g)
            self.mba_scores.append(mba.unsqueeze(0))

    def _compute_mba(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute Mean Boundary Accuracy for a single sample.

        Evaluates accuracy at multiple boundary scales and returns the average.
        Small ground truth regions are handled with a neutral score.

        Args:
            pred: Single prediction tensor of shape (H, W).
            gt: Single ground truth tensor of shape (H, W).

        Returns:
            Scalar tensor with the MBA score.
        """
        h, w = gt.shape

        if pred.shape != gt.shape:
            pred_tensor = pred.unsqueeze(0).unsqueeze(0)
            pred_tensor = kornia.geometry.transform.resize(
                pred_tensor,
                (h, w),
                interpolation="bilinear",
                align_corners=False,
            )
            pred = pred_tensor.squeeze(0).squeeze(0)

        # Avoid unstable measurements on very small foreground regions
        if gt.sum().item() < 32 * 32:
            return torch.tensor(0.5, device=pred.device, dtype=torch.float32)

        pred_bin = (pred > 0.5).float()
        gt_bin = (gt > 0.5).float()

        min_radius = 1
        max_radius = (w + h) / 300
        pred_acc = []
        gt_tensor = gt_bin.unsqueeze(0).unsqueeze(0)

        for i in range(self.num_steps):
            curr_radius = min_radius + int((max_radius - min_radius) / self.num_steps * i)
            curr_radius = max(1, curr_radius)
            kernel = self._disk_kernel(curr_radius, device=pred.device, dtype=gt_bin.dtype)

            # Morphological gradient defines the boundary region
            dilated = km.dilation(gt_tensor, kernel, border_type="reflect")
            eroded = km.erosion(gt_tensor, kernel, border_type="reflect")
            boundary_region = (dilated - eroded) > 0
            boundary_region = boundary_region.squeeze(0).squeeze(0)

            num_edge_pixels = int(boundary_region.sum().item())
            if num_edge_pixels == 0:
                pred_acc.append(0.5)
                continue

            gt_in_bound = gt_bin[boundary_region]
            pred_in_bound = pred_bin[boundary_region]
            num_pred_gd_pix = (
                ((gt_in_bound * pred_in_bound) + ((1 - gt_in_bound) * (1 - pred_in_bound)))
                .sum()
                .item()
            )

            pred_acc.append(num_pred_gd_pix / num_edge_pixels)

        mba = sum(pred_acc) / self.num_steps
        return torch.tensor(mba, device=pred.device, dtype=torch.float32)

    @staticmethod
    def _disk_kernel(radius: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create a circular (disk) structuring element for morphological operations.

        Args:
            radius: Radius of the disk in pixels.
            device: Target device for the kernel tensor.
            dtype: Data type for the kernel tensor.

        Returns:
            Binary disk kernel of shape (2*radius+1, 2*radius+1).
        """
        size = radius * 2 + 1
        coords = torch.arange(size, device=device, dtype=dtype) - radius
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = (xx**2 + yy**2) <= radius**2
        return kernel.to(dtype)

    def compute(self) -> Tensor:
        """Compute mean boundary accuracy over all accumulated samples.

        Returns:
            Scalar tensor with the mean MBA score.
        """
        if len(self.mba_scores) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.cat(self.mba_scores).mean()

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

        gt = (gt > 0.5).float()

        return pred, gt
