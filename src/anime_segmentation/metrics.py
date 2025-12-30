"""BiRefNet-style evaluation metrics for segmentation.

Implements S-measure, E-measure, F-measure, Weighted F-measure, and MAE
for comprehensive segmentation quality evaluation.

Based on BiRefNet evaluation design.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist
from torch import Tensor
from torch.func import vmap
from torchmetrics import Metric

_EPS = 1e-8


def _prepare_data(pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    """Prepare prediction and ground truth for metric computation.

    Args:
        pred: Predicted mask (0-1 range or 0-255)
        gt: Ground truth mask (0-1 range or 0-255)

    Returns:
        Normalized pred (0-1) and binary gt

    Note:
        Uses torch.where instead of if-else for vmap compatibility.
    """
    # Binarize GT at 0.5 threshold
    gt = (gt > 0.5).float()

    # Normalize pred to 0-1 range
    pred_min = pred.min()
    pred_max = pred.max()
    denom = pred_max - pred_min
    # Use where to avoid data-dependent control flow
    pred = torch.where(denom > 0, (pred - pred_min) / (denom + _EPS), pred)

    return pred, gt


def _get_adaptive_threshold(matrix: Tensor, max_value: float = 1.0) -> Tensor:
    """Get adaptive threshold as 2 * mean, clamped to max_value."""
    return torch.clamp(2 * matrix.mean(), max=max_value)


class MAEMetric(Metric):
    """Mean Absolute Error metric."""

    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        pred, gt = _prepare_data(pred, gt)
        mae = torch.abs(pred - gt).mean()
        self.sum_mae += mae
        self.count += 1

    def compute(self) -> Tensor:
        return self.sum_mae / self.count.clamp(min=1)


class SMeasure(Metric):
    """Structure-measure for segmentation evaluation.

    Combines object-aware and region-aware structural similarity.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, alpha: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.add_state("sum_sm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        # Process each sample in batch
        for i in range(pred.shape[0]):
            p = pred[i].squeeze()
            g = gt[i].squeeze()
            p, g = _prepare_data(p, g)
            sm = self._cal_sm(p, g)
            self.sum_sm += sm
            self.count += 1

    def _cal_sm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Calculate S-measure for a single image."""
        y = gt.mean()
        if y == 0:
            sm = 1 - pred.mean()
        elif y == 1:
            sm = pred.mean()
        else:
            sm = self.alpha * self._object(pred, gt) + (1 - self.alpha) * self._region(pred, gt)
            sm = torch.clamp(sm, min=0)
        return sm

    def _object(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Object-aware structural similarity."""
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = gt.mean()
        return u * self._s_object(fg, gt) + (1 - u) * self._s_object(bg, 1 - gt)

    def _s_object(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Object structural similarity score."""
        mask = gt > 0.5
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        x = pred[mask].mean()
        sigma_x = pred[mask].std()
        return 2 * x / (x.pow(2) + 1 + sigma_x + _EPS)

    def _region(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Region-aware structural similarity using SSIM-like measure."""
        x, y = self._centroid(gt)
        h, w = gt.shape

        # Divide into 4 quadrants
        gt_lt = gt[:y, :x]
        gt_rt = gt[:y, x:]
        gt_lb = gt[y:, :x]
        gt_rb = gt[y:, x:]

        pred_lt = pred[:y, :x]
        pred_rt = pred[:y, x:]
        pred_lb = pred[y:, :x]
        pred_rb = pred[y:, x:]

        # Weights based on area
        area = h * w
        w1 = x * y / area
        w2 = (w - x) * y / area
        w3 = x * (h - y) / area
        w4 = 1 - w1 - w2 - w3

        return (
            w1 * self._ssim(pred_lt, gt_lt)
            + w2 * self._ssim(pred_rt, gt_rt)
            + w3 * self._ssim(pred_lb, gt_lb)
            + w4 * self._ssim(pred_rb, gt_rb)
        )

    def _centroid(self, gt: Tensor) -> tuple[int, int]:
        """Calculate centroid of ground truth."""
        h, w = gt.shape
        if gt.sum() == 0:
            return w // 2, h // 2

        # Find indices where gt is True
        indices = torch.nonzero(gt > 0.5, as_tuple=True)
        if len(indices[0]) == 0:
            return w // 2, h // 2

        y = int(indices[0].float().mean().round().item()) + 1
        x = int(indices[1].float().mean().round().item()) + 1
        return min(x, w - 1), min(y, h - 1)

    def _ssim(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Simple SSIM-like measure for region comparison."""
        if pred.numel() == 0 or gt.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        n = pred.numel()
        x = pred.mean()
        y = gt.mean()

        sigma_x = ((pred - x).pow(2).sum()) / max(n - 1, 1)
        sigma_y = ((gt - y).pow(2).sum()) / max(n - 1, 1)
        sigma_xy = ((pred - x) * (gt - y)).sum() / max(n - 1, 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x.pow(2) + y.pow(2)) * (sigma_x + sigma_y)

        if alpha != 0:
            return alpha / (beta + _EPS)
        if beta == 0:
            return torch.tensor(1.0, device=pred.device)
        return torch.tensor(0.0, device=pred.device)

    def compute(self) -> Tensor:
        return self.sum_sm / self.count.clamp(min=1)


def _cal_em_single(pred: Tensor, gt: Tensor) -> Tensor:
    """Calculate adaptive E-measure for a single sample (vmap-compatible)."""
    pred, gt = _prepare_data(pred, gt)
    adaptive_threshold = _get_adaptive_threshold(pred)

    gt_fg_numel = gt.sum()
    gt_size = float(gt.numel())

    binarized_pred = (pred >= adaptive_threshold).float()
    fg_fg_numel = (binarized_pred * gt).sum()
    fg_bg_numel = (binarized_pred * (1 - gt)).sum()

    fg_numel = fg_fg_numel + fg_bg_numel
    bg_numel = gt_size - fg_numel

    # Calculate enhanced alignment (general case)
    bg_fg_numel = gt_fg_numel - fg_fg_numel
    bg_bg_numel = bg_numel - bg_fg_numel

    mean_pred_value = fg_numel / gt_size
    mean_gt_value = gt_fg_numel / gt_size

    demeaned_pred_fg = 1 - mean_pred_value
    demeaned_pred_bg = -mean_pred_value
    demeaned_gt_fg = 1 - mean_gt_value
    demeaned_gt_bg = -mean_gt_value

    # Vectorized computation of all 4 parts
    part_numels = torch.stack([fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel])
    pred_vals = torch.stack(
        [demeaned_pred_fg, demeaned_pred_fg, demeaned_pred_bg, demeaned_pred_bg]
    )
    gt_vals = torch.stack([demeaned_gt_fg, demeaned_gt_bg, demeaned_gt_fg, demeaned_gt_bg])

    align_matrix_values = 2 * pred_vals * gt_vals / (pred_vals**2 + gt_vals**2 + _EPS)
    enhanced_matrix_values = (align_matrix_values + 1) ** 2 / 4
    enhanced_matrix_sum_general = (enhanced_matrix_values * part_numels).sum()

    # Use torch.where to handle edge cases without data-dependent control flow
    enhanced_matrix_sum = torch.where(
        gt_fg_numel == 0,
        bg_numel,
        torch.where(gt_fg_numel == gt_size, fg_numel, enhanced_matrix_sum_general),
    )

    return enhanced_matrix_sum / (gt_size - 1 + _EPS)


class EMeasure(Metric):
    """Enhanced-alignment measure for segmentation evaluation.

    Optimized with vmap for efficient batched computation.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_em", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        # Squeeze channel dim and use vmap for batched computation
        pred_squeezed = pred.squeeze(1) if pred.dim() == 4 else pred
        gt_squeezed = gt.squeeze(1) if gt.dim() == 4 else gt

        ems = vmap(_cal_em_single)(pred_squeezed, gt_squeezed)
        self.sum_em += ems.sum()
        self.count += pred.shape[0]

    def compute(self) -> Tensor:
        return self.sum_em / self.count.clamp(min=1)


def _cal_adaptive_fm_single(pred: Tensor, gt: Tensor, beta: float) -> Tensor:
    """Calculate adaptive F-measure for a single sample."""
    pred, gt = _prepare_data(pred, gt)
    adaptive_threshold = _get_adaptive_threshold(pred)
    binary_pred = (pred >= adaptive_threshold).float()

    area_intersection = (binary_pred * gt).sum()
    # Use where instead of if to keep vmap-compatible (no data-dependent control flow)
    pre = area_intersection / binary_pred.sum().clamp(min=_EPS)
    rec = area_intersection / gt.sum().clamp(min=_EPS)
    fm = (1 + beta) * pre * rec / (beta * pre + rec + _EPS)
    return torch.where(area_intersection == 0, torch.zeros_like(fm), fm)


class FMeasure(Metric):
    """F-measure (F-beta score) for segmentation evaluation.

    Uses adaptive thresholding for binarization.
    Optimized with vmap for efficient batched computation.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, beta: float = 0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.add_state("sum_fm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        # Squeeze channel dim and use vmap for batched computation
        pred_squeezed = pred.squeeze(1) if pred.dim() == 4 else pred
        gt_squeezed = gt.squeeze(1) if gt.dim() == 4 else gt

        from functools import partial

        fm_fn = partial(_cal_adaptive_fm_single, beta=self.beta)
        fms = vmap(fm_fn)(pred_squeezed, gt_squeezed)
        self.sum_fm += fms.sum()
        self.count += pred.shape[0]

    def compute(self) -> Tensor:
        return self.sum_fm / self.count.clamp(min=1)


class WeightedFMeasure(Metric):
    """Weighted F-measure for segmentation evaluation.

    Applies distance-based weighting to penalize errors near boundaries.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, beta: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.add_state("sum_wfm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        for i in range(pred.shape[0]):
            p = pred[i].squeeze()
            g = gt[i].squeeze()
            p, g = _prepare_data(p, g)

            wfm = torch.tensor(0.0, device=pred.device) if g.sum() == 0 else self._cal_wfm(p, g)
            self.sum_wfm += wfm
            self.count += 1

    def _cal_wfm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Calculate weighted F-measure."""
        device = pred.device

        # Convert to numpy for distance transform
        gt_np = (gt > 0.5).cpu().to(torch.float32).numpy()
        pred_np = pred.cpu().to(torch.float32).numpy()

        # Distance transform
        dst, idxt = bwdist(gt_np == 0, return_indices=True)

        # Pixel dependency
        e = abs(pred_np - gt_np)
        et = e.copy()
        et[gt_np == 0] = et[idxt[0][gt_np == 0], idxt[1][gt_np == 0]]

        # Gaussian smoothing
        k = self._matlab_style_gauss2d((7, 7), sigma=5).astype(np.float32)
        ea = convolve(et.astype(np.float32), weights=k, mode="constant", cval=0)

        # Min of E and EA
        min_e_ea = e.copy()
        mask = gt_np.astype(bool) & (ea < e)
        min_e_ea[mask] = ea[mask]

        # Pixel importance (distance-based weighting)

        b = np.where(gt_np == 0, 2 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(gt_np))
        ew = min_e_ea * b

        # Weighted metrics
        tpw = gt_np.sum() - ew[gt_np == 1].sum()
        fpw = ew[gt_np == 0].sum()

        r = 1 - ew[gt_np == 1].mean() if gt_np.sum() > 0 else 0
        p = tpw / (tpw + fpw + _EPS)

        q = (1 + self.beta) * r * p / (r + self.beta * p + _EPS)
        return torch.tensor(q, device=device, dtype=pred.dtype)

    @staticmethod
    def _matlab_style_gauss2d(shape: tuple[int, int] = (7, 7), sigma: float = 5) -> np.ndarray:
        """Create MATLAB-style 2D Gaussian kernel."""
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
        return self.sum_wfm / self.count.clamp(min=1)
