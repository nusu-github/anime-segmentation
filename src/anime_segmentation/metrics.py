"""BiRefNet-style evaluation metrics for segmentation.

Implements S-measure, E-measure, F-measure, Weighted F-measure, and MAE
for comprehensive segmentation quality evaluation.

Based on BiRefNet evaluation design.
"""

from typing import Any, Literal

import cv2
import numpy as np
import torch
from scipy.ndimage import convolve, distance_transform_edt
from skimage import measure, morphology
from torch import Tensor
from torchmetrics import Metric

TYPE = np.float64
EPS = torch.finfo(torch.float64).eps


def _ensure_batch(pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    return pred, gt


def _normalize_torch(pred: Tensor, gt: Tensor, normalize: bool) -> tuple[Tensor, Tensor]:
    if normalize:
        gt = gt > 128
        pred = pred.to(torch.float64) / 255.0
        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())
    else:
        if not pred.is_floating_point():
            raise TypeError(f"pred must be float when normalize=False, got {pred.dtype}")
        if pred.min() < 0 or pred.max() > 1:
            raise ValueError("pred values must be in [0, 1]")
        if gt.dtype != torch.bool:
            raise TypeError(f"gt must be bool when normalize=False, got {gt.dtype}")
        pred = pred.to(torch.float64)
    return pred, gt.bool()


def _normalize_numpy(
    pred: np.ndarray, gt: np.ndarray, normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    if normalize:
        gt = gt > 128
        pred = pred.astype(np.float64) / 255.0
        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())
    else:
        if not np.issubdtype(pred.dtype, np.floating):
            raise TypeError(f"pred must be float when normalize=False, got {pred.dtype}")
        if pred.min() < 0 or pred.max() > 1:
            raise ValueError("pred values must be in [0, 1]")
        if gt.dtype != bool:
            raise TypeError(f"gt must be bool when normalize=False, got {gt.dtype}")
        pred = pred.astype(np.float64)
    return pred.astype(np.float64), gt.astype(bool)


class FMeasure(Metric):
    """F-measure evaluator for salient object detection (torchmetrics compatible).
    Computes precision, recall, and F-measure at multiple thresholds,
    supporting both adaptive and dynamic evaluation modes.
    Reference:
        Achanta et al., "Frequency-tuned salient region detection", CVPR 2009
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    num_samples: Tensor

    def __init__(
        self,
        beta: float = 0.3,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            beta: Weight of precision in F-measure calculation.
            normalize: If True, normalize pred from [0,255] and gt from uint8.
                      If False, expect pred in [0,1] float and gt as bool.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.normalize = normalize
        self.add_state(
            "adaptive_fm_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "precision_sum", default=torch.zeros(256, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "recall_sum", default=torch.zeros(256, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "changeable_fm_sum", default=torch.zeros(256, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and ground truth.

        Args:
            pred: Prediction tensor, shape (H, W) or (B, H, W).
                  If normalize=True: uint8 [0, 255]
                  If normalize=False: float [0, 1]
            gt: Ground truth tensor, shape (H, W) or (B, H, W).
                If normalize=True: uint8 (threshold at 128)
                If normalize=False: bool
        """
        pred, gt = _ensure_batch(pred, gt)
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p, g = pred[i], gt[i]
            p, g = self._validate_and_normalize(p.squeeze(0), g.squeeze(0))
            adaptive_fm = self._cal_adaptive_fm(p, g)
            self.adaptive_fm_sum += adaptive_fm
            precisions, recalls, changeable_fms = self._cal_pr(p, g)
            self.precision_sum += precisions
            self.recall_sum += recalls
            self.changeable_fm_sum += changeable_fms
            self.num_samples += 1

    def _validate_and_normalize(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        """Validate and normalize input tensors."""
        return _normalize_torch(pred, gt, self.normalize)

    def _cal_adaptive_fm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Calculate adaptive F-measure for a single sample."""
        adaptive_threshold = min(2 * pred.mean().item(), 1.0)
        binary_pred = pred >= adaptive_threshold
        area_intersection = binary_pred[gt].sum().to(torch.float64)
        if area_intersection == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=pred.device)
        num_pred_fg = binary_pred.sum().to(torch.float64)
        num_gt_fg = gt.sum().to(torch.float64)
        precision = area_intersection / num_pred_fg
        recall = area_intersection / num_gt_fg
        return (1 + self.beta) * precision * recall / (self.beta * precision + recall)

    def _cal_pr(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Calculate precision, recall, and F-measure at 256 thresholds."""
        pred_uint8 = (pred * 255).to(torch.uint8)
        fg_hist = torch.histc(pred_uint8[gt].float(), bins=256, min=0, max=255)
        bg_hist = torch.histc(pred_uint8[~gt].float(), bins=256, min=0, max=255)
        fg_w_thrs = torch.flip(fg_hist, dims=[0]).cumsum(dim=0).to(torch.float64)
        bg_w_thrs = torch.flip(bg_hist, dims=[0]).cumsum(dim=0).to(torch.float64)
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs
        Ps = torch.where(Ps == 0, torch.ones_like(Ps), Ps)
        T = max(gt.sum().item(), 1)
        precisions = TPs / Ps
        recalls = TPs / T
        numerator = (1 + self.beta) * precisions * recalls
        denominator = torch.where(
            numerator == 0, torch.ones_like(numerator), self.beta * precisions + recalls
        )
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def compute(self) -> Tensor:
        """Compute final metrics.
        Returns:
            Tensor: Adaptive F-measure (scalar)
        """
        n = self.num_samples.to(torch.float64)
        if n == 0:
            n = torch.tensor(1.0, dtype=torch.float64)
        return self.adaptive_fm_sum / n


class WeightedFMeasure(Metric):
    """Weighted F-measure evaluator for salient object detection (torchmetrics compatible).
    Considers both pixel dependency and pixel importance when evaluating foreground maps,
    weighting pixels by their distance from the foreground boundary.
    Note: Uses CPU-bound distance transform with index return, which has no efficient GPU equivalent.
    Reference:
        Margolin et al., "How to eval foreground maps?", CVPR 2014
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    num_samples: Tensor

    def __init__(
        self,
        beta: float = 1.0,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            beta: Weight for balancing precision and recall. Defaults to 1.0 (F1-score).
            normalize: If True, normalize pred from [0,255] and gt from uint8.
                      If False, expect pred in [0,1] float and gt as bool.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.normalize = normalize
        self.add_state(
            "wfm_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and ground truth.
        Args:
            pred: Prediction tensor, shape (H, W) or (B, H, W).
            gt: Ground truth tensor, shape (H, W) or (B, H, W).
        """
        pred, gt = _ensure_batch(pred, gt)
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p = pred[i].detach().cpu().numpy()
            g = gt[i].detach().cpu().numpy()
            p, g = self._validate_and_normalize(p.squeeze(0), g.squeeze(0))
            # All background case
            wfm = self._cal_wfm(p, g) if np.any(g) else 0.0
            self.wfm_sum += wfm
            self.num_samples += 1

    def _validate_and_normalize(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize input arrays."""
        return _normalize_numpy(pred, gt, self.normalize)

    def _cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate weighted F-measure (CPU-bound)."""
        # Distance transform with nearest foreground indices
        Dst, Idxt = distance_transform_edt(gt == 0, return_indices=True)
        # Pixel dependency
        E = np.abs(pred - gt)
        Et = np.copy(E)
        # Use error at closest GT edge for background pixels
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]
        # Gaussian smoothing
        K = self._matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # Take minimum of original and smoothed error in foreground
        MIN_E_EA = np.where(gt & (EA < E), EA, E)
        # Pixel importance weighting
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt, dtype=np.float64))
        Ew = MIN_E_EA * B
        # Weighted TP and FP
        TPw = np.sum(gt) - np.sum(Ew[gt])
        FPw = np.sum(Ew[~gt])
        # Weighted Recall and Precision
        R = 1 - np.mean(Ew[gt])
        P = TPw / (TPw + FPw + EPS)
        # Weighted F-measure
        Q = (1 + self.beta) * R * P / (R + self.beta * P + EPS)
        return float(Q)

    @staticmethod
    def _matlab_style_gauss2D(shape: tuple = (7, 7), sigma: float = 5.0) -> np.ndarray:
        """Generate 2D Gaussian kernel compatible with MATLAB's fspecial."""
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def compute(self) -> Tensor:
        """Compute final weighted F-measure.
        Returns:
            Tensor: Weighted F-measure (scalar)
        """
        n = self.num_samples.to(torch.float64)
        if n == 0:
            n = torch.tensor(1.0, dtype=torch.float64)
        return self.wfm_sum / n


class SMeasure(Metric):
    """S-measure evaluator for salient object detection (torchmetrics compatible).
    Evaluates foreground maps by considering both object-aware and region-aware
    structural similarity between prediction and ground truth.
    Reference:
        Fan et al., "Structure-measure: A new way to eval foreground maps", ICCV 2017
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    num_samples: Tensor

    def __init__(
        self,
        alpha: float = 0.5,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            alpha: Weight for balancing object and region scores.
                   Higher values give more weight to object-level similarity.
                   Valid range: [0, 1]. Defaults to 0.5.
            normalize: If True, normalize pred from [0,255] and gt from uint8.
                      If False, expect pred in [0,1] float and gt as bool.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.normalize = normalize
        self.add_state(
            "sm_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and ground truth.
        Args:
            pred: Prediction tensor, shape (H, W) or (B, H, W).
            gt: Ground truth tensor, shape (H, W) or (B, H, W).
        """
        pred, gt = _ensure_batch(pred, gt)
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p, g = pred[i], gt[i]
            p, g = self._validate_and_normalize(p.squeeze(0), g.squeeze(0))
            sm = self._cal_sm(p, g)
            self.sm_sum += sm
            self.num_samples += 1

    def _validate_and_normalize(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        """Validate and normalize input tensors."""
        return _normalize_torch(pred, gt, self.normalize)

    def _cal_sm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Calculate S-measure score."""
        y = gt.float().mean()
        if y == 0:
            # All background
            sm = 1 - pred.mean()
        elif y == 1:
            # All foreground
            sm = pred.mean()
        else:
            object_score = self._object(pred, gt) * self.alpha
            region_score = self._region(pred, gt) * (1 - self.alpha)
            sm = torch.clamp(object_score + region_score, min=0.0)
        return sm.to(torch.float64)

    def _s_object(self, x: Tensor) -> Tensor:
        """Calculate object-aware score for a region."""
        if x.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=x.device)
        mean = x.mean()
        # ddof=1 equivalent: std with unbiased=True (default)
        std = (
            x.std(unbiased=True)
            if x.numel() > 1
            else torch.tensor(0.0, dtype=x.dtype, device=x.device)
        )
        return 2 * mean / (mean.pow(2) + 1 + std + EPS)

    def _object(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Calculate object-level structural similarity score."""
        gt_mean = gt.float().mean()
        fg_score = self._s_object(pred[gt]) * gt_mean
        bg_score = self._s_object((1 - pred)[~gt]) * (1 - gt_mean)
        return fg_score + bg_score

    def _region(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Calculate region-level structural similarity score."""
        h, w = gt.shape
        area = h * w
        # Calculate centroid of foreground
        if gt.sum() == 0:
            cy = round(h / 2)
            cx = round(w / 2)
        else:
            # argwhere returns (N, 2) with [row, col] indices
            fg_coords = torch.argwhere(gt).float()
            cy = int(fg_coords[:, 0].mean().round().item())
            cx = int(fg_coords[:, 1].mean().round().item())
        # +1 for MATLAB compatibility
        cy, cx = cy + 1, cx + 1
        # Divide into four quadrants and compute weighted SSIM
        w_lt = cx * cy / area
        w_rt = cy * (w - cx) / area
        w_lb = (h - cy) * cx / area
        w_rb = 1 - w_lt - w_rt - w_lb
        score_lt = self._ssim(pred[0:cy, 0:cx], gt[0:cy, 0:cx]) * w_lt
        score_rt = self._ssim(pred[0:cy, cx:w], gt[0:cy, cx:w]) * w_rt
        score_lb = self._ssim(pred[cy:h, 0:cx], gt[cy:h, 0:cx]) * w_lb
        score_rb = self._ssim(pred[cy:h, cx:w], gt[cy:h, cx:w]) * w_rb
        return score_lt + score_rt + score_lb + score_rb

    def _ssim(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Calculate SSIM (Structural Similarity Index) score."""
        if pred.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=pred.device)
        h, w = pred.shape
        N = h * w
        x = pred.mean()
        y = gt.float().mean()
        sigma_x = (
            ((pred - x) ** 2).sum() / (N - 1)
            if N > 1
            else torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        )
        sigma_y = (
            ((gt.float() - y) ** 2).sum() / (N - 1)
            if N > 1
            else torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        )
        sigma_xy = (
            ((pred - x) * (gt.float() - y)).sum() / (N - 1)
            if N > 1
            else torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        )
        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)
        if alpha != 0:
            return alpha / (beta + EPS)
        if beta == 0:
            return torch.tensor(1.0, dtype=torch.float64, device=pred.device)
        return torch.tensor(0.0, dtype=torch.float64, device=pred.device)

    def compute(self) -> Tensor:
        """Compute final S-measure.
        Returns:
            Tensor: S-measure (scalar)
        """
        n = self.num_samples.to(torch.float64)
        if n == 0:
            n = torch.tensor(1.0, dtype=torch.float64)
        return self.sm_sum / n


class EMeasure(Metric):
    """E-measure evaluator for salient object detection (torchmetrics compatible).
    Assesses binary foreground map quality by measuring the alignment between
    prediction and ground truth using an enhanced alignment matrix.
    Reference:
        Fan et al., "Enhanced-alignment Measure for Binary Foreground Map Evaluation", IJCAI 2018
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    num_samples: Tensor

    def __init__(
        self,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            normalize: If True, normalize pred from [0,255] and gt from uint8.
                      If False, expect pred in [0,1] float and gt as bool.
        """
        super().__init__(**kwargs)
        self.normalize = normalize
        self.add_state(
            "adaptive_em_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "changeable_em_sum", default=torch.zeros(256, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and ground truth.
        Args:
            pred: Prediction tensor, shape (H, W) or (B, H, W).
            gt: Ground truth tensor, shape (H, W) or (B, H, W).
        """
        pred, gt = _ensure_batch(pred, gt)
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p, g = pred[i], gt[i]
            p, g = self._validate_and_normalize(p.squeeze(0), g.squeeze(0))
            # Cache for current sample
            gt_fg_numel = g.sum().item()
            gt_size = g.shape[0] * g.shape[1]
            adaptive_em = self._cal_adaptive_em(p, g, gt_fg_numel, gt_size)
            self.adaptive_em_sum += adaptive_em
            changeable_ems = self._cal_changeable_em(p, g, gt_fg_numel, gt_size)
            self.changeable_em_sum += changeable_ems
            self.num_samples += 1

    def _validate_and_normalize(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        """Validate and normalize input tensors."""
        return _normalize_torch(pred, gt, self.normalize)

    def _cal_adaptive_em(self, pred: Tensor, gt: Tensor, gt_fg_numel: int, gt_size: int) -> Tensor:
        """Calculate adaptive E-measure using adaptive threshold."""
        adaptive_threshold = min(2 * pred.mean().item(), 1.0)
        return self._cal_em_with_threshold(pred, gt, adaptive_threshold, gt_fg_numel, gt_size)

    def _cal_changeable_em(
        self, pred: Tensor, gt: Tensor, gt_fg_numel: int, gt_size: int
    ) -> Tensor:
        """Calculate E-measure scores across 256 thresholds."""
        return self._cal_em_with_cumsumhistogram(pred, gt, gt_fg_numel, gt_size)

    def _cal_em_with_threshold(
        self, pred: Tensor, gt: Tensor, threshold: float, gt_fg_numel: int, gt_size: int
    ) -> Tensor:
        """Calculate E-measure for a specific threshold."""
        device = pred.device
        binarized_pred = pred >= threshold
        fg_fg_numel = (binarized_pred & gt).sum().item()
        fg_bg_numel = (binarized_pred & ~gt).sum().item()
        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = gt_size - fg___numel
        if gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif gt_fg_numel == gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self._generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
                gt_fg_numel=gt_fg_numel,
                gt_size=gt_size,
            )
            enhanced_matrix_sum = 0.0
            for part_numel, (comb_pred, comb_gt) in zip(parts_numel, combinations, strict=False):
                align_matrix_value = 2 * comb_pred * comb_gt / (comb_pred**2 + comb_gt**2 + EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                enhanced_matrix_sum += enhanced_matrix_value * part_numel
        return torch.tensor(
            enhanced_matrix_sum / (gt_size - 1 + EPS), dtype=torch.float64, device=device
        )

    def _cal_em_with_cumsumhistogram(
        self, pred: Tensor, gt: Tensor, gt_fg_numel: int, gt_size: int
    ) -> Tensor:
        """Calculate E-measure for 256 thresholds using cumulative histogram."""
        device = pred.device
        pred_uint8 = (pred * 255).to(torch.uint8)
        fg_fg_hist = torch.histc(pred_uint8[gt].float(), bins=256, min=0, max=255)
        fg_bg_hist = torch.histc(pred_uint8[~gt].float(), bins=256, min=0, max=255)
        fg_fg_numel_w_thrs = torch.flip(fg_fg_hist, dims=[0]).cumsum(dim=0)
        fg_bg_numel_w_thrs = torch.flip(fg_bg_hist, dims=[0]).cumsum(dim=0)
        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = gt_size - fg___numel_w_thrs
        if gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif gt_fg_numel == gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self._generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
                gt_fg_numel=gt_fg_numel,
                gt_size=gt_size,
            )
            results_parts = torch.zeros(4, 256, dtype=torch.float64, device=device)
            for i, (part_numel, (comb_pred, comb_gt)) in enumerate(
                zip(parts_numel_w_thrs, combinations, strict=False)
            ):
                align_matrix_value = 2 * comb_pred * comb_gt / (comb_pred**2 + comb_gt**2 + EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(dim=0)
        return (enhanced_matrix_sum / (gt_size - 1 + EPS)).to(torch.float64)

    def _generate_parts_numel_combinations(
        self,
        fg_fg_numel,
        fg_bg_numel,
        pred_fg_numel,
        pred_bg_numel,
        gt_fg_numel: int,
        gt_size: int,
    ):
        """Generate element counts and demeaned value combinations for four regions."""
        bg_fg_numel = gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel
        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]
        mean_pred_value = pred_fg_numel / gt_size
        mean_gt_value = gt_fg_numel / gt_size
        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value
        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def compute(self) -> Tensor:
        """Compute final E-measure metrics.
        Returns:
            Tensor: Adaptive E-measure (scalar)
        """
        n = self.num_samples.to(torch.float64)
        if n == 0:
            n = torch.tensor(1.0, dtype=torch.float64)
        return self.adaptive_em_sum / n


class HCEMeasure(Metric):
    """Human Correction Effort Measure for Dichotomous Image Segmentation (torchmetrics compatible).
    Note: This metric requires CPU processing due to skimage/OpenCV dependencies.
    Reference:
        Qin et al., "Highly Accurate Dichotomous Image Segmentation", ECCV 2022
    """

    is_differentiable = False
    higher_is_better = False  # Lower HCE is better
    full_state_update = False
    num_samples: Tensor

    def __init__(
        self,
        relax: int = 5,
        epsilon: float = 2.0,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            relax: Number of morphological relaxation iterations. Defaults to 5.
            epsilon: RDP approximation tolerance for polygon simplification. Defaults to 2.0.
            normalize: If True, normalize pred from [0,255] and gt from uint8.
                      If False, expect pred in [0,1] float and gt as bool.
        """
        super().__init__(**kwargs)
        self.relax = relax
        self.epsilon = epsilon
        self.normalize = normalize
        self.morphology_kernel = morphology.disk(1)
        self.add_state(
            "hce_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and ground truth.
        Args:
            pred: Prediction tensor, shape (H, W) or (B, H, W).
            gt: Ground truth tensor, shape (H, W) or (B, H, W).
        """
        pred, gt = _ensure_batch(pred, gt)
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p, g = pred[i], gt[i]
            p_np, g_np = self._validate_and_normalize(p.squeeze(0), g.squeeze(0))
            hce = self._cal_hce(p_np, g_np)
            self.hce_sum += hce
            self.num_samples += 1

    def _validate_and_normalize(self, pred: Tensor, gt: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Validate, normalize, and convert to NumPy."""
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        return _normalize_numpy(pred_np, gt_np, self.normalize)

    def _cal_hce(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Human Correction Effort."""
        gt_skeleton = morphology.skeletonize(gt).astype(bool)
        pred = pred > 0.5
        union = np.logical_or(gt, pred)
        TP = np.logical_and(gt, pred)
        FP = np.logical_xor(pred, TP)
        FN = np.logical_xor(gt, TP)
        # Relax the union
        eroded_union = cv2.erode(
            union.astype(np.uint8), self.morphology_kernel, iterations=self.relax
        )
        # Relaxed FP regions
        FP_ = np.logical_and(FP, eroded_union)
        for _ in range(self.relax):
            FP_ = cv2.dilate(FP_.astype(np.uint8), self.morphology_kernel)
            FP_ = np.logical_and(FP_.astype(bool), ~gt)
        FP_ = np.logical_and(FP, FP_)
        # Relaxed FN regions
        FN_ = np.logical_and(FN, eroded_union)
        for _ in range(self.relax):
            FN_ = cv2.dilate(FN_.astype(np.uint8), self.morphology_kernel)
            FN_ = np.logical_and(FN_, ~pred)
        FN_ = np.logical_and(FN, FN_)
        FN_ = np.logical_or(FN_, np.logical_xor(gt_skeleton, np.logical_and(TP, gt_skeleton)))
        # Find contours and control points
        contours_FP, _ = cv2.findContours(
            FP_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        condition_FP = np.logical_or(TP, FN_)
        bdies_FP, indep_cnt_FP = self._filter_conditional_boundary(contours_FP, FP_, condition_FP)
        contours_FN, _ = cv2.findContours(
            FN_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        condition_FN = 1 - np.logical_or(np.logical_or(TP, FP_), FN_)
        bdies_FN, indep_cnt_FN = self._filter_conditional_boundary(contours_FN, FN_, condition_FN)
        poly_FP_point_cnt = self._count_polygon_control_points(bdies_FP, epsilon=self.epsilon)
        poly_FN_point_cnt = self._count_polygon_control_points(bdies_FN, epsilon=self.epsilon)
        return poly_FP_point_cnt + indep_cnt_FP + poly_FN_point_cnt + indep_cnt_FN

    def _filter_conditional_boundary(
        self, contours: list, mask: np.ndarray, condition: np.ndarray
    ) -> tuple[list, int]:
        """Filter boundary segments based on condition mask."""
        condition = cv2.dilate(condition.astype(np.uint8), self.morphology_kernel)
        labels = measure.label(mask)
        independent_flags = np.ones(labels.max() + 1, dtype=int)
        independent_flags[0] = 0
        boundaries = []
        visited_map = np.zeros(condition.shape[:2], dtype=int)
        for item in contours:
            temp_boundaries = []
            temp_boundary = []
            for pt in item:
                row, col = pt[0, 1], pt[0, 0]
                if condition[row, col].sum() == 0 or visited_map[row, col] != 0:
                    if temp_boundary:
                        temp_boundaries.append(temp_boundary)
                        temp_boundary = []
                    continue
                temp_boundary.append([col, row])
                visited_map[row, col] += 1
                independent_flags[labels[row, col]] = 0
            if temp_boundary:
                temp_boundaries.append(temp_boundary)
            # Check if first and last boundaries are connected
            if len(temp_boundaries) > 1:
                first_x, first_y = temp_boundaries[0][0]
                last_x, last_y = temp_boundaries[-1][-1]
                if (
                    (abs(first_x - last_x) == 1 and first_y == last_y)
                    or (first_x == last_x and abs(first_y - last_y) == 1)
                    or (abs(first_x - last_x) == 1 and abs(first_y - last_y) == 1)
                ):
                    temp_boundaries[-1].extend(temp_boundaries[0][::-1])
                    del temp_boundaries[0]
            for k in range(len(temp_boundaries)):
                temp_boundaries[k] = np.array(temp_boundaries[k])[:, np.newaxis, :]
            if temp_boundaries:
                boundaries.extend(temp_boundaries)
        return boundaries, independent_flags.sum()

    def _count_polygon_control_points(self, boundaries: list, epsilon: float = 1.0) -> int:
        """Count control points using RDP algorithm."""
        num_points = 0
        for boundary in boundaries:
            approx_poly = cv2.approxPolyDP(boundary, epsilon, False)
            num_points += len(approx_poly)
        return num_points

    def compute(self) -> Tensor:
        """Compute final HCE.
        Returns:
            Tensor: HCE (scalar)
        """
        n = self.num_samples.to(torch.float64)
        if n == 0:
            n = torch.tensor(1.0, dtype=torch.float64)
        return self.hce_sum / n


class MBAMeasure(Metric):
    """Mean Boundary Accuracy Measure (torchmetrics compatible).
    Note: This metric requires CPU processing due to OpenCV dependencies.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    num_samples: Tensor

    def __init__(
        self,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            normalize: If True, normalize pred from [0,255] and gt from uint8.
                      If False, expect pred in [0,1] float and gt as bool.
        """
        super().__init__(**kwargs)
        self.normalize = normalize
        self.add_state(
            "ba_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and ground truth."""
        pred, gt = _ensure_batch(pred, gt)
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p, g = pred[i], gt[i]
            p_np, g_np = self._to_numpy(p.squeeze(0), g.squeeze(0))
            ba = self._cal_ba(p_np, g_np)
            self.ba_sum += ba
            self.num_samples += 1

    def _to_numpy(self, pred: Tensor, gt: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Convert to NumPy and apply threshold."""
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        if self.normalize:
            # [0, 255] → bool (threshold at 128)
            pred_np = pred_np > 128
            gt_np = gt_np > 128
        else:
            # [0, 1] → bool (threshold at 0.5)
            pred_np = pred_np > 0.5
            if gt_np.dtype != bool:
                gt_np = gt_np.astype(bool)
        return pred_np, gt_np

    def _get_disk_kernel(self, radius: int) -> np.ndarray:
        """Get elliptical structuring element."""
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

    def _cal_ba(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate boundary accuracy."""
        gt = gt.astype(np.uint8)
        pred = pred.astype(np.uint8)
        h, w = gt.shape
        min_radius = 1
        max_radius = (w + h) / 300
        num_steps = 5
        pred_acc = []
        for i in range(num_steps):
            curr_radius = min_radius + int((max_radius - min_radius) / num_steps * i)
            kernel = self._get_disk_kernel(curr_radius)
            boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0
            num_edge_pixels = boundary_region.sum()
            if num_edge_pixels == 0:
                pred_acc.append(1.0)
                continue
            gt_in_bound = gt[boundary_region]
            pred_in_bound = pred[boundary_region]
            num_pred_gd_pix = (
                gt_in_bound * pred_in_bound + (1 - gt_in_bound) * (1 - pred_in_bound)
            ).sum()
            pred_acc.append(num_pred_gd_pix / num_edge_pixels)
        return sum(pred_acc) / num_steps

    def compute(self) -> Tensor:
        """Compute final MBA.
        Returns:
            Tensor: MBA (scalar)
        """
        n = self.num_samples.to(torch.float64)
        if n == 0:
            n = torch.tensor(1.0, dtype=torch.float64)
        return self.ba_sum / n


class BIoUMeasure(Metric):
    """Boundary IoU Measure (torchmetrics compatible).
    Note: This metric requires CPU processing due to OpenCV dependencies.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    num_samples: Tensor

    def __init__(
        self,
        mode: Literal["mean", "max"] = "mean",
        dilation_ratio: float = 0.02,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            dilation_ratio: Ratio of image diagonal for boundary dilation. Defaults to 0.02.
            normalize: If True, normalize pred from [0,255] and gt from uint8.
                      If False, expect pred in [0,1] float and gt as bool.
        """
        super().__init__(**kwargs)
        self.mode = mode
        self.dilation_ratio = dilation_ratio
        self.normalize = normalize
        self.add_state(
            "biou_sum", default=torch.zeros(256, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """Update state with predictions and ground truth."""
        pred, gt = _ensure_batch(pred, gt)
        batch_size = pred.shape[0]
        for i in range(batch_size):
            p, g = pred[i], gt[i]
            p_np, g_np = self._validate_and_normalize(p.squeeze(0), g.squeeze(0))
            biou = self._cal_biou(p_np, g_np)
            self.biou_sum += torch.from_numpy(biou).to(self.biou_sum)
            self.num_samples += 1

    def _validate_and_normalize(self, pred: Tensor, gt: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Validate, normalize, and convert to NumPy."""
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        return _normalize_numpy(pred_np, gt_np, self.normalize)

    def _mask_to_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to boundary region."""
        h, w = mask.shape
        img_diag = np.sqrt(h**2 + w**2)
        dilation = max(round(self.dilation_ratio * img_diag), 1)
        # Pad to handle border-truncated boundaries
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        return mask - mask_erode

    def _cal_biou(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Calculate Boundary IoU curve."""
        # Convert to boundary
        pred_boundary = (pred * 255).astype(np.uint8)
        pred_boundary = self._mask_to_boundary(pred_boundary)
        gt_boundary = (gt.astype(np.float64) * 255).astype(np.uint8)
        gt_boundary = self._mask_to_boundary(gt_boundary)
        gt_boundary = gt_boundary > 128
        # Histogram-based threshold sweep
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred_boundary[gt_boundary], bins=bins)
        bg_hist, _ = np.histogram(pred_boundary[~gt_boundary], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        TPs = fg_w_thrs
        T = max(np.count_nonzero(gt_boundary), 1)
        # IoU = TP / (T + FP) where FP = bg_w_thrs
        return (TPs / (T + bg_w_thrs)).astype(np.float64)

    def compute(self) -> Tensor:
        """Compute final Boundary IoU.
        Returns:
            dict: {"biou": Tensor (scalar)}
        """
        n = self.num_samples.to(torch.float64)
        if n == 0:
            n = torch.tensor(1.0, dtype=torch.float64)
        if self.mode == "mean":
            biou_curve = self.biou_sum / n
            biou_value = biou_curve.mean()
        else:  # self.mode == "max"
            biou_curve = self.biou_sum / n
            biou_value = biou_curve.max()
        return biou_value
