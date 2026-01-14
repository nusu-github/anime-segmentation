"""TorchMetrics implementations for segmentation evaluation.

These metrics are GPU-accelerated and distributed-training friendly.
For comprehensive evaluation with curves, use the NumPy-based metrics in metrics.py.
"""

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection


def _normalize_tensors(
    preds: Tensor,
    target: Tensor,
    binarize_target: bool = True,
) -> tuple[Tensor, Tensor]:
    """Normalize prediction and target tensors for metric computation.

    Args:
        preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
        target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].
        binarize_target: Whether to binarize target with threshold 0.5.

    Returns:
        Tuple of normalized (preds, target) tensors with shape [B, H, W].

    """
    preds = preds.float()
    target = target.float()

    # Squeeze channel dimension if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    # Binarize target if requested
    if binarize_target:
        target = (target > 0.5).float()

    return preds, target


class IoUMetric(Metric):
    """Intersection over Union (Jaccard Index) metric.

    Computes IoU between predicted and ground truth binary masks.
    Uses adaptive thresholding on predictions.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize IoU metric.

        Args:
            threshold: Fixed threshold for binarization. If None, uses adaptive threshold.

        """
        super().__init__()
        self.threshold = threshold
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state.

        Args:
            preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
            target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].

        """
        preds, target_bin = _normalize_tensors(preds, target)

        # Process each sample
        for pred, gt in zip(preds, target_bin, strict=True):
            # Adaptive or fixed threshold
            thresh = min(2 * pred.mean(), 1.0) if self.threshold is None else self.threshold

            pred_bin = (pred >= thresh).float()

            intersection = (pred_bin * gt).sum()
            union = pred_bin.sum() + gt.sum() - intersection

            self.intersection += intersection
            self.union += union

    def compute(self) -> Tensor:
        """Compute IoU."""
        return self.intersection / (self.union + 1e-8)


class MAEMetric(Metric):
    """Mean Absolute Error metric for segmentation."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state.

        Args:
            preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
            target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].

        """
        preds, target_bin = _normalize_tensors(preds, target)

        # Compute MAE
        self.sum_error += torch.abs(preds - target_bin).sum()
        self.total_pixels += target_bin.numel()

    def compute(self) -> Tensor:
        """Compute MAE."""
        return self.sum_error / self.total_pixels


class FMeasureMetric(Metric):
    """Adaptive F-measure metric.

    Uses adaptive thresholding (2 * mean) for binarization.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, beta: float = 0.3) -> None:
        """Initialize F-measure metric.

        Args:
            beta: Weight for precision in F-measure formula.

        """
        super().__init__()
        self.beta = beta
        self.add_state("fm_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state.

        Args:
            preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
            target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].

        """
        preds, target_bin = _normalize_tensors(preds, target)
        target_bin = target_bin > 0.5  # Convert back to bool

        # Process each sample
        for pred, gt in zip(preds, target_bin, strict=True):
            # Adaptive threshold
            thresh = min(2 * pred.mean().item(), 1.0)
            pred_bin = pred >= thresh

            # Compute intersection
            intersection = (pred_bin & gt).sum().float()

            if intersection == 0:
                fm = torch.tensor(0.0, device=preds.device)
            else:
                precision = intersection / max(pred_bin.sum().float(), 1e-8)
                recall = intersection / max(gt.sum().float(), 1e-8)
                fm = (1 + self.beta) * precision * recall / (self.beta * precision + recall + 1e-8)

            self.fm_sum += fm
            self.count += 1

    def compute(self) -> Tensor:
        """Compute mean F-measure."""
        return self.fm_sum / (self.count + 1e-8)


class SMeasureMetric(Metric):
    """Structure Measure metric.

    Evaluates structural similarity between prediction and ground truth.
    Combines object-aware and region-aware structural similarity.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, alpha: float = 0.5) -> None:
        """Initialize S-measure metric.

        Args:
            alpha: Weight between object and region similarity.

        """
        super().__init__()
        self.alpha = alpha
        self.add_state("sm_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state.

        Args:
            preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
            target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].

        """
        preds, target_bin = _normalize_tensors(preds, target)

        # Process each sample
        for pred, gt in zip(preds, target_bin, strict=True):
            sm = self._compute_sm(pred, gt)
            self.sm_sum += sm
            self.count += 1

    def _compute_sm(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute S-measure for a single sample."""
        y = gt.mean()

        if y == 0:
            # No foreground in GT
            return 1 - pred.mean()
        if y == 1:
            # All foreground in GT
            return pred.mean()
        # Combine object and region similarity
        so = self._object_similarity(pred, gt)
        sr = self._region_similarity(pred, gt)
        return self.alpha * so + (1 - self.alpha) * sr

    def _object_similarity(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute object-aware structural similarity."""
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = gt.mean()

        fg_score = self._s_object(fg, gt)
        bg_score = self._s_object(bg, 1 - gt)

        return u * fg_score + (1 - u) * bg_score

    def _s_object(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute object similarity component."""
        mask = gt > 0.5
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        x = pred[mask].mean()
        sigma_x = pred[mask].std()

        return 2 * x / (x.pow(2) + 1 + sigma_x + 1e-8)

    def _region_similarity(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute region-aware structural similarity."""
        # Find centroid of foreground
        h, w = gt.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=gt.device),
            torch.arange(w, device=gt.device),
            indexing="ij",
        )

        fg_sum = gt.sum()
        if fg_sum == 0:
            cx, cy = w // 2, h // 2
        else:
            cx = (x_coords.float() * gt).sum() / fg_sum
            cy = (y_coords.float() * gt).sum() / fg_sum
            cx, cy = int(cx.round()), int(cy.round())

        # Ensure valid indices
        cx = max(1, min(cx, w - 1))
        cy = max(1, min(cy, h - 1))

        # Divide into 4 regions and compute SSIM for each
        weights = []
        scores = []

        regions = [
            (slice(0, cy), slice(0, cx)),  # Top-left
            (slice(0, cy), slice(cx, w)),  # Top-right
            (slice(cy, h), slice(0, cx)),  # Bottom-left
            (slice(cy, h), slice(cx, w)),  # Bottom-right
        ]

        total_area = h * w
        for region in regions:
            pred_region = pred[region]
            gt_region = gt[region]
            area = pred_region.numel()

            if area == 0:
                continue

            weight = area / total_area
            score = self._ssim(pred_region, gt_region)
            weights.append(weight)
            scores.append(score)

        if not scores:
            return torch.tensor(0.0, device=pred.device)

        weights = torch.tensor(weights, device=pred.device)
        scores = torch.stack(scores)
        return (weights * scores).sum()

    def _ssim(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute SSIM between two regions."""
        x = pred.mean()
        y = gt.mean()
        n = pred.numel()

        if n <= 1:
            return torch.tensor(1.0, device=pred.device)

        sigma_x = ((pred - x).pow(2)).sum() / (n - 1)
        sigma_y = ((gt - y).pow(2)).sum() / (n - 1)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x.pow(2) + y.pow(2)) * (sigma_x + sigma_y)

        if alpha != 0:
            return alpha / (beta + 1e-8)
        if beta == 0:
            return torch.tensor(1.0, device=pred.device)
        return torch.tensor(0.0, device=pred.device)

    def compute(self) -> Tensor:
        """Compute mean S-measure."""
        result = self.sm_sum / (self.count + 1e-8)
        return torch.clamp(result, min=0)


class EMeasureMetric(Metric):
    """Enhanced-alignment Measure metric (adaptive version).

    Evaluates the alignment between prediction and ground truth
    considering both pixel-level and image-level information.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("em_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state.

        Args:
            preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
            target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].

        """
        preds, target_bin = _normalize_tensors(preds, target)

        # Process each sample
        for pred, gt in zip(preds, target_bin, strict=True):
            em = self._compute_em(pred, gt)
            self.em_sum += em
            self.count += 1

    def _compute_em(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute E-measure for a single sample with adaptive threshold."""
        # Adaptive threshold
        thresh = min(2 * pred.mean().item(), 1.0)
        pred_bin = (pred >= thresh).float()

        h, w = gt.shape
        gt_size = h * w
        gt_fg_numel = gt.sum()

        fg_match = (pred_bin * gt).sum()  # True positives
        fg_mismatch = (pred_bin * (1 - gt)).sum()  # False positives
        pred_fg_numel = pred_bin.sum()
        pred_bg_numel = gt_size - pred_fg_numel

        if gt_fg_numel == 0:
            # No foreground: E-measure is proportion of true negatives
            return pred_bg_numel / (gt_size - 1 + 1e-8)
        if gt_fg_numel == gt_size:
            # All foreground: E-measure is proportion of true positives
            return pred_fg_numel / (gt_size - 1 + 1e-8)
        # General case: compute enhanced alignment matrix
        bg_fg = gt_fg_numel - fg_match  # False negatives
        bg_bg = pred_bg_numel - bg_fg  # True negatives

        mean_pred = pred_fg_numel / gt_size
        mean_gt = gt_fg_numel / gt_size

        # Demeaned values
        d_pred_fg = 1 - mean_pred
        d_pred_bg = -mean_pred
        d_gt_fg = 1 - mean_gt
        d_gt_bg = -mean_gt

        # Compute enhanced matrix for each part
        parts = [
            (fg_match, d_pred_fg, d_gt_fg),  # TP
            (fg_mismatch, d_pred_fg, d_gt_bg),  # FP
            (bg_fg, d_pred_bg, d_gt_fg),  # FN
            (bg_bg, d_pred_bg, d_gt_bg),  # TN
        ]

        enhanced_sum = torch.tensor(0.0, device=pred.device)
        for numel, d_pred, d_gt in parts:
            align = 2 * d_pred * d_gt / (d_pred**2 + d_gt**2 + 1e-8)
            enhanced = (align + 1) ** 2 / 4
            enhanced_sum += enhanced * numel

        return enhanced_sum / (gt_size - 1 + 1e-8)

    def compute(self) -> Tensor:
        """Compute mean E-measure."""
        return self.em_sum / (self.count + 1e-8)


class SegmentationMetrics(MetricCollection):
    """Collection of all segmentation metrics.

    Usage:
        metrics = SegmentationMetrics()
        metrics.update(preds, targets)
        results = metrics.compute()  # Returns dict with all metric values

    """

    def __init__(self, prefix: str = "") -> None:
        """Initialize metric collection.

        Args:
            prefix: Prefix for metric names (e.g., "val_" or "test_").

        """
        super().__init__(
            {
                "iou": IoUMetric(),
                "mae": MAEMetric(),
                "fm_adp": FMeasureMetric(beta=0.3),
                "sm": SMeasureMetric(alpha=0.5),
                "em_adp": EMeasureMetric(),
            },
            prefix=prefix,
        )
