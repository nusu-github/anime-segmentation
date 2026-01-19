"""TorchMetrics implementations for segmentation evaluation.

These metrics are GPU-accelerated and distributed-training friendly.
For comprehensive evaluation with curves, use the NumPy-based metrics in metrics.py.
"""

import torch
from torch import Tensor
from torch.func import vmap
from torchmetrics import Metric, MetricCollection


def _normalize_tensors(
    preds: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Normalize prediction and target tensors for metric computation.

    Args:
        preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
        target: Ground truth [B, 1, H, W] or [B, H, W], values in {0, 1}.

    Returns:
        Tuple of normalized (preds, target) tensors with shape [B, H, W].

    """
    # Detach to avoid retaining autograd graphs when metrics are used during training.
    preds = preds.detach().float()
    target = target.detach().float()

    # Squeeze channel dimension if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

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

        if self.threshold is None:
            # Adaptive threshold per sample: [B]
            adaptive_thresh = torch.clamp(2 * preds.mean(dim=(1, 2)), max=1.0)
            thresh = adaptive_thresh.view(-1, 1, 1)
        else:
            thresh = self.threshold

        pred_bin = (preds >= thresh).float()

        # Batch computation: [B] -> sum to scalar
        intersection = (pred_bin * target_bin).sum(dim=(1, 2))
        union = pred_bin.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2)) - intersection

        self.intersection += intersection.sum()
        self.union += union.sum()

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
        target_bin = target_bin.bool()

        # Adaptive threshold per sample: [B]
        adaptive_thresh = torch.clamp(2 * preds.mean(dim=(1, 2)), max=1.0)
        thresh = adaptive_thresh.view(-1, 1, 1)
        pred_bin = preds >= thresh

        # Batch computation: [B]
        intersection = (pred_bin & target_bin).sum(dim=(1, 2)).float()
        pred_sum = pred_bin.sum(dim=(1, 2)).float()
        gt_sum = target_bin.sum(dim=(1, 2)).float()

        precision = intersection / (pred_sum + 1e-8)
        recall = intersection / (gt_sum + 1e-8)
        fm = (1 + self.beta) * precision * recall / (self.beta * precision + recall + 1e-8)

        # Replace conditional with torch.where: fm = 0 when intersection == 0
        fm = torch.where(intersection > 0, fm, torch.zeros_like(fm))

        self.fm_sum += fm.sum()
        self.count += preds.shape[0]

    def compute(self) -> Tensor:
        """Compute mean F-measure."""
        return self.fm_sum / (self.count + 1e-8)


class SMeasureMetric(Metric):
    """Structure Measure (S-measure) for salient object detection.

    Evaluates structural similarity via two complementary components:
    - Object-aware (So): similarity within foreground/background regions
    - Region-aware (Sr): SSIM computed over 4 quadrants centered on the mask centroid

    Final score: alpha * So + (1 - alpha) * Sr
    Reference: Fan et al., "Structure-measure: A New Way to Evaluate Foreground Maps" (ICCV 2017)
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

        # Compute object and region similarity
        so = self._object_similarity(pred, gt)
        sr = self._region_similarity(pred, gt)
        combined = self.alpha * so + (1 - self.alpha) * sr

        # Replace conditionals with torch.where
        return torch.where(
            y == 0,
            1 - pred.mean(),
            torch.where(y == 1, pred.mean(), combined),
        )

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
        mask = gt.bool()
        mask_sum = mask.sum()

        # Safe indexing with fallback
        masked_pred = pred[mask] if mask_sum > 0 else pred.new_zeros(1)
        x = masked_pred.mean()
        sigma_x = masked_pred.std() if mask_sum > 1 else pred.new_zeros(())

        score = 2 * x / (x.pow(2) + 1 + sigma_x + 1e-8)

        # Return 0 if no foreground
        return torch.where(mask_sum > 0, score, pred.new_zeros(()))

    def _region_similarity(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute region-aware structural similarity."""
        # Find centroid of foreground
        h, w = gt.shape
        fg_sum = gt.sum()
        if fg_sum == 0:
            cx, cy = w // 2, h // 2
        else:
            # Use 1D projections to avoid allocating full HxW coordinate grids.
            y_coords = torch.arange(h, device=gt.device, dtype=gt.dtype)
            x_coords = torch.arange(w, device=gt.device, dtype=gt.dtype)
            cy = (gt.sum(dim=1) * y_coords).sum() / fg_sum
            cx = (gt.sum(dim=0) * x_coords).sum() / fg_sum
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

        # Handle edge case: n <= 1
        if n <= 1:
            return pred.new_ones(())

        sigma_x = ((pred - x).pow(2)).sum() / (n - 1)
        sigma_y = ((gt - y).pow(2)).sum() / (n - 1)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x.pow(2) + y.pow(2)) * (sigma_x + sigma_y)

        # Replace conditionals with torch.where
        normal_result = alpha / (beta + 1e-8)
        return torch.where(
            alpha != 0,
            normal_result,
            torch.where(beta == 0, pred.new_ones(()), pred.new_zeros(())),
        )

    def compute(self) -> Tensor:
        """Compute mean S-measure."""
        result = self.sm_sum / (self.count + 1e-8)
        return torch.clamp(result, min=0)


class EMeasureMetric(Metric):
    """Enhanced-alignment Measure (E-measure) for binary segmentation.

    Computes alignment between prediction and ground truth by measuring
    how well the prediction's global mean and local values match the target.
    Uses an enhanced alignment matrix that penalizes both false positives and negatives.

    Reference: Fan et al., "Enhanced-alignment Measure for Binary Foreground Map Evaluation" (IJCAI 2018)
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

        em_values = vmap(self._compute_em)(preds, target_bin)
        self.em_sum += em_values.sum()
        self.count += preds.shape[0]

    def _compute_em(self, pred: Tensor, gt: Tensor) -> Tensor:
        """Compute E-measure for a single sample with adaptive threshold."""
        # Adaptive threshold (no .item() call)
        thresh = torch.clamp(2 * pred.mean(), max=1.0)
        pred_bin = (pred >= thresh).float()

        h, w = gt.shape
        gt_size = h * w
        gt_fg_numel = gt.sum()

        fg_match = (pred_bin * gt).sum()  # True positives
        fg_mismatch = (pred_bin * (1 - gt)).sum()  # False positives
        pred_fg_numel = pred_bin.sum()
        pred_bg_numel = gt_size - pred_fg_numel

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

        # Compute enhanced matrix for each part (vectorized, no loop)
        numels = torch.stack([fg_match, fg_mismatch, bg_fg, bg_bg])
        d_preds = torch.stack([d_pred_fg, d_pred_fg, d_pred_bg, d_pred_bg])
        d_gts = torch.stack([d_gt_fg, d_gt_bg, d_gt_fg, d_gt_bg])

        align = 2 * d_preds * d_gts / (d_preds**2 + d_gts**2 + 1e-8)
        enhanced = (align + 1) ** 2 / 4
        enhanced_sum = (enhanced * numels).sum()

        general_result = enhanced_sum / (gt_size - 1 + 1e-8)

        # Handle edge cases with torch.where
        case_no_fg = pred_bg_numel / (gt_size - 1 + 1e-8)
        case_all_fg = pred_fg_numel / (gt_size - 1 + 1e-8)

        return torch.where(
            gt_fg_numel == 0,
            case_no_fg,
            torch.where(gt_fg_numel == gt_size, case_all_fg, general_result),
        )

    def compute(self) -> Tensor:
        """Compute mean E-measure."""
        return self.em_sum / (self.count + 1e-8)


class BoundaryFScoreMetric(Metric):
    """Boundary F-score metric for edge quality evaluation.

    Measures how well the predicted boundaries match the ground truth boundaries.
    Uses morphological operations to extract boundaries.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        boundary_width: int = 2,
        beta: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        """Initialize boundary F-score metric.

        Args:
            boundary_width: Width of boundary zone in pixels.
            beta: F-score beta parameter (1.0 for F1).
            threshold: Threshold for binarizing predictions.

        """
        super().__init__()
        self.boundary_width = boundary_width
        self.beta = beta
        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _extract_boundary(self, mask: Tensor) -> Tensor:
        """Extract boundary from binary mask using dilation/erosion.

        Args:
            mask: Binary mask [B, H, W] or [H, W].

        Returns:
            Boundary mask of same shape.

        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Kernel size for morphological operations
        k = 2 * self.boundary_width + 1

        # Pad for valid convolution
        padded = torch.nn.functional.pad(
            mask.unsqueeze(1).float(),
            (self.boundary_width, self.boundary_width, self.boundary_width, self.boundary_width),
            mode="replicate",
        )

        # Dilation (max pooling approximation)
        dilated = torch.nn.functional.max_pool2d(padded, k, stride=1)

        # Erosion (min pooling = neg max neg)
        eroded = -torch.nn.functional.max_pool2d(-padded, k, stride=1)

        # Boundary = dilated - eroded
        boundary = (dilated - eroded).squeeze(1)

        return (boundary > 0).float()

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state.

        Args:
            preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
            target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].

        """
        preds, target = _normalize_tensors(preds, target)

        # Binarize predictions
        pred_bin = (preds >= self.threshold).float()

        # Extract boundaries
        pred_boundary = self._extract_boundary(pred_bin)
        gt_boundary = self._extract_boundary(target)

        # Compute TP, FP, FN
        tp = (pred_boundary * gt_boundary).sum()
        fp = (pred_boundary * (1 - gt_boundary)).sum()
        fn = ((1 - pred_boundary) * gt_boundary).sum()

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self) -> Tensor:
        """Compute boundary F-score."""
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)

        beta_sq = self.beta**2
        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8)


class NegativeFPRateMetric(Metric):
    """False positive rate metric for negative examples.

    Measures the rate of false positive predictions on images
    that should have no foreground (negative examples).
    """

    is_differentiable: bool = False
    higher_is_better: bool = False  # Lower is better
    full_state_update: bool = False

    def __init__(
        self,
        threshold: float = 0.5,
        area_threshold: float = 0.01,
    ) -> None:
        """Initialize negative FP rate metric.

        Args:
            threshold: Threshold for binarizing predictions.
            area_threshold: Minimum area ratio to count as false positive.

        """
        super().__init__()
        self.threshold = threshold
        self.area_threshold = area_threshold

        self.add_state("fp_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Tensor,
        target: Tensor,
        is_negative: Tensor | None = None,
    ) -> None:
        """Update metric state.

        Args:
            preds: Predictions [B, 1, H, W] or [B, H, W], values in [0, 1].
            target: Ground truth [B, 1, H, W] or [B, H, W], values in [0, 1].
            is_negative: Optional boolean tensor [B] indicating negative samples.
                If None, infers from target (empty masks are negative).

        """
        preds, target = _normalize_tensors(preds, target)

        # Infer negative samples if not provided
        if is_negative is None:
            # Samples with no foreground in GT are negative
            is_negative = target.sum(dim=(1, 2)) == 0

        if not is_negative.any():
            return

        # Filter to negative samples only
        neg_preds = preds[is_negative]

        # Binarize predictions
        pred_bin = (neg_preds >= self.threshold).float()

        # Check for false positives (any prediction on negative samples)
        pred_area = pred_bin.sum(dim=(1, 2)) / (pred_bin.shape[1] * pred_bin.shape[2])
        fp_samples = (pred_area > self.area_threshold).sum()

        self.fp_count += fp_samples
        self.total_negatives += is_negative.sum()

    def compute(self) -> Tensor:
        """Compute false positive rate on negative samples."""
        if self.total_negatives == 0:
            return torch.tensor(0.0, device=self.fp_count.device)

        return self.fp_count.float() / self.total_negatives.float()


class SegmentationMetrics(MetricCollection):
    """Collection of all segmentation metrics.

    Usage:
        metrics = SegmentationMetrics()
        metrics.update(preds, targets)
        results = metrics.compute()  # Returns dict with all metric values

    """

    def __init__(self, prefix: str = "", include_boundary: bool = False) -> None:
        """Initialize metric collection.

        Args:
            prefix: Prefix for metric names (e.g., "val_" or "test_").
            include_boundary: Whether to include boundary F-score metric.

        """
        metrics_dict = {
            "iou": IoUMetric(),
            "mae": MAEMetric(),
            "fm_adp": FMeasureMetric(beta=0.3),
            "sm": SMeasureMetric(alpha=0.5),
            "em_adp": EMeasureMetric(),
        }

        if include_boundary:
            metrics_dict["boundary_f"] = BoundaryFScoreMetric()

        super().__init__(metrics_dict, prefix=prefix)
