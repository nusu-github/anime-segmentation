"""BiRefNet-style evaluation metrics for segmentation.

Implements S-measure, E-measure, F-measure, Weighted F-measure, and MAE
for comprehensive segmentation quality evaluation.

Based on BiRefNet evaluation design.
"""

from functools import partial

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_erosion, convolve, maximum_filter
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import label
from skimage.morphology import disk, skeletonize
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


def _prepare_data_np(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt = gt > 128
    pred = pred / 255.0
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel boundary from a binary mask."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return np.logical_and(mask, np.logical_not(eroded))


def _boundary_match_stats(
    pred_boundary: np.ndarray, gt_boundary: np.ndarray, tolerance: int
) -> tuple[float, float]:
    """Boundary matching precision/recall with distance tolerance."""
    pred_count = pred_boundary.sum()
    gt_count = gt_boundary.sum()
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0, 0.0

    dt_gt = bwdist(gt_boundary == 0)
    dt_pred = bwdist(pred_boundary == 0)

    pred_match = pred_boundary & (dt_gt <= tolerance)
    gt_match = gt_boundary & (dt_pred <= tolerance)

    precision = pred_match.sum() / max(pred_count, 1)
    recall = gt_match.sum() / max(gt_count, 1)
    return float(precision), float(recall)


def _skeleton_from_distance(mask: np.ndarray) -> np.ndarray:
    """Approximate skeleton via distance-transform ridge."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    dist = bwdist(mask == 0)
    if dist.max() <= 0:
        return np.zeros_like(mask, dtype=bool)
    local_max = maximum_filter(dist, size=3, mode="constant")
    return np.logical_and(dist > 0, dist == local_max)


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


class HCE(Metric):
    """Human Correction Effort (boundary distance of errors).

    Computes the mean distance from misclassified pixels to the GT boundary,
    normalized by max(H, W). Lower is better.
    """

    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_hce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        for i in range(pred.shape[0]):
            pred_ary = (pred[i].squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            gt_ary = (gt[i].squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            gt_ske = skeletonize(gt_ary > 128)
            hce = self._cal_hce(pred_ary, gt_ary, gt_ske)
            self.sum_hce += torch.tensor(hce, device=pred.device, dtype=pred.dtype)
            self.count += 1

    def compute(self) -> Tensor:
        return self.sum_hce / self.count.clamp(min=1)

    @staticmethod
    def _cal_hce(
        pred: np.ndarray, gt: np.ndarray, gt_ske: np.ndarray, relax: int = 5, epsilon: float = 2.0
    ) -> float:
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        gt = (gt > 128).astype(np.uint8)

        if len(pred.shape) > 2:
            pred = pred[:, :, 0]
        pred = (pred > 128).astype(np.uint8)

        union = np.logical_or(gt, pred)
        tp = np.logical_and(gt, pred)
        fp = pred - tp
        fn = gt - tp

        union_erode = cv2.erode(union.astype(np.uint8), disk(1), iterations=relax)

        fp_relaxed = np.logical_and(fp, union_erode)
        for _ in range(relax):
            fp_relaxed = cv2.dilate(fp_relaxed.astype(np.uint8), disk(1))
            fp_relaxed = np.logical_and(fp_relaxed, 1 - np.logical_or(tp, fn))
        fp_relaxed = np.logical_and(fp, fp_relaxed)

        fn_relaxed = np.logical_and(fn, union_erode)
        for _ in range(relax):
            fn_relaxed = cv2.dilate(fn_relaxed.astype(np.uint8), disk(1))
            fn_relaxed = np.logical_and(fn_relaxed, 1 - np.logical_or(tp, fp))
        fn_relaxed = np.logical_and(fn, fn_relaxed)
        fn_relaxed = np.logical_or(fn_relaxed, np.logical_xor(gt_ske, np.logical_and(tp, gt_ske)))

        ctrs_fp = cv2.findContours(
            fp_relaxed.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        ctrs_fp = ctrs_fp[0] if len(ctrs_fp) == 2 else ctrs_fp[1]
        bdies_fp, indep_cnt_fp = HCE._filter_bdy_cond(
            ctrs_fp, fp_relaxed, np.logical_or(tp, fn_relaxed)
        )

        ctrs_fn = cv2.findContours(
            fn_relaxed.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        ctrs_fn = ctrs_fn[0] if len(ctrs_fn) == 2 else ctrs_fn[1]
        bdies_fn, indep_cnt_fn = HCE._filter_bdy_cond(
            ctrs_fn,
            fn_relaxed,
            1 - np.logical_or(np.logical_or(tp, fp_relaxed), fn_relaxed),
        )

        _, _, poly_fp_point_cnt = HCE._approximate_rdp(bdies_fp, epsilon=epsilon)
        _, _, poly_fn_point_cnt = HCE._approximate_rdp(bdies_fn, epsilon=epsilon)

        return float(poly_fp_point_cnt + indep_cnt_fp + poly_fn_point_cnt + indep_cnt_fn)

    @staticmethod
    def _filter_bdy_cond(
        bdy: list[np.ndarray], mask: np.ndarray, cond: np.ndarray
    ) -> tuple[list[np.ndarray], float]:
        cond = cv2.dilate(cond.astype(np.uint8), disk(1))
        labels = label(mask)
        indep = np.ones(lbls.shape[0])
        indep[0] = 0

        boundaries: list[np.ndarray] = []
        h, w = cond.shape[:2]
        ind_map = np.zeros((h, w))

        for item in bdy:
            tmp_bdies = []
            tmp_bdy = []
            for j in range(item.shape[0]):
                r, c = item[j, 0, 1], item[j, 0, 0]

                if (np.sum(cond[r, c]) == 0) or (ind_map[r, c] != 0):
                    if len(tmp_bdy) > 0:
                        tmp_bdies.append(tmp_bdy)
                        tmp_bdy = []
                    continue
                tmp_bdy.append([c, r])
                ind_map[r, c] = ind_map[r, c] + 1
                indep[labels[r, c]] = 0
            if len(tmp_bdy) > 0:
                tmp_bdies.append(tmp_bdy)

            if len(tmp_bdies) > 1:
                first_x, first_y = tmp_bdies[0][0]
                last_x, last_y = tmp_bdies[-1][-1]
                if (
                    (abs(first_x - last_x) == 1 and first_y == last_y)
                    or (first_x == last_x and abs(first_y - last_y) == 1)
                    or (abs(first_x - last_x) == 1 and abs(first_y - last_y) == 1)
                ):
                    tmp_bdies[-1].extend(tmp_bdies[0][::-1])
                    del tmp_bdies[0]

            for k in range(len(tmp_bdies)):
                tmp_bdies[k] = np.array(tmp_bdies[k])[:, np.newaxis, :]
            if tmp_bdies:
                boundaries.extend(tmp_bdies)

        return boundaries, float(np.sum(indep))

    @staticmethod
    def _approximate_rdp(
        boundaries: list[np.ndarray], epsilon: float = 1.0
    ) -> tuple[list[np.ndarray], list[int], int]:
        boundaries_len_ = []
        pixel_cnt = 0

        boundaries_ = [cv2.approxPolyDP(item, epsilon, False) for item in boundaries]
        for item_ in boundaries_:
            boundaries_len_.append(len(item_))
            pixel_cnt = pixel_cnt + len(item_)

        return boundaries_, boundaries_len_, pixel_cnt


class BoundaryIoU(Metric):
    """Boundary IoU with dilation-based tolerance."""

    higher_is_better = True
    full_state_update = False

    def __init__(self, dilation_ratio: float = 0.02, mode: str = "mean", **kwargs) -> None:
        super().__init__(**kwargs)
        if mode not in {"mean", "max"}:
            raise ValueError("BoundaryIoU mode must be 'mean' or 'max'.")
        self.dilation_ratio = dilation_ratio
        self.mode = mode
        self.add_state("sum_biou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        for i in range(pred.shape[0]):
            pred_ary = (pred[i].squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            gt_ary = (gt[i].squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            pred_norm, gt_mask = _prepare_data_np(pred_ary, gt_ary)
            ious = self._cal_biou(pred_norm, gt_mask)
            biou = float(ious.max()) if self.mode == "max" else float(ious.mean())
            self.sum_biou += torch.tensor(biou, device=pred.device, dtype=pred.dtype)
            self.count += 1

    def compute(self) -> Tensor:
        return self.sum_biou / self.count.clamp(min=1)

    def _mask_to_boundary(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        img_diag = np.sqrt(h**2 + w**2)
        dilation = round(self.dilation_ratio * img_diag)
        dilation = max(dilation, 1)
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        return mask - mask_erode

    def _cal_biou(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = (pred * 255).astype(np.uint8)
        pred = self._mask_to_boundary(pred)
        gt = (gt * 255).astype(np.uint8)
        gt = self._mask_to_boundary(gt)
        gt = gt > 128

        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        tps = fg_w_thrs
        ps = fg_w_thrs + bg_w_thrs
        ps[ps == 0] = 1
        t = max(np.count_nonzero(gt), 1)

        return tps / (t + bg_w_thrs)


class BoundaryFMeasure(Metric):
    """Boundary F-measure with distance tolerance."""

    higher_is_better = True
    full_state_update = False

    def __init__(self, tolerance: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.add_state("sum_bf", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        for i in range(pred.shape[0]):
            p = pred[i].squeeze()
            g = gt[i].squeeze()
            p, g = _prepare_data(p, g)
            adaptive_threshold = _get_adaptive_threshold(p)
            p_bin = (p >= adaptive_threshold).cpu().numpy().astype(bool)
            g_bin = (g > 0.5).cpu().numpy().astype(bool)

            p_b = _binary_boundary(p_bin)
            g_b = _binary_boundary(g_bin)

            precision, recall = _boundary_match_stats(p_b, g_b, self.tolerance)
            denom = precision + recall
            bf = 0.0 if denom == 0 else 2 * precision * recall / denom
            self.sum_bf += torch.tensor(bf, device=pred.device, dtype=pred.dtype)
            self.count += 1

    def compute(self) -> Tensor:
        return self.sum_bf / self.count.clamp(min=1)


class MeanBoundaryAccuracy(Metric):
    """Mean Boundary Accuracy (average of boundary precision and recall)."""

    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_mba", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        for i in range(pred.shape[0]):
            pred_ary = (pred[i].squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            gt_ary = (gt[i].squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            mba = self._cal_mba(pred_ary, gt_ary)
            self.sum_mba += torch.tensor(mba, device=pred.device, dtype=pred.dtype)
            self.count += 1

    def compute(self) -> Tensor:
        return self.sum_mba / self.count.clamp(min=1)

    @staticmethod
    def _get_disk_kernel(radius: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

    @classmethod
    def _cal_ba(cls, pred: np.ndarray, gt: np.ndarray) -> float:
        gt = gt.astype(np.uint8)
        pred = pred.astype(np.uint8)

        h, w = gt.shape
        min_radius = 1
        max_radius = (w + h) / 300
        num_steps = 5

        pred_acc: list[float] = []
        for i in range(num_steps):
            curr_radius = min_radius + int((max_radius - min_radius) / num_steps * i)
            kernel = cls._get_disk_kernel(curr_radius)
            boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

            gt_in_bound = gt[boundary_region]
            pred_in_bound = pred[boundary_region]

            num_edge_pixels = boundary_region.sum()
            num_pred_gd_pix = (
                (gt_in_bound) * (pred_in_bound) + (1 - gt_in_bound) * (1 - pred_in_bound)
            ).sum()
            pred_acc.append(num_pred_gd_pix / num_edge_pixels)

        return float(sum(pred_acc) / num_steps)

    @classmethod
    def _cal_mba(cls, pred: np.ndarray, gt: np.ndarray) -> float:
        pred_bin = pred > 128
        gt_bin = gt > 128
        return cls._cal_ba(pred_bin, gt_bin)


class SkeletonFMeasure(Metric):
    """Skeleton F-measure using distance-transform ridge skeletons."""

    higher_is_better = True
    full_state_update = False

    def __init__(self, tolerance: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.add_state("sum_skel_f", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, gt: Tensor) -> None:
        for i in range(pred.shape[0]):
            p = pred[i].squeeze()
            g = gt[i].squeeze()
            p, g = _prepare_data(p, g)
            adaptive_threshold = _get_adaptive_threshold(p)
            p_bin = (p >= adaptive_threshold).cpu().numpy().astype(bool)
            g_bin = (g > 0.5).cpu().numpy().astype(bool)

            p_skel = _skeleton_from_distance(p_bin)
            g_skel = _skeleton_from_distance(g_bin)
            precision, recall = _boundary_match_stats(p_skel, g_skel, self.tolerance)
            denom = precision + recall
            skel_f = 0.0 if denom == 0 else 2 * precision * recall / denom
            self.sum_skel_f += torch.tensor(skel_f, device=pred.device, dtype=pred.dtype)
            self.count += 1

    def compute(self) -> Tensor:
        return self.sum_skel_f / self.count.clamp(min=1)
