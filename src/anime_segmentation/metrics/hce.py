"""Human Correction Effort (HCE) metric for segmentation evaluation.

This module implements the Human Correction Effort metric from the DIS
dataset evaluation protocol. HCE estimates the number of control points
a human annotator would need to correct prediction errors, providing
an intuitive measure of segmentation quality from a practical editing
perspective.

The implementation uses scikit-image for morphological operations and
polygon approximation, running on CPU.

Reference:
    Qin et al., "Highly Accurate Dichotomous Image Segmentation", ECCV 2022.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from skimage.measure import approximate_polygon, find_contours, label
from skimage.morphology import dilation as sk_dilation
from skimage.morphology import disk, skeletonize
from skimage.morphology import erosion as sk_erosion
from torch import Tensor
from torchmetrics import Metric

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HumanCorrectionEffort(Metric):
    """Human Correction Effort (HCE) metric for segmentation evaluation.

    Measures the number of control points needed to correct prediction errors,
    providing an intuitive proxy for the practical effort required to fix
    a segmentation result. Uses relaxed boundary regions to tolerate minor
    boundary variations and RDP polygon approximation to estimate control points.

    As input to ``forward`` and ``update`` the metric accepts:
        - ``pred``: Tensor of shape (N, 1, H, W) or (N, H, W) with values in [0, 255] or [0, 1]
        - ``gt``: Tensor of shape (N, 1, H, W) or (N, H, W) with binary values
        - ``gt_ske``: Optional pre-computed skeleton of ground truth

    As output of ``forward`` and ``compute`` the metric returns:
        - ``HCE``: Mean human correction effort score (lower is better)

    Args:
        relax: Number of erosion iterations for boundary relaxation.
               Higher values tolerate more boundary variation. Default is 5.
        epsilon: Tolerance for Ramer-Douglas-Peucker polygon approximation.
                 Higher values produce fewer control points. Default is 2.0.
        **kwargs: Additional arguments passed to the base Metric class.

    Attributes:
        hce_scores: List of per-sample HCE scores.

    Note:
        This metric uses scikit-image for morphological operations and runs
        on CPU. The skeleton computation can be expensive for large masks.
        Pre-computing skeletons and passing them via gt_ske can improve
        performance when evaluating multiple models on the same ground truth.

    Example:
        >>> metric = HumanCorrectionEffort(relax=5, epsilon=2.0)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> gt = (torch.rand(4, 1, 256, 256) > 0.5).float()
        >>> metric.update(pred, gt)
        >>> hce = metric.compute()
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    hce_scores: list[Tensor]

    def __init__(
        self,
        relax: int = 5,
        epsilon: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Human Correction Effort metric.

        Args:
            relax: Boundary relaxation iterations for error tolerance.
            epsilon: RDP polygon approximation tolerance.
            **kwargs: Additional arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)
        self.relax = relax
        self.epsilon = epsilon
        self.add_state("hce_scores", default=[], dist_reduce_fx="cat")

    def update(
        self,
        pred: Tensor,
        gt: Tensor,
        gt_ske: Tensor | None = None,
    ) -> None:
        """Update state with predictions and targets.

        Args:
            pred: Predicted mask tensor.
            gt: Ground truth mask tensor.
            gt_ske: Optional pre-computed skeleton of ground truth.
                    If None, skeleton is computed automatically.
        """
        pred, gt = self._prepare_inputs(pred, gt)

        batch_size = pred.shape[0] if pred.ndim > 2 else 1
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            if gt_ske is not None:
                gt_ske = gt_ske.unsqueeze(0)

        for i in range(batch_size):
            p, g = pred[i], gt[i]
            ske = gt_ske[i] if gt_ske is not None else None
            hce = self._compute_hce(p, g, ske)
            self.hce_scores.append(hce.unsqueeze(0))

    def _compute_hce(
        self,
        pred: Tensor,
        gt: Tensor,
        gt_ske: Tensor | None = None,
    ) -> Tensor:
        """Compute HCE for a single sample.

        Args:
            pred: Single prediction tensor of shape (H, W).
            gt: Single ground truth tensor of shape (H, W).
            gt_ske: Optional pre-computed skeleton tensor.

        Returns:
            Scalar tensor with the HCE score (number of control points).
        """
        pred_np = pred.cpu().numpy().astype(np.uint8)
        gt_np = gt.cpu().numpy().astype(np.uint8)

        pred_np = (pred_np > 0.5).astype(np.uint8)
        gt_np = (gt_np > 0.5).astype(np.uint8)

        ske_np = skeletonize(gt_np > 0) if gt_ske is None else gt_ske.cpu().numpy() > 0

        hce = self._relax_hce(gt_np, pred_np, ske_np)

        return torch.tensor(hce, device=pred.device, dtype=torch.float32)

    def _relax_hce(
        self,
        gt: NDArray[np.uint8],
        pred: NDArray[np.uint8],
        gt_ske: NDArray[np.bool_],
    ) -> float:
        """Compute relaxed Human Correction Effort.

        Applies boundary relaxation to tolerate minor variations, then
        counts control points needed to correct remaining errors.

        Args:
            gt: Binary ground truth mask.
            pred: Binary prediction mask.
            gt_ske: Ground truth skeleton for structure preservation.

        Returns:
            Total number of control points for correction.
        """
        union_mask = np.logical_or(gt, pred)
        tp = np.logical_and(gt, pred)
        fp = pred.astype(np.int16) - tp.astype(np.int16)
        fn = gt.astype(np.int16) - tp.astype(np.int16)
        fp = fp > 0
        fn = fn > 0

        union_erode = union_mask.copy().astype(np.uint8)
        disk_kernel = disk(1)
        for _ in range(self.relax):
            union_erode = sk_erosion(union_erode, disk_kernel)

        # Relax false positives by dilating and constraining
        fp_relaxed = np.logical_and(fp, union_erode)
        for _ in range(self.relax):
            fp_relaxed = sk_dilation(fp_relaxed.astype(np.uint8), disk_kernel)
            fp_relaxed = np.logical_and(fp_relaxed, ~np.logical_or(tp, fn))
        fp_relaxed = np.logical_and(fp, fp_relaxed)

        # Relax false negatives similarly
        fn_relaxed = np.logical_and(fn, union_erode)
        for _ in range(self.relax):
            fn_relaxed = sk_dilation(fn_relaxed.astype(np.uint8), disk_kernel)
            fn_relaxed = np.logical_and(fn_relaxed, ~np.logical_or(tp, fp))
        fn_relaxed = np.logical_and(fn, fn_relaxed)

        # Ensure skeleton structure is preserved in false negatives
        fn_relaxed = np.logical_or(fn_relaxed, np.logical_xor(gt_ske, np.logical_and(tp, gt_ske)))

        ctrs_fp = self._find_contours_cv2_format(fp_relaxed.astype(np.uint8))
        bdies_fp, indep_cnt_fp = self._filter_bdy_cond(
            ctrs_fp, fp_relaxed, np.logical_or(tp, fn_relaxed)
        )

        ctrs_fn = self._find_contours_cv2_format(fn_relaxed.astype(np.uint8))
        bdies_fn, indep_cnt_fn = self._filter_bdy_cond(
            ctrs_fn,
            fn_relaxed,
            ~np.logical_or(np.logical_or(tp, fp_relaxed), fn_relaxed),
        )

        _, _, poly_fp_point_cnt = self._approximate_rdp(bdies_fp, self.epsilon)
        _, _, poly_fn_point_cnt = self._approximate_rdp(bdies_fn, self.epsilon)

        return float(poly_fp_point_cnt + indep_cnt_fp + poly_fn_point_cnt + indep_cnt_fn)

    @staticmethod
    def _find_contours_cv2_format(mask: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
        """Find contours and convert to OpenCV-compatible format.

        Args:
            mask: Binary mask as uint8 array.

        Returns:
            List of contours in (N, 1, 2) format with (x, y) coordinates.
        """
        sk_contours = find_contours(mask, level=0.5)
        cv2_contours = []
        for contour in sk_contours:
            converted = contour[:, ::-1][:, np.newaxis, :].astype(np.int32)
            cv2_contours.append(converted)
        return cv2_contours

    def _filter_bdy_cond(
        self,
        bdy_: list[NDArray[np.int32]],
        mask: NDArray[np.bool_],
        cond: NDArray[np.bool_],
    ) -> tuple[list[NDArray[np.int32]], int]:
        """Filter boundary segments based on adjacency conditions.

        Identifies boundary segments that are adjacent to the condition mask
        and counts independent (isolated) error regions.

        Args:
            bdy_: List of boundary contours.
            mask: Error region mask.
            cond: Adjacency condition mask.

        Returns:
            Tuple of (filtered boundaries, count of independent regions).
        """
        cond = sk_dilation(cond.astype(np.uint8), disk(1))
        labels = label(mask)
        lbls = np.unique(labels)
        indep = np.ones(lbls.shape[0])
        indep[0] = 0

        boundaries: list[NDArray[np.int32]] = []
        h, w = cond.shape[:2]
        ind_map = np.zeros((h, w))

        for item in bdy_:
            tmp_bdies: list[list[list[int]]] = []
            tmp_bdy: list[list[int]] = []

            for j in range(item.shape[0]):
                r, c = item[j, 0, 1], item[j, 0, 0]

                if np.sum(cond[r, c]) == 0 or ind_map[r, c] != 0:
                    if len(tmp_bdy) > 0:
                        tmp_bdies.append(tmp_bdy)
                        tmp_bdy = []
                    continue

                tmp_bdy.append([c, r])
                ind_map[r, c] += 1
                indep[labels[r, c]] = 0

            if len(tmp_bdy) > 0:
                tmp_bdies.append(tmp_bdy)

            # Merge adjacent boundary segments for closed contours
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

        return boundaries, int(np.sum(indep))

    @staticmethod
    def _approximate_rdp(
        boundaries: list[NDArray[np.int32]], epsilon: float = 1.0
    ) -> tuple[list[NDArray[np.float64]], list[int], int]:
        """Approximate boundaries using Ramer-Douglas-Peucker algorithm.

        Simplifies boundary contours to reduce the number of control points
        while preserving overall shape within the specified tolerance.

        Args:
            boundaries: List of boundary contours.
            epsilon: Approximation tolerance in pixels.

        Returns:
            Tuple of (simplified boundaries, lengths per boundary, total point count).
        """
        boundaries_ = []
        for boundary in boundaries:
            coords = boundary.reshape(-1, 2).astype(np.float64)
            approx = approximate_polygon(coords, tolerance=epsilon)
            boundaries_.append(approx[:, np.newaxis, :])

        boundaries_len_ = []
        pixel_cnt_ = 0

        for item in boundaries_:
            boundaries_len_.append(len(item))
            pixel_cnt_ += len(item)

        return boundaries_, boundaries_len_, pixel_cnt_

    def compute(self) -> Tensor:
        """Compute mean HCE score over all accumulated samples.

        Returns:
            Scalar tensor with the mean HCE score.
        """
        if len(self.hce_scores) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.cat(self.hce_scores).mean()

    def _prepare_inputs(self, pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        """Normalize inputs and apply standard preprocessing.

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
