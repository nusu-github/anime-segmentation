import torch

from anime_segmentation.training.metrics import SMeasureMetric


def _s_measure_reference(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    y = gt.mean()

    def _s_object(pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        mask = gt_obj.bool()
        mask_sum = mask.sum()
        if mask_sum > 0:
            masked_pred = pred_obj[mask]
            x = masked_pred.mean()
            sigma_x = masked_pred.std() if mask_sum > 1 else pred_obj.new_zeros(())
            return 2 * x / (x.pow(2) + 1 + sigma_x + 1e-8)
        return pred_obj.new_zeros(())

    def _object_similarity(pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        fg = pred_obj * gt_obj
        bg = (1 - pred_obj) * (1 - gt_obj)
        u = gt_obj.mean()
        fg_score = _s_object(fg, gt_obj)
        bg_score = _s_object(bg, 1 - gt_obj)
        return u * fg_score + (1 - u) * bg_score

    def _ssim(pred_region: torch.Tensor, gt_region: torch.Tensor) -> torch.Tensor:
        x = pred_region.mean()
        y_region = gt_region.mean()
        n = pred_region.numel()
        if n <= 1:
            return pred_region.new_ones(())

        sigma_x = ((pred_region - x).pow(2)).sum() / (n - 1)
        sigma_y = ((gt_region - y_region).pow(2)).sum() / (n - 1)
        sigma_xy = ((pred_region - x) * (gt_region - y_region)).sum() / (n - 1)

        alpha_ssim = 4 * x * y_region * sigma_xy
        beta = (x.pow(2) + y_region.pow(2)) * (sigma_x + sigma_y)
        normal_result = alpha_ssim / (beta + 1e-8)
        return torch.where(
            alpha_ssim != 0,
            normal_result,
            torch.where(beta == 0, pred_region.new_ones(()), pred_region.new_zeros(())),
        )

    def _region_similarity(pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        h, w = gt_obj.shape
        fg_sum = gt_obj.sum()
        if fg_sum == 0:
            cx, cy = w // 2, h // 2
        else:
            y_coords = torch.arange(h, device=gt_obj.device, dtype=gt_obj.dtype)
            x_coords = torch.arange(w, device=gt_obj.device, dtype=gt_obj.dtype)
            cy = (gt_obj.sum(dim=1) * y_coords).sum() / fg_sum
            cx = (gt_obj.sum(dim=0) * x_coords).sum() / fg_sum
            cx, cy = int(cx.round()), int(cy.round())

        cx = max(1, min(cx, w - 1))
        cy = max(1, min(cy, h - 1))

        regions = [
            (slice(0, cy), slice(0, cx)),
            (slice(0, cy), slice(cx, w)),
            (slice(cy, h), slice(0, cx)),
            (slice(cy, h), slice(cx, w)),
        ]

        total_area = h * w
        weights = []
        scores = []
        for region in regions:
            pred_region = pred_obj[region]
            gt_region = gt_obj[region]
            area = pred_region.numel()
            if area == 0:
                continue
            weights.append(area / total_area)
            scores.append(_ssim(pred_region, gt_region))

        if not scores:
            return torch.tensor(0.0, device=pred_obj.device)

        weights_t = torch.tensor(weights, device=pred_obj.device)
        scores_t = torch.stack(scores)
        return (weights_t * scores_t).sum()

    so = _object_similarity(pred, gt)
    sr = _region_similarity(pred, gt)
    combined = alpha * so + (1 - alpha) * sr
    return torch.where(y == 0, 1 - pred.mean(), torch.where(y == 1, pred.mean(), combined))


def test_s_measure_batch_matches_reference() -> None:
    torch.manual_seed(0)
    preds = torch.rand(3, 1, 5, 7)
    targets = torch.randint(0, 2, (3, 1, 5, 7)).float()

    targets[0] = 0
    targets[1] = 1

    metric = SMeasureMetric(alpha=0.5)
    metric.update(preds, targets)
    batch_result = metric.compute()

    ref_vals = torch.stack(
        [_s_measure_reference(preds[i, 0], targets[i, 0], alpha=0.5) for i in range(3)],
    )

    assert torch.allclose(batch_result, ref_vals.mean(), atol=1e-6)
