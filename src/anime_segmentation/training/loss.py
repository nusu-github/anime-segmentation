"""Loss functions for segmentation training."""

import kornia
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Valid loss keys for PixLoss
VALID_PIX_LOSS_KEYS = frozenset(
    {"bce", "iou", "iou_patch", "mae", "mse", "reg", "ssim", "cnt", "structure"},
)

# Valid loss keys for ClsLoss
VALID_CLS_LOSS_KEYS = frozenset({"ce"})


# =============================================================================
# Base class for loss functions with reduction
# =============================================================================


class BaseLoss(nn.Module):
    """Base class for loss functions with common reduction interface.

    Provides a unified reduction mechanism for loss functions.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        """Initialize base loss.

        Args:
            eps: Small epsilon for numerical stability.
            reduction: Reduction mode ('none', 'sum', 'mean').

        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor.

        Args:
            loss: Per-sample loss tensor.

        Returns:
            Reduced loss tensor.

        """
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


# =============================================================================
# Individual loss functions
# =============================================================================


class ContourLoss(nn.Module):
    def __init__(self, weight: float = 10.0, mode: str = "diff", eps: float = 1e-8) -> None:
        """Contour loss using kornia spatial gradient for edge detection.

        Args:
            weight: Weight for the length term.
            mode: Gradient mode for kornia.filters.spatial_gradient ('sobel' or 'diff').
            eps: Small epsilon to avoid sqrt(0).

        """
        super().__init__()
        self.weight = weight
        self.mode = mode
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute contour loss.

        Args:
            pred: Predicted tensor of shape (B, C, H, W), probability (0..1).
            target: Target tensor of shape (B, C, H, W), where region_in_contour == 1, region_out == 0.

        Returns:
            Contour loss value.

        """
        grad = kornia.filters.spatial_gradient(pred, mode=self.mode, order=1, normalized=True)
        grad_mag = torch.sqrt(grad.pow(2).sum(dim=2) + self.eps)
        length = grad_mag.mean()
        region_in = (pred * (target - 1.0).pow(2)).mean()
        region_out = ((1.0 - pred) * target.pow(2)).mean()
        region = region_in + region_out
        return self.weight * length + region


class IoULoss(BaseLoss):
    """Intersection over Union loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            IoU loss value.

        """
        dims = tuple(range(1, pred.ndim))
        inter = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
        loss = 1.0 - (inter + self.eps) / (union + self.eps)
        return self._reduce(loss)


class StructureLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, target):
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target,
        )
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class PatchIoULoss(BaseLoss):
    """Patch-based IoU loss for local structure preservation."""

    def __init__(
        self,
        patch_size: tuple[int, int] = (64, 64),
        stride: tuple[int, int] | None = None,
        eps: float = 1e-7,
        reduction: str = "mean",
    ) -> None:
        """Initialize patch IoU loss.

        Args:
            patch_size: Size of patches (height, width).
            stride: Stride for patch extraction. Defaults to patch_size.
            eps: Small epsilon for numerical stability.
            reduction: Reduction mode ('none', 'sum', 'mean').

        """
        super().__init__(eps=eps, reduction=reduction)
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute patch IoU loss.

        Args:
            pred: Predicted tensor.
            target: Target tensor.

        Returns:
            Patch IoU loss value.

        """
        ph, pw = self.patch_size
        sh, sw = self.stride
        pred_u = F.unfold(pred, kernel_size=(ph, pw), stride=(sh, sw))
        targ_u = F.unfold(target, kernel_size=(ph, pw), stride=(sh, sw))
        inter = (pred_u * targ_u).sum(dim=1)
        union = pred_u.sum(dim=1) + targ_u.sum(dim=1) - inter
        loss = 1.0 - (inter + self.eps) / (union + self.eps)
        return self._reduce(loss)


class ThresholdRegularizationLoss(torch.nn.Module):
    """Threshold regularization loss.

    Encourages predictions to be close to 0 or 1 by penalizing values
    in between. This helps produce sharper, more binary-like outputs.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """Compute threshold regularization loss.

        Args:
            pred: Predicted tensor of shape (B, C, H, W), probability (0..1).
            target: Unused, kept for interface compatibility.

        Returns:
            Regularization loss value.

        """
        del target  # Unused
        return torch.mean(1 - (pred**2 + (pred - 1) ** 2))


class ClsLoss(nn.Module):
    """Auxiliary classification loss for each refined class output."""

    def __init__(self, lambdas_cls: dict[str, float]) -> None:
        super().__init__()
        # Validate keys (fail-fast)
        invalid_keys = set(lambdas_cls.keys()) - VALID_CLS_LOSS_KEYS
        if invalid_keys:
            msg = f"Unknown classification loss keys: {invalid_keys}. Valid: {VALID_CLS_LOSS_KEYS}"
            raise ValueError(msg)

        self.lambdas_cls = lambdas_cls
        self.criterions_last = {
            "ce": nn.CrossEntropyLoss(),
        }

    def forward(self, preds: list[Tensor | None], gt: Tensor) -> Tensor:
        loss = gt.new_zeros(())  # Initialize as Tensor, not float
        for pred_lvl in preds:
            if pred_lvl is None:
                continue
            for criterion_name, criterion in self.criterions_last.items():
                loss = loss + criterion(pred_lvl, gt) * self.lambdas_cls[criterion_name]
        return loss


class SSIMLoss(torch.nn.Module):
    """SSIM Loss using kornia implementation."""

    def __init__(self, window_size: int = 11, max_val: float = 1.0, padding: str = "same") -> None:
        super().__init__()
        self.loss = kornia.losses.SSIMLoss(
            window_size=window_size,
            max_val=max_val,
            padding=padding,
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self.loss(img1.float(), img2.float())  # AMP compatibility


class PixLoss(nn.Module):
    """Pixel loss for each refined map output."""

    def __init__(self, loss_weights: dict[str, float]) -> None:
        super().__init__()
        # Validate keys (fail-fast)
        invalid_keys = set(loss_weights.keys()) - VALID_PIX_LOSS_KEYS
        if invalid_keys:
            msg = f"Unknown pixel loss keys: {invalid_keys}. Valid: {VALID_PIX_LOSS_KEYS}"
            raise ValueError(msg)

        self.lambdas = loss_weights
        # Losses expecting probability input (after sigmoid)
        self.loss_prob = nn.ModuleDict()
        # Note: BCELoss moved to loss_logit as BCEWithLogitsLoss for AMP compatibility
        if self.lambdas.get("iou"):
            self.loss_prob["iou"] = IoULoss()
        if self.lambdas.get("iou_patch"):
            self.loss_prob["iou_patch"] = PatchIoULoss()
        if self.lambdas.get("ssim"):
            self.loss_prob["ssim"] = SSIMLoss()
        if self.lambdas.get("mae"):
            self.loss_prob["mae"] = nn.L1Loss()
        if self.lambdas.get("mse"):
            self.loss_prob["mse"] = nn.MSELoss()
        if self.lambdas.get("reg"):
            self.loss_prob["reg"] = ThresholdRegularizationLoss()
        if self.lambdas.get("cnt"):
            self.loss_prob["cnt"] = ContourLoss()
        # Losses expecting logits input (before sigmoid)
        self.loss_logit = nn.ModuleDict()
        if self.lambdas.get("bce"):
            self.loss_logit["bce"] = nn.BCEWithLogitsLoss()
        if self.lambdas.get("structure"):
            self.loss_logit["structure"] = StructureLoss()

    def forward(
        self,
        scaled_preds: list[torch.Tensor],
        gt: torch.Tensor,
        pix_loss_lambda: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute pixel loss across all scales.

        Args:
            scaled_preds: List of predictions at different scales.
            gt: Ground truth tensor.
            pix_loss_lambda: Global loss weight multiplier.

        Returns:
            Tuple of (total_loss, per_component_loss_dict).

        """
        n = len(scaled_preds)
        gt_size = gt.shape[2:]

        # Pre-resize all predictions to GT size (batch operation)
        resized_logits = [
            F.interpolate(p, size=gt_size, mode="bilinear", align_corners=True)
            if p.shape[2:] != gt_size
            else p
            for p in scaled_preds
        ]

        # Stack for efficient computation: [num_scales, B, C, H, W]
        stacked_logits = torch.stack(resized_logits)
        stacked_probs = stacked_logits.sigmoid()

        # Pre-initialize log accumulators for efficiency
        log_accum: dict[str, torch.Tensor] = {
            name: gt.new_zeros(())
            for name in list(self.loss_prob.keys()) + list(self.loss_logit.keys())
        }

        total = gt.new_zeros(())

        # Compute losses for all scales at once where possible
        for i in range(n):
            probs = stacked_probs[i]
            logits = stacked_logits[i]

            # Probability-based losses
            for name, crit in self.loss_prob.items():
                w = self.lambdas[name] * pix_loss_lambda
                loss_val = crit(probs, gt) * w
                total = total + loss_val
                log_accum[name] = log_accum[name] + (loss_val.detach() / n)

            # Logit-based losses
            for name, crit in self.loss_logit.items():
                w = self.lambdas[name] * pix_loss_lambda
                loss_val = crit(logits, gt) * w
                total = total + loss_val
                log_accum[name] = log_accum[name] + (loss_val.detach() / n)

        loss_dict = {k: v.item() for k, v in log_accum.items() if k in self.lambdas}
        return total, loss_dict
