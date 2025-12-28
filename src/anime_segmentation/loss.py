"""Hybrid loss functions for anime segmentation.

Implements BCE/IoU/SSIM/Structure/Contour losses to constrain
pixel/region/boundary simultaneously.

Based on BiRefNet loss design.
"""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def gaussian(window_size: int, sigma: float) -> Tensor:
    """Create 1D Gaussian kernel."""
    gauss = torch.tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Tensor:
    """Create 2D Gaussian window for SSIM computation."""
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d_window.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(
    img1: Tensor,
    img2: Tensor,
    window: Tensor,
    window_size: int,
    channel: int,
    *,
    size_average: bool = True,
) -> Tensor:
    """Compute SSIM between two images."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()  # type: ignore[return-value]
    return ssim_map.mean(1).mean(1).mean(1)


class IoULoss(nn.Module):
    """Intersection over Union loss for region-level supervision."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        b = pred.shape[0]
        iou_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for i in range(b):
            iand = torch.sum(target[i] * pred[i])
            ior = torch.sum(target[i]) + torch.sum(pred[i]) - iand
            iou = iand / (ior + 1e-8)
            iou_loss = iou_loss + (1 - iou)
        return iou_loss / b


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss."""

    window: Tensor

    def __init__(self, window_size: int = 11, *, size_average: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer("window", create_window(window_size, self.channel))

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        _, channel, _, _ = img1.size()
        window: Tensor = self.window
        if channel != self.channel or window.data.type() != img1.data.type():
            window = create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)

        ssim_val = _ssim(
            img1, img2, window, self.window_size, channel, size_average=self.size_average
        )
        # Convert SSIM (1=identical) to loss (0=identical)
        return 1 - (1 + ssim_val) / 2


class StructureLoss(nn.Module):
    """Boundary-weighted BCE + IoU loss.

    Uses avg_pool to detect boundaries and apply higher weights there.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Boundary weight: 1 + 5 * |avg_pool(target) - target|
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )

        # Weighted BCE (pred is logits)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU
        pred_sig = torch.sigmoid(pred)
        inter = ((pred_sig * target) * weit).sum(dim=(2, 3))
        union = ((pred_sig + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class ContourLoss(nn.Module):
    """Gradient-based contour/edge loss.

    Combines length term (edge smoothness) and region term (inside/outside).
    """

    def forward(self, pred: Tensor, target: Tensor, weight: float = 10.0) -> Tensor:
        # Horizontal and vertical gradients
        delta_r = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # (B, C, H-1, W)
        delta_c = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # (B, C, H, W-1)

        # Squared gradients (cropped to same size)
        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        # Length term: smooth edges
        epsilon = 1e-8
        length = torch.mean(torch.sqrt(delta_pred + epsilon))

        # Region term: inside/outside consistency
        c_in = torch.ones_like(pred)
        c_out = torch.zeros_like(pred)

        region_in = torch.mean(pred * (target - c_in) ** 2)
        region_out = torch.mean((1 - pred) * (target - c_out) ** 2)
        region = region_in + region_out

        return weight * length + region


class HybridLoss(nn.Module):
    """Hybrid loss combining BCE/IoU/SSIM/Structure/Contour.

    Args:
        bce_weight: Weight for BCE loss (pixel-level)
        iou_weight: Weight for IoU loss (region-level)
        ssim_weight: Weight for SSIM loss (structure-level)
        structure_weight: Weight for Structure loss (boundary-weighted)
        contour_weight: Weight for Contour loss (edge-level)
    """

    def __init__(
        self,
        bce_weight: float = 30.0,
        iou_weight: float = 0.5,
        ssim_weight: float = 10.0,
        structure_weight: float = 0.0,
        contour_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.ssim_weight = ssim_weight
        self.structure_weight = structure_weight
        self.contour_weight = contour_weight

        self.iou_loss = IoULoss()
        self.ssim_loss = SSIMLoss()
        self.structure_loss = StructureLoss()
        self.contour_loss = ContourLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute hybrid loss.

        Args:
            pred: Predicted logits (before sigmoid), shape (B, 1, H, W)
            target: Ground truth mask, shape (B, 1, H, W)

        Returns:
            Combined loss scalar
        """
        # Interpolate if shapes don't match
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=True)

        pred_sig = torch.sigmoid(pred)
        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # BCE loss (with logits)
        if self.bce_weight > 0:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
            loss = loss + self.bce_weight * bce

        # IoU loss
        if self.iou_weight > 0:
            iou = self.iou_loss(pred_sig, target)
            loss = loss + self.iou_weight * iou

        # SSIM loss
        if self.ssim_weight > 0:
            ssim = self.ssim_loss(pred_sig, target)
            loss = loss + self.ssim_weight * ssim

        # Structure loss (uses logits internally for BCE part)
        if self.structure_weight > 0:
            structure = self.structure_loss(pred, target)
            loss = loss + self.structure_weight * structure

        # Contour loss
        if self.contour_weight > 0:
            contour = self.contour_loss(pred_sig, target)
            loss = loss + self.contour_weight * contour

        return loss


# Default loss weights matching BiRefNet configuration
DEFAULT_LOSS_WEIGHTS = {
    "bce": 30.0,
    "iou": 0.5,
    "ssim": 10.0,
    "structure": 0.0,  # Disabled (overlaps with BCE+IoU)
    "contour": 0.0,  # Disabled
}

# Global configured weights (can be overridden before model initialization)
_configured_weights: dict[str, float] | None = None


def configure_loss_weights(weights: dict[str, float]) -> None:
    """Configure global loss weights before model initialization.

    This should be called before any model is created to ensure
    all models use the same loss weights.

    Args:
        weights: Dictionary with keys 'bce', 'iou', 'ssim', 'structure', 'contour'
    """
    global _configured_weights
    _configured_weights = weights


def get_configured_weights() -> dict[str, float]:
    """Get currently configured weights, or defaults if not configured."""
    if _configured_weights is not None:
        return _configured_weights
    return DEFAULT_LOSS_WEIGHTS


def get_hybrid_loss(weights: dict[str, float] | None = None) -> HybridLoss:
    """Create HybridLoss with specified, configured, or default weights."""
    if weights is None:
        weights = get_configured_weights()
    return HybridLoss(
        bce_weight=weights.get("bce", 30.0),
        iou_weight=weights.get("iou", 0.5),
        ssim_weight=weights.get("ssim", 10.0),
        structure_weight=weights.get("structure", 0.0),
        contour_weight=weights.get("contour", 0.0),
    )
