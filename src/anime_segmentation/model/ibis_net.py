"""IBIS-Net: IS-Net enhanced with BiRefNet-inspired components."""

from dataclasses import dataclass, field

import kornia.filters as KF
import kornia.losses as KL
import kornia.morphology as KM
import kornia.color as KC
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .isnet import RSU4, RSU4F, RSU5, RSU6, RSU7, _upsample_like


@dataclass
class IBISNetConfig:
    img_size: int = 1024

    # OutRef
    use_outref: bool = True
    outref_stages: list[int] = field(default_factory=lambda: [1, 2, 3])
    outref_hidden_channels: int = 16
    outref_dilation_kernel: int = 5

    # InRef
    use_inref: bool = True
    inref_stages: list[int] = field(default_factory=lambda: [1, 2, 3])

    # ReconstructionBlock (not implemented yet)
    use_reconstruction_block: bool = False

    # Loss weights
    lambda_bce: float = 1.0
    lambda_fs: float = 1.0
    lambda_grad: float = 1.0
    lambda_iou: float = 0.5
    lambda_ssim: float = 0.5

    # Loss application
    iou_stages: list[int] = field(default_factory=lambda: [1, 2])
    ssim_stages: list[int] = field(default_factory=lambda: [1])

    # Fine-tuning (reserved)
    finetune_last_epochs: int = 0
    finetune_iou_boost: float = 2.0
    finetune_bce_decay: float = 0.5

    # Logging (reserved)
    log_loss_magnitudes: bool = True
    log_magnitude_iterations: int = 500


class GradientLabelGenerator(nn.Module):
    """Generate gradient labels using Kornia's Sobel filter."""

    def forward(self, image: Tensor) -> Tensor:
        gray = KC.rgb_to_grayscale(image)

        # Kornia sobel returns gradient magnitude directly: (B, C, H, W)
        gradient = KF.sobel(gray)

        # Normalize per image
        grad_max = gradient.amax(dim=(2, 3), keepdim=True)
        return gradient / (grad_max + 1e-8)


class GradientOutRefBlock(nn.Module):
    """Gradient-aware output refinement block using Kornia for morphology."""

    dilation_kernel: Tensor

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        dilation_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.dilation_kernel_size = dilation_kernel_size

        self.local_logit_conv = nn.Conv2d(in_channels, 1, 1, 1, 0)
        self.grad_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.grad_pred_head = nn.Conv2d(hidden_channels, 1, 1, 1, 0)
        self.grad_attn_head = nn.Conv2d(hidden_channels, 1, 1, 1, 0)

        # Register dilation kernel as buffer
        kernel = torch.ones(dilation_kernel_size, dilation_kernel_size)
        self.register_buffer("dilation_kernel", kernel)

    def compute_gradient_label(self, image: Tensor) -> Tensor:
        """Compute gradient using Kornia sobel."""
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        # Kornia sobel returns gradient magnitude directly
        gradient = KF.sobel(gray)

        grad_max = gradient.amax(dim=(2, 3), keepdim=True)
        return gradient / (grad_max + 1e-8)

    def dilate_mask(self, mask: Tensor) -> Tensor:
        """Dilate mask using Kornia morphology."""
        return KM.dilation(mask, self.dilation_kernel)

    def forward(
        self,
        features: Tensor,
        image: Tensor,
        grad_gt_fullres: Tensor | None = None,
        training: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
        _, _c, height, width = features.shape

        local_logit = self.local_logit_conv(features)

        grad_feat = self.grad_conv(features)
        grad_attn = self.grad_attn_head(grad_feat).sigmoid()
        features_attn = features * grad_attn

        if training:
            grad_pred = self.grad_pred_head(grad_feat)

            if grad_gt_fullres is not None:
                grad_gt = F.interpolate(
                    grad_gt_fullres, size=(height, width), mode="bilinear", align_corners=True
                )
            else:
                image_resized = F.interpolate(
                    image, size=(height, width), mode="bilinear", align_corners=True
                )
                grad_gt = self.compute_gradient_label(image_resized)

            mask_sigmoid = local_logit.sigmoid().detach()
            mask_dilated = self.dilate_mask(mask_sigmoid)
            grad_label = grad_gt * mask_dilated

            return features_attn, local_logit, grad_pred, grad_label, grad_attn

        return features_attn, local_logit, None, None, grad_attn


class InRefFusion(nn.Module):
    def __init__(self, feature_channels: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        input_ref_channels = 3 * scale * scale
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_channels + input_ref_channels, feature_channels, 1, 1, 0),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, features: Tensor, image: Tensor) -> Tensor:
        _, _c, height, width = features.shape
        expected_h = height * self.scale
        expected_w = width * self.scale
        if image.shape[2] != expected_h or image.shape[3] != expected_w:
            raise ValueError(
                "Input size mismatch for InRefFusion: "
                f"expected ({expected_h}, {expected_w}), got ({image.shape[2]}, {image.shape[3]}). "
                f"Input size must be divisible by {self.scale}."
            )

        image_ref = self.pixel_unshuffle(image)
        return self.fusion_conv(torch.cat([features, image_ref], dim=1))


class GradientLoss(nn.Module):
    """Gradient prediction loss using SmoothL1."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction="mean")

    def forward(self, grad_preds: list[Tensor | None], grad_labels: list[Tensor | None]) -> Tensor:
        device = None
        dtype = None
        for pred in grad_preds:
            if pred is not None:
                device = pred.device
                dtype = pred.dtype
                break
        if device is None or dtype is None:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        valid_count = 0
        for pred, label in zip(grad_preds, grad_labels, strict=False):
            if pred is None or label is None:
                continue
            pred_sigmoid = pred.sigmoid()
            if pred_sigmoid.shape != label.shape:
                label = F.interpolate(
                    label, size=pred_sigmoid.shape[2:], mode="bilinear", align_corners=True
                )
            loss = loss + self.loss_fn(pred_sigmoid, label)
            valid_count += 1
        return loss / max(valid_count, 1)


class IoULoss(nn.Module):
    """Intersection over Union loss."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_sigmoid = pred.sigmoid()
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return (1 - iou).mean()


class IBISNet(nn.Module):
    """IBIS-Net: IS-Net with BiRefNet-inspired enhancements.

    Uses Kornia for SSIM loss computation.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        config: IBISNetConfig | None = None,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or IBISNetConfig()
        if img_size is not None:
            self.config.img_size = img_size

        self.use_outref = self.config.use_outref
        self.use_inref = self.config.use_inref
        self.outref_stages = set(self.config.outref_stages)
        self.inref_stages = set(self.config.inref_stages)

        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        if self.use_outref:
            self.grad_label_generator = GradientLabelGenerator()
            self.outref1 = (
                GradientOutRefBlock(
                    64,
                    hidden_channels=self.config.outref_hidden_channels,
                    dilation_kernel_size=self.config.outref_dilation_kernel,
                )
                if 1 in self.outref_stages
                else None
            )
            self.outref2 = (
                GradientOutRefBlock(
                    64,
                    hidden_channels=self.config.outref_hidden_channels,
                    dilation_kernel_size=self.config.outref_dilation_kernel,
                )
                if 2 in self.outref_stages
                else None
            )
            self.outref3 = (
                GradientOutRefBlock(
                    128,
                    hidden_channels=self.config.outref_hidden_channels,
                    dilation_kernel_size=self.config.outref_dilation_kernel,
                )
                if 3 in self.outref_stages
                else None
            )
        else:
            self.grad_label_generator = None
            self.outref1 = None
            self.outref2 = None
            self.outref3 = None

        if self.use_inref:
            self.inref1 = InRefFusion(64, scale=2) if 1 in self.inref_stages else None
            self.inref2 = InRefFusion(64, scale=4) if 2 in self.inref_stages else None
            self.inref3 = InRefFusion(128, scale=8) if 3 in self.inref_stages else None
        else:
            self.inref1 = None
            self.inref2 = None
            self.inref3 = None

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.grad_loss_fn = GradientLoss()
        self.iou_loss_fn = IoULoss()
        # Kornia SSIM loss (returns 1 - SSIM, so higher is worse)
        self.ssim_loss_fn = KL.SSIMLoss(window_size=11, reduction="mean")
        self.fea_loss = nn.MSELoss(reduction="mean")

    def _check_inref_size(self, image: Tensor) -> None:
        if not self.use_inref:
            return
        if (
            self.training
            and self.config.img_size is not None
            and (image.shape[2] != self.config.img_size or image.shape[3] != self.config.img_size)
        ):
            raise ValueError(
                f"IBISNet expects img_size={self.config.img_size} during training when InRef "
                f"is enabled, got ({image.shape[2]}, {image.shape[3]})."
            )

    def _bce_multi_scale(self, preds: list[Tensor], target: Tensor) -> tuple[Tensor, Tensor]:
        loss0 = torch.tensor(0.0, device=target.device, dtype=target.dtype)
        loss = torch.tensor(0.0, device=target.device, dtype=target.dtype)
        for idx, pred in enumerate(preds):
            if pred.shape[2:] != target.shape[2:]:
                target_resized = F.interpolate(
                    target, size=pred.shape[2:], mode="bilinear", align_corners=True
                )
            else:
                target_resized = target
            bce = self.bce_loss(pred, target_resized)
            loss = loss + bce
            if idx == 0:
                loss0 = bce
        return loss0, loss

    def _feature_sync_loss(self, dfs: list[Tensor], fs: list[Tensor]) -> Tensor:
        loss = torch.tensor(0.0, device=dfs[0].device, dtype=dfs[0].dtype)
        for df, fs_i in zip(dfs, fs, strict=False):
            if df.shape != fs_i.shape:
                if df.shape[1] != fs_i.shape[1]:
                    raise ValueError(f"Feature channels mismatch: df={df.shape}, fs={fs_i.shape}")
                fs_i = F.interpolate(fs_i, size=df.shape[2:], mode="bilinear", align_corners=True)
            loss = loss + self.fea_loss(df, fs_i)
        return loss

    def forward(self, x: Tensor) -> dict[str, list[Tensor] | list[Tensor | None]]:
        self._check_inref_size(x)

        hxin = self.conv_in(x)

        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        if self.inref3 is not None:
            hx3d = self.inref3(hx3d, x)

        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        if self.inref2 is not None:
            hx2d = self.inref2(hx2d, x)

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        if self.inref1 is not None:
            hx1d = self.inref1(hx1d, x)

        grad_gt_fullres = None
        if self.use_outref and self.training and self.grad_label_generator is not None:
            grad_gt_fullres = self.grad_label_generator(x)

        grad_preds: list[Tensor | None] = [None, None, None]
        grad_labels: list[Tensor | None] = [None, None, None]
        grad_attns: list[Tensor | None] = [None, None, None]
        local_logits: list[Tensor | None] = [None, None, None]

        hx3d_attn = hx3d
        if self.outref3 is not None:
            hx3d_attn, local_logit, grad_pred, grad_label, grad_attn = self.outref3(
                hx3d, x, grad_gt_fullres=grad_gt_fullres, training=self.training
            )
            local_logits[2] = local_logit
            grad_preds[2] = grad_pred
            grad_labels[2] = grad_label
            grad_attns[2] = grad_attn

        hx2d_attn = hx2d
        if self.outref2 is not None:
            hx2d_attn, local_logit, grad_pred, grad_label, grad_attn = self.outref2(
                hx2d, x, grad_gt_fullres=grad_gt_fullres, training=self.training
            )
            local_logits[1] = local_logit
            grad_preds[1] = grad_pred
            grad_labels[1] = grad_label
            grad_attns[1] = grad_attn

        hx1d_attn = hx1d
        if self.outref1 is not None:
            hx1d_attn, local_logit, grad_pred, grad_label, grad_attn = self.outref1(
                hx1d, x, grad_gt_fullres=grad_gt_fullres, training=self.training
            )
            local_logits[0] = local_logit
            grad_preds[0] = grad_pred
            grad_labels[0] = grad_label
            grad_attns[0] = grad_attn

        d1 = self.side1(hx1d_attn)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d_attn)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d_attn)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return {
            "ds": [d1, d2, d3, d4, d5, d6],
            "dfs_raw": [hx1d, hx2d, hx3d, hx4d, hx5d, hx6],
            "dfs_attn": [hx1d_attn, hx2d_attn, hx3d_attn, hx4d, hx5d, hx6],
            "grad_preds": grad_preds,
            "grad_labels": grad_labels,
            "grad_attns": grad_attns,
            "local_logits": local_logits,
        }

    def compute_loss(self, args) -> tuple[Tensor, Tensor, dict[str, float]]:
        if len(args) == 2:
            outputs, labels = args
            fs = None
        else:
            outputs, labels, fs = args

        ds = outputs["ds"]
        dfs_raw = outputs["dfs_raw"]

        bce_loss0, bce_loss_total = self._bce_multi_scale(ds, labels)
        total_loss = self.config.lambda_bce * bce_loss_total
        loss_dict: dict[str, float] = {
            "loss_pix_raw": bce_loss_total.detach().item(),
            "loss_pix_w": (self.config.lambda_bce * bce_loss_total).detach().item(),
            "loss_pix_main_raw": bce_loss0.detach().item(),
        }
        if fs is not None:
            fs_loss = self._feature_sync_loss(dfs_raw, fs)
            total_loss = total_loss + self.config.lambda_fs * fs_loss
            loss_dict["loss_fs_raw"] = fs_loss.detach().item()
            loss_dict["loss_fs_w"] = (self.config.lambda_fs * fs_loss).detach().item()

        if any(
            pred is not None and label is not None
            for pred, label in zip(outputs["grad_preds"], outputs["grad_labels"], strict=False)
        ):
            grad_loss = self.grad_loss_fn(outputs["grad_preds"], outputs["grad_labels"])
            total_loss = total_loss + self.config.lambda_grad * grad_loss
            loss_dict["loss_grad_raw"] = grad_loss.detach().item()
            loss_dict["loss_grad_w"] = (self.config.lambda_grad * grad_loss).detach().item()

        if self.config.lambda_iou > 0 and self.config.iou_stages:
            iou_losses: list[Tensor] = []
            for stage in self.config.iou_stages:
                idx = stage - 1
                if idx < 0 or idx >= len(ds):
                    continue
                pred = ds[idx]
                target = (
                    F.interpolate(labels, size=pred.shape[2:], mode="bilinear", align_corners=True)
                    if pred.shape[2:] != labels.shape[2:]
                    else labels
                )
                iou_losses.append(self.iou_loss_fn(pred, target))
            if iou_losses:
                iou_loss_avg: Tensor = sum(iou_losses) / len(iou_losses)  # type: ignore[assignment]
                total_loss = total_loss + self.config.lambda_iou * iou_loss_avg
                loss_dict["loss_region_raw"] = iou_loss_avg.detach().item()
                loss_dict["loss_region_w"] = (self.config.lambda_iou * iou_loss_avg).detach().item()

        if self.config.lambda_ssim > 0 and self.config.ssim_stages:
            ssim_losses = []
            for stage in self.config.ssim_stages:
                idx = stage - 1
                if idx < 0 or idx >= len(ds):
                    continue
                pred = ds[idx]
                target = (
                    F.interpolate(labels, size=pred.shape[2:], mode="bilinear", align_corners=True)
                    if pred.shape[2:] != labels.shape[2:]
                    else labels
                )
                # Kornia SSIM expects inputs in [0, 1], apply sigmoid to pred
                ssim_losses.append(self.ssim_loss_fn(pred.sigmoid(), target))
            if ssim_losses:
                ssim_loss_avg: Tensor = sum(ssim_losses) / len(ssim_losses)  # type: ignore[assignment]
                total_loss = total_loss + self.config.lambda_ssim * ssim_loss_avg
                loss_dict["loss_boundary_raw"] = ssim_loss_avg.detach().item()
                loss_dict["loss_boundary_w"] = (
                    (self.config.lambda_ssim * ssim_loss_avg).detach().item()
                )

        loss_dict["loss_total"] = total_loss.detach().item()
        return self.config.lambda_bce * bce_loss0, total_loss, loss_dict
