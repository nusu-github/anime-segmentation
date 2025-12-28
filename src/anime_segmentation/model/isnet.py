# Codes are borrowed from
# https://github.com/xuebinqin/DIS/blob/main/IS-Net/models/isnet.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anime_segmentation.loss import HybridLoss, get_hybrid_loss

# Shared hybrid loss instance (initialized lazily per model)
_hybrid_loss: HybridLoss | None = None


def _get_hybrid_loss() -> HybridLoss:
    """Get or create the shared HybridLoss instance."""
    global _hybrid_loss
    if _hybrid_loss is None:
        _hybrid_loss = get_hybrid_loss()
    return _hybrid_loss


def multi_loss_fusion(preds: list[Tensor], target: Tensor) -> tuple[Tensor, Tensor]:
    """Compute hybrid loss across multiple prediction scales.

    Args:
        preds: List of predictions at different scales (logits)
        target: Ground truth mask

    Returns:
        Tuple of (loss at first scale, total loss across all scales)
    """
    hybrid_loss = _get_hybrid_loss()
    scale_losses = [hybrid_loss(pred, target) for pred in preds]
    loss0 = scale_losses[0]
    loss = torch.stack(scale_losses).mean()
    return loss0, loss


# Feature distillation losses
fea_loss = nn.MSELoss(reduction="mean")
kl_loss = nn.KLDivLoss(reduction="batchmean")
l1_loss = nn.L1Loss(reduction="mean")
smooth_l1_loss = nn.SmoothL1Loss(reduction="mean")


class EMALossBalancer:
    def __init__(
        self,
        target_ratio: float = 0.1,
        momentum: float = 0.99,
        eps: float = 1e-8,
        w_min: float = 0.0,
        w_max: float = 1e4,
    ) -> None:
        self.target_ratio = target_ratio
        self.momentum = momentum
        self.eps = eps
        self.w_min = w_min
        self.w_max = w_max
        self.ema_pixel: float | None = None
        self.ema_feature: float | None = None

    def get_feature_weight(self, pixel_loss: Tensor, feature_loss: Tensor) -> Tensor:
        pixel_val = float(pixel_loss.detach())
        feature_val = float(feature_loss.detach())

        feature_val = max(feature_val, self.eps)

        if self.ema_pixel is None or self.ema_feature is None:
            self.ema_pixel = pixel_val
            self.ema_feature = feature_val
        else:
            self.ema_pixel = self.momentum * self.ema_pixel + (1 - self.momentum) * pixel_val
            self.ema_feature = self.momentum * self.ema_feature + (1 - self.momentum) * feature_val

        ema_pixel = max(self.ema_pixel, self.eps)
        ema_feature = max(self.ema_feature, self.eps)

        w = self.target_ratio * ema_pixel / ema_feature
        w = min(max(w, self.w_min), self.w_max)

        return pixel_loss.new_tensor(w)


# Shared EMA balancer instance for isnet_is
_ema_balancer: EMALossBalancer | None = None


def _get_ema_balancer() -> EMALossBalancer:
    """Get or create the shared EMA balancer instance."""
    global _ema_balancer
    if _ema_balancer is None:
        _ema_balancer = EMALossBalancer(target_ratio=0.1, momentum=0.99)
    return _ema_balancer


def multi_loss_fusion_kl(
    preds: list[Tensor],
    target: Tensor,
    dfs: list[Tensor],
    fs: list[Tensor],
    mode: str = "MSE",
) -> tuple[Tensor, Tensor]:
    """Compute hybrid loss with feature distillation (KL/MSE).

    Uses EMA-based dynamic balancing to maintain feature loss contribution.

    Args:
        preds: List of predictions at different scales (logits)
        target: Ground truth mask
        dfs: Decoder features from main model
        fs: Features from GT encoder (distillation targets)
        mode: Feature loss type ("MSE", "KL", "MAE", "SmoothL1")

    Returns:
        Tuple of (loss at first scale, total loss including feature loss)
    """
    loss0, pixel_loss = multi_loss_fusion(preds, target)

    fea_terms = []
    for df, fs_i in zip(dfs, fs, strict=True):
        match mode:
            case "MSE":
                fea_terms.append(fea_loss(df, fs_i))
            case "KL":
                fea_terms.append(kl_loss(F.log_softmax(df, dim=1), F.softmax(fs_i, dim=1)))
            case "MAE":
                fea_terms.append(l1_loss(df, fs_i))
            case "SmoothL1":
                fea_terms.append(smooth_l1_loss(df, fs_i))

    feature_loss = torch.stack(fea_terms).mean()

    balancer = _get_ema_balancer()
    feature_weight = balancer.get_feature_weight(pixel_loss, feature_loss)

    total_loss = pixel_loss + feature_weight * feature_loss
    return loss0, total_loss


class RebnConv(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1, stride: int = 1) -> None:
        super().__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate, stride=stride
        )
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        return self.relu_s1(self.bn_s1(self.conv_s1(hx)))


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar) -> Tensor:
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


# RSU-7
class RSU7(nn.Module):
    def __init__(
        self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3, img_size: int = 512
    ) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = RebnConv(in_ch, out_ch, dirate=1)  # 1 -> 1/2

        self.rebnconv1 = RebnConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = RebnConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = RebnConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RebnConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        _b, _c, _h, _w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-6
class RSU6(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = RebnConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RebnConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = RebnConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = RebnConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RebnConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-5
class RSU5(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = RebnConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RebnConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = RebnConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = RebnConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RebnConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4
class RSU4(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = RebnConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RebnConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = RebnConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = RebnConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = RebnConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = RebnConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = RebnConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4F
class RSU4F(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = RebnConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = RebnConv(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = RebnConv(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = RebnConv(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = RebnConv(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = RebnConv(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = RebnConv(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = RebnConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class MyRebnConv(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.rl(self.bn(self.conv(x)))


class ISNetGTEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1) -> None:
        super().__init__()

        # nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.conv_in = MyRebnConv(in_ch, 16, 3, stride=2, padding=1)

        self.stage1 = RSU7(16, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 32, 256)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 64, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 64, 512)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    @staticmethod
    def compute_loss(args):
        preds, targets = args
        return multi_loss_fusion(preds, targets)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        # side output
        d1 = self.side1(hx1)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        # return [torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)], [hx1, hx2, hx3, hx4, hx5, hx6]
        return [d1, d2, d3, d4, d5, d6], [hx1, hx2, hx3, hx4, hx5, hx6]


class ISNetDIS(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__()

        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

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

        # decoder
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

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    @staticmethod
    def compute_loss_kl(preds, targets, dfs, fs, mode: str = "MSE"):
        return multi_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)

    @staticmethod
    def compute_loss(args):
        if len(args) == 3:
            ds, dfs, labels = args
            return multi_loss_fusion(ds, labels)
        ds, dfs, labels, fs = args
        return multi_loss_fusion_kl(ds, labels, dfs, fs, mode="MSE")

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)
        hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        # return [torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
        return [d1, d2, d3, d4, d5, d6], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
