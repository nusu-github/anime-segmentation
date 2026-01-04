"""ISNet architecture for Dichotomous Image Segmentation (DIS).

This module implements the ISNet model as described in:
    "Highly Accurate Dichotomous Image Segmentation" (ECCV 2022)
    https://arxiv.org/abs/2203.03041

The architecture uses a U-Net style encoder-decoder with Residual U-blocks (RSU)
at multiple scales, combined with intermediate supervision for accurate boundary
detection and object segmentation.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation

_BCE_WITH_LOGITS_LOSS = nn.BCEWithLogitsLoss(reduction="mean")
_FEA_LOSS = nn.MSELoss(reduction="mean")
_KL_LOSS = nn.KLDivLoss(reduction="mean")
_L1_LOSS = nn.L1Loss(reduction="mean")
_SMOOTH_L1_LOSS = nn.SmoothL1Loss(reduction="mean")


def multi_loss_fusion(
    preds: list[Tensor],
    target: Tensor,
) -> tuple[float, float]:
    """Compute multi-scale BCE loss with deep supervision.

    Aggregates binary cross-entropy loss across all intermediate predictions,
    automatically handling resolution mismatches via bilinear interpolation.

    Args:
        preds: List of prediction tensors (logits) from each decoder stage.
        target: Ground truth segmentation mask.

    Returns:
        A tuple of (first_stage_loss, total_loss) where total_loss is the
        sum of BCE losses across all prediction scales.
    """
    loss0 = 0.0
    loss = 0.0

    for i in range(len(preds)):
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            tmp_target = F.interpolate(
                target, size=preds[i].size()[2:], mode="bilinear", align_corners=True
            )
            loss += _BCE_WITH_LOGITS_LOSS(preds[i], tmp_target)
        else:
            loss += _BCE_WITH_LOGITS_LOSS(preds[i], target)
        if i == 0:
            loss0 = loss
    return loss0, loss


def multi_loss_fusion_kl(
    preds: list[Tensor],
    target: Tensor,
    dfs: list[Tensor],
    fs: list[Tensor],
    mode: str = "MSE",
) -> tuple[float, float]:
    """Compute multi-scale loss with feature distillation.

    Combines BCE loss for predictions with feature-level distillation loss
    between student (dfs) and teacher (fs) feature maps. This enables
    knowledge transfer from the GT encoder to the main segmentation network.

    Args:
        preds: List of prediction tensors (logits) from each decoder stage.
        target: Ground truth segmentation mask.
        dfs: Feature maps from the main network (student).
        fs: Feature maps from the GT encoder (teacher).
        mode: Feature distillation loss type. One of:
            - "MSE": Mean squared error (default)
            - "KL": KL divergence on softmax distributions
            - "MAE": Mean absolute error (L1)
            - "SmoothL1": Smooth L1 loss

    Returns:
        A tuple of (first_stage_loss, total_loss) including both
        prediction and feature distillation losses.
    """
    loss0 = 0.0
    loss = 0.0

    for i in range(len(preds)):
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            tmp_target = F.interpolate(
                target, size=preds[i].size()[2:], mode="bilinear", align_corners=True
            )
            loss += _BCE_WITH_LOGITS_LOSS(preds[i], tmp_target)
        else:
            loss += _BCE_WITH_LOGITS_LOSS(preds[i], target)
        if i == 0:
            loss0 = loss

    for i in range(len(dfs)):
        match mode:
            case "MSE":
                loss += _FEA_LOSS(dfs[i], fs[i])
            case "KL":
                loss += _KL_LOSS(F.log_softmax(dfs[i], dim=1), F.softmax(fs[i], dim=1))
            case "MAE":
                loss += _L1_LOSS(dfs[i], fs[i])
            case "SmoothL1":
                loss += _SMOOTH_L1_LOSS(dfs[i], fs[i])

    return loss0, loss


class REBNConv(Conv2dNormActivation):
    """Residual-Enhanced Batch-Normalized Convolution block.

    A basic building block consisting of Conv2d -> BatchNorm2d -> ReLU.
    Supports dilated convolutions for multi-scale feature extraction.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        dirate: Dilation rate for the convolution kernel.
        stride: Stride for the convolution.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1, stride: int = 1) -> None:
        super().__init__(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=stride,
            padding=dirate,
            dilation=dirate,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            inplace=True,
        )


def _upsample_like(
    src: Tensor,
    tar: Tensor,
) -> Tensor:
    """Upsample source tensor to match target tensor's spatial dimensions."""
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=True)


class RSU7(nn.Module):
    """Residual U-block with 7 encoder-decoder levels.

    A nested U-structure block that captures multi-scale contextual information
    through progressive downsampling and upsampling with skip connections.
    The deepest level uses dilated convolution instead of pooling.

    Args:
        in_ch: Number of input channels.
        mid_ch: Number of channels in intermediate layers.
        out_ch: Number of output channels.
        _img_size: Unused parameter (kept for API compatibility).
    """

    def __init__(
        self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3, _img_size: int = 512
    ) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        # Encoder path
        self.rebnconvin = REBNConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNConv(mid_ch, mid_ch, dirate=1)

        # Bottleneck with dilation to expand receptive field
        self.rebnconv7 = REBNConv(mid_ch, mid_ch, dirate=2)

        # Decoder path
        self.rebnconv6d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: Tensor) -> Tensor:
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


class RSU6(nn.Module):
    """Residual U-block with 6 encoder-decoder levels.

    Similar to RSU7 but with one fewer pooling stage. Used in shallower
    parts of the network where spatial resolution is already reduced.

    Args:
        in_ch: Number of input channels.
        mid_ch: Number of channels in intermediate layers.
        out_ch: Number of output channels.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = REBNConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: Tensor) -> Tensor:
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


class RSU5(nn.Module):
    """Residual U-block with 5 encoder-decoder levels.

    Args:
        in_ch: Number of input channels.
        mid_ch: Number of channels in intermediate layers.
        out_ch: Number of output channels.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = REBNConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: Tensor) -> Tensor:
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


class RSU4(nn.Module):
    """Residual U-block with 4 encoder-decoder levels.

    Args:
        in_ch: Number of input channels.
        mid_ch: Number of channels in intermediate layers.
        out_ch: Number of output channels.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = REBNConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNConv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNConv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNConv(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNConv(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNConv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: Tensor) -> Tensor:
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


class RSU4F(nn.Module):
    """Residual U-block with 4 levels using only dilated convolutions.

    Unlike RSU4-RSU7 which use pooling for downsampling, RSU4F maintains
    spatial resolution throughout using progressively increasing dilation
    rates (1, 2, 4, 8). This preserves fine-grained spatial information
    at the deepest encoder stages.

    Args:
        in_ch: Number of input channels.
        mid_ch: Number of channels in intermediate layers.
        out_ch: Number of output channels.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        super().__init__()

        self.rebnconvin = REBNConv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNConv(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNConv(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNConv(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNConv(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNConv(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNConv(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNConv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: Tensor) -> Tensor:
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


class MyREBNConv(Conv2dNormActivation):
    """Configurable Conv-BN-ReLU block with full parameter control.

    Extended version of REBNConv that exposes all convolution parameters
    for flexible network construction in the GT encoder.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.
        dilation: Dilation rate of the convolution.
        groups: Number of blocked connections for grouped convolution.
    """

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
        super().__init__(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU,
            inplace=True,
        )


class ISNetGTEncoder(nn.Module):
    """Ground Truth Encoder for ISNet feature distillation.

    An encoder-only network that processes ground truth masks to extract
    hierarchical feature representations. These features serve as targets
    for distillation training, guiding the main ISNetDIS network to learn
    discriminative representations aligned with ground truth structure.

    The architecture mirrors the encoder path of ISNetDIS but operates
    on single-channel mask inputs.

    Args:
        in_ch: Number of input channels (typically 1 for binary masks).
        out_ch: Number of output channels for side predictions.
        ckpt_path: Optional path to load pretrained weights.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, ckpt_path: str | None = None) -> None:
        super().__init__()

        self.conv_in = MyREBNConv(in_ch, 16, 3, stride=2, padding=1)

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

        # Side output convolutions for multi-scale predictions
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        if ckpt_path is not None:
            self._load_from_checkpoint(ckpt_path)

    def _load_from_checkpoint(self, ckpt_path: str) -> None:
        """Load weights from a PyTorch Lightning checkpoint.

        Handles the Lightning checkpoint format where model weights are stored
        under 'state_dict' with a 'model.' prefix added by LightningModule.

        Args:
            ckpt_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value

        self.load_state_dict(new_state_dict)

    def compute_loss(
        self,
        preds: list[Tensor],
        targets: Tensor,
    ) -> tuple[float, float]:
        """Compute multi-scale supervision loss for encoder training."""
        return multi_loss_fusion(preds, targets)

    def forward(
        self,
        x: Tensor,
    ) -> tuple[
        list[Tensor],
        list[Tensor],
    ]:
        """Forward pass through the GT encoder.

        Returns:
            A tuple containing:
                - List of side output predictions at each scale
                - List of intermediate feature maps for distillation
        """
        hx = x

        hxin = self.conv_in(hx)

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

        # Generate side outputs at each scale, upsampled to input resolution
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

        return [d1, d2, d3, d4, d5, d6], [hx1, hx2, hx3, hx4, hx5, hx6]


class ISNetDIS(nn.Module):
    """ISNet for Dichotomous Image Segmentation.

    A U-Net style encoder-decoder network using Residual U-blocks (RSU)
    for high-accuracy binary segmentation. Features deep supervision through
    multi-scale side outputs and optional feature distillation from the
    GT encoder for improved training.

    The network progressively downsamples input through 6 encoder stages,
    then upsamples through 5 decoder stages with skip connections. Each
    stage produces a side output for intermediate supervision.

    Args:
        in_ch: Number of input channels (typically 3 for RGB images).
        out_ch: Number of output channels (typically 1 for binary masks).
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__()

        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Encoder stages with progressive channel expansion
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

        # Decoder stages with skip connections from encoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # Side output convolutions for multi-scale predictions
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def compute_loss_kl(
        self,
        preds: list[Tensor],
        targets: Tensor,
        dfs: list[Tensor],
        fs: list[Tensor],
        mode: str = "MSE",
    ) -> tuple[float, float]:
        """Compute loss with feature distillation from GT encoder."""
        return multi_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)

    def compute_loss(
        self,
        preds: list[Tensor],
        targets: Tensor,
    ) -> tuple[float, float]:
        """Compute multi-scale supervision loss without distillation."""
        return multi_loss_fusion(preds, targets)

    def forward(
        self,
        x: Tensor,
    ) -> tuple[
        list[Tensor],
        list[Tensor],
    ]:
        """Forward pass through the encoder-decoder network.

        Returns:
            A tuple containing:
                - List of side output predictions at each decoder scale
                - List of decoder feature maps for optional distillation
        """
        hx = x

        hxin = self.conv_in(hx)

        # Encoder path
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

        # Decoder path with skip connections
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # Generate side outputs at each scale, upsampled to input resolution
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

        return [d1, d2, d3, d4, d5, d6], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
