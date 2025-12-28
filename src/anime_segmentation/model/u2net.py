# Codes are borrowed from
# https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net_refactor.py
# Refactored for TorchScript compatibility

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..loss import HybridLoss, get_hybrid_loss

__all__ = ["U2Net", "U2NetFull", "U2NetFull2", "U2NetLite", "U2NetLite2"]

# Shared hybrid loss instance
_hybrid_loss: HybridLoss | None = None


def _get_hybrid_loss() -> HybridLoss:
    """Get or create the shared HybridLoss instance."""
    global _hybrid_loss
    if _hybrid_loss is None:
        _hybrid_loss = get_hybrid_loss()
    return _hybrid_loss


def _upsample_like(x: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _compute_sizes(h: int, w: int, height: int) -> list[tuple[int, int]]:
    """Compute sizes for each height level. Index 0 is unused placeholder."""
    sizes: list[tuple[int, int]] = [(0, 0)]  # placeholder for index 0
    curr_h, curr_w = h, w
    for _ in range(1, height):
        sizes.append((curr_h, curr_w))
        curr_h = math.ceil(curr_h / 2)
        curr_w = math.ceil(curr_w / 2)
    return sizes


class RebnConv(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, dilate: int = 1) -> None:
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    """RSU block with TorchScript-compatible forward pass."""

    def __init__(
        self, name: str, height: int, in_ch: int, mid_ch: int, out_ch: int, dilated: bool = False
    ) -> None:
        super().__init__()
        self.name = name
        self.height = height
        self.dilated = dilated

        # Input conv
        self.rebnconvin = RebnConv(in_ch, out_ch)
        self.downsample = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Encoder layers (indices 0 to height-1)
        self.enc_convs = nn.ModuleList()
        self.enc_convs.append(RebnConv(out_ch, mid_ch))  # height=1

        for i in range(2, height):
            dilate = 2 ** (i - 1) if dilated else 1
            self.enc_convs.append(RebnConv(mid_ch, mid_ch, dilate=dilate))

        # Bottom conv
        dilate = 2 ** (height - 1) if dilated else 2
        self.enc_convs.append(RebnConv(mid_ch, mid_ch, dilate=dilate))

        # Decoder layers - stored in reverse order (height-1 to 1) for forward iteration
        # dec_convs[0] corresponds to height-1, dec_convs[-1] corresponds to height=1
        self.dec_convs = nn.ModuleList()
        for i in range(height - 1, 1, -1):  # height-1, height-2, ..., 2
            dilate = 2 ** (i - 1) if dilated else 1
            self.dec_convs.append(RebnConv(mid_ch * 2, mid_ch, dilate=dilate))
        self.dec_convs.append(RebnConv(mid_ch * 2, out_ch))  # height=1

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        sizes = _compute_sizes(h, w, self.height)

        hx = self.rebnconvin(x)
        residual = hx

        # Encoder: collect features at each level
        features: list[Tensor] = []
        num_enc = self.height - 1
        for i, conv in enumerate(self.enc_convs):
            if i < num_enc:
                hx = conv(hx)
                features.append(hx)
                if not self.dilated and i < num_enc - 1:
                    hx = self.downsample(hx)
            else:
                # Bottom conv
                hx = conv(hx)

        # Decoder: fuse features from bottom to top
        # dec_convs[i] corresponds to level (height-1-i) in the original structure
        # features[j] corresponds to level (j+1) in the original structure
        # We need to match dec_convs[i] with features[height-2-i]
        num_features = len(features)
        for i, conv in enumerate(self.dec_convs):
            feat_idx = num_features - 1 - i  # Start from last feature, go backwards
            hx = torch.cat([hx, features[feat_idx]], dim=1)
            hx = conv(hx)
            # Upsample if not dilated and not at the final output level
            if not self.dilated and i < num_features - 1:
                hx = _upsample_like(hx, sizes[feat_idx])

        return hx + residual


class U2Net(nn.Module):
    """U2Net with TorchScript-compatible forward pass."""

    def __init__(
        self,
        enc_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]],
        dec_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]],
        side_channels: list[int],
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.num_stages = len(enc_cfgs)

        self.downsample = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Encoder stages
        self.enc_stages = nn.ModuleList()
        for i, cfg in enumerate(enc_cfgs):
            height, in_ch, mid_ch, out_ch_stage = cfg[:4]
            dilated = cfg[4] if len(cfg) > 4 else False
            self.enc_stages.append(RSU(f"En_{i + 1}", height, in_ch, mid_ch, out_ch_stage, dilated))

        # Decoder stages
        self.dec_stages = nn.ModuleList()
        for i, cfg in enumerate(dec_cfgs):
            height, in_ch, mid_ch, out_ch_stage = cfg[:4]
            dilated = cfg[4] if len(cfg) > 4 else False
            self.dec_stages.append(
                RSU(f"De_{self.num_stages - i}", height, in_ch, mid_ch, out_ch_stage, dilated)
            )

        # Side output convolutions - split into bottom and decoder sides
        # First one is for the bottom encoder stage, rest are for decoder stages
        self.side_bottom = nn.Conv2d(side_channels[0], out_ch, 3, padding=1)
        self.side_dec = nn.ModuleList()
        for ch in side_channels[1:]:
            self.side_dec.append(nn.Conv2d(ch, out_ch, 3, padding=1))

        # Fuse layer
        self.outconv = nn.Conv2d(len(side_channels) * out_ch, out_ch, 1)

    def forward(self, x: Tensor) -> list[Tensor]:
        _, _, h, w = x.shape
        original_size = (h, w)

        # Compute sizes for each level
        sizes: list[tuple[int, int]] = [original_size]
        curr_h, curr_w = h, w
        for _ in range(self.num_stages - 1):
            curr_h = math.ceil(curr_h / 2)
            curr_w = math.ceil(curr_w / 2)
            sizes.append((curr_h, curr_w))

        # Encoder pass
        enc_features: list[Tensor] = []
        hx = x
        num_enc = len(self.enc_stages)
        for i, stage in enumerate(self.enc_stages):
            hx = stage(hx)
            enc_features.append(hx)
            if i < num_enc - 1:
                hx = self.downsample(hx)

        # Decoder pass with side outputs
        side_outputs: list[Tensor] = []

        # Bottom stage side output
        bottom_feat = enc_features[-1]
        hx = bottom_feat
        side_out = self.side_bottom(bottom_feat)
        side_out = _upsample_like(side_out, original_size)
        side_outputs.append(side_out)

        # Process decoder stages with their side outputs
        num_features = len(enc_features)
        for i, (stage, side_conv) in enumerate(zip(self.dec_stages, self.side_dec, strict=True)):
            # enc_idx goes from num_stages-2 down to 0
            enc_idx = num_features - 2 - i
            hx = _upsample_like(hx, sizes[enc_idx])
            hx = torch.cat([hx, enc_features[enc_idx]], dim=1)
            hx = stage(hx)

            # Side output
            side_out = side_conv(hx)
            side_out = _upsample_like(side_out, original_size)
            side_outputs.append(side_out)

        # Fuse all side outputs
        side_outputs.reverse()  # Now ordered from stage1d to stage6
        fused = torch.cat(side_outputs, dim=1)
        d0 = self.outconv(fused)

        # Return [d0, d1, d2, d3, d4, d5, d6]
        return [d0, *side_outputs]

    @staticmethod
    def compute_loss(
        args: tuple[list[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute hybrid loss across all decoder outputs.

        Args:
            args: Tuple of (predictions list, ground truth labels)

        Returns:
            Tuple of (loss at d0, total loss across all scales)
        """
        preds, labels_v = args
        hybrid_loss = _get_hybrid_loss()

        loss = torch.tensor(0.0, device=preds[0].device, dtype=preds[0].dtype)
        loss0 = torch.tensor(0.0, device=preds[0].device, dtype=preds[0].dtype)

        for i, pred in enumerate(preds):
            scale_loss = hybrid_loss(pred, labels_v)
            loss = loss + scale_loss
            if i == 0:
                loss0 = scale_loss

        return loss0, loss


def U2NetFull() -> U2Net:
    # Encoder configs: (height, in_ch, mid_ch, out_ch, [dilated])
    enc_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (7, 3, 32, 64),
        (6, 64, 32, 128),
        (5, 128, 64, 256),
        (4, 256, 128, 512),
        (4, 512, 256, 512, True),
        (4, 512, 256, 512, True),
    ]
    # Decoder configs: (height, in_ch, mid_ch, out_ch, [dilated])
    dec_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (4, 1024, 256, 512, True),
        (4, 1024, 128, 256),
        (5, 512, 64, 128),
        (6, 256, 32, 64),
        (7, 128, 16, 64),
    ]
    # Side channels: stage6, stage5d, stage4d, stage3d, stage2d, stage1d
    side_channels = [512, 512, 256, 128, 64, 64]
    return U2Net(enc_cfgs, dec_cfgs, side_channels, out_ch=1)


def U2NetFull2() -> U2Net:
    enc_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (8, 3, 32, 64),
        (7, 64, 32, 128),
        (6, 128, 64, 256),
        (5, 256, 128, 512),
        (5, 512, 256, 512, True),
        (5, 512, 256, 512, True),
    ]
    dec_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (5, 1024, 256, 512, True),
        (5, 1024, 128, 256),
        (6, 512, 64, 128),
        (7, 256, 32, 64),
        (8, 128, 16, 64),
    ]
    side_channels = [512, 512, 256, 128, 64, 64]
    return U2Net(enc_cfgs, dec_cfgs, side_channels, out_ch=1)


def U2NetLite() -> U2Net:
    enc_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (7, 3, 16, 64),
        (6, 64, 16, 64),
        (5, 64, 16, 64),
        (4, 64, 16, 64),
        (4, 64, 16, 64, True),
        (4, 64, 16, 64, True),
    ]
    dec_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (4, 128, 16, 64, True),
        (4, 128, 16, 64),
        (5, 128, 16, 64),
        (6, 128, 16, 64),
        (7, 128, 16, 64),
    ]
    side_channels = [64, 64, 64, 64, 64, 64]
    return U2Net(enc_cfgs, dec_cfgs, side_channels, out_ch=1)


def U2NetLite2() -> U2Net:
    enc_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (8, 3, 16, 64),
        (7, 64, 16, 64),
        (6, 64, 16, 64),
        (5, 64, 16, 64),
        (5, 64, 16, 64, True),
        (5, 64, 16, 64, True),
    ]
    dec_cfgs: list[tuple[int, int, int, int] | tuple[int, int, int, int, bool]] = [
        (5, 128, 16, 64, True),
        (5, 128, 16, 64),
        (6, 128, 16, 64),
        (7, 128, 16, 64),
        (8, 128, 16, 64),
    ]
    side_channels = [64, 64, 64, 64, 64, 64]
    return U2Net(enc_cfgs, dec_cfgs, side_channels, out_ch=1)
