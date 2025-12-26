# Codes are borrowed from
# https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net_refactor.py

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.types import Tensor

__all__ = ["U2Net", "U2NetFull", "U2NetFull2", "U2NetLite", "U2NetLite2"]

bce_loss = nn.BCEWithLogitsLoss(reduction="mean")


def _upsample_like(x, size) -> Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _size_map(x, height: int):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class RebnConv(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, dilate: int = 1) -> None:
        super().__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated: bool = False) -> None:
        super().__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f"rebnconv{height}")(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(self.downsample(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f"rebnconv{height}d")(torch.cat((x2, x1), 1))
                return (
                    _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
                )
            return getattr(self, f"rebnconv{height}")(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False) -> None:
        self.add_module("rebnconvin", RebnConv(in_ch, out_ch))
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module("rebnconv1", RebnConv(out_ch, mid_ch))
        self.add_module("rebnconv1d", RebnConv(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 2 ** (i - 1) if dilated else 1
            self.add_module(f"rebnconv{i}", RebnConv(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f"rebnconv{i}d", RebnConv(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 ** (height - 1) if dilated else 2
        self.add_module(f"rebnconv{height}", RebnConv(mid_ch, mid_ch, dilate=dilate))


class U2Net(nn.Module):
    def __init__(self, cfgs, out_ch) -> None:
        super().__init__()
        self.out_ch = out_ch
        self._make_layers(cfgs)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f"stage{height}")(x)
                x2 = unet(self.downsample(x1), height + 1)
                x = getattr(self, f"stage{height}d")(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            x = getattr(self, f"stage{height}")(x)
            side(x, height)
            return _upsample_like(x, sizes[height - 1])

        def side(x, h) -> None:
            # side output saliency map (before sigmoid)
            x = getattr(self, f"side{h}")(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = self.outconv(x)
            maps.insert(0, x)
            # return [torch.sigmoid(x) for x in maps]
            return list(maps)

        unet(x)
        maps = fuse()
        return maps

    @staticmethod
    def compute_loss(args):
        preds, labels_v = args
        d0, d1, d2, d3, d4, d5, d6 = preds
        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss0, loss

    def _make_layers(self, cfgs) -> None:
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f"side{v[0][-1]}", nn.Conv2d(v[2], self.out_ch, 3, padding=1))
        # build fuse layer
        self.add_module("outconv", nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1))


def U2NetFull() -> U2Net:
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (7, 3, 32, 64), -1],
        "stage2": ["En_2", (6, 64, 32, 128), -1],
        "stage3": ["En_3", (5, 128, 64, 256), -1],
        "stage4": ["En_4", (4, 256, 128, 512), -1],
        "stage5": ["En_5", (4, 512, 256, 512, True), -1],
        "stage6": ["En_6", (4, 512, 256, 512, True), 512],
        "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (4, 1024, 128, 256), 256],
        "stage3d": ["De_3", (5, 512, 64, 128), 128],
        "stage2d": ["De_2", (6, 256, 32, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2Net(cfgs=full, out_ch=1)


def U2NetFull2() -> U2Net:
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (8, 3, 32, 64), -1],
        "stage2": ["En_2", (7, 64, 32, 128), -1],
        "stage3": ["En_3", (6, 128, 64, 256), -1],
        "stage4": ["En_4", (5, 256, 128, 512), -1],
        "stage5": ["En_5", (5, 512, 256, 512, True), -1],
        "stage6": ["En_6", (5, 512, 256, 512, True), 512],
        "stage5d": ["De_5", (5, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (5, 1024, 128, 256), 256],
        "stage3d": ["De_3", (6, 512, 64, 128), 128],
        "stage2d": ["De_2", (7, 256, 32, 64), 64],
        "stage1d": ["De_1", (8, 128, 16, 64), 64],
    }
    return U2Net(cfgs=full, out_ch=1)


def U2NetLite() -> U2Net:
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (7, 3, 16, 64), -1],
        "stage2": ["En_2", (6, 64, 16, 64), -1],
        "stage3": ["En_3", (5, 64, 16, 64), -1],
        "stage4": ["En_4", (4, 64, 16, 64), -1],
        "stage5": ["En_5", (4, 64, 16, 64, True), -1],
        "stage6": ["En_6", (4, 64, 16, 64, True), 64],
        "stage5d": ["De_5", (4, 128, 16, 64, True), 64],
        "stage4d": ["De_4", (4, 128, 16, 64), 64],
        "stage3d": ["De_3", (5, 128, 16, 64), 64],
        "stage2d": ["De_2", (6, 128, 16, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2Net(cfgs=lite, out_ch=1)


def U2NetLite2() -> U2Net:
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (8, 3, 16, 64), -1],
        "stage2": ["En_2", (7, 64, 16, 64), -1],
        "stage3": ["En_3", (6, 64, 16, 64), -1],
        "stage4": ["En_4", (5, 64, 16, 64), -1],
        "stage5": ["En_5", (5, 64, 16, 64, True), -1],
        "stage6": ["En_6", (5, 64, 16, 64, True), 64],
        "stage5d": ["De_5", (5, 128, 16, 64, True), 64],
        "stage4d": ["De_4", (5, 128, 16, 64), 64],
        "stage3d": ["De_3", (6, 128, 16, 64), 64],
        "stage2d": ["De_2", (7, 128, 16, 64), 64],
        "stage1d": ["De_1", (8, 128, 16, 64), 64],
    }
    return U2Net(cfgs=lite, out_ch=1)
