# Codes are borrowed from https://github.com/plemeri/InSPyReNet

import math
from operator import xor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.morphology import dilation, erosion
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_
from torch import nn
from torch.nn.modules.activation import GELU
from torch.nn.modules.container import Sequential
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
from torch.types import Tensor
from torch.utils import checkpoint, model_zoo


class ImagePyramid:
    def __init__(self, ksize: int = 7, sigma: int = 1, channels: int = 1) -> None:
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels

        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)

    def expand(self, x: Tensor) -> Tensor:
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (self.ksize // 2,) * 4, mode="reflect")
        return F.conv2d(x, self.kernel * 4, groups=self.channels)

    def reduce(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.ksize // 2,) * 4, mode="reflect")
        x = F.conv2d(x, self.kernel, groups=self.channels)
        return x[:, :, ::2, ::2]

    def deconstruct(self, x: Tensor) -> tuple[Tensor, Tensor]:
        reduced_x = self.reduce(x)
        expanded_reduced_x = self.expand(reduced_x)

        if x.shape != expanded_reduced_x.shape:
            expanded_reduced_x = F.interpolate(expanded_reduced_x, x.shape[-2:])

        laplacian_x = x - expanded_reduced_x
        return reduced_x, laplacian_x

    def reconstruct(self, x: Tensor, laplacian_x: Tensor) -> Tensor:
        expanded_x = self.expand(x)
        if laplacian_x.shape != expanded_x:
            laplacian_x = F.interpolate(
                laplacian_x, expanded_x.shape[-2:], mode="bilinear", align_corners=True
            )
        return expanded_x + laplacian_x

    def _apply(self, fn) -> None:
        self.kernel = fn(self.kernel)


class Transition:
    def __init__(self, k: int = 3) -> None:
        self.kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).float()

    def __call__(self, x: Tensor) -> Tensor:
        x = torch.sigmoid(x)
        dx = dilation(x, self.kernel)
        ex = erosion(x, self.kernel)

        return ((dx - ex) > 0.5).float()

    def _apply(self, fn) -> None:
        self.kernel = fn(self.kernel)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: str = "same",
        bias: bool = False,
        bn: bool = True,
        relu: bool = False,
    ) -> None:
        super().__init__()
        if "__iter__" not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if "__iter__" not in dir(stride):
            stride = (stride, stride)
        if "__iter__" not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == "same":
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == "valid":
            width_pad_size = 0
            height_pad_size = 0
        elif "__iter__" in dir(padding):
            width_pad_size = padding[0] * 2
            height_pad_size = padding[1] * 2
        else:
            width_pad_size = padding * 2
            height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias
        )

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, mode: str = "hw", stage_size=None) -> None:
        super().__init__()

        self.mode = mode

        self.query_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.stage_size = stage_size

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if "h" in self.mode:
            axis *= height
        if "w" in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        projected_query = (projected_query.shape[2] ** -0.5) * projected_query
        projected_key = (projected_key.shape[1] ** -0.5) * projected_key
        # ↑note: different from original project, to avoid overflow and variance explosion
        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        return self.gamma * out + x


class PAA_d(nn.Module):
    def __init__(
        self, in_channel, out_channel: int = 1, depth: int = 64, base_size=None, stage=None
    ) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channel, depth, 3)
        self.conv2 = Conv2d(depth, depth, 3)
        self.conv3 = Conv2d(depth, depth, 3)
        self.conv4 = Conv2d(depth, depth, 3)
        self.conv5 = Conv2d(depth, out_channel, 3, bn=False)

        self.base_size = base_size
        self.stage = stage

        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2**stage), base_size[1] // (2**stage))
        else:
            self.stage_size = [None, None]

        self.Hattn = SelfAttention(depth, "h", self.stage_size[0])
        self.Wattn = SelfAttention(depth, "w", self.stage_size[1])

        self.upsample = lambda img, size: F.interpolate(
            img, size=size, mode="bilinear", align_corners=True
        )

    def forward(self, fs):  # f3 f4 f5 -> f3 f2 f1
        fx = fs[0]
        for i in range(1, len(fs)):
            fs[i] = self.upsample(fs[i], fx.shape[-2:])
        fx = torch.cat(fs[::-1], dim=1)

        fx = self.conv1(fx)

        Hfx = self.Hattn(fx)
        Wfx = self.Wattn(fx)
        fx = self.conv2(Hfx + Wfx)
        fx = self.conv3(fx)
        fx = self.conv4(fx)
        out = self.conv5(fx)
        return fx, out


class PAA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size, stage_size=None) -> None:
        super().__init__()
        self.conv0 = Conv2d(in_channel, out_channel, 1)
        self.conv1 = Conv2d(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = Conv2d(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = Conv2d(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = SelfAttention(
            out_channel, "h", stage_size[0] if stage_size is not None else None
        )
        self.Wattn = SelfAttention(
            out_channel, "w", stage_size[1] if stage_size is not None else None
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        return self.conv3(Hx + Wx)


class PAA_e(nn.Module):
    def __init__(self, in_channel, out_channel, base_size=None, stage=None) -> None:
        super().__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2**stage), base_size[1] // (2**stage))
        else:
            self.stage_size = None

        self.branch0 = Conv2d(in_channel, out_channel, 1)
        self.branch1 = PAA_kernel(in_channel, out_channel, 3, self.stage_size)
        self.branch2 = PAA_kernel(in_channel, out_channel, 5, self.stage_size)
        self.branch3 = PAA_kernel(in_channel, out_channel, 7, self.stage_size)

        self.conv_cat = Conv2d(4 * out_channel, out_channel, 3)
        self.conv_res = Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))


class SICA(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel: int = 1,
        depth: int = 64,
        base_size=None,
        stage=None,
        lmap_in: bool = False,
    ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.lmap_in = lmap_in
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2**stage), base_size[1] // (2**stage))
        else:
            self.stage_size = None

        self.conv_query = nn.Sequential(
            Conv2d(in_channel, depth, 3, relu=True), Conv2d(depth, depth, 3, relu=True)
        )
        self.conv_key = nn.Sequential(
            Conv2d(in_channel, depth, 1, relu=True), Conv2d(depth, depth, 1, relu=True)
        )
        self.conv_value = nn.Sequential(
            Conv2d(in_channel, depth, 1, relu=True), Conv2d(depth, depth, 1, relu=True)
        )

        self.ctx = 5 if self.lmap_in else 3
        self.conv_out1 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out2 = Conv2d(in_channel + depth, depth, 3, relu=True)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out4 = Conv2d(depth, out_channel, 1)

        self.threshold = Parameter(torch.tensor([0.5]))

        if self.lmap_in is True:
            self.lthreshold = Parameter(torch.tensor([0.5]))

    def forward(self, x, smap, lmap: torch.Tensor | None = None):
        assert not xor(self.lmap_in is True, lmap is not None)
        b, _c, h, w = x.shape

        # compute class probability
        smap = F.interpolate(smap, size=x.shape[-2:], mode="bilinear", align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold

        fg = torch.clip(p, 0, 1)  # foreground
        bg = torch.clip(-p, 0, 1)  # background
        cg = self.threshold - torch.abs(p)  # confusion area

        if self.lmap_in is True and lmap is not None:
            lmap = F.interpolate(lmap, size=x.shape[-2:], mode="bilinear", align_corners=False)
            lmap = torch.sigmoid(lmap)
            lp = lmap - self.lthreshold
            fp = torch.clip(lp, 0, 1)  # foreground
            bp = torch.clip(-lp, 0, 1)  # background

            prob = [fg, bg, cg, fp, bp]
        else:
            prob = [fg, bg, cg]

        prob = torch.cat(prob, dim=1)

        # reshape feature & prob
        if self.stage_size is not None:
            shape = self.stage_size
            shape_mul = self.stage_size[0] * self.stage_size[1]
        else:
            shape = (h, w)
            shape_mul = h * w

        f = F.interpolate(x, size=shape, mode="bilinear", align_corners=False).view(
            b, shape_mul, -1
        )
        prob = F.interpolate(prob, size=shape, mode="bilinear", align_corners=False).view(
            b, self.ctx, shape_mul
        )

        # compute context vector
        prob = (
            1 / shape_mul
        ) * prob  # note: different from original project, to avoid overflow and variance explosion
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 3, c
        # k q v compute
        query = self.conv_query(x).view(b, self.depth, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.depth, -1)
        value = self.conv_value(context).view(b, self.depth, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.depth**-0.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        return x, out


model_urls = {
    "res2net50_v1b_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
    "res2net101_v1b_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth",
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride: int = 1,
        dilation: int = 1,
        downsample=None,
        baseWidth: int = 26,
        scale: int = 4,
        stype: str = "normal",
    ) -> None:
        super().__init__()

        width = math.floor(planes * (baseWidth / 64.0))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for _i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=3,
                    stride=stride,
                    dilation=dilation,
                    padding=dilation,
                    bias=False,
                )
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 or self.stype == "stage" else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class Res2Net(nn.Module):
    def __init__(
        self,
        block,
        layers,
        baseWidth: int = 26,
        scale: int = 4,
        num_classes: int = 1000,
        output_stride: int = 32,
    ) -> None:
        self.inplanes = 64
        super().__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.output_stride = output_stride
        if self.output_stride == 8:
            self.grid = [1, 2, 1]
            self.stride = [1, 2, 1, 1]
            self.dilation = [1, 1, 2, 4]
        elif self.output_stride == 16:
            self.grid = [1, 2, 4]
            self.stride = [1, 2, 2, 1]
            self.dilation = [1, 1, 1, 2]
        elif self.output_stride == 32:
            self.grid = [1, 2, 4]
            self.stride = [1, 2, 2, 2]
            self.dilation = [1, 1, 2, 4]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=self.stride[0], dilation=self.dilation[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=self.stride[1], dilation=self.dilation[1]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=self.stride[2], dilation=self.dilation[2]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=self.stride[3], dilation=self.dilation[3], grid=self.grid
        )

    def _make_layer(
        self, block, planes: int, blocks, stride=1, dilation=1, grid=None
    ) -> Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
                ),
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                dilation,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        ]
        self.inplanes = planes * block.expansion

        if grid is not None:
            assert len(grid) == blocks
        else:
            grid = [1] * blocks

        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation * grid[i],
                    baseWidth=self.baseWidth,
                    scale=self.scale,
                )
            )

        return nn.Sequential(*layers)

    def change_stride(self, output_stride: int = 16) -> None:
        if output_stride == self.output_stride:
            return
        self.output_stride = output_stride
        if self.output_stride == 8:
            self.grid = [1, 2, 1]
            self.stride = [1, 2, 1, 1]
            self.dilation = [1, 1, 2, 4]
        elif self.output_stride == 16:
            self.grid = [1, 2, 4]
            self.stride = [1, 2, 2, 1]
            self.dilation = [1, 1, 1, 2]
        elif self.output_stride == 32:
            self.grid = [1, 2, 4]
            self.stride = [1, 2, 2, 2]
            self.dilation = [1, 1, 2, 4]

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for j, block in enumerate(layer):
                if block.downsample is not None:
                    block.downsample[0].kernel_size = (self.stride[i], self.stride[i])
                    block.downsample[0].stride = (self.stride[i], self.stride[i])
                    if hasattr(block, "pool"):
                        block.pool.stride = (self.stride[i], self.stride[i])
                    for conv in block.convs:
                        conv.stride = (self.stride[i], self.stride[i])
                for conv in block.convs:
                    d = self.dilation[i] if i != 3 else self.dilation[i] * self.grid[j]
                    conv.dilation = (d, d)
                    conv.padding = (d, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out = [x]

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)

        return out


def res2net50_v1b(pretrained: bool = False, **kwargs) -> Res2Net:
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net50_v1b_26w_4s"]), strict=False)

    return model


def res2net101_v1b(pretrained: bool = False, **kwargs) -> Res2Net:
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net101_v1b_26w_4s"]))

    return model


def res2net50_v1b_26w_4s(pretrained: bool = False, **kwargs) -> Res2Net:
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(
            torch.load("data/backbone_ckpt/res2net50_v1b_26w_4s-3cf99910.pth", map_location="cpu")
        )

    return model


def res2net101_v1b_26w_4s(pretrained: bool = True, **kwargs) -> Res2Net:
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(
            torch.load("data/backbone_ckpt/res2net101_v1b_26w_4s-0812c246.pth", map_location="cpu")
        )

    return model


def res2net152_v1b_26w_4s(pretrained: bool = False, **kwargs) -> Res2Net:
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["res2net152_v1b_26w_4s"]))

    return model


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer: type[GELU] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def window_partition(x: Tensor, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias: bool = True,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[GELU] = nn.GELU,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer: type[LayerNorm] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        return self.reduction(x)


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96, norm_layer=None
    ) -> None:
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths=None,
        num_heads=None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        use_checkpoint: bool = False,
    ) -> None:
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = x + absolute_pos_embed  # B Wh*Ww C

        outs = [x.contiguous()]
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()


def SwinT(pretrained: bool = True) -> SwinTransformer:
    model = SwinTransformer(
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7
    )
    if pretrained:
        model.load_state_dict(
            torch.load("data/backbone_ckpt/swin_tiny_patch4_window7_224.pth", map_location="cpu")[
                "model"
            ],
            strict=False,
        )

    return model


def SwinS(pretrained: bool = True) -> SwinTransformer:
    model = SwinTransformer(
        embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=7
    )
    if pretrained:
        model.load_state_dict(
            torch.load("data/backbone_ckpt/swin_small_patch4_window7_224.pth", map_location="cpu")[
                "model"
            ],
            strict=False,
        )

    return model


def SwinB(pretrained: bool = True) -> SwinTransformer:
    model = SwinTransformer(
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12
    )
    if pretrained:
        model.load_state_dict(
            torch.load(
                "data/backbone_ckpt/swin_base_patch4_window12_384_22kto1k.pth", map_location="cpu"
            )["model"],
            strict=False,
        )

    return model


def SwinL(pretrained: bool = True) -> SwinTransformer:
    model = SwinTransformer(
        embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12
    )
    if pretrained:
        model.load_state_dict(
            torch.load(
                "data/backbone_ckpt/swin_large_patch4_window12_384_22kto1k.pth", map_location="cpu"
            )["model"],
            strict=False,
        )

    return model


def weighted_bce_loss_with_logits(pred, mask, reduction="mean") -> Tensor:
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    return F.binary_cross_entropy_with_logits(pred, mask, weight, reduction=reduction)


def iou_loss(pred: Tensor, mask, reduction="mean"):
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)
    if reduction == "mean":
        iou = iou.mean()
    return iou


def iou_loss_with_logits(pred, mask, reduction="none"):
    return iou_loss(torch.sigmoid(pred), mask, reduction=reduction)


class InSPyReNet(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels,
        depth: int = 64,
        base_size: tuple[int, int] = (384, 384),
        threshold: int | None = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        self.threshold = threshold

        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=self.base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=self.base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=self.base_size, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=self.base_size, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=self.base_size, stage=4)

        self.decoder = PAA_d(self.depth * 3, depth=self.depth, base_size=base_size, stage=2)

        self.attention0 = SICA(
            self.depth, depth=self.depth, base_size=self.base_size, stage=0, lmap_in=True
        )
        self.attention1 = SICA(
            self.depth * 2, depth=self.depth, base_size=self.base_size, stage=1, lmap_in=True
        )
        self.attention2 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=2)

        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(
            x, y, reduction="mean"
        ) + iou_loss_with_logits(x, y, reduction="mean")
        self.pc_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(
            x, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        self.res = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode="nearest")

        self.image_pyramid = ImagePyramid(7, 1)

        self.transition0 = Transition(17)
        self.transition1 = Transition(9)
        self.transition2 = Transition(5)

        self.forward = self.forward_inference

    def _apply(self, fn):
        super()._apply(fn)
        self.image_pyramid._apply(fn)
        self.transition0._apply(fn)
        self.transition1._apply(fn)
        self.transition2._apply(fn)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.forward = self.forward_train if mode else self.forward_inference
        return self

    def forward_inspyre(self, x: Tensor):
        _B, _, H, W = x.shape

        x1, x2, x3, x4, x5 = self.backbone(x)

        x1 = self.context1(x1)  # 4
        x2 = self.context2(x2)  # 4
        x3 = self.context3(x3)  # 8
        x4 = self.context4(x4)  # 16
        x5 = self.context5(x5)  # 32

        f3, d3 = self.decoder([x3, x4, x5])  # 16
        f3 = self.res(f3, (H // 4, W // 4))
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3.detach())

        d2 = self.image_pyramid.reconstruct(d3.detach(), p2)  # 4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))

        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2.detach(), p2.detach())  # 2
        d1 = self.image_pyramid.reconstruct(d2.detach(), p1)  # 2

        f1 = self.res(f1, (H, W))
        _, p0 = self.attention0(f1, d1.detach(), p1.detach())  # 2
        d0 = self.image_pyramid.reconstruct(d1.detach(), p0)  # 2

        return {"saliency": [d3, d2, d1, d0], "laplacian": [p2, p1, p0]}

    def forward_train(self, x, y):
        _B, _, H, W = x.shape
        out = self.forward_inspyre(x)

        d3, d2, d1, d0 = out["saliency"]
        p2, p1, p0 = out["laplacian"]

        y1 = self.image_pyramid.reduce(y)
        y2 = self.image_pyramid.reduce(y1)
        y3 = self.image_pyramid.reduce(y2)

        loss = (
            self.pc_loss_fn(
                self.des(d3, (H, W)), self.des(self.image_pyramid.reduce(d2), (H, W)).detach()
            )
            * 0.0001
        )

        loss += (
            self.pc_loss_fn(
                self.des(d2, (H, W)), self.des(self.image_pyramid.reduce(d1), (H, W)).detach()
            )
            * 0.0001
        )

        loss += (
            self.pc_loss_fn(
                self.des(d1, (H, W)), self.des(self.image_pyramid.reduce(d0), (H, W)).detach()
            )
            * 0.0001
        )

        loss += self.sod_loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
        loss += self.sod_loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
        loss += self.sod_loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
        loss0 = self.sod_loss_fn(self.des(d0, (H, W)), self.des(y, (H, W)))
        loss += loss0

        pred = torch.sigmoid(d0)

        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return {
            "pred": pred,
            "loss": loss,
            "loss0": loss0,
            "saliency": [d3, d2, d1, d0],
            "laplacian": [p2, p1, p0],
        }

    def forward_inference(self, x):
        _B, _, H, W = x.shape

        if self.threshold is None:
            out = self.forward_inspyre(x)
            d3, d2, d1, d0 = out["saliency"]
            p2, p1, p0 = out["laplacian"]

        elif self.threshold >= H or self.threshold >= W:
            out = self.forward_inspyre(self.res(x, self.base_size))

            d3, d2, d1, d0 = out["saliency"]
            p2, p1, p0 = out["laplacian"]

        else:
            # LR Saliency Pyramid
            lr_out = self.forward_inspyre(self.res(x, self.base_size))
            _lr_d3, _lr_d2, _lr_d1, lr_d0 = lr_out["saliency"]
            _lr_p2, _lr_p1, _lr_p0 = lr_out["laplacian"]

            # HR Saliency Pyramid
            if H % 32 != 0 or W % 32 != 0:
                x = self.res(x, ((H // 32) * 32, (W // 32) * 32))
            hr_out = self.forward_inspyre(x)
            hr_d3, _hr_d2, _hr_d1, _hr_d0 = hr_out["saliency"]
            hr_p2, hr_p1, hr_p0 = hr_out["laplacian"]

            # Pyramid Blending
            d3 = self.ret(lr_d0, hr_d3)

            t2 = self.ret(self.transition2(d3), hr_p2)
            p2 = t2 * hr_p2
            d2 = self.image_pyramid.reconstruct(d3, p2)

            t1 = self.ret(self.transition1(d2), hr_p1)
            p1 = t1 * hr_p1
            d1 = self.image_pyramid.reconstruct(d2, p1)

            t0 = self.ret(self.transition0(d1), hr_p0)
            p0 = t0 * hr_p0
            d0 = self.image_pyramid.reconstruct(d1, p0)

        if d0.shape[2] != H or d0.shape[3] != 2:
            d0 = self.res(d0, (H, W))
        pred = torch.sigmoid(d0)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return {"pred": pred, "loss": 0, "saliency": [d3, d2, d1, d0], "laplacian": [p2, p1, p0]}

    @staticmethod
    def compute_loss(sample):
        return sample["loss0"], sample["loss"]


def InSPyReNet_Res2Net50(
    depth: int = 64,
    pretrained: bool = True,
    base_size: int | tuple[int, int] | None = None,
    **kwargs,
) -> InSPyReNet:
    if base_size is None:
        base_size = (384, 384)
    if isinstance(base_size, int):
        base_size = (base_size, base_size)
    return InSPyReNet(
        res2net50_v1b(pretrained=pretrained),
        [64, 256, 512, 1024, 2048],
        depth,
        base_size,
        threshold=None,
        **kwargs,
    )


def InSPyReNet_SwinB(
    depth: int = 64,
    pretrained: bool = False,
    base_size: int | tuple[int, int] | None = None,
    **kwargs,
) -> InSPyReNet:
    if base_size is None:
        base_size = (384, 384)
    if isinstance(base_size, int):
        base_size = (base_size, base_size)
    return InSPyReNet(
        SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs
    )
