import torch
import torch.nn.functional as F
from timm.layers import ConvNormAct
from torch import nn

from .deform_conv import DeformableConv2d
from .norms import group_norm


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context aggregation."""

    def __init__(self, in_channels=64, out_channels=None, output_stride=16, use_norm=True) -> None:
        super().__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        norm_layer = group_norm

        self.aspp1 = ConvNormAct(
            in_channels,
            self.in_channelster,
            1,
            padding=0,
            dilation=dilations[0],
            norm_layer=norm_layer,
            apply_norm=use_norm,
            bias=False,
        )
        self.aspp2 = ConvNormAct(
            in_channels,
            self.in_channelster,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            norm_layer=norm_layer,
            apply_norm=use_norm,
            bias=False,
        )
        self.aspp3 = ConvNormAct(
            in_channels,
            self.in_channelster,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            norm_layer=norm_layer,
            apply_norm=use_norm,
            bias=False,
        )
        self.aspp4 = ConvNormAct(
            in_channels,
            self.in_channelster,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            norm_layer=norm_layer,
            apply_norm=use_norm,
            bias=False,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvNormAct(
                in_channels,
                self.in_channelster,
                1,
                norm_layer=norm_layer,
                apply_norm=use_norm,
                bias=False,
            ),
        )
        self.conv1 = ConvNormAct(
            self.in_channelster * 5,
            out_channels,
            1,
            norm_layer=norm_layer,
            apply_norm=use_norm,
            bias=False,
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        return self.dropout(x)


class ASPPDeformable(nn.Module):
    """ASPP variant using deformable convolutions for adaptive spatial sampling."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        parallel_block_sizes=None,
        use_norm=True,
    ) -> None:
        if parallel_block_sizes is None:
            parallel_block_sizes = [1, 3, 7]
        super().__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        norm_layer = group_norm

        self.aspp1 = nn.Sequential(
            DeformableConv2d(
                in_channels,
                self.in_channelster,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            group_norm(self.in_channelster) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.aspp_deforms = nn.ModuleList(
            [
                nn.Sequential(
                    DeformableConv2d(
                        in_channels,
                        self.in_channelster,
                        conv_size,
                        padding=int(conv_size // 2),
                        bias=False,
                    ),
                    group_norm(self.in_channelster) if use_norm else nn.Identity(),
                    nn.ReLU(inplace=True),
                )
                for conv_size in parallel_block_sizes
            ],
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvNormAct(
                in_channels,
                self.in_channelster,
                1,
                norm_layer=norm_layer,
                apply_norm=use_norm,
                bias=False,
            ),
        )
        self.conv1 = ConvNormAct(
            self.in_channelster * (2 + len(self.aspp_deforms)),
            out_channels,
            1,
            norm_layer=norm_layer,
            apply_norm=use_norm,
            bias=False,
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)

        x = self.conv1(x)
        return self.dropout(x)
