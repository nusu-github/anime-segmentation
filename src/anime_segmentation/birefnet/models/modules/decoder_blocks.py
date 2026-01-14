from timm.layers import ConvNormAct
from torch import nn

from .aspp import ASPP, ASPPDeformable


class BasicDecBlk(nn.Module):
    """Basic decoder block with optional ASPP attention."""

    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        inter_channels=64,
        dec_channels_inter="fixed",
        attention_type=None,
        use_bn=True,
    ) -> None:
        super().__init__()
        inter_channels = in_channels // 4 if dec_channels_inter == "adap" else 64

        norm_layer = nn.BatchNorm2d
        self.conv_in = ConvNormAct(
            in_channels,
            inter_channels,
            3,
            padding=1,
            norm_layer=norm_layer,
            apply_norm=use_bn,
        )

        if attention_type == "ASPP":
            self.dec_att = ASPP(in_channels=inter_channels, use_bn=use_bn)
        elif attention_type == "ASPPDeformable":
            self.dec_att = ASPPDeformable(in_channels=inter_channels, use_bn=use_bn)

        self.conv_out = ConvNormAct(
            inter_channels,
            out_channels,
            3,
            padding=1,
            norm_layer=norm_layer,
            apply_norm=use_bn,
            apply_act=False,
        )

    def forward(self, x):
        x = self.conv_in(x)
        if hasattr(self, "dec_att"):
            x = self.dec_att(x)
        return self.conv_out(x)


class ResBlk(nn.Module):
    """Residual decoder block with skip connection and optional ASPP attention."""

    def __init__(
        self,
        in_channels=64,
        out_channels=None,
        inter_channels=64,
        dec_channels_inter="fixed",
        attention_type=None,
        use_bn=True,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_channels = in_channels // 4 if dec_channels_inter == "adap" else 64

        norm_layer = nn.BatchNorm2d
        self.conv_in = ConvNormAct(
            in_channels,
            inter_channels,
            3,
            padding=1,
            norm_layer=norm_layer,
            apply_norm=use_bn,
        )

        if attention_type == "ASPP":
            self.dec_att = ASPP(in_channels=inter_channels, use_bn=use_bn)
        elif attention_type == "ASPPDeformable":
            self.dec_att = ASPPDeformable(in_channels=inter_channels, use_bn=use_bn)

        self.conv_out = ConvNormAct(
            inter_channels,
            out_channels,
            3,
            padding=1,
            norm_layer=norm_layer,
            apply_norm=use_bn,
            apply_act=False,
        )
        self.conv_resi = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        _x = self.conv_resi(x)
        x = self.conv_in(x)
        if hasattr(self, "dec_att"):
            x = self.dec_att(x)
        x = self.conv_out(x)
        return x + _x
