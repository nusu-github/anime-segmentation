from torch import nn


class BasicLatBlk(nn.Module):
    """1x1 conv for lateral feature fusion in FPN-style decoders."""

    def __init__(self, in_channels=64, out_channels=64, ks=1, s=1, p=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ks, s, p)

    def forward(self, x):
        return self.conv(x)
