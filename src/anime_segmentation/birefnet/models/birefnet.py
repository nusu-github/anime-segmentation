import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from kornia.filters import laplacian
from timm.layers import ConvNormAct
from torch import nn

from .backbones.build_backbone import build_backbone
from .config import get_lateral_channels
from .modules.decoder_blocks import BasicDecBlk, ResBlk
from .modules.lateral_blocks import BasicLatBlk
from .modules.norms import group_norm

_ACT_LAYER_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
}


def _resolve_act_layer(act_layer: str | None = None, act_kwargs=None):
    if act_layer is None:
        act_layer = "relu"
    if isinstance(act_layer, str):
        key = act_layer.replace("-", "_").lower()
        if key not in _ACT_LAYER_MAP:
            msg = f"Unknown activation '{act_layer}'. Available: {', '.join(_ACT_LAYER_MAP)}"
            raise ValueError(msg)
        act_layer: type[nn.Module] = _ACT_LAYER_MAP[key]
    if act_kwargs is None:
        match act_layer:
            case nn.ReLU | nn.SiLU:
                act_kwargs = {"inplace": True}
            case nn.GELU:
                act_kwargs = {"approximate": "tanh"}
    return act_layer, act_kwargs


def image2patches(
    image,
    grid_h=2,
    grid_w=2,
    patch_ref=None,
    transformation="b c (hg h) (wg w) -> (b hg wg) c h w",
):
    if patch_ref is not None:
        grid_h, grid_w = (
            image.shape[-2] // patch_ref.shape[-2],
            image.shape[-1] // patch_ref.shape[-1],
        )
    return rearrange(image, transformation, hg=grid_h, wg=grid_w)


def patches2image(
    patches,
    grid_h=2,
    grid_w=2,
    patch_ref=None,
    transformation="(b hg wg) c h w -> b c (hg h) (wg w)",
):
    if patch_ref is not None:
        grid_h, grid_w = (
            patch_ref.shape[-2] // patches[0].shape[-2],
            patch_ref.shape[-1] // patches[0].shape[-1],
        )
    return rearrange(patches, transformation, hg=grid_h, wg=grid_w)


class BiRefNet(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="birefnet",
    repo_url="https://github.com/ZhengPeng7/BiRefNet",
    tags=[
        "Image Segmentation",
        "Background Removal",
        "Mask Generation",
        "Dichotomous Image Segmentation",
        "Camouflaged Object Detection",
        "Salient Object Detection",
    ],
):
    """Bilateral Reference Network for dichotomous image segmentation.

    Encoder-decoder architecture with multi-scale supervision and gradient-aware loss.
    See: https://github.com/ZhengPeng7/BiRefNet
    """

    def __init__(
        self,
        bb_name="swin_v1_l",
        bb_pretrained=True,
        auxiliary_classification=False,
        squeeze_block="BasicDecBlk_x1",
        dec_blk="BasicDecBlk",
        lat_blk="BasicLatBlk",
        mul_scl_ipt="cat",
        cxt_num=3,
        dec_att="ASPPDeformable",
        dec_ipt=True,
        dec_ipt_split=True,
        use_norm=True,
        ms_supervision=True,
        out_ref=True,
        dec_channels_inter="fixed",
        act_layer="relu",
        act_kwargs=None,
        num_classes=None,
    ) -> None:
        super().__init__()
        self.bb_name = bb_name
        self.mul_scl_ipt = mul_scl_ipt
        self.cxt_num = cxt_num
        self.auxiliary_classification = auxiliary_classification
        self.squeeze_block_name = squeeze_block
        self.out_ref = out_ref
        self.ms_supervision = ms_supervision
        self.dec_ipt = dec_ipt
        self.dec_att = dec_att
        self.act_layer, self.act_kwargs = _resolve_act_layer(act_layer, act_kwargs)

        self.bb = build_backbone(bb_name, pretrained=bb_pretrained)

        # Get lateral channels: auto-infer from backbone or fallback to known map
        channels = get_lateral_channels(bb_name, backbone=self.bb)

        if mul_scl_ipt == "cat":
            channels = [c * 2 for c in channels]

        self.cxt = channels[1:][::-1][-cxt_num:] if cxt_num else []

        if auxiliary_classification:
            if num_classes is None:
                # Raise error or set default. Assuming user knows what they are doing if enabling this.
                msg = "num_classes must be specified when auxiliary_classification is True"
                raise ValueError(msg)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_head = nn.Sequential(
                nn.Linear(channels[0], num_classes),
            )

        if squeeze_block:
            block_type, num_blocks = squeeze_block.split("_x")
            num_blocks = int(num_blocks)
            block_map = {"BasicDecBlk": BasicDecBlk, "ResBlk": ResBlk}
            BlockClass = block_map[block_type]
            self.squeeze_module = nn.Sequential(
                *[
                    BlockClass(
                        channels[0] + sum(self.cxt),
                        channels[0],
                        use_norm=use_norm,
                        attention_type=dec_att,
                        dec_channels_inter=dec_channels_inter,
                        act_layer=self.act_layer,
                        act_kwargs=self.act_kwargs,
                    )
                    for _ in range(num_blocks)
                ],
            )

        self.decoder = Decoder(
            channels,
            mul_scl_ipt=mul_scl_ipt,
            dec_blk=dec_blk,
            lat_blk=lat_blk,
            dec_att=dec_att,
            dec_ipt=dec_ipt,
            dec_ipt_split=dec_ipt_split,
            use_norm=use_norm,
            ms_supervision=ms_supervision,
            out_ref=out_ref,
            dec_channels_inter=dec_channels_inter,
            bb_name=bb_name,
            act_layer=self.act_layer,
            act_kwargs=self.act_kwargs,
        )

    def forward_enc(self, x):
        x1, x2, x3, x4 = self.bb(x)  # 4-level feature pyramid

        if self.mul_scl_ipt:
            _B, _C, H, W = x.shape
            x_pyramid = F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=True)
            if self.mul_scl_ipt == "cat":
                x1_, x2_, x3_, x4_ = self.bb(x_pyramid)
                x1 = torch.cat(
                    [
                        x1,
                        F.interpolate(x1_, size=x1.shape[2:], mode="bilinear", align_corners=True),
                    ],
                    dim=1,
                )
                x2 = torch.cat(
                    [
                        x2,
                        F.interpolate(x2_, size=x2.shape[2:], mode="bilinear", align_corners=True),
                    ],
                    dim=1,
                )
                x3 = torch.cat(
                    [
                        x3,
                        F.interpolate(x3_, size=x3.shape[2:], mode="bilinear", align_corners=True),
                    ],
                    dim=1,
                )
                x4 = torch.cat(
                    [
                        x4,
                        F.interpolate(x4_, size=x4.shape[2:], mode="bilinear", align_corners=True),
                    ],
                    dim=1,
                )
            elif self.mul_scl_ipt == "add":
                x1_, x2_, x3_, x4_ = self.bb(x_pyramid)
                x1 = x1 + F.interpolate(x1_, size=x1.shape[2:], mode="bilinear", align_corners=True)
                x2 = x2 + F.interpolate(x2_, size=x2.shape[2:], mode="bilinear", align_corners=True)
                x3 = x3 + F.interpolate(x3_, size=x3.shape[2:], mode="bilinear", align_corners=True)
                x4 = x4 + F.interpolate(x4_, size=x4.shape[2:], mode="bilinear", align_corners=True)
        class_preds = (
            self.cls_head(self.avgpool(x4).view(x4.shape[0], -1))
            if self.training and self.auxiliary_classification
            else None
        )
        if self.cxt:
            features_to_cat = [x1, x2, x3][-len(self.cxt) :]
            upsampled_features = [
                F.interpolate(f, size=x4.shape[2:], mode="bilinear", align_corners=True)
                for f in features_to_cat
            ]
            x4 = torch.cat([*upsampled_features, x4], dim=1)
        return (x1, x2, x3, x4), class_preds

    def forward_ori(self, x):
        # Encoder
        (x1, x2, x3, x4), class_preds = self.forward_enc(x)
        if self.squeeze_block_name:
            x4 = self.squeeze_module(x4)

        # Decoder
        features = [x, x1, x2, x3, x4]
        if self.training and self.out_ref:
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))
        scaled_preds = self.decoder(features)
        return scaled_preds, class_preds

    def forward(self, x):
        scaled_preds, class_preds = self.forward_ori(x)
        if self.training:
            return scaled_preds, [class_preds]
        return scaled_preds


class Decoder(nn.Module):
    """Progressive decoder with multi-scale supervision and gradient-aware refinement."""

    def __init__(
        self,
        channels,
        mul_scl_ipt,
        dec_blk,
        lat_blk,
        dec_att,
        dec_ipt,
        dec_ipt_split,
        use_norm,
        ms_supervision,
        out_ref,
        dec_channels_inter,
        bb_name,
        act_layer,
        act_kwargs,
    ) -> None:
        super().__init__()
        self.mul_scl_ipt = mul_scl_ipt
        self.dec_ipt = dec_ipt
        self.split = dec_ipt_split
        self.ms_supervision = ms_supervision
        self.out_ref = out_ref
        self.act_layer = act_layer
        self.act_kwargs = act_kwargs

        DecoderBlock = {"BasicDecBlk": BasicDecBlk, "ResBlk": ResBlk}[dec_blk]
        LateralBlock = {"BasicLatBlk": BasicLatBlk}[lat_blk]

        self.bbs_without_pyramid = ["vit", "dino"]
        self.use_pyramid_neck = any(
            bb_without_pyramid in bb_name for bb_without_pyramid in self.bbs_without_pyramid
        )
        if self.use_pyramid_neck:
            # Default channels based on swin_v1_l for ViT/DINOv3 backbones
            self.manually_controlled_decoder_in_channels = [
                c * (1 + int(self.mul_scl_ipt == "cat")) for c in (1536, 768, 384, 192)
            ]
            self.pyramid_neck_x4 = LateralBlock(
                channels[0],
                self.manually_controlled_decoder_in_channels[0],
            )
            self.pyramid_neck_x3 = LateralBlock(
                channels[1],
                self.manually_controlled_decoder_in_channels[1],
            )
            self.pyramid_neck_x2 = LateralBlock(
                channels[2],
                self.manually_controlled_decoder_in_channels[2],
            )
            self.pyramid_neck_x1 = LateralBlock(
                channels[3],
                self.manually_controlled_decoder_in_channels[3],
            )

        if self.dec_ipt:
            N_dec_ipt = 64
            DBlock = SimpleConvs
            ic = 64
            ipt_cha_opt = 1
            ipt_blk_in_channels = [2**i * 3 for i in (10, 8, 6, 4, 0)] if self.split else [3] * 5
            ipt_blk_out_channels = [[N_dec_ipt, channels[i] // 8][ipt_cha_opt] for i in range(4)]
            self.ipt_blk5 = DBlock(
                ipt_blk_in_channels[0],
                ipt_blk_out_channels[0],
                inter_channels=ic,
            )
            self.ipt_blk4 = DBlock(
                ipt_blk_in_channels[1],
                ipt_blk_out_channels[0],
                inter_channels=ic,
            )
            self.ipt_blk3 = DBlock(
                ipt_blk_in_channels[2],
                ipt_blk_out_channels[1],
                inter_channels=ic,
            )
            self.ipt_blk2 = DBlock(
                ipt_blk_in_channels[3],
                ipt_blk_out_channels[2],
                inter_channels=ic,
            )
            self.ipt_blk1 = DBlock(
                ipt_blk_in_channels[4],
                ipt_blk_out_channels[3],
                inter_channels=ic,
            )

        if self.use_pyramid_neck:
            bb_neck_out_channels = list(self.manually_controlled_decoder_in_channels)
        else:
            bb_neck_out_channels = channels.copy()
        dec_blk_out_channels = [*list(bb_neck_out_channels[1:]), bb_neck_out_channels[-1] // 2]
        if self.dec_ipt:
            dec_blk_in_channels = [
                bb_neck_out_channels[i] + ipt_blk_out_channels[max(0, i - 1)]
                for i in range(len(bb_neck_out_channels))
            ]
        else:
            dec_blk_in_channels = bb_neck_out_channels

        self.decoder_block4 = DecoderBlock(
            dec_blk_in_channels[0],
            dec_blk_out_channels[0],
            attention_type=dec_att,
            use_norm=use_norm,
            dec_channels_inter=dec_channels_inter,
            act_layer=self.act_layer,
            act_kwargs=self.act_kwargs,
        )
        self.decoder_block3 = DecoderBlock(
            dec_blk_in_channels[1],
            dec_blk_out_channels[1],
            attention_type=dec_att,
            use_norm=use_norm,
            dec_channels_inter=dec_channels_inter,
            act_layer=self.act_layer,
            act_kwargs=self.act_kwargs,
        )
        self.decoder_block2 = DecoderBlock(
            dec_blk_in_channels[2],
            dec_blk_out_channels[2],
            attention_type=dec_att,
            use_norm=use_norm,
            dec_channels_inter=dec_channels_inter,
            act_layer=self.act_layer,
            act_kwargs=self.act_kwargs,
        )
        self.decoder_block1 = DecoderBlock(
            dec_blk_in_channels[3],
            dec_blk_out_channels[3],
            attention_type=dec_att,
            use_norm=use_norm,
            dec_channels_inter=dec_channels_inter,
            act_layer=self.act_layer,
            act_kwargs=self.act_kwargs,
        )
        self.conv_out1 = nn.Sequential(
            nn.Conv2d(
                dec_blk_out_channels[3] + (ipt_blk_out_channels[3] if self.dec_ipt else 0),
                1,
                1,
                1,
                0,
            ),
        )

        self.lateral_block4 = LateralBlock(bb_neck_out_channels[1], dec_blk_out_channels[0])
        self.lateral_block3 = LateralBlock(bb_neck_out_channels[2], dec_blk_out_channels[1])
        self.lateral_block2 = LateralBlock(bb_neck_out_channels[3], dec_blk_out_channels[2])

        if self.ms_supervision:
            self.conv_ms_spvn_4 = nn.Conv2d(dec_blk_out_channels[0], 1, 1, 1, 0)
            self.conv_ms_spvn_3 = nn.Conv2d(dec_blk_out_channels[1], 1, 1, 1, 0)
            self.conv_ms_spvn_2 = nn.Conv2d(dec_blk_out_channels[2], 1, 1, 1, 0)

            if self.out_ref:
                _N = 16
                self.gdt_convs_4 = ConvNormAct(
                    dec_blk_out_channels[0],
                    _N,
                    3,
                    padding=1,
                    norm_layer=group_norm,
                    apply_norm=use_norm,
                    act_layer=self.act_layer,
                    act_kwargs=self.act_kwargs,
                )
                self.gdt_convs_3 = ConvNormAct(
                    dec_blk_out_channels[1],
                    _N,
                    3,
                    padding=1,
                    norm_layer=group_norm,
                    apply_norm=use_norm,
                    act_layer=self.act_layer,
                    act_kwargs=self.act_kwargs,
                )
                self.gdt_convs_2 = ConvNormAct(
                    dec_blk_out_channels[2],
                    _N,
                    3,
                    padding=1,
                    norm_layer=group_norm,
                    apply_norm=use_norm,
                    act_layer=self.act_layer,
                    act_kwargs=self.act_kwargs,
                )

                self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

                self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
                self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def forward(self, features):
        if self.training and self.out_ref:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features
        size_x1_to_x4_template = [
            (x.shape[2] // (2**i), x.shape[3] // (2**i)) for i in (2, 3, 4, 5)
        ]
        if self.use_pyramid_neck:
            x1 = F.interpolate(
                x1,
                size=size_x1_to_x4_template[0],
                mode="bilinear",
                align_corners=True,
            )
            x1 = self.pyramid_neck_x1(x1)

            x2 = F.interpolate(
                x2,
                size=size_x1_to_x4_template[1],
                mode="bilinear",
                align_corners=True,
            )
            x2 = self.pyramid_neck_x2(x2)

            x3 = F.interpolate(
                x3,
                size=size_x1_to_x4_template[2],
                mode="bilinear",
                align_corners=True,
            )
            x3 = self.pyramid_neck_x3(x3)

            x4 = F.interpolate(
                x4,
                size=size_x1_to_x4_template[3],
                mode="bilinear",
                align_corners=True,
            )
            x4 = self.pyramid_neck_x4(x4)
        outs = []

        if self.dec_ipt:
            patches_batch = (
                image2patches(
                    x,
                    patch_ref=x4,
                    transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
                )
                if self.split
                else x
            )
            x4 = torch.cat(
                (
                    x4,
                    self.ipt_blk5(
                        F.interpolate(
                            patches_batch,
                            size=x4.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        ),
                    ),
                ),
                1,
            )
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.ms_supervision and self.training else None
        if self.out_ref:
            # Gradient-aware refinement at scale 4
            p4_gdt = self.gdt_convs_4(p4)
            if self.training:
                m4_dia = m4
                assert m4_dia is not None
                gdt_label_main_4 = gdt_gt * F.interpolate(
                    m4_dia,
                    size=gdt_gt.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
                outs_gdt_label.append(gdt_label_main_4)
                gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
                outs_gdt_pred.append(gdt_pred_4)
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            p4 = p4 * gdt_attn_4
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        if self.dec_ipt:
            patches_batch = (
                image2patches(
                    x,
                    patch_ref=_p3,
                    transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
                )
                if self.split
                else x
            )
            _p3 = torch.cat(
                (
                    _p3,
                    self.ipt_blk4(
                        F.interpolate(
                            patches_batch,
                            size=x3.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        ),
                    ),
                ),
                1,
            )
        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.ms_supervision and self.training else None
        if self.out_ref:
            # Gradient-aware refinement at scale 3
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                m3_dia = m3
                assert m3_dia is not None
                gdt_label_main_3 = gdt_gt * F.interpolate(
                    m3_dia,
                    size=gdt_gt.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
                outs_gdt_label.append(gdt_label_main_3)
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            p3 = p3 * gdt_attn_3
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        if self.dec_ipt:
            patches_batch = (
                image2patches(
                    x,
                    patch_ref=_p2,
                    transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
                )
                if self.split
                else x
            )
            _p2 = torch.cat(
                (
                    _p2,
                    self.ipt_blk3(
                        F.interpolate(
                            patches_batch,
                            size=x2.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        ),
                    ),
                ),
                1,
            )
        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.ms_supervision and self.training else None
        if self.out_ref:
            # Gradient-aware refinement at scale 2
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                m2_dia = m2
                assert m2_dia is not None
                gdt_label_main_2 = gdt_gt * F.interpolate(
                    m2_dia,
                    size=gdt_gt.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
                outs_gdt_label.append(gdt_label_main_2)
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            p2 = p2 * gdt_attn_2
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        if self.dec_ipt:
            patches_batch = (
                image2patches(
                    x,
                    patch_ref=_p1,
                    transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
                )
                if self.split
                else x
            )
            _p1 = torch.cat(
                (
                    _p1,
                    self.ipt_blk2(
                        F.interpolate(
                            patches_batch,
                            size=x1.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        ),
                    ),
                ),
                1,
            )
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode="bilinear", align_corners=True)

        if self.dec_ipt:
            patches_batch = (
                image2patches(
                    x,
                    patch_ref=_p1,
                    transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
                )
                if self.split
                else x
            )
            _p1 = torch.cat(
                (
                    _p1,
                    self.ipt_blk1(
                        F.interpolate(
                            patches_batch,
                            size=x.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        ),
                    ),
                ),
                1,
            )
        p1_out = self.conv_out1(_p1)

        if self.ms_supervision and self.training:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)
        return (
            outs
            if not (self.out_ref and self.training)
            else ([outs_gdt_pred, outs_gdt_label], outs)
        )


class SimpleConvs(nn.Module):
    """Two-layer conv block for input projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels=64,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))
