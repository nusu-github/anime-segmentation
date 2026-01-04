from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import optim

from .metrics.collection import SegmentationMetrics
from .model import ISNetDIS, ISNetGTEncoder

net_names = [
    "isnet_is",
    "isnet",
    "isnet_gt",
]


def get_net(net_name: str, _img_size: int | None) -> torch.nn.Module:
    if net_name in {"isnet", "isnet_is"}:
        return ISNetDIS()
    if net_name == "isnet_gt":
        return ISNetGTEncoder()
    raise NotImplementedError


def _torch_load(path: str, map_location: Any | None = None) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(ckpt_or_state: Any) -> dict[str, Any]:
    if isinstance(ckpt_or_state, dict) and "state_dict" in ckpt_or_state:
        sd = ckpt_or_state["state_dict"]
        if isinstance(sd, dict):
            return sd
    if isinstance(ckpt_or_state, dict):
        return ckpt_or_state
    msg = "Unsupported checkpoint format"
    raise TypeError(msg)


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


class AnimeSegmentation(
    pl.LightningModule,
    PyTorchModelHubMixin,
    library_name="anime_segmentation",
    repo_url="https://github.com/SkyTNT/anime-segmentation",
    tags=["image-segmentation"],
):
    def __init__(
        self,
        net_name: str,
        img_size: int | None = None,
        lr: float = 1e-4,
        gt_encoder_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        if net_name not in net_names:
            msg = f"Unknown net_name: {net_name!r}. Expected one of {net_names}."
            raise ValueError(msg)

        self.net_name = net_name
        self.img_size = img_size
        self.lr = float(lr)
        self.gt_encoder_ckpt = gt_encoder_ckpt

        self.net = get_net(net_name, img_size)

        self.val_metrics = SegmentationMetrics(
            metrics=("F", "WF", "MAE", "S", "HCE", "MBA", "BIoU"),
        )

        self._loaded_from_ckpt = False

        if net_name == "isnet_is":
            self.gt_encoder = get_net("isnet_gt", img_size)
            self.gt_encoder.requires_grad_(False)
        else:
            self.gt_encoder = None

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint
        self._loaded_from_ckpt = True

    def setup(self, stage: str) -> None:
        if stage != "fit":
            return
        if self.net_name != "isnet_is":
            return
        if self.gt_encoder is None:
            msg = "gt_encoder is required for isnet_is"
            raise RuntimeError(msg)

        if self._loaded_from_ckpt:
            return

        if not self.gt_encoder_ckpt:
            msg = (
                "gt_encoder_ckpt is required for stage2 training (net_name=isnet_is). "
                "Run stage1 (isnet_gt) first and set model.gt_encoder_ckpt to the produced ckpt."
            )
            raise ValueError(msg)

        ckpt_path = Path(self.gt_encoder_ckpt)
        if not ckpt_path.exists():
            msg = f"gt_encoder_ckpt not found: {ckpt_path}"
            raise FileNotFoundError(msg)

        ckpt = _torch_load(str(ckpt_path), map_location="cpu")
        sd = _extract_state_dict(ckpt)

        # stage1 checkpoint comes from AnimeSegmentation(net_name=isnet_gt) -> weights under "net.".
        sd = _strip_prefix(sd, "net.")

        missing, unexpected = self.gt_encoder.load_state_dict(sd, strict=True)
        if missing or unexpected:
            msg = f"Failed to load gt_encoder: missing={missing}, unexpected={unexpected}"
            raise RuntimeError(msg)

        self.gt_encoder.requires_grad_(False)

    @classmethod
    def try_load(
        cls,
        net_name: str,
        ckpt_path: str,
        map_location: Any | None = None,
        img_size: int | None = None,
    ):
        state = _torch_load(ckpt_path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            return cls.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                net_name=net_name,
                img_size=img_size,
                map_location=map_location,
            )

        model = cls(net_name, img_size)
        sd = _extract_state_dict(state)

        if any(k.startswith("net.") for k in sd):
            model.load_state_dict(sd)
        else:
            model.net.load_state_dict(sd)
        return model

    def configure_optimizers(self):
        return optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net, ISNetDIS):
            return self.net(x)[0][0].sigmoid()
        if isinstance(self.net, ISNetGTEncoder):
            return self.net(x)[0][0].sigmoid()
        raise NotImplementedError

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetDIS):
            ds, dfs = self.net(images)
            loss_args: list[Any] = [ds, dfs, labels]
            if self.gt_encoder is not None:
                fs = self.gt_encoder(labels)[1]
                loss_args.append(fs)
        elif isinstance(self.net, ISNetGTEncoder):
            ds = self.net(labels)[0]
            loss_args = [ds, labels]
        else:
            raise NotImplementedError

        loss0, loss = self.net.compute_loss(loss_args)
        loss_t = torch.as_tensor(loss)
        loss0_t = torch.as_tensor(loss0)
        self.log("train/loss", loss_t, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_tar", loss0_t, on_step=True, on_epoch=True)
        return loss_t

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]
        if isinstance(self.net, ISNetGTEncoder):
            preds = self.forward(labels)
        else:
            preds = self.forward(images)
        self.val_metrics.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        results = self.val_metrics.compute()

        key_map: dict[str, tuple[str, bool]] = {
            "StructuralMeasure": ("val_Sm", True),
            "MeanAbsoluteError": ("val_MAE", True),
            "WeightedFMeasure": ("val_wFmeasure", False),
            "FMeasure/maxF": ("val_maxFm", False),
            "FMeasure/meanF": ("val_meanFm", False),
            "FMeasure/adpF": ("val_adpFm", False),
            "HumanCorrectionEffort": ("val_HCE", False),
            "MeanBoundaryAccuracy": ("val_mBA", False),
            "BoundaryIoU/maxBIoU": ("val_maxBIoU", False),
            "BoundaryIoU/meanBIoU": ("val_meanBIoU", False),
        }

        for src_key, (dst_key, prog_bar) in key_map.items():
            if src_key in results:
                self.log(dst_key, results[src_key], prog_bar=prog_bar)

        self.val_metrics.reset()
