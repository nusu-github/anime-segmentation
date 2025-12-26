import argparse
import math
from argparse import Namespace
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
from huggingface_hub import PyTorchModelHubMixin
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch import optim
from torch.optim import Optimizer
from torchmetrics import MeanAbsoluteError, MetricCollection
from torchmetrics.classification import BinaryFBetaScore, BinaryPrecision, BinaryRecall

from .data_module import AnimeSegDataModule
from .model import (
    InSPyReNet,
    InSPyReNet_Res2Net50,
    InSPyReNet_SwinB,
    ISNetDIS,
    ISNetGTEncoder,
    MODNet,
    U2Net,
    U2NetFull2,
    U2NetLite2,
)

# warnings.filterwarnings("ignore")


def get_precision(fp32: bool, bf16: bool) -> Literal["32-true", "bf16-mixed", "16-mixed"]:
    """Get precision string for Trainer based on flags."""
    if fp32:
        return "32-true"
    return "bf16-mixed" if bf16 else "16-mixed"


def get_strategy(devices: int) -> DDPStrategy | str:
    """Get optimized training strategy based on device count."""
    if devices > 1:
        return DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
    return "auto"


NET_NAMES = [
    "isnet_is",
    "isnet",
    "isnet_gt",
    "u2net",
    "u2netl",
    "modnet",
    "inspyrnet_res",
    "inspyrnet_swin",
]


def get_net(net_name: str, img_size) -> ISNetDIS | ISNetGTEncoder | InSPyReNet | MODNet | U2Net:
    match net_name:
        case "isnet" | "isnet_is":
            return ISNetDIS()
        case "isnet_gt":
            return ISNetGTEncoder()
        case "u2net":
            return U2NetFull2()
        case "u2netl":
            return U2NetLite2()
        case "modnet":
            return MODNet()
        case "inspyrnet_res":
            return InSPyReNet_Res2Net50(base_size=img_size)
        case "inspyrnet_swin":
            return InSPyReNet_SwinB(base_size=img_size)
        case _:
            raise NotImplementedError


class AnimeSegmentation(
    pl.LightningModule,
    PyTorchModelHubMixin,
    library_name="anime_segmentation",
    repo_url="https://github.com/SkyTNT/anime-segmentation",
    tags=["image-segmentation"],
):
    def __init__(self, net_name: str, img_size: int | None = None, lr: float = 1e-3) -> None:
        super().__init__()
        assert net_name in NET_NAMES
        self.save_hyperparameters()
        self.net = get_net(net_name, img_size)
        if self.hparams["net_name"] == "isnet_is":
            self.gt_encoder = get_net("isnet_gt", img_size)
            self.gt_encoder.requires_grad_(False)
        else:
            self.gt_encoder = None

        # Validation metrics
        # Original f1_torch used beta^2=0.3, so beta=sqrt(0.3)
        self.val_metrics = MetricCollection(
            {
                "precision": BinaryPrecision(multidim_average="global"),
                "recall": BinaryRecall(multidim_average="global"),
                "f1": BinaryFBetaScore(beta=math.sqrt(0.3), multidim_average="global"),
            },
            prefix="val/",
        )
        self.val_mae = MeanAbsoluteError()

    @classmethod
    def try_load(cls, net_name, ckpt_path, map_location: str | None = None, img_size=None):
        state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if "epoch" in state_dict:
            return cls.load_from_checkpoint(
                ckpt_path, net_name=net_name, img_size=img_size, map_location=map_location
            )
        model = cls(net_name, img_size)
        if any(k.startswith("net.") for k, v in state_dict.items()):
            model.load_state_dict(state_dict)
        else:
            model.net.load_state_dict(state_dict)
        return model

    def configure_optimizers(self) -> Optimizer:
        return optim.Adam(
            self.net.parameters(),
            lr=self.hparams["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.net:
            case ISNetDIS() | ISNetGTEncoder():
                return self.net(x)[0][0].sigmoid()
            case U2Net():
                return self.net(x)[0].sigmoid()
            case MODNet():
                return self.net(x, True)[2]
            case InSPyReNet():
                return self.net.forward_inference(x)["pred"]
            case _:
                raise NotImplementedError

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        match self.net:
            case ISNetDIS():
                ds, dfs = self.net(images)
                loss_args = [ds, dfs, labels]
                if self.gt_encoder is not None:
                    fs = self.gt_encoder(labels)[1]
                    loss_args.append(fs)
            case ISNetGTEncoder():
                ds = self.net(labels)[0]
                loss_args = [ds, labels]
            case U2Net():
                ds = self.net(images)
                loss_args = [ds, labels]
            case MODNet():
                trimaps = batch["trimap"]
                pred_semantic, pred_detail, pred_matte = self.net(images, False)
                loss_args = [pred_semantic, pred_detail, pred_matte, images, trimaps, labels]
            case InSPyReNet():
                loss_args = self.net.forward_train(images, labels)
            case _:
                raise NotImplementedError

        loss0, loss = self.net.compute_loss(loss_args)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_tar", loss0)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]
        match self.net:
            case ISNetGTEncoder():
                preds = self.forward(labels)
            case _:
                preds = self.forward(images)

        # Clean predictions for metric computation
        preds_clean = preds.nan_to_num(nan=0, posinf=1, neginf=0)

        # Flatten for global binary metrics (batch, channels, height, width) -> (N,)
        preds_flat = preds_clean.view(-1)
        labels_flat = labels.view(-1)

        # Update metrics
        self.val_metrics.update(preds_flat, labels_flat)
        self.val_mae.update(preds_clean, labels)

        # Log metrics (torchmetrics handles sync automatically)
        self.log_dict(self.val_metrics, prog_bar=True)
        self.log("val/mae", self.val_mae, prog_bar=True)

    def predict_step(
        self, batch: dict[str, torch.Tensor] | torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        images = batch["image"] if isinstance(batch, dict) else batch
        return self(images)


def get_gt_encoder(
    datamodule: AnimeSegDataModule, opt: Namespace
) -> ISNetDIS | ISNetGTEncoder | InSPyReNet | MODNet | U2Net:
    """Train ground truth encoder for ISNet with intermediate supervision."""
    print("---start train ground truth encoder---")
    gt_encoder = AnimeSegmentation("isnet_gt")
    trainer = Trainer(
        precision=get_precision(opt.fp32, opt.bf16),
        accelerator=opt.accelerator,
        devices=opt.devices,
        max_epochs=opt.gt_epoch,
        benchmark=opt.benchmark,
        accumulate_grad_batches=opt.acc_step,
        check_val_every_n_epoch=opt.val_epoch,
        log_every_n_steps=opt.log_step,
        strategy=get_strategy(opt.devices),
    )
    trainer.fit(gt_encoder, datamodule=datamodule)
    return gt_encoder.net


def main(opt: Namespace) -> None:
    Path("lightning_logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    # Create DataModule
    datamodule = AnimeSegDataModule(
        data_dir=opt.data_dir,
        fg_dir=opt.fg_dir,
        bg_dir=opt.bg_dir,
        img_dir=opt.img_dir,
        mask_dir=opt.mask_dir,
        fg_ext=opt.fg_ext,
        bg_ext=opt.bg_ext,
        img_ext=opt.img_ext,
        mask_ext=opt.mask_ext,
        data_split=opt.data_split,
        img_size=opt.img_size,
        batch_size_train=opt.batch_size_train,
        batch_size_val=opt.batch_size_val,
        num_workers_train=opt.workers_train,
        num_workers_val=opt.workers_val,
        with_trimap=opt.net == "modnet",
        cache_ratio=opt.cache,
        cache_update_epoch=opt.cache_epoch,
    )

    print("---define model---")
    if opt.pretrained_ckpt == "":
        anime_seg = AnimeSegmentation(opt.net, opt.img_size, lr=opt.lr)
    else:
        anime_seg = AnimeSegmentation.try_load(opt.net, opt.pretrained_ckpt, "cpu", opt.img_size)
        anime_seg.hparams["lr"] = opt.lr

    if not opt.pretrained_ckpt and not opt.resume_ckpt and opt.net == "isnet_is":
        assert anime_seg.gt_encoder is not None
        anime_seg.gt_encoder.load_state_dict(get_gt_encoder(datamodule, opt).state_dict())

    # Configure loggers
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="anime_seg")
    csv_logger = CSVLogger(save_dir="lightning_logs", name="anime_seg")

    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            monitor="val/f1",
            mode="max",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
            filename="epoch={epoch:02d}-f1={val/f1:.4f}",
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/f1",
            mode="max",
            patience=5,
            min_delta=0.001,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    print("---start train---")
    trainer = Trainer(
        precision=get_precision(opt.fp32, opt.bf16),
        accelerator=opt.accelerator,
        devices=opt.devices,
        max_epochs=opt.epoch,
        benchmark=opt.benchmark,
        accumulate_grad_batches=opt.acc_step,
        check_val_every_n_epoch=opt.val_epoch,
        log_every_n_steps=opt.log_step,
        strategy=get_strategy(opt.devices),
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
    )
    trainer.fit(anime_seg, datamodule=datamodule, ckpt_path=opt.resume_ckpt or None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument(
        "--net",
        type=str,
        default="isnet_is",
        choices=NET_NAMES,
        help="isnet_is: Train ISNet with intermediate feature supervision, "
        "isnet: Train ISNet, "
        "u2net: Train U2Net full, "
        "u2netl: Train U2Net lite, "
        "modnet: Train MODNet"
        "inspyrnet_res: Train InSPyReNet_Res2Net50"
        "inspyrnet_swin: Train InSPyReNet_SwinB",
    )
    parser.add_argument("--pretrained-ckpt", type=str, default="", help="load form pretrained ckpt")
    parser.add_argument("--resume-ckpt", type=str, default="", help="resume training from ckpt")
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="image size for training and validation,"
        "1024 recommend for ISNet,"
        "384 recommend for InSPyReNet"
        "640 recommend for others,",
    )

    # dataset args
    parser.add_argument(
        "--data-dir", type=str, default="../../dataset/anime-seg", help="root dir of dataset"
    )
    parser.add_argument("--fg-dir", type=str, default="fg", help="relative dir of foreground")
    parser.add_argument("--bg-dir", type=str, default="bg", help="relative dir of background")
    parser.add_argument("--img-dir", type=str, default="imgs", help="relative dir of images")
    parser.add_argument("--mask-dir", type=str, default="masks", help="relative dir of masks")
    parser.add_argument("--fg-ext", type=str, default=".png", help="extension name of foreground")
    parser.add_argument("--bg-ext", type=str, default=".jpg", help="extension name of background")
    parser.add_argument("--img-ext", type=str, default=".jpg", help="extension name of images")
    parser.add_argument("--mask-ext", type=str, default=".jpg", help="extension name of masks")
    parser.add_argument(
        "--data-split", type=float, default=0.95, help="split rate for training and validation"
    )

    # training args
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epoch", type=int, default=40, help="epoch num")
    parser.add_argument(
        "--gt-epoch",
        type=int,
        default=4,
        help="epoch for training ground truth encoder when net is isnet_is",
    )
    parser.add_argument("--batch-size-train", type=int, default=2, help="batch size for training")
    parser.add_argument("--batch-size-val", type=int, default=2, help="batch size for val")
    parser.add_argument(
        "--workers-train", type=int, default=4, help="workers num for training dataloader"
    )
    parser.add_argument(
        "--workers-val", type=int, default=4, help="workers num for validation dataloader"
    )
    parser.add_argument("--acc-step", type=int, default=4, help="gradient accumulation step")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu", "ipu", "hpu", "auto"],
        help="accelerator",
    )
    parser.add_argument("--devices", type=int, default=1, help="devices num")
    parser.add_argument("--fp32", action="store_true", default=False, help="disable mix precision")
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="use bfloat16 mixed precision (for Ampere+ GPUs)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="enable cudnn benchmark (recommended for fixed input sizes)",
    )
    parser.add_argument("--log-step", type=int, default=2, help="log training loss every n steps")
    parser.add_argument("--val-epoch", type=int, default=1, help="valid and save every n epoch")
    parser.add_argument("--cache-epoch", type=int, default=3, help="update cache every n epoch")
    parser.add_argument(
        "--cache",
        type=float,
        default=0,
        help="ratio (cache to entire training dataset), "
        "higher values require more memory, set 0 to disable cache",
    )

    opt = parser.parse_args()
    print(opt)

    main(opt)
