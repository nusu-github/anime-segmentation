import math
from argparse import Namespace
from typing import Any

import lightning.pytorch as pl
import torch
from huggingface_hub import ModelCard, PyTorchModelHubMixin
from lightning import Trainer
from lightning.pytorch.cli import ArgsType, LightningCLI, OptimizerCallable
from lightning.pytorch.strategies import DDPStrategy
from torch.optim import Optimizer
from torchmetrics import MeanAbsoluteError, MetricCollection
from torchmetrics.classification import BinaryFBetaScore, BinaryPrecision, BinaryRecall

from .data_module import AnimeSegDataModule
from .loss import configure_loss_weights
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


def get_net(
    net_name: str, img_size: int | tuple[int, int] | None
) -> ISNetDIS | ISNetGTEncoder | InSPyReNet | MODNet | U2Net:
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
    pipeline_tag="image-segmentation",
    license="apache-2.0",
    tags=["image-segmentation", "anime", "background-removal", "matting"],
):
    def __init__(
        self,
        net_name: str,
        img_size: int | None = None,
        lr: float = 1e-3,
        optimizer: OptimizerCallable = torch.optim.Adam,
        loss_bce: float = 30.0,
        loss_iou: float = 0.5,
        loss_ssim: float = 10.0,
        loss_structure: float = 5.0,
        loss_contour: float = 5.0,
    ) -> None:
        super().__init__()
        assert net_name in NET_NAMES
        self.save_hyperparameters()

        # Configure loss weights before model initialization
        loss_weights = {
            "bce": loss_bce,
            "iou": loss_iou,
            "ssim": loss_ssim,
            "structure": loss_structure,
            "contour": loss_contour,
        }
        configure_loss_weights(loss_weights)

        self.net = get_net(net_name, img_size)
        if self.hparams["net_name"] == "isnet_is":
            self.gt_encoder = get_net("isnet_gt", img_size)
            self.gt_encoder.requires_grad_(requires_grad=False)
        else:
            self.gt_encoder = None

        self.optimizer_factory = optimizer

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
        # If optimizer is passed as class path in CLI, it becomes a partial/callable
        # We invoke it with parameters. `lr` is passed if not already fixed in the callable.
        # But usually we want to respect the `lr` argument of __init__ if provided.
        # If the callable already has 'lr' bound (from config), this might override or duplicate.
        # Typically, config `init_args` are bound. We pass `lr` as override or addition.
        return self.optimizer_factory(self.parameters(), lr=self.hparams["lr"])

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

    def generate_model_card(self, *args, **kwargs) -> ModelCard:
        """Generate model card with training configuration and metrics."""
        card = super().generate_model_card(*args, **kwargs)

        # Add model architecture info to card text
        net_name = self.hparams.get("net_name", "unknown")
        img_size = self.hparams.get("img_size", "unknown")

        card.text += f"""
## Model Details

- **Architecture**: {net_name}
- **Input Size**: {img_size}x{img_size}
- **Task**: Anime character segmentation / background removal

## Usage

```python
from anime_segmentation import AnimeSegmentation

# Load model from Hub
model = AnimeSegmentation.from_pretrained("your-username/model-name")

# Or load from local checkpoint
model = AnimeSegmentation.try_load("{net_name}", "path/to/checkpoint.ckpt")

# Inference
import torch
from PIL import Image
from torchvision.transforms import functional as F

image = Image.open("anime_image.png").convert("RGB")
input_tensor = F.to_tensor(image).unsqueeze(0)

with torch.no_grad():
    mask = model(input_tensor)
```

## Training

Trained using the [anime-segmentation](https://github.com/SkyTNT/anime-segmentation) pipeline.
"""
        return card


def get_gt_encoder(
    datamodule: AnimeSegDataModule, trainer_config: dict[str, Any], gt_epoch: int
) -> torch.nn.Module:
    """Train ground truth encoder for ISNet with intermediate supervision."""
    print("---start train ground truth encoder---")
    gt_encoder = AnimeSegmentation("isnet_gt")

    # Extract relevant trainer args from config
    # We use a simple DDP or auto strategy for this aux task to avoid complex config issues
    devices = trainer_config.get("devices", 1)
    strategy = "auto"
    if isinstance(devices, int) and devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )

    trainer = Trainer(
        precision=trainer_config.get("precision", "16-mixed"),
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=devices,
        max_epochs=gt_epoch,
        benchmark=trainer_config.get("benchmark", True),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 1),
        check_val_every_n_epoch=trainer_config.get("check_val_every_n_epoch", 1),
        log_every_n_steps=trainer_config.get("log_every_n_steps", 50),
        strategy=strategy,
        enable_checkpointing=False,  # Don't save checkpoints for this aux task
        logger=False,  # Don't log this aux task to main logger to avoid confusion
    )
    trainer.fit(gt_encoder, datamodule=datamodule)
    return gt_encoder.net


class AnimeSegmentationCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--gt_epoch",
            type=int,
            default=4,
            help="Epochs for training ground truth encoder (isnet_is only)",
        )

    def before_fit(self):
        # Handle isnet_is pretraining
        # We need to access the parsed config. In before_fit, self.config is available.
        # It is a Namespace or Dict depending on setup. LightningCLI usually provides Namespace.
        config = self.config["fit"]
        model_config = config.get("model", {})
        trainer_config = config.get("trainer", {})

        net_name = model_config.get("net_name")
        ckpt_path = config.get("ckpt_path")

        # Check if we need to train GT encoder
        if net_name == "isnet_is" and not ckpt_path:
            # We assume self.model and self.datamodule are already instantiated by CLI
            # but we need to pass the datamodule to the aux trainer.
            # self.datamodule might be None if datamodule_class wasn't used or something else.
            # But we passed it to CLI, so it should be in self.datamodule.
            if self.datamodule is None:
                raise RuntimeError("DataModule is not instantiated. Cannot train GT encoder.")

            assert self.model.gt_encoder is not None

            # Extract GT epochs from config (added in add_arguments_to_parser)
            gt_epoch = config.get("gt_epoch", 4)

            # We need to convert Namespace to dict if needed, or get attributes
            # trainer_config might be a Namespace or Dict.
            if isinstance(trainer_config, Namespace):
                t_cfg = vars(trainer_config)
            else:
                t_cfg = trainer_config

            trained_net = get_gt_encoder(self.datamodule, t_cfg, gt_epoch)
            self.model.gt_encoder.load_state_dict(trained_net.state_dict())


def cli_main(args: ArgsType = None):
    AnimeSegmentationCLI(
        model_class=AnimeSegmentation,
        datamodule_class=AnimeSegDataModule,
        save_config_kwargs={"overwrite": True},
        args=args,
    )


if __name__ == "__main__":
    cli_main()
