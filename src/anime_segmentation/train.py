from argparse import Namespace
from typing import Any, cast

import lightning.pytorch as pl
import torch
from huggingface_hub import ModelCard, PyTorchModelHubMixin
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import ArgsType, LightningCLI, OptimizerCallable
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer.states import TrainerFn
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from .data_module import AnimeSegDataModule
from .metrics import EMeasure, FMeasure, MAEMetric, SMeasure, WeightedFMeasure
from .model import IBISNet, ISNetDIS, ISNetGTEncoder

NET_NAMES = [
    "ibisnet_is",
    "ibisnet",
    "isnet_is",
    "isnet",
    "isnet_gt",
]


def get_net(
    net_name: str, img_size: int | tuple[int, int] | None
) -> ISNetDIS | ISNetGTEncoder | IBISNet:
    match net_name:
        case "ibisnet" | "ibisnet_is":
            ibis_img_size = img_size[0] if isinstance(img_size, tuple) else img_size
            return IBISNet(img_size=ibis_img_size)
        case "isnet" | "isnet_is":
            return ISNetDIS()
        case "isnet_gt":
            return ISNetGTEncoder()
        case _:
            msg = f"Unknown net_name: {net_name}"
            raise ValueError(msg)


class ScheduleFreeCallback(Callback):
    """Schedule-Free Optimizer Callback for Lightning."""

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[ScheduleFreeCallback] {msg}")

    def _is_schedule_free(self, opt) -> bool:
        """Check if it is a Schedule-Free optimizer with train()/eval() methods"""
        result = hasattr(opt, "train") and hasattr(opt, "eval")
        self._log(f"_is_schedule_free({type(opt).__name__}) = {result}")
        return result

    def _get_train_mode(self, opt) -> bool:
        """Retrieve the current mode (differs between Standard and Wrapper versions)."""
        # Wrapper version: optimizer.train_mode
        if hasattr(opt, "train_mode"):
            return opt.train_mode
        # Standard version: param_groups[0]['train_mode']
        if hasattr(opt, "param_groups") and opt.param_groups:
            return opt.param_groups[0].get("train_mode", False)
        return False

    def _set_mode(self, trainer: "pl.Trainer", mode: str) -> None:
        self._log(f"_set_mode called: mode={mode}, num_optimizers={len(trainer.optimizers)}")
        for i, opt in enumerate(trainer.optimizers):
            if self._is_schedule_free(opt):
                self._log(f"  opt[{i}]: calling {mode}()")
                getattr(opt, mode)()

    # === Training ===
    def on_fit_start(self, trainer, pl_module):  # noqa: ARG002
        self._log("on_fit_start called")
        self._set_mode(trainer, "train")

    def on_train_start(self, trainer, pl_module):  # noqa: ARG002
        self._log("on_train_start called")
        self._set_mode(trainer, "train")

    # === Validation ===
    def on_validation_start(self, trainer, pl_module):  # noqa: ARG002
        self._log(f"on_validation_start called (sanity={trainer.sanity_checking})")
        self._set_mode(trainer, "eval")

    def on_validation_end(self, trainer, pl_module):  # noqa: ARG002
        # During fit, trainer.training is False while validating, so check state.fn instead
        is_fitting = trainer.state.fn == TrainerFn.FITTING
        self._log(f"on_validation_end called (fitting={is_fitting})")
        if is_fitting:
            self._set_mode(trainer, "train")

    # === Test / Predict ===
    def on_test_start(self, trainer, pl_module):  # noqa: ARG002
        self._log("on_test_start called")
        self._set_mode(trainer, "eval")

    def on_predict_start(self, trainer, pl_module):  # noqa: ARG002
        self._log("on_predict_start called")
        self._set_mode(trainer, "eval")

    # === Checkpoint ===
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):  # noqa: ARG002
        """Save the optimizer state in eval mode"""
        new_states = []
        for i, opt in enumerate(trainer.optimizers):
            if self._is_schedule_free(opt):
                was_train = self._get_train_mode(opt)
                if was_train:
                    # Type Safety: switch to eval mode
                    opt.eval()  # pyright: ignore[reportAttributeAccessIssue]
                new_states.append(trainer.strategy.optimizer_state(opt))
                if was_train:
                    # Type Safety: restore to train mode
                    opt.train()  # pyright: ignore[reportAttributeAccessIssue]
            else:
                new_states.append(checkpoint["optimizer_states"][i])
        checkpoint["optimizer_states"] = new_states


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
        loss_bce: float = 1.0,
        loss_iou: float = 0.5,
        loss_ssim: float = 0.5,
        compile: bool = False,
        compile_mode: str | None = None,
    ) -> None:
        super().__init__()
        if net_name not in NET_NAMES:
            msg = f"net_name must be one of {NET_NAMES}, got {net_name}"
            raise ValueError(msg)
        self.save_hyperparameters()

        self.net = get_net(net_name, img_size)

        # Apply loss weights to IBISNet config
        if isinstance(self.net, IBISNet):
            self.net.config.lambda_bce = loss_bce
            self.net.config.lambda_iou = loss_iou
            self.net.config.lambda_ssim = loss_ssim

        # Apply torch.compile to the network if enabled
        if compile:
            self.net = torch.compile(self.net, mode=compile_mode)

        if self.hparams["net_name"] in {"isnet_is", "ibisnet_is"}:
            self.gt_encoder = get_net("isnet_gt", img_size)
            self.freeze_gt_encoder()
        else:
            self.gt_encoder = None

        self.optimizer_factory = optimizer

        # Cache the original (unwrapped) network type for pattern matching
        self._net_unwrapped = (
            self.net._orig_mod if isinstance(self.net, OptimizedModule) else self.net
        )

        # BiRefNet-style validation metrics
        self.val_metrics = MetricCollection(
            {
                "Sm": SMeasure(),
                "Em": EMeasure(),
                "Fm": FMeasure(beta=0.3),
                "wFm": WeightedFMeasure(beta=1.0),
                "MAE": MAEMetric(),
            },
            prefix="val/",
        )

    @classmethod
    def try_load(cls, net_name, ckpt_path, map_location: str | None = None, img_size=None):
        state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if "epoch" in state_dict:
            return cls.load_from_checkpoint(
                ckpt_path, net_name=net_name, img_size=img_size, map_location=map_location
            )
        model = cls(net_name, img_size)
        if any(k.startswith("net.") for k in state_dict):
            model.load_state_dict(state_dict)
        else:
            model.net.load_state_dict(state_dict)
        return model

    def configure_optimizers(self) -> Optimizer:
        lr: float = self.hparams["lr"]
        return self.optimizer_factory(self.parameters(), lr=lr)

    def freeze_gt_encoder(self) -> None:
        if self.gt_encoder is None:
            return
        self.gt_encoder.eval()
        self.gt_encoder.requires_grad_(requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self._net_unwrapped:
            case IBISNet():
                outputs = self.net(x)
                return outputs["ds"][0].sigmoid()
            case ISNetDIS() | ISNetGTEncoder():
                return self.net(x)[0][0].sigmoid()
            case _:
                msg = f"Unsupported network type: {type(self._net_unwrapped)}"
                raise TypeError(msg)

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]

        # Get GT encoder features with inference_mode for performance
        # (teacher model is frozen, no gradients needed)
        gt_features = None
        if self.gt_encoder is not None:
            with torch.inference_mode():
                gt_features = self.gt_encoder(labels)[1]

        match self._net_unwrapped:
            case IBISNet():
                outputs = self.net(images)
                loss_args = [outputs, labels]
                if gt_features is not None:
                    loss_args.append(gt_features)
            case ISNetDIS():
                ds, dfs = self.net(images)
                loss_args = [ds, dfs, labels]
                if gt_features is not None:
                    loss_args.append(gt_features)
            case ISNetGTEncoder():
                ds = self.net(labels)[0]
                loss_args = [ds, labels]
            case _:
                msg = f"Unsupported network type: {type(self._net_unwrapped)}"
                raise TypeError(msg)

        loss_result = self._net_unwrapped.compute_loss(loss_args)

        # Handle IBISNet's extended return value (includes loss_dict)
        if len(loss_result) == 3:
            loss0, loss, loss_dict = loss_result
            for name, val in loss_dict.items():
                self.log(f"train/loss_{name}", val)
        else:
            loss0, loss = loss_result

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_tar", loss0)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]

        match self._net_unwrapped:
            case ISNetGTEncoder():
                preds = self.forward(labels)
            case _:
                preds = self.forward(images)

        # Clean predictions for metric computation
        preds_clean = preds.nan_to_num(nan=0, posinf=1, neginf=0)

        # Update BiRefNet-style metrics (expects batch of images)
        self.val_metrics.update(preds_clean, labels)

        # Log metrics (torchmetrics handles sync automatically)
        self.log_dict(self.val_metrics, prog_bar=True)

    def predict_step(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        images = batch["image"] if isinstance(batch, dict) else batch
        return self(images)

    def generate_model_card(self, *args, **kwargs) -> ModelCard:
        """Generate model card with training configuration and metrics."""
        card = super().generate_model_card(*args, **kwargs)

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
    datamodule: AnimeSegDataModule,
    trainer_config: dict[str, Any],
    gt_epoch: int,
    compile: bool = False,
    compile_mode: str | None = None,
) -> ISNetGTEncoder:
    """Train ground truth encoder for ISNet with intermediate supervision."""
    print("---start train ground truth encoder---")
    gt_encoder = AnimeSegmentation("isnet_gt", compile=compile, compile_mode=compile_mode)

    # Extract relevant trainer args from config
    devices = trainer_config.get("devices", 1)
    strategy = "auto"
    if isinstance(devices, int) and devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )

    trainer = Trainer(
        callbacks=[ScheduleFreeCallback()],
        precision=trainer_config.get("precision", "16-mixed"),
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=devices,
        max_epochs=gt_epoch,
        benchmark=trainer_config.get("benchmark", True),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 1),
        check_val_every_n_epoch=trainer_config.get("check_val_every_n_epoch", 1),
        log_every_n_steps=trainer_config.get("log_every_n_steps", 50),
        strategy=strategy,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(gt_encoder, datamodule=datamodule)
    # Unwrap compiled module to get clean state_dict without _orig_mod prefix
    net = gt_encoder.net
    if isinstance(net, OptimizedModule):
        net = net._orig_mod
    return cast("ISNetGTEncoder", net)


class AnimeSegmentationCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--gt_epoch",
            type=int,
            default=4,
            help="Epochs for training ground truth encoder (isnet_is/ibisnet_is only)",
        )

    def before_fit(self):
        config = self.config["fit"]
        model_config = config.get("model", {})
        trainer_config = config.get("trainer", {})

        net_name = model_config.get("net_name")
        ckpt_path = config.get("ckpt_path")

        # Check if we need to train GT encoder
        if net_name in {"isnet_is", "ibisnet_is"} and not ckpt_path:
            if self.datamodule is None:
                raise RuntimeError("DataModule is not instantiated. Cannot train GT encoder.")

            assert self.model.gt_encoder is not None

            gt_epoch = config.get("gt_epoch", 4)

            if isinstance(trainer_config, Namespace):
                t_cfg = vars(trainer_config)
            else:
                t_cfg = trainer_config

            compile_enabled = model_config.get("compile", False)
            compile_mode = model_config.get("compile_mode", None)

            trained_net = get_gt_encoder(
                self.datamodule, t_cfg, gt_epoch, compile_enabled, compile_mode
            )
            self.model.gt_encoder.load_state_dict(trained_net.state_dict())


def cli_main(args: ArgsType = None):
    AnimeSegmentationCLI(
        model_class=AnimeSegmentation,
        datamodule_class=AnimeSegDataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "callbacks": [ScheduleFreeCallback(debug=True)],
        },
        args=args,
    )


if __name__ == "__main__":
    cli_main()
