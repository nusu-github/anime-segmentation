"""LightningCLI for BiRefNet training.

Usage:
    # Train with config file
    python -m anime_segmentation.training.train fit --config configs/birefnet.yaml

    # Train with command line overrides
    python -m anime_segmentation.training.train fit \
        --config configs/birefnet.yaml \
        --trainer.max_epochs 200 \
        --data.batch_size 16

    # Validate
    python -m anime_segmentation.training.train validate \
        --config configs/birefnet.yaml \
        --ckpt_path path/to/checkpoint.ckpt

    # Test
    python -m anime_segmentation.training.train test \
        --config configs/birefnet.yaml \
        --ckpt_path path/to/checkpoint.ckpt

    # Predict
    python -m anime_segmentation.training.train predict \
        --config configs/birefnet.yaml \
        --ckpt_path path/to/checkpoint.ckpt
"""

import os

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from .callbacks import (
    BackboneFreezeCallback,
    FinetuneCallback,
    ScheduleFreeCallback,
    VisualizationCallback,
)
from .datamodule import BiRefNetDataModule
from .lightning_module import BiRefNetLightning

# Enable expandable segments for CUDA memory allocation (PyTorch 2.5+)
if tuple(map(int, torch.__version__.split("+")[0].split(".")[:3])) >= (2, 5, 0):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


class BiRefNetCLI(LightningCLI):
    """Custom LightningCLI for BiRefNet training."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments and link arguments between components.

        Args:
            parser: Lightning argument parser.

        """
        # Set default trainer arguments for better debugging/profiling
        parser.set_defaults(
            {
                "trainer.num_sanity_val_steps": 2,
                "trainer.log_every_n_steps": 10,
                "trainer.precision": "16-mixed",
            },
        )

        # Add default callbacks
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                "checkpoint.dirpath": "ckpts/",
                "checkpoint.filename": "birefnet-epoch={epoch:03d}-val_loss={val/loss:.4f}",
                "checkpoint.auto_insert_metric_name": False,
                "checkpoint.save_top_k": 3,
                "checkpoint.monitor": "val/loss",
                "checkpoint.mode": "min",
                "checkpoint.save_last": True,
            },
        )

        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults(
            {
                "lr_monitor.logging_interval": "epoch",
            },
        )

        parser.add_lightning_class_args(FinetuneCallback, "finetune")
        parser.set_defaults(
            {
                "finetune.finetune_last_epochs": -40,
            },
        )

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val/loss",
                "early_stopping.patience": 20,
                "early_stopping.mode": "min",
            },
        )

        parser.add_lightning_class_args(VisualizationCallback, "visualization")
        parser.set_defaults(
            {
                "visualization.num_samples": 4,
                "visualization.log_every_n_epochs": 1,
            },
        )

        parser.add_lightning_class_args(BackboneFreezeCallback, "backbone_freeze")
        # Default to disabled (unfreeze at epoch 0)
        parser.set_defaults(
            {
                "backbone_freeze.unfreeze_at_epoch": 0,
            },
        )

        parser.add_lightning_class_args(GradientAccumulationScheduler, "accumulate_grad")

        parser.add_lightning_class_args(ScheduleFreeCallback, "schedule_free")

    def before_instantiate_classes(self) -> None:
        """Hook called before instantiating classes."""
        # Set random seed if specified
        config = self.config[self.subcommand]  # ty:ignore[invalid-argument-type]
        if hasattr(config, "seed_everything") and config.seed_everything is not None:
            L.seed_everything(config.seed_everything, workers=True)


def main() -> None:
    """Main entry point for CLI."""
    # Set float32 matrix multiplication precision for better performance on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")

    _cli = BiRefNetCLI(
        BiRefNetLightning,
        BiRefNetDataModule,
        seed_everything_default=7,
        auto_configure_optimizers=False,
        parser_kwargs={
            "default_env": True,
        },
    )


if __name__ == "__main__":
    main()
