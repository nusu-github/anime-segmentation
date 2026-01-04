"""Command-line interface for IS-Net training and inference.

This module provides a LightningCLI-based entry point for training, validation,
testing, and inference of ISNet models. It supports YAML configuration files
with command-line overrides for flexible experiment management.

Example usage:
    # Training
    python -m anime_segmentation.train_cli fit --config configs/base.yaml

    # Validation with checkpoint
    python -m anime_segmentation.train_cli validate --config configs/base.yaml --ckpt_path model.ckpt

    # Testing
    python -m anime_segmentation.train_cli test --config configs/base.yaml --ckpt_path model.ckpt

    # Inference
    python -m anime_segmentation.train_cli predict --config configs/base.yaml --ckpt_path model.ckpt

Subcommands:
    fit: Train the model from scratch or resume from checkpoint.
    validate: Run validation loop on validation dataset.
    test: Run test loop for final evaluation.
    predict: Run inference on input data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI

from anime_segmentation.lightning import AnimeSegDataModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import ArgsType

# LightningCLI configuration constants
# Exposed as module-level variables for testing and customization

# Base class for model selection; enables switching between LightningModule subclasses
MODEL_CLASS = LightningModule

# Data module class; single type used across all experiments
DATAMODULE_CLASS = AnimeSegDataModule

# Optimizer configuration is handled by the LightningModule.configure_optimizers method
AUTO_CONFIGURE_OPTIMIZERS = False

# Configuration is saved automatically for experiment reproducibility
SAVE_CONFIG_CALLBACK = True

# Enable model subclass selection via class_path in config
SUBCLASS_MODE_MODEL = True

# Disable data subclass selection; AnimeSegDataModule is used directly
SUBCLASS_MODE_DATA = False

# Available CLI subcommands
SUPPORTED_SUBCOMMANDS = ("fit", "validate", "test", "predict")


def main(args: ArgsType = None) -> None:
    """Entry point for the IS-Net command-line interface.

    Initializes LightningCLI with configured model and data module classes.
    The CLI parses command-line arguments and YAML configuration files to
    set up training, validation, testing, or inference runs.

    Configuration is automatically saved to the output directory for
    experiment reproducibility. Optimizer configuration is delegated to
    the LightningModule's configure_optimizers method.

    Args:
        args: Optional list of strings, dict, or Namespace to override sys.argv.
    """
    LightningCLI(
        model_class=MODEL_CLASS,
        datamodule_class=DATAMODULE_CLASS,
        subclass_mode_model=SUBCLASS_MODE_MODEL,
        subclass_mode_data=SUBCLASS_MODE_DATA,
        auto_configure_optimizers=AUTO_CONFIGURE_OPTIMIZERS,
        args=args,
    )


if __name__ == "__main__":
    main()
