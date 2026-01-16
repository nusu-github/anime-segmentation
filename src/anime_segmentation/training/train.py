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
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from .callbacks import ScheduleFreeCallback
from .datamodule import AnimeSegmentationDataModule
from .lightning_module import BiRefNetLightning

# Register AnimeSegmentationDataModule for CLI discovery
__all__ = ["AnimeSegmentationDataModule", "BiRefNetLightning"]

# Enable expandable segments for CUDA memory allocation (PyTorch 2.5+)
if tuple(map(int, torch.__version__.split("+")[0].split(".")[:3])) >= (2, 5, 0):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


class BiRefNetCLI(LightningCLI):
    """Custom LightningCLI for BiRefNet training."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add callback arguments to the parser.

        Only transparent callbacks that have no effect when unused are registered here.
        Other callbacks should be configured via trainer.callbacks in config files.

        Args:
            parser: Lightning argument parser.

        """
        parser.add_lightning_class_args(ScheduleFreeCallback, "schedule_free")


def _configure_cuda_backends() -> None:
    """Configure CUDA backends for optimal performance and memory efficiency.

    Settings applied:
    - TF32 for faster matrix operations on Ampere+ GPUs
    - cuDNN benchmark mode for optimized convolution algorithms
    - SDPA backends optimized for memory efficiency (flash/mem_efficient preferred)
    - BF16 reduced precision for mixed-precision training
    """
    # Float32 matrix multiplication precision
    torch.set_float32_matmul_precision("high")

    # TF32 for Ampere+ GPUs (faster matmul with minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN optimization (beneficial for fixed input sizes)
    torch.backends.cudnn.benchmark = True

    # SDPA (Scaled Dot-Product Attention) memory optimization
    # Prioritize memory-efficient backends, disable math backend (highest VRAM usage)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    # Allow reduced precision accumulation for BF16/FP16 mixed precision training
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


def main() -> None:
    """Main entry point for CLI."""
    _configure_cuda_backends()

    _cli = BiRefNetCLI(
        BiRefNetLightning,
        L.LightningDataModule,
        seed_everything_default=7,
        auto_configure_optimizers=False,
        subclass_mode_data=True,
        parser_kwargs={
            "default_env": True,
            "default_config_files": ["configs/default.yaml"],
        },
    )


if __name__ == "__main__":
    main()
