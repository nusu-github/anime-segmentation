"""PyTorch Lightning modules for IS-Net model training and inference.

This module provides LightningCLI-compatible training wrappers that handle:

- Model forward passes and loss computation
- GPU-based augmentation via KorniaAugmentationPipeline
- Metric computation during validation
- Optimizer and scheduler configuration via dependency injection
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from anime_segmentation.augmentation import KorniaAugmentationPipeline
from anime_segmentation.metrics import SegmentationMetrics

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from anime_segmentation.data_loader import AugmentationConfig

# torch.compile mode options
CompileMode = Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]

ERR_GT_ENCODER_MISSING = "gt_encoder is required when interm_sup is enabled."


class ISNetLightningModule(LightningModule):
    """CLI-compatible training wrapper for ISNet segmentation models.

    This module accepts pre-instantiated model instances via dependency injection,
    enabling configuration through LightningCLI YAML files with class_path/init_args patterns.

    GPU augmentation is enabled by default using KorniaAugmentationPipeline.

    Args:
        model: Main model instance (e.g., ISNetDIS). Injected via YAML class_path.
        gt_encoder: GT encoder instance. Injected via YAML class_path (optional).
            Required when interm_sup=True.
        interm_sup: Enable intermediate supervision with GT encoder.
        fs_loss_mode: Feature matching loss mode for intermediate supervision.
        optimizer: Optimizer callable for configure_optimizers. Configured via
            YAML class_path/init_args pattern (e.g., torch.optim.AdamW with lr, betas).
        scheduler: Optional LR scheduler callable. Configured via YAML class_path.
        metric_names: Metrics to compute during validation.
        enable_metrics: Whether to compute metrics during validation.
        aug_config: Augmentation config. If None, uses default training augmentation.
        normalize_mean: RGB normalization mean for augmentation pipeline.
        normalize_std: RGB normalization std for augmentation pipeline.
        compile_model: Enable torch.compile for model acceleration.
        compile_mode: Compilation mode. Options:
            - "default": Balance between performance and compile time.
            - "reduce-overhead": Reduces Python/CUDA overhead via CUDA graphs.
            - "max-autotune": Maximum optimization with Triton autotuning.
            - "max-autotune-no-cudagraphs": Max-autotune without CUDA graphs.
        compile_fullgraph: If True, requires the entire model to compile as one graph.
        compile_dynamic: Enable dynamic shape support. None uses heuristics.
        compile_backend: Compilation backend (default: "inductor").

    Raises:
        ValueError: If interm_sup=True but gt_encoder is None.

    Note:
        Optimizer parameters (lr, betas, weight_decay) are configured in YAML under
        `model.optimizer.init_args`, NOT as separate module parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        gt_encoder: nn.Module | None = None,
        interm_sup: bool = False,
        fs_loss_mode: Literal["MSE", "KL", "MAE", "SmoothL1"] = "MSE",
        optimizer: OptimizerCallable = torch.optim.Adam,  # type: ignore[assignment]
        scheduler: LRSchedulerCallable | None = None,
        metric_names: Iterable[str] = ("F", "WF", "MAE", "S", "HCE", "MBA", "BIoU"),
        enable_metrics: bool = True,
        aug_config: AugmentationConfig | None = None,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        compile_model: bool = False,
        compile_mode: CompileMode = "default",
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = None,
        compile_backend: str = "inductor",
    ) -> None:
        super().__init__()

        # Validate gt_encoder requirement for intermediate supervision
        if interm_sup and gt_encoder is None:
            raise ValueError(ERR_GT_ENCODER_MISSING)

        self.save_hyperparameters(ignore=["model", "gt_encoder", "optimizer", "scheduler"])

        self.interm_sup = interm_sup
        self.fs_loss_mode = fs_loss_mode

        # Apply torch.compile if enabled
        if compile_model:
            self.model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
                backend=compile_backend,
            )
            if gt_encoder is not None:
                self.gt_encoder = torch.compile(
                    gt_encoder,
                    mode=compile_mode,
                    fullgraph=compile_fullgraph,
                    dynamic=compile_dynamic,
                    backend=compile_backend,
                )
            else:
                self.gt_encoder = None
        else:
            self.model = model
            self.gt_encoder = gt_encoder
        self._optimizer_callable = optimizer
        self._scheduler_callable = scheduler
        self.enable_metrics = enable_metrics

        # Initialize Kornia augmentation pipeline (default enabled)
        from anime_segmentation.data_loader import AugmentationConfig as AugConfig

        if aug_config is None:
            aug_config = AugConfig.training_default()
        self.kornia_aug = KorniaAugmentationPipeline(
            aug_config,
            normalize_mean=list(normalize_mean),
            normalize_std=list(normalize_std),
        )

        if self.gt_encoder is not None:
            self.gt_encoder.eval()
            for param in self.gt_encoder.parameters():
                param.requires_grad = False

        self._val_metrics: list[SegmentationMetrics] = []
        self._metric_names = tuple(metric_names)

    @property
    def optimizer(self) -> OptimizerCallable:
        """Return the optimizer callable for use in configure_optimizers."""
        return self._optimizer_callable

    @property
    def scheduler(self) -> LRSchedulerCallable | None:
        """Return the LR scheduler callable, if configured."""
        return self._scheduler_callable

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass through the segmentation model.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Model output (predictions and intermediate features).
        """
        return self.model(x)

    def _compute_loss(
        self, preds: list[torch.Tensor], labels: torch.Tensor, dfs: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute segmentation loss with optional intermediate supervision.

        Args:
            preds: List of prediction tensors from multi-scale outputs.
            labels: Ground truth segmentation masks.
            dfs: Deep feature tensors for intermediate supervision (optional).

        Returns:
            Tuple of (target_loss, total_loss) where target_loss is the primary
            segmentation loss and total_loss includes intermediate supervision.

        Raises:
            RuntimeError: If intermediate supervision is enabled but gt_encoder is None.
        """
        if self.interm_sup:
            if self.gt_encoder is None:
                raise RuntimeError(ERR_GT_ENCODER_MISSING)
            with torch.no_grad():
                _, fs = self.gt_encoder(labels)
            loss2, loss = self.model.compute_loss_kl(  # type: ignore[attr-defined]
                preds, labels, dfs, fs, mode=self.fs_loss_mode
            )
        else:
            loss2, loss = self.model.compute_loss(preds, labels)  # type: ignore[attr-defined]
        return loss2, loss

    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Applies GPU augmentation to inputs, performs forward pass, computes loss,
        and logs training metrics.

        Args:
            batch: Dictionary containing 'image' and 'label' tensors.
            _batch_idx: Index of the current batch (unused).

        Returns:
            Total training loss for backpropagation.
        """
        images = batch["image"]
        labels = batch["label"]

        with torch.no_grad():
            images, labels = self.kornia_aug(images, labels)

        preds, dfs = self.model(images)
        loss2, loss = self._compute_loss(preds, labels, dfs=dfs)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_target", loss2, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset metric accumulators at the start of each validation epoch."""
        self._val_metrics = []

    def validation_step(
        self, batch: dict[str, Any], _batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Execute a single validation step.

        Computes loss and updates segmentation metrics. Predictions are resized
        to original image dimensions when shape information is available.

        Args:
            batch: Dictionary containing 'image', 'label', and optionally 'shape'.
            _batch_idx: Index of the current batch (unused).
            dataloader_idx: Index of the dataloader for multi-dataset validation.
        """
        images = batch["image"]
        labels = batch["label"]
        shapes = batch.get("shape")

        preds, dfs = self.model(images)
        loss2, loss = self._compute_loss(preds, labels, dfs=dfs)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
        )
        self.log(
            "val_loss_target",
            loss2,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=True,
        )

        if not self.enable_metrics:
            return

        if len(self._val_metrics) <= dataloader_idx:
            self._val_metrics.append(
                SegmentationMetrics(metrics=self._metric_names).to(self.device)
            )

        metrics = self._val_metrics[dataloader_idx]
        d1 = torch.sigmoid(preds[0])

        for i in range(d1.shape[0]):
            pred = d1[i : i + 1]
            if shapes is not None:
                height = int(shapes[i][0].item())
                width = int(shapes[i][1].item())
                pred = F.interpolate(
                    pred, size=(height, width), mode="bilinear", align_corners=True
                )
                gt = F.interpolate(
                    labels[i : i + 1],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                gt = labels[i : i + 1]

            pred_scaled = pred * 255.0
            gt_scaled = gt * 255.0
            metrics.update(pred_scaled, gt_scaled)

    def on_validation_epoch_end(self) -> None:
        """Log aggregated metrics at the end of each validation epoch."""
        if not self.enable_metrics:
            return
        for idx, metrics in enumerate(self._val_metrics):
            results = metrics.get_results()
            prefix = f"val{idx}_" if len(self._val_metrics) > 1 else "val_"
            for key, value in results.items():
                self.log(
                    f"{prefix}{key}",
                    value,
                    on_epoch=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )

    def predict_step(
        self, batch: dict[str, Any], _batch_idx: int, _dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Execute a single prediction step.

        Args:
            batch: Dictionary containing 'image' and optional metadata.
            _batch_idx: Index of the current batch (unused).
            _dataloader_idx: Index of the dataloader (unused).

        Returns:
            Dictionary with prediction tensor and batch metadata for post-processing.
        """
        images = batch["image"]
        preds, _dfs = self.model(images)
        pred = torch.sigmoid(preds[0])

        return {
            "pred": pred,
            "shape": batch.get("shape"),
            "imidx": batch.get("imidx"),
            "path": batch.get("path"),
        }

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and optional scheduler using injected callables.

        Returns:
            Optimizer instance if scheduler is None, otherwise dict with optimizer
            and lr_scheduler keys.
        """
        optimizer = self._optimizer_callable(self.parameters())

        if self._scheduler_callable is None:
            return optimizer

        scheduler = self._scheduler_callable(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class GTEncoderLightningModule(LightningModule):
    """CLI-compatible training wrapper for GT encoder models.

    This module accepts pre-instantiated model instances via dependency injection,
    enabling configuration through LightningCLI YAML files.

    Args:
        model: GT encoder model instance (e.g., ISNetGTEncoder). Injected via YAML class_path.
        optimizer: Optimizer callable. Configured via YAML class_path/init_args pattern.
        scheduler: Optional LR scheduler callable. Configured via YAML class_path.
        compile_model: Enable torch.compile for model acceleration.
        compile_mode: Compilation mode. Options:
            - "default": Balance between performance and compile time.
            - "reduce-overhead": Reduces Python/CUDA overhead via CUDA graphs.
            - "max-autotune": Maximum optimization with Triton autotuning.
            - "max-autotune-no-cudagraphs": Max-autotune without CUDA graphs.
        compile_fullgraph: If True, requires the entire model to compile as one graph.
        compile_dynamic: Enable dynamic shape support. None uses heuristics.
        compile_backend: Compilation backend (default: "inductor").

    Note:
        Optimizer parameters configured in YAML under `model.optimizer.init_args`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizerCallable = torch.optim.Adam,  # type: ignore[assignment]
        scheduler: LRSchedulerCallable | None = None,
        compile_model: bool = False,
        compile_mode: CompileMode = "default",
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = None,
        compile_backend: str = "inductor",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler"])

        # Apply torch.compile if enabled
        if compile_model:
            self.model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
                backend=compile_backend,
            )
        else:
            self.model = model

        self._optimizer_callable = optimizer
        self._scheduler_callable = scheduler

    @property
    def optimizer(self) -> OptimizerCallable:
        """Return the optimizer callable for use in configure_optimizers."""
        return self._optimizer_callable

    @property
    def scheduler(self) -> LRSchedulerCallable | None:
        """Return the LR scheduler callable, if configured."""
        return self._scheduler_callable

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass through the GT encoder model.

        Args:
            x: Input tensor (ground truth mask) of shape (N, 1, H, W).

        Returns:
            Model output (reconstructed mask and intermediate features).
        """
        return self.model(x)

    def training_step(self, batch: dict[str, Any] | list, _batch_idx: int) -> torch.Tensor:
        """Execute a single training step for GT encoder.

        The GT encoder learns to reconstruct ground truth masks, enabling
        feature extraction for intermediate supervision in the main model.

        Args:
            batch: Dictionary or list of dictionaries containing 'label' tensor.
            _batch_idx: Index of the current batch (unused).

        Returns:
            Total training loss for backpropagation.
        """
        if isinstance(batch, list):
            batch = batch[0]
        labels = batch["label"]
        preds, _fs = self.model(labels)
        loss2, loss = self.model.compute_loss(preds, labels)  # type: ignore[attr-defined]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_target", loss2, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self, batch: dict[str, Any] | list, _batch_idx: int, _dataloader_idx: int = 0
    ) -> None:
        """Execute a single validation step for GT encoder.

        Args:
            batch: Dictionary or list of dictionaries containing 'label' tensor.
            _batch_idx: Index of the current batch (unused).
            _dataloader_idx: Index of the dataloader (unused).
        """
        if isinstance(batch, list):
            batch = batch[0]
        labels = batch["label"]
        preds, _fs = self.model(labels)
        loss2, loss = self.model.compute_loss(preds, labels)  # type: ignore[attr-defined]
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=True,
        )
        self.log(
            "val_loss_target",
            loss2,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and optional scheduler using injected callables.

        Returns:
            Optimizer instance if scheduler is None, otherwise dict with optimizer
            and lr_scheduler keys.
        """
        optimizer = self._optimizer_callable(self.parameters())

        if self._scheduler_callable is None:
            return optimizer

        scheduler = self._scheduler_callable(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
