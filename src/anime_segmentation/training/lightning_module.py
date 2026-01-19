from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

import lightning as L
import torch
from huggingface_hub import ModelCard, PyTorchModelHubMixin, constants
from huggingface_hub.file_download import hf_hub_download
from lightning.pytorch.utilities import grad_norm
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import nn

from anime_segmentation.models import BiRefNet

from .config import (
    BackboneConfig,
    ClassificationLossConfig,
    DecoderConfig,
    LossConfig,
)
from .loss import ClsLoss, PixLoss
from .metrics import SegmentationMetrics
from .model_card_template import MODEL_CARD_TEMPLATE

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

# torch.compile modes
CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]


class BiRefNetLightning(
    L.LightningModule,
    PyTorchModelHubMixin,
    library_name="anime_segmentation",
    repo_url="https://github.com/nusu-github/anime-segmentation",
    pipeline_tag="image-segmentation",
    license="apache-2.0",
    tags=["image-segmentation", "background-removal", "anime", "birefnet", "pytorch"],
    model_card_template=MODEL_CARD_TEMPLATE,
):
    def __init__(
        self,
        backbone: BackboneConfig | None = None,
        decoder: DecoderConfig | None = None,
        loss: LossConfig | None = None,
        cls_loss: ClassificationLossConfig | None = None,
        auxiliary_classification: bool = False,
        num_classes: int | None = None,
        strict_loading: bool = True,
        compile: bool = False,
        compile_mode: CompileMode = "default",
        optimizer: OptimizerCallable = torch.optim.AdamW,
        scheduler: LRSchedulerCallable | None = None,
        scheduler_interval: Literal["epoch", "step"] = "epoch",
    ) -> None:
        """Initialize the BiRefNet Lightning module."""
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])
        self.optimizer_callable = optimizer
        self.scheduler_callable = scheduler

        self.backbone_cfg = backbone if backbone is not None else BackboneConfig()
        self.decoder_cfg = decoder if decoder is not None else DecoderConfig()
        self.loss_cfg = loss if loss is not None else LossConfig()
        self.cls_loss_cfg = cls_loss if cls_loss is not None else ClassificationLossConfig()
        self.auxiliary_classification = auxiliary_classification
        self.num_classes = num_classes
        self.strict_loading = strict_loading
        self._compile = compile
        self._compile_mode = compile_mode

        self.model: BiRefNet | None = None
        self.example_input_array: torch.Tensor | None = None

        pix_weights = {
            "bce": self.loss_cfg.bce,
            "iou": self.loss_cfg.iou,
            "iou_patch": self.loss_cfg.iou_patch,
            "mae": self.loss_cfg.mae,
            "mse": self.loss_cfg.mse,
            "reg": self.loss_cfg.reg,
            "ssim": self.loss_cfg.ssim,
            "cnt": self.loss_cfg.cnt,
            "structure": self.loss_cfg.structure,
        }
        self.lambdas_pix = pix_weights
        self.pix_loss = PixLoss(loss_weights=pix_weights)

        cls_weights = {"ce": self.cls_loss_cfg.ce}
        self.lambdas_cls = cls_weights
        self.cls_loss = ClsLoss(lambdas_cls=cls_weights)

        self.criterion_gdt: nn.BCEWithLogitsLoss | None = (
            nn.BCEWithLogitsLoss() if self.decoder_cfg.out_ref else None
        )

        self.val_metrics = SegmentationMetrics(prefix="val/")
        self.test_metrics = SegmentationMetrics(prefix="test/")

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _require_model(self) -> BiRefNet:
        """Ensure model is initialized and return it.

        Returns:
            The initialized BiRefNet model.

        Raises:
            RuntimeError: If model is not initialized.

        """
        if self.model is None:
            msg = "Model not initialized. Call configure_model() first."
            raise RuntimeError(msg)
        return self.model

    # =========================================================================
    # Properties for type-safe access (Protocol support)
    # =========================================================================

    @property
    def backbone(self) -> nn.Module:
        """Return the backbone module for freezing callbacks.

        Implements HasBackbone protocol from callbacks module.

        Raises:
            RuntimeError: If model is not initialized.

        """
        return self._require_model().bb

    @property
    def out_ref(self) -> bool:
        """Whether output refinement is enabled."""
        return self.decoder_cfg.out_ref

    @property
    def ms_supervision(self) -> bool:
        """Whether multi-scale supervision is enabled."""
        return self.decoder_cfg.ms_supervision

    # =========================================================================
    # Core methods
    # =========================================================================

    def enter_finetune_phase(self) -> None:
        """Adjust loss weights for the fine-tuning phase.

        This method adjusts loss weights to focus on perceptual quality metrics
        (SSIM, MAE) over pixel-level accuracy (BCE). Called by FinetuneCallback
        when entering the final training epochs.
        """
        self.pix_loss.lambdas["bce"] = 0.0
        self.pix_loss.lambdas["ssim"] *= 2.0
        self.pix_loss.lambdas["iou"] *= 0.5
        if "mae" in self.pix_loss.lambdas:
            self.pix_loss.lambdas["mae"] *= 0.9

    def configure_model(self) -> None:
        """Initialize and optionally compile the model."""
        if self.model is not None:
            return

        self.model = BiRefNet(
            bb_name=self.backbone_cfg.name,
            bb_pretrained=self.backbone_cfg.pretrained,
            ms_supervision=self.decoder_cfg.ms_supervision,
            out_ref=self.decoder_cfg.out_ref,
            dec_ipt=self.decoder_cfg.dec_ipt,
            dec_ipt_split=self.decoder_cfg.dec_ipt_split,
            cxt_num=self.decoder_cfg.cxt_num,
            mul_scl_ipt=self.decoder_cfg.mul_scl_ipt,
            dec_att=self.decoder_cfg.dec_att,
            squeeze_block=self.decoder_cfg.squeeze_block,
            dec_blk=self.decoder_cfg.dec_blk,
            lat_blk=self.decoder_cfg.lat_blk,
            dec_channels_inter=self.decoder_cfg.dec_channels_inter,
            use_norm=self.decoder_cfg.use_norm,
            auxiliary_classification=self.auxiliary_classification,
            num_classes=self.num_classes,
        )

        if self._compile:
            self.model = torch.compile(self.model, mode=self._compile_mode)

        self.example_input_array = torch.randn(1, 3, 1024, 1024)

    def forward(self, x):
        return self._require_model()(x)

    def training_step(self, batch, batch_idx):
        """Training step with modular loss computation.

        Args:
            batch: Tuple of (inputs, gts, class_labels).
            batch_idx: Batch index.

        Returns:
            Total loss for backpropagation.

        """
        inputs, gts, class_labels = batch

        # Forward pass
        scaled_preds, class_preds_lst = self._forward_train(inputs)

        # Compute all losses
        losses = self._compute_training_losses(
            inputs=inputs,
            gts=gts,
            class_labels=class_labels,
            scaled_preds=scaled_preds,
            class_preds_lst=class_preds_lst,
        )

        # Log metrics
        self._log_training_metrics(losses)

        return losses["total"]

    def _forward_train(self, inputs: torch.Tensor) -> tuple:
        """Execute forward pass during training.

        Args:
            inputs: Input tensor [B, 3, H, W].

        Returns:
            Tuple of (scaled_preds, class_preds_lst).

        Raises:
            RuntimeError: If model is not initialized.

        """
        model = self._require_model()
        with self.trainer.profiler.profile("model_forward"):  # ty:ignore[unresolved-attribute]
            return model(inputs)

    def _compute_training_losses(
        self,
        inputs: torch.Tensor,
        gts: torch.Tensor,
        class_labels: torch.Tensor,
        scaled_preds: tuple,
        class_preds_lst: list,
    ) -> dict[str, torch.Tensor | dict]:
        """Compute all training losses.

        Args:
            inputs: Input tensor.
            gts: Ground truth tensor.
            class_labels: Class labels tensor.
            scaled_preds: Model predictions at multiple scales.
            class_preds_lst: Class predictions list.

        Returns:
            Dictionary containing total, pix, cls, gdt losses and pix_details.

        """
        with self.trainer.profiler.profile("loss_calculation"):  # ty:ignore[unresolved-attribute]
            # Gradient Loss (if out_ref is enabled)
            loss_gdt = inputs.new_zeros(())
            if self.out_ref:
                (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
                loss_gdt = self._calculate_gdt_loss(outs_gdt_pred, outs_gdt_label)

            # Classification Loss
            loss_cls = self._compute_cls_loss(class_preds_lst, class_labels, inputs.device)

            # Pixel Loss
            loss_pix, loss_dict_pix = self.pix_loss(
                scaled_preds,
                gts,
                pix_loss_lambda=1.0,
            )

            # Total Loss
            total = loss_pix + loss_cls + loss_gdt

        return {
            "total": total,
            "pix": loss_pix,
            "cls": loss_cls,
            "gdt": loss_gdt,
            "pix_details": loss_dict_pix,
        }

    def _compute_cls_loss(
        self,
        class_preds_lst: list,
        class_labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute classification loss if predictions are available.

        Args:
            class_preds_lst: List of class predictions.
            class_labels: Ground truth class labels.
            device: Device for tensor creation.

        Returns:
            Classification loss tensor.

        """
        if None in class_preds_lst:
            return torch.zeros((), device=device)
        return self.cls_loss(class_preds_lst, class_labels)

    def _log_training_metrics(self, losses: dict) -> None:
        """Log training metrics to the configured logger.

        Args:
            losses: Dictionary containing loss values.

        """
        # Main loss with progress bar
        self.log(
            "train/loss",
            losses["total"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Build detailed log dict
        train_logs: dict[str, torch.Tensor] = {"train/loss_pix": losses["pix"]}

        if losses["cls"] > 0:
            train_logs["train/loss_cls"] = losses["cls"]
        if self.out_ref:
            train_logs["train/loss_gdt"] = losses["gdt"]

        # Per-component pixel losses
        for k, v in losses["pix_details"].items():
            train_logs[f"train/pix_{k}"] = v

        self.log_dict(train_logs, on_step=False, on_epoch=True)

    def on_before_optimizer_step(self, optimizer) -> None:
        """Compute and log gradient norms.

        This helps in detecting exploding gradients early.
        """
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:  # ty:ignore[unresolved-attribute]
            norms = grad_norm(self.model, norm_type=2)  # ty:ignore[invalid-argument-type]
            self.log_dict(norms)

    def _calculate_gdt_loss(self, outs_gdt_pred, outs_gdt_label) -> torch.Tensor:
        """Calculate gradient loss for multiple scales.

        Args:
            outs_gdt_pred: Predicted gradients.
            outs_gdt_label: Target gradients.

        Returns:
            Mean gradient loss.

        """
        if not outs_gdt_pred:
            return torch.tensor(0.0, device=self.device)

        # Compute losses for each scale and stack
        losses = []
        for gdt_pred, gdt_label in zip(outs_gdt_pred, outs_gdt_label, strict=False):
            # Resize prediction to match label size
            if gdt_pred.shape[2:] != gdt_label.shape[2:]:
                gdt_pred = nn.functional.interpolate(
                    gdt_pred,
                    size=gdt_label.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            # BCEWithLogitsLoss expects logits for pred, probabilities for target
            losses.append(self.criterion_gdt(gdt_pred, gdt_label.sigmoid()))

        # Stack and compute mean efficiently
        return torch.stack(losses).mean()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        The optimizer and scheduler are configured via the LightningCLI using
        OptimizerCallable and LRSchedulerCallable type hints. This allows any
        optimizer or scheduler to be used simply by specifying it in the YAML
        config file.

        Raises:
            RuntimeError: If model is not initialized.

        """
        model = self._require_model()

        # Create optimizer using the callable
        optimizer = self.optimizer_callable(model.parameters())

        # Return optimizer only if no scheduler is configured
        if self.scheduler_callable is None:
            return optimizer

        # Create scheduler using the callable
        scheduler = self.scheduler_callable(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams["scheduler_interval"],
            },
        }

    # =========================================================================
    # Hub integration methods (PyTorchModelHubMixin overrides)
    # =========================================================================

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save BiRefNet model weights to the specified directory.

        This override ensures that only the inner BiRefNet model weights are saved,
        not the full Lightning module (loss functions, metrics, etc.).

        Args:
            save_directory: Directory path to save the model.

        """
        # Ensure model is initialized
        if self.model is None:
            self.configure_model()

        # Model must be initialized after configure_model()
        model = self._require_model()

        # Save only the BiRefNet model weights
        save_model_as_safetensor(
            model,
            str(save_directory / constants.SAFETENSORS_SINGLE_FILE),
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs: Any,
    ) -> BiRefNetLightning:
        """Load BiRefNetLightning from pretrained weights.

        This override handles the lazy model initialization and ensures weights
        are loaded into self.model (not the full Lightning module).

        Args:
            model_id: Model identifier (Hub repo ID or local path).
            revision: Model revision to load.
            cache_dir: Directory for cached downloads.
            force_download: Force re-download even if cached.
            local_files_only: Only use local files, no downloads.
            token: HuggingFace token for private repos.
            map_location: Device to load weights to.
            strict: Strict weight loading mode.
            **model_kwargs: Additional arguments for model initialization.

        Returns:
            Initialized BiRefNetLightning instance with loaded weights.

        """
        # Create instance with hyperparameters from config
        # The parent class already populates model_kwargs from config.json
        instance = cls(**model_kwargs)

        # Initialize the model (would normally be done by trainer)
        instance.configure_model()

        # Determine model file path
        model_file = (
            os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            if os.path.isdir(model_id)
            else hf_hub_download(
                repo_id=model_id,
                filename=constants.SAFETENSORS_SINGLE_FILE,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
        )

        # Load weights into the inner BiRefNet model
        # Use _require_model() for type safety
        model = instance._require_model()
        load_model_as_safetensor(
            model,
            model_file,
            strict=strict,
            device=map_location,
        )

        # Set to eval mode
        instance.eval()

        return instance

    def generate_model_card(self, **kwargs: Any) -> ModelCard:
        """Generate a model card with dynamic training information.

        This override uses a custom template and includes training configuration
        and metrics when available.

        Args:
            **kwargs: Additional template variables.

        Returns:
            ModelCard instance with populated content.

        """
        # Build training config dict from hyperparameters
        training_config: dict[str, Any] = {
            "Backbone": self.backbone_cfg.name,
            "Multi-scale Supervision": self.decoder_cfg.ms_supervision,
            "Output Refinement": self.decoder_cfg.out_ref,
        }

        # Add loss weights if available
        if hasattr(self, "lambdas_pix") and self.lambdas_pix:
            training_config["BCE Loss Weight"] = self.lambdas_pix.get("bce", 0)
            training_config["SSIM Loss Weight"] = self.lambdas_pix.get("ssim", 0)
            training_config["IoU Loss Weight"] = self.lambdas_pix.get("iou", 0)

        # Collect metrics if available (from trainer)
        metrics: dict[str, str] = {}
        if hasattr(self, "trainer") and self.trainer is not None:
            callback_metrics = getattr(self.trainer, "callback_metrics", {})
            for key in ["val/loss", "val/IoU", "val/Sm", "val/MAE", "val/Em", "val/Fm"]:
                if key in callback_metrics:
                    value = callback_metrics[key]
                    metrics[key] = f"{float(value):.4f}"

        # Use custom template via parent class
        return ModelCard.from_template(
            card_data=self._hub_mixin_info.model_card_data,
            template_str=self._hub_mixin_info.model_card_template,
            repo_url=self._hub_mixin_info.repo_url,
            paper_url=self._hub_mixin_info.paper_url or "https://arxiv.org/abs/2401.03407",
            docs_url=self._hub_mixin_info.docs_url,
            # Custom template variables
            backbone_name=self.backbone_cfg.name,
            ms_supervision=self.decoder_cfg.ms_supervision,
            out_ref=self.decoder_cfg.out_ref,
            training_config=training_config or None,
            metrics=metrics or None,
            **kwargs,
        )

    # =========================================================================
    # Evaluation methods (validation, test, predict)
    # =========================================================================

    def _shared_eval_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        compute_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Shared evaluation logic for validation and test steps.

        Args:
            batch: Tuple of (inputs, gts, class_labels).
            compute_loss: Whether to compute pixel loss.

        Returns:
            Tuple of (pred, loss, gts) where loss is None if compute_loss is False.

        Raises:
            RuntimeError: If model is not initialized.

        """
        inputs, gts, _ = batch
        model = self._require_model()

        # Forward pass (in eval mode, model returns only scaled_preds)
        scaled_preds = model(inputs)

        # Get final prediction with sigmoid
        pred = scaled_preds[-1].sigmoid()

        # Optionally compute loss
        loss = None
        if compute_loss:
            loss, _ = self.pix_loss(
                scaled_preds,
                gts,
                pix_loss_lambda=1.0,
            )

        return pred, loss, gts

    def validation_step(self, batch, batch_idx):
        """Validation step with loss computation and metrics update.

        Args:
            batch: Tuple of (inputs, gts, class_labels).
            batch_idx: Batch index.

        Returns:
            Validation loss.

        """
        pred, loss, gts = self._shared_eval_step(batch, compute_loss=True)

        # Update metrics
        self.val_metrics.update(pred, gts)

        # Log loss and metrics
        self.log(
            "val/loss",
            loss,  # ty:ignore[invalid-argument-type]
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx) -> None:
        """Test step with metrics update only.

        Args:
            batch: Tuple of (inputs, gts, class_labels).
            batch_idx: Batch index.

        """
        pred, _, gts = self._shared_eval_step(batch, compute_loss=False)

        # Update metrics
        self.test_metrics.update(pred, gts)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        """Prediction step.

        Args:
            batch: Input tensor [B, 3, H, W] or tuple with inputs as first element.
            batch_idx: Batch index.

        Returns:
            Prediction tensor [B, 1, H, W] with values in [0, 1].

        Raises:
            RuntimeError: If model is not initialized.

        """
        # Handle both tensor input and tuple input
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        model = self._require_model()

        # In eval mode, model returns only scaled_preds (no class_preds)
        scaled_preds = model(inputs)
        return scaled_preds[-1].sigmoid()
