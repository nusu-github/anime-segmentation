from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import lightning as L
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from einops import rearrange, repeat
from huggingface_hub import HfApi
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.trainer.states import TrainerFn

from anime_segmentation.constants import IMAGENET_MEAN, IMAGENET_STD

from .protocols import Finetunable, HasBackbone

if TYPE_CHECKING:
    from .lightning_module import BiRefNetLightning

logger = logging.getLogger(__name__)


class FinetuneCallback(L.Callback):
    """Callback for adjusting loss weights in final epochs.

    This implements BiRefNet's fine-tuning strategy where loss weights
    are adjusted during the final epochs of training to focus on
    perceptual quality metrics.
    """

    def __init__(self, finetune_last_epochs: int = -40) -> None:
        """Initialize finetune callback.

        Args:
            finetune_last_epochs: Number of epochs before end to start finetuning.
                Negative value means epochs from the end (e.g., -40 = last 40 epochs).
                Set to 0 to disable.

        """
        super().__init__()
        self.finetune_last_epochs = finetune_last_epochs
        self._finetuning_started = False

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Adjust loss weights at the start of each epoch.

        Args:
            trainer: Lightning trainer.
            pl_module: BiRefNet Lightning module with enter_finetune_phase method.

        """
        if self.finetune_last_epochs == 0 or trainer.max_epochs is None:
            return

        epoch = trainer.current_epoch + 1
        threshold = trainer.max_epochs + self.finetune_last_epochs

        if epoch > threshold and not self._finetuning_started:
            self._finetuning_started = True

            # Type-safe delegation using Protocol
            if isinstance(pl_module, Finetunable):
                pl_module.enter_finetune_phase()
                logger.info("Entered finetune phase at epoch %d", epoch)


class BackboneFreezeCallback(BaseFinetuning):
    """Callback for freezing/unfreezing backbone during training.

    Starts with a frozen backbone and gradually unfreezes it
    after a specified number of epochs.
    """

    def __init__(
        self,
        unfreeze_at_epoch: int = 10,
        backbone_lr_scale: float = 0.1,
    ) -> None:
        """Initialize backbone freeze callback.

        Args:
            unfreeze_at_epoch: Epoch at which to unfreeze the backbone.
            backbone_lr_scale: Learning rate multiplier for backbone parameters
                when unfreezing (relative to main learning rate).

        """
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.backbone_lr_scale = backbone_lr_scale

    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        """Freeze backbone before training starts.

        Args:
            pl_module: BiRefNet Lightning module implementing HasBackbone protocol.

        """
        # Skip if unfreezing is disabled (unfreeze_at_epoch=0 means no freezing)
        if self.unfreeze_at_epoch == 0:
            return

        if isinstance(pl_module, HasBackbone):
            # Ensure model is initialized before accessing backbone
            if hasattr(pl_module, "configure_model"):
                pl_module.configure_model()
            self.freeze(pl_module.backbone)

    def finetune_function(
        self,
        pl_module: L.LightningModule,
        epoch: int,
        optimizer,
    ) -> None:
        """Unfreeze backbone at the specified epoch.

        Args:
            pl_module: BiRefNet Lightning module implementing HasBackbone protocol.
            epoch: Current training epoch.
            optimizer: Optimizer instance.

        """
        if epoch != self.unfreeze_at_epoch:
            return

        if not isinstance(pl_module, HasBackbone):
            return

        # Get current learning rate from optimizer
        current_lr = optimizer.param_groups[0]["lr"]
        backbone_lr = current_lr * self.backbone_lr_scale

        # Unfreeze and add to optimizer with lower learning rate
        self.unfreeze_and_add_param_group(
            modules=pl_module.backbone,
            optimizer=optimizer,
            lr=backbone_lr,
        )


class VisualizationCallback(L.Callback):
    """Callback for visualizing predictions during validation.

    Logs a grid of images (input, GT, prediction) to the logger.
    Memory-optimized with optional downsampling for large images.
    """

    def __init__(
        self,
        num_samples: int = 4,
        log_every_n_epochs: int = 1,
        max_resolution: int | None = 512,
    ) -> None:
        """Initialize visualization callback.

        Args:
            num_samples: Number of samples to visualize.
            log_every_n_epochs: Frequency of logging (in epochs).
            max_resolution: Max resolution for visualization. None to keep original.

        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.max_resolution = max_resolution
        self._val_inputs: torch.Tensor | None = None
        self._val_gts: torch.Tensor | None = None

    def _should_log(self, trainer: L.Trainer) -> bool:
        """Check if visualization should be logged this epoch."""
        return trainer.current_epoch % self.log_every_n_epochs == 0

    def _downsample_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Downsample tensor if larger than max_resolution."""
        if self.max_resolution is None:
            return tensor
        _, _, h, w = tensor.shape
        if max(h, w) <= self.max_resolution:
            return tensor
        scale = self.max_resolution / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )

    def _cleanup(self) -> None:
        """Explicitly cleanup captured samples to free memory."""
        self._val_inputs = None
        self._val_gts = None

    def on_validation_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Reset state at the start of each validation epoch."""
        self._cleanup()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Capture samples from the first validation batch.

        Args:
            trainer: Lightning trainer.
            pl_module: BiRefNet Lightning module.
            outputs: Outputs from validation_step.
            batch: Input batch.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.

        """
        if not self._should_log(trainer):
            return

        if batch_idx == 0 and self._val_inputs is None:
            inputs, gts, _ = batch
            num_to_capture = min(self.num_samples, inputs.size(0))

            # Downsample for memory efficiency
            inputs_small = self._downsample_if_needed(inputs[:num_to_capture])
            gts_small = self._downsample_if_needed(gts[:num_to_capture])

            # Store detached CPU copies
            self._val_inputs = inputs_small.detach().cpu()
            self._val_gts = gts_small.detach().cpu()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log visualizations at the end of validation epoch.

        Args:
            trainer: Lightning trainer.
            pl_module: BiRefNet Lightning module.

        """
        if not self._should_log(trainer) or self._val_inputs is None:
            return

        try:
            self._generate_and_log_visualization(trainer, pl_module)
        finally:
            # Always cleanup, even on error
            self._cleanup()

    def _generate_and_log_visualization(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Generate predictions and log visualization grid."""
        inputs_captured = self._val_inputs
        gts_captured = self._val_gts

        if inputs_captured is None or gts_captured is None:
            return

        # Generate predictions with memory-efficient context
        with torch.no_grad():
            inputs_device = inputs_captured.to(pl_module.device)
            preds = pl_module(inputs_device)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            preds = preds.sigmoid().cpu()
            del inputs_device  # Free GPU memory immediately

        # Denormalize inputs for visualization
        mean = rearrange(torch.tensor(IMAGENET_MEAN), "c -> 1 c 1 1")
        std = rearrange(torch.tensor(IMAGENET_STD), "c -> 1 c 1 1")
        inputs_viz = torch.clamp(inputs_captured * std + mean, 0, 1)

        # Create visualization grid: grayscale to RGB by repeating channel
        gts_rgb = repeat(gts_captured, "b 1 h w -> b 3 h w")
        preds_rgb = repeat(preds, "b 1 h w -> b 3 h w")
        # Concatenate input, ground truth, prediction along width
        viz_batch = torch.cat([inputs_viz, gts_rgb, preds_rgb], dim=3)
        viz_list = list(viz_batch)

        grid = vutils.make_grid(viz_list, nrow=1, padding=2)

        # Log to available loggers
        self._log_grid(trainer, grid)

    def _log_grid(self, trainer: L.Trainer, grid: torch.Tensor) -> None:
        """Log visualization grid to the configured logger."""
        viz_logger = trainer.logger
        if viz_logger is None:
            return

        if hasattr(viz_logger, "log_image"):
            # Wandb, Comet, etc.
            viz_logger.log_image(
                key="val_samples",
                images=[grid],
                caption=[f"Epoch {trainer.current_epoch}"],
            )
        elif hasattr(viz_logger, "experiment") and hasattr(viz_logger.experiment, "add_image"):
            # TensorBoard
            viz_logger.experiment.add_image(
                "val_samples",
                grid,
                global_step=trainer.global_step,
            )
        # CSVLogger doesn't support images, so we skip


class ScheduleFreeCallback(L.Callback):
    """Schedule-Free Optimizer Callback for Lightning."""

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[ScheduleFreeCallback] {msg}")

    def _is_schedule_free(self, opt) -> bool:
        """Check if it is a Schedule-Free optimizer with train()/eval() methods."""
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

    def _set_mode(self, trainer: L.Trainer, mode: str) -> None:
        self._log(f"_set_mode called: mode={mode}, num_optimizers={len(trainer.optimizers)}")
        for i, opt in enumerate(trainer.optimizers):
            if self._is_schedule_free(opt):
                self._log(f"  opt[{i}]: calling {mode}()")
                getattr(opt, mode)()

    # === Training ===
    def on_fit_start(self, trainer, pl_module) -> None:
        self._log("on_fit_start called")
        self._set_mode(trainer, "train")

    def on_train_start(self, trainer, pl_module) -> None:
        self._log("on_train_start called")
        self._set_mode(trainer, "train")

    # === Validation ===
    def on_validation_start(self, trainer, pl_module) -> None:
        self._log(f"on_validation_start called (sanity={trainer.sanity_checking})")
        self._set_mode(trainer, "eval")

    def on_validation_end(self, trainer, pl_module) -> None:
        # During fit, trainer.training is False while validating, so check state.fn instead
        is_fitting = trainer.state.fn == TrainerFn.FITTING
        self._log(f"on_validation_end called (fitting={is_fitting})")
        if is_fitting:
            self._set_mode(trainer, "train")

    # === Test / Predict ===
    def on_test_start(self, trainer, pl_module) -> None:
        self._log("on_test_start called")
        self._set_mode(trainer, "eval")

    def on_predict_start(self, trainer, pl_module) -> None:
        self._log("on_predict_start called")
        self._set_mode(trainer, "eval")

    # === Checkpoint ===
    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        """Save the optimizer state in eval mode."""
        new_states = []
        for i, opt in enumerate(trainer.optimizers):
            if self._is_schedule_free(opt):
                was_train = self._get_train_mode(opt)
                if was_train:
                    # Type Safety: switch to eval mode
                    opt.eval()
                new_states.append(trainer.strategy.optimizer_state(opt))
                if was_train:
                    # Type Safety: restore to train mode
                    opt.train()
            else:
                new_states.append(checkpoint["optimizer_states"][i])
        checkpoint["optimizer_states"] = new_states


class HubUploadCallback(L.Callback):
    """Upload model checkpoints to Hugging Face Hub during training.

    This callback uploads model weights to the Hub at configurable intervals:
    - On validation metric improvement (best model)
    - Every N epochs
    - At the end of training

    The model must inherit from PyTorchModelHubMixin (like AnimeSegmentation).

    Example config:
        callbacks:
          - class_path: anime_segmentation.train.HubUploadCallback
            init_args:
              repo_id: "username/anime-segmentation-model"
              upload_best: true
              upload_every_n_epochs: 5
              monitor: "val/Sm"
              mode: "max"
    """

    def __init__(
        self,
        repo_id: str,
        *,
        upload_best: bool = True,
        upload_last: bool = True,
        upload_every_n_epochs: int | None = None,
        monitor: str = "val/loss",
        mode: str = "min",
        private: bool = False,
        branch: str | None = None,
        token: str | None = None,
        blocking: bool = False,
    ) -> None:
        """Initialize the Hub upload callback.

        Args:
            repo_id: Hugging Face Hub repository ID (e.g., "username/model-name").
            upload_best: Upload when monitored metric improves.
            upload_last: Upload at the end of training.
            upload_every_n_epochs: Upload every N epochs. None to disable.
            monitor: Metric to monitor for best model uploads.
            mode: "min" or "max" - whether lower or higher metric is better.
            private: Whether to create a private repository.
            branch: Git branch to upload to. None for main branch.
            token: HuggingFace token. None to use cached token.
            blocking: If False, uploads run in background without blocking training.

        """
        super().__init__()
        self.repo_id = repo_id
        self.upload_best = upload_best
        self.upload_last = upload_last
        self.upload_every_n_epochs = upload_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self.private = private
        self.branch = branch
        self.token = token
        self.blocking = blocking

        self.best_score: float | None = None
        self.api = HfApi(token=token)
        self._repo_created = False

    def _is_improvement(self, current: float) -> bool:
        """Check if current score is an improvement over best."""
        if self.best_score is None:
            return True
        match self.mode:
            case "min":
                return current < self.best_score
            case _:
                return current > self.best_score

    def _should_upload_best(self, trainer: L.Trainer) -> bool:
        """Check if best model should be uploaded based on current state."""
        return self.upload_best and not trainer.sanity_checking

    def _should_upload_periodic(self, trainer: L.Trainer) -> bool:
        """Check if periodic upload should occur at current epoch."""
        if self.upload_every_n_epochs is None:
            return False
        epoch = trainer.current_epoch + 1
        return epoch % self.upload_every_n_epochs == 0

    def _ensure_repo_exists(self) -> None:
        """Create the repository if it doesn't exist."""
        if self._repo_created:
            return
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            private=self.private,
            exist_ok=True,
        )
        self._repo_created = True

    def _upload_model(
        self,
        trainer: L.Trainer,
        pl_module: BiRefNetLightning,
        commit_message: str,
    ) -> None:
        """Upload model to Hub with comprehensive metadata."""
        # Only upload from rank 0 in distributed training
        if trainer.global_rank != 0:
            return

        self._ensure_repo_exists()

        # Save model to temporary directory and upload
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)

            # Build comprehensive config from hyperparameters
            config = {
                # Model architecture
                "bb_name": pl_module.hparams.get("bb_name"),
                "bb_pretrained": pl_module.hparams.get("bb_pretrained"),
                "out_ref": pl_module.hparams.get("out_ref"),
                "ms_supervision": pl_module.hparams.get("ms_supervision"),
                # Loss weights
                "lambdas_pix": pl_module.hparams.get("lambdas_pix"),
                "lambdas_cls": pl_module.hparams.get("lambdas_cls"),
                # Compile settings (for reproducibility)
                "compile": pl_module.hparams.get("compile"),
                "compile_mode": pl_module.hparams.get("compile_mode"),
            }

            # Build model card kwargs with dynamic information
            model_card_kwargs: dict = {
                "model_name": f"Anime Segmentation - {pl_module.hparams.get('bb_name', 'BiRefNet')}",
                "repo_id": self.repo_id,
            }

            # Add metrics if available
            metrics: dict[str, str] = {}
            for key in ["val/loss", "val/IoU", "val/Sm", "val/MAE", "val/Em", "val/Fm"]:
                if key in trainer.callback_metrics:
                    metrics[key] = f"{float(trainer.callback_metrics[key]):.4f}"
            if metrics:
                model_card_kwargs["metrics"] = metrics

            # Use PyTorchModelHubMixin's save_pretrained
            pl_module.save_pretrained(
                save_path,
                config=config,
                model_card_kwargs=model_card_kwargs,
            )

            # Upload folder to Hub
            if self.blocking:
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(save_path),
                    commit_message=commit_message,
                    revision=self.branch,
                )
            else:
                # Non-blocking upload
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(save_path),
                    commit_message=commit_message,
                    revision=self.branch,
                    run_as_future=True,
                )

    def _get_monitored_value(self, trainer: L.Trainer) -> float | None:
        """Get the monitored metric value from trainer's callback metrics.

        Args:
            trainer: Lightning trainer.

        Returns:
            Metric value or None if not available.

        """
        callback_metrics = trainer.callback_metrics

        if self.monitor not in callback_metrics:
            logger.warning(
                "Monitored metric '%s' not found in callback_metrics. Available: %s",
                self.monitor,
                list(callback_metrics.keys()),
            )
            return None

        value = callback_metrics[self.monitor]
        return float(value.item() if hasattr(value, "item") else value)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Upload model if validation metric improved.

        Args:
            trainer: Lightning trainer.
            pl_module: BiRefNet Lightning module.

        """
        if not self._should_upload_best(trainer):
            return

        current_score = self._get_monitored_value(trainer)
        if current_score is None:
            return

        if self._is_improvement(current_score):
            self.best_score = current_score
            commit_message = (
                f"Best model (epoch {trainer.current_epoch}, {self.monitor}={current_score:.4f})"
            )
            self._upload_model(trainer, pl_module, commit_message)
            logger.info(
                "Uploaded best model: %s=%.4f",
                self.monitor,
                current_score,
            )

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Upload model at epoch intervals if configured.

        Args:
            trainer: Lightning trainer.
            pl_module: BiRefNet Lightning module.

        """
        if not self._should_upload_periodic(trainer):
            return

        epoch = trainer.current_epoch + 1
        commit_message = f"Checkpoint at epoch {epoch}"
        self._upload_model(trainer, pl_module, commit_message)
        logger.info("Uploaded periodic checkpoint at epoch %d", epoch)

    def on_fit_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Upload final model at end of training if configured.

        Args:
            trainer: Lightning trainer.
            pl_module: BiRefNet Lightning module.

        """
        if not self.upload_last:
            return

        commit_message = f"Final model (epoch {trainer.current_epoch})"
        self._upload_model(trainer, pl_module, commit_message)
        logger.info("Uploaded final model")
