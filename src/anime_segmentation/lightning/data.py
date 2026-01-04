"""Lightning DataModule for IS-Net training and evaluation pipelines.

This module provides a LightningCLI-compatible data module that handles dataset
loading, normalization, and dataloader creation for all training stages.

Augmentation is intentionally NOT applied in the DataModule. Instead, GPU-based
augmentation is handled by :class:`KorniaAugmentationPipeline` in the LightningModule
to maximize throughput and utilize GPU parallelism.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from lightning import LightningDataModule

from anime_segmentation.data_loader import (
    AugmentationConfig,
    GOSNormalize,
    create_dataloaders,
    get_im_gt_name_dict,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset


ERR_TRAIN_DATASETS_REQUIRED_FIT = "train_datasets is required for fit stage."
ERR_VAL_DATASETS_REQUIRED_FIT = "val_datasets is required for fit stage."
ERR_VAL_DATASETS_REQUIRED_VALIDATE = "val_datasets is required for validate stage."
ERR_TEST_DATASETS_REQUIRED = "test_datasets must be configured for test stage."
ERR_PREDICT_DATASETS_REQUIRED = "predict_datasets must be configured for predict stage."
ERR_IM_DIR_NOT_FOUND = "Image directory not found for dataset '{name}': {path}"


def _resolve_augmentation_preset(
    preset: Literal["default", "light", "aggressive", "hflip_only", "none"],
) -> AugmentationConfig:
    """Map an augmentation preset name to its corresponding configuration.

    This function translates user-friendly preset names into fully configured
    AugmentationConfig instances for use with the Kornia augmentation pipeline.

    Args:
        preset: Name of the augmentation preset. Options are:
            - ``"default"``: Standard training augmentation with moderate intensity.
            - ``"light"``: Minimal augmentation for datasets with limited variation.
            - ``"aggressive"``: Strong augmentation for maximum data diversity.
            - ``"hflip_only"``: Horizontal flip only, useful for debugging.
            - ``"none"``: No augmentation (used for validation/test).

    Returns:
        Configured AugmentationConfig instance. Falls back to ``training_default()``
        for unrecognized preset names.
    """
    match preset:
        case "default":
            return AugmentationConfig.training_default()
        case "light":
            return AugmentationConfig.light()
        case "aggressive":
            return AugmentationConfig.aggressive()
        case "hflip_only":
            return AugmentationConfig.hflip_only()
        case "none":
            return AugmentationConfig.validation_default()
        case _:
            return AugmentationConfig.training_default()


class AnimeSegDataModule(LightningDataModule):
    """CLI-compatible data pipeline for IS-Net training.

    This module accepts all configuration as explicit keyword arguments,
    enabling configuration through LightningCLI YAML files.

    Augmentation is handled on GPU via KorniaAugmentationPipeline in the
    LightningModule. This DataModule provides the aug_config property to
    pass the resolved augmentation configuration.

    Args:
        train_datasets: List of training dataset dicts with keys:
            name, im_dir, gt_dir, im_ext, gt_ext
        val_datasets: List of validation dataset dicts (same format).
        test_datasets: List of test dataset dicts.
        predict_datasets: List of dataset dicts for inference.
        image_size: Target image size as (height, width).
        batch_size_train: Training batch size.
        batch_size_valid: Validation batch size.
        normalize_mean: RGB normalization mean.
        normalize_std: RGB normalization std.
        aug_preset_train: Augmentation preset for training ("default", "light",
            "aggressive", "none").
        num_workers: DataLoader workers.

    Raises:
        ValueError: If required datasets are not configured for the requested stage.
        FileNotFoundError: If dataset im_dir does not exist.
    """

    def __init__(
        self,
        train_datasets: list[dict[str, Any]],
        val_datasets: list[dict[str, Any]],
        test_datasets: list[dict[str, Any]] | None = None,
        predict_datasets: list[dict[str, Any]] | None = None,
        image_size: tuple[int, int] = (1024, 1024),
        batch_size_train: int = 4,
        batch_size_valid: int = 1,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        aug_preset_train: Literal[
            "default", "light", "aggressive", "hflip_only", "none"
        ] = "default",
        num_workers: int = 4,
    ) -> None:
        if test_datasets is None:
            test_datasets = []
        if predict_datasets is None:
            predict_datasets = []
        super().__init__()

        self._train_datasets_config = train_datasets
        self._val_datasets_config = val_datasets
        self._test_datasets_config = test_datasets
        self._predict_datasets_config = predict_datasets

        self._image_size = list(image_size)

        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid

        self._normalize_mean = list(normalize_mean)
        self._normalize_std = list(normalize_std)

        self.num_workers = num_workers

        self._resolved_aug_train = _resolve_augmentation_preset(aug_preset_train)

        self._train_dataloaders: list[DataLoader] | None = None
        self._val_dataloaders: list[DataLoader] | None = None
        self._test_dataloaders: list[DataLoader] | None = None
        self._predict_dataloaders: list[DataLoader] | None = None
        self._train_datasets: list[Dataset] | None = None
        self._val_datasets: list[Dataset] | None = None
        self._test_datasets_obj: list[Dataset] | None = None
        self._predict_datasets_obj: list[Dataset] | None = None

    def _validate_datasets(
        self, datasets: list[dict[str, Any]], stage: str, *, require_gt: bool = True
    ) -> None:
        """Validate dataset configurations and verify that directories exist.

        Args:
            datasets: List of dataset configuration dictionaries. Each dict must
                contain at minimum ``name`` and ``im_dir`` keys.
            stage: Stage name (e.g., "train", "validation") for error messages.
            require_gt: Whether ground truth directories are required. Set to
                ``False`` for prediction stage where labels may not exist.

        Raises:
            FileNotFoundError: If the image directory does not exist.

        Note:
            Ground truth directory validation is currently lenient to allow
            flexibility during validation when gt_dir may be optional.
        """
        for dataset in datasets:
            im_dir = Path(dataset["im_dir"])
            if not im_dir.exists():
                raise FileNotFoundError(
                    ERR_IM_DIR_NOT_FOUND.format(name=dataset["name"], path=im_dir)
                )

    def _setup_train_dataloader(self) -> None:
        """Create training dataloaders from configured datasets.

        No transforms are applied here because augmentation is performed on GPU
        via KorniaAugmentationPipeline in the LightningModule for better throughput.
        """
        train_nm_im_gt_list = get_im_gt_name_dict(self._train_datasets_config, flag="train")

        self._train_dataloaders, self._train_datasets = create_dataloaders(
            train_nm_im_gt_list,
            image_size=self._image_size,
            my_transforms=None,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def _setup_val_dataloader(self) -> None:
        """Create validation dataloaders with normalization transforms applied."""
        val_nm_im_gt_list = get_im_gt_name_dict(self._val_datasets_config, flag="valid")
        val_transforms = [GOSNormalize(self._normalize_mean, self._normalize_std)]

        self._val_dataloaders, self._val_datasets = create_dataloaders(
            val_nm_im_gt_list,
            image_size=self._image_size,
            my_transforms=val_transforms,
            batch_size=self.batch_size_valid,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _setup_test_dataloader(self) -> None:
        """Create test dataloaders with normalization transforms applied."""
        test_nm_im_gt_list = get_im_gt_name_dict(self._test_datasets_config, flag="test")
        test_transforms = [GOSNormalize(self._normalize_mean, self._normalize_std)]

        self._test_dataloaders, self._test_datasets_obj = create_dataloaders(
            test_nm_im_gt_list,
            image_size=self._image_size,
            my_transforms=test_transforms,
            batch_size=self.batch_size_valid,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _setup_predict_dataloader(self) -> None:
        """Create prediction dataloaders with normalization transforms applied."""
        predict_nm_im_gt_list = get_im_gt_name_dict(self._predict_datasets_config, flag="predict")
        predict_transforms = [GOSNormalize(self._normalize_mean, self._normalize_std)]

        self._predict_dataloaders, self._predict_datasets_obj = create_dataloaders(
            predict_nm_im_gt_list,
            image_size=self._image_size,
            my_transforms=predict_transforms,
            batch_size=self.batch_size_valid,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def setup(self, stage: str | None = None) -> None:
        """Configure dataloaders for the specified training stage.

        This method is called by Lightning before each stage begins. It validates
        that required datasets are configured and creates the appropriate dataloaders.

        Args:
            stage: Lightning stage name. One of ``"fit"``, ``"validate"``,
                ``"test"``, ``"predict"``, or ``None`` (defaults to fit behavior).

        Raises:
            ValueError: If required datasets are not configured for the stage.
            FileNotFoundError: If dataset directories do not exist.
        """
        match stage:
            case "fit":
                if not self._train_datasets_config:
                    raise ValueError(ERR_TRAIN_DATASETS_REQUIRED_FIT)
                if not self._val_datasets_config:
                    raise ValueError(ERR_VAL_DATASETS_REQUIRED_FIT)
                self._validate_datasets(self._train_datasets_config, "train")
                self._validate_datasets(self._val_datasets_config, "validation")
                self._setup_train_dataloader()
                self._setup_val_dataloader()

            case "validate":
                if not self._val_datasets_config:
                    raise ValueError(ERR_VAL_DATASETS_REQUIRED_VALIDATE)
                self._validate_datasets(self._val_datasets_config, "validation")
                if self._val_dataloaders is None:
                    self._setup_val_dataloader()

            case "test":
                if not self._test_datasets_config:
                    raise ValueError(ERR_TEST_DATASETS_REQUIRED)
                self._validate_datasets(self._test_datasets_config, "test")
                self._setup_test_dataloader()

            case "predict":
                if not self._predict_datasets_config:
                    raise ValueError(ERR_PREDICT_DATASETS_REQUIRED)
                self._validate_datasets(self._predict_datasets_config, "predict", require_gt=False)
                self._setup_predict_dataloader()

            case None:
                if self._train_datasets_config and self._val_datasets_config:
                    self._validate_datasets(self._train_datasets_config, "train")
                    self._validate_datasets(self._val_datasets_config, "validation")
                    self._setup_train_dataloader()
                    self._setup_val_dataloader()

    def train_dataloader(self) -> list[DataLoader] | None:
        """Return training dataloaders."""
        return self._train_dataloaders

    def val_dataloader(self) -> list[DataLoader] | None:
        """Return validation dataloaders."""
        return self._val_dataloaders

    def test_dataloader(self) -> list[DataLoader] | None:
        """Return test dataloaders."""
        return self._test_dataloaders

    def predict_dataloader(self) -> list[DataLoader] | None:
        """Return prediction dataloaders."""
        return self._predict_dataloaders

    @property
    def train_datasets(self) -> list[Dataset] | None:
        """Return the underlying training Dataset objects."""
        return self._train_datasets

    @property
    def val_datasets(self) -> list[Dataset] | None:
        """Return the underlying validation Dataset objects."""
        return self._val_datasets

    @property
    def aug_config(self) -> AugmentationConfig:
        """Return the resolved augmentation configuration for GPU augmentation.

        This config is passed to the LightningModule's KorniaAugmentationPipeline.
        """
        return self._resolved_aug_train

    @property
    def normalize_mean(self) -> list[float]:
        """Return RGB normalization mean values (ImageNet defaults)."""
        return self._normalize_mean

    @property
    def normalize_std(self) -> list[float]:
        """Return RGB normalization standard deviation values (ImageNet defaults)."""
        return self._normalize_std
