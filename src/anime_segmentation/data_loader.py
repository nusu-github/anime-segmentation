"""Data loading utilities for image segmentation datasets.

This module provides dataset classes and data loading utilities for training
and evaluating salient object detection models. It supports on-the-fly image
loading, preprocessing, and configurable data augmentation.

Key components:
    - AugmentationConfig: Dataclass for configuring augmentation parameters.
    - GOSDataset: PyTorch Dataset for loading image-mask pairs.
    - GOSNormalize: Transform for ImageNet-style normalization.
    - create_dataloaders: Factory function for creating DataLoader instances.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import normalize, to_dtype


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline.

    This dataclass defines all parameters for the augmentation pipeline used
    during training. Augmentations are organized into groups (geometric, color,
    blur, cutout) that can be enabled/disabled independently.

    Each augmentation has a probability parameter controlling how often it is
    applied, and additional parameters defining the augmentation strength/range.

    Attributes:
        enable_geometric: Master switch for geometric augmentations.
        enable_color: Master switch for color/intensity augmentations.
        enable_blur: Master switch for blur augmentations.
        enable_cutout: Master switch for cutout/erasing augmentation.
        horizontal_flip_prob: Probability of horizontal flip.
        vertical_flip_prob: Probability of vertical flip.
        rotation_prob: Probability of rotation.
        rotation_angle_range: Maximum rotation angle in degrees (symmetric).
        scale_jitter_prob: Probability of scale jitter.
        scale_range: Scale factor range as (min_scale, max_scale).
        shear_prob: Probability of shear transform.
        shear_range: Maximum shear factor (symmetric).
        translate_prob: Probability of translation.
        translate_range: Maximum translation as fraction of image size.
        brightness_prob: Probability of brightness adjustment.
        brightness_range: Brightness factor range as (min, max).
        contrast_prob: Probability of contrast adjustment.
        contrast_range: Contrast factor range as (min, max).
        saturation_prob: Probability of saturation adjustment.
        saturation_range: Saturation factor range as (min, max).
        hue_prob: Probability of hue adjustment.
        hue_range: Hue shift range in degrees.
        gaussian_blur_prob: Probability of Gaussian blur.
        gaussian_sigma_range: Sigma range for Gaussian blur.
        motion_blur_prob: Probability of motion blur.
        motion_blur_kernel_range: Kernel size range for motion blur.
        grayscale_prob: Probability of converting to grayscale.
        cutout_prob: Probability of cutout.
        cutout_num_regions: Number of cutout regions.
        cutout_size_range: Size range of cutout as fraction of image size.
    """

    # Master switches for augmentation groups
    enable_geometric: bool = False
    enable_color: bool = False
    enable_blur: bool = False
    enable_cutout: bool = False

    # Flip augmentations
    horizontal_flip_prob: float = 0.0
    vertical_flip_prob: float = 0.0

    # Rotation augmentation
    rotation_prob: float = 0.0
    rotation_angle_range: float = 15.0

    # Scale jitter augmentation
    scale_jitter_prob: float = 0.0
    scale_range: tuple[float, float] = (0.8, 1.2)

    # Shear augmentation
    shear_prob: float = 0.0
    shear_range: float = 0.1

    # Translation augmentation
    translate_prob: float = 0.0
    translate_range: float = 0.1

    # Brightness augmentation
    brightness_prob: float = 0.0
    brightness_range: tuple[float, float] = (0.8, 1.2)

    # Contrast augmentation
    contrast_prob: float = 0.0
    contrast_range: tuple[float, float] = (0.8, 1.2)

    # Saturation augmentation
    saturation_prob: float = 0.0
    saturation_range: tuple[float, float] = (0.8, 1.2)

    # Hue augmentation
    hue_prob: float = 0.0
    hue_range: float = 18.0

    # Gaussian blur augmentation
    gaussian_blur_prob: float = 0.0
    gaussian_sigma_range: tuple[float, float] = (0.1, 2.0)

    # Motion blur augmentation
    motion_blur_prob: float = 0.0
    motion_blur_kernel_range: tuple[int, int] = (3, 7)

    # Grayscale augmentation
    grayscale_prob: float = 0.0

    # Cutout/random erasing augmentation
    cutout_prob: float = 0.0
    cutout_num_regions: int = 1
    cutout_size_range: tuple[float, float] = (0.02, 0.1)

    @classmethod
    def training_default(cls) -> AugmentationConfig:
        """Create default training configuration with balanced augmentation.

        Returns:
            AugmentationConfig with moderate augmentation strength suitable
            for most training scenarios.
        """
        return cls(
            enable_geometric=True,
            enable_color=True,
            enable_blur=True,
            enable_cutout=True,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.0,
            rotation_prob=0.3,
            rotation_angle_range=15.0,
            scale_jitter_prob=0.3,
            scale_range=(0.8, 1.2),
            shear_prob=0.2,
            shear_range=0.1,
            translate_prob=0.2,
            translate_range=0.1,
            brightness_prob=0.3,
            brightness_range=(0.8, 1.2),
            contrast_prob=0.3,
            contrast_range=(0.8, 1.2),
            saturation_prob=0.3,
            saturation_range=(0.8, 1.2),
            hue_prob=0.2,
            hue_range=18.0,
            gaussian_blur_prob=0.2,
            gaussian_sigma_range=(0.1, 2.0),
            motion_blur_prob=0.1,
            motion_blur_kernel_range=(3, 7),
            grayscale_prob=0.1,
            cutout_prob=0.2,
            cutout_num_regions=1,
            cutout_size_range=(0.02, 0.1),
        )

    @classmethod
    def validation_default(cls) -> AugmentationConfig:
        """Create validation configuration with no augmentation.

        Returns:
            AugmentationConfig with all augmentations disabled for
            deterministic evaluation.
        """
        return cls()

    @classmethod
    def light(cls) -> AugmentationConfig:
        """Create light augmentation configuration for fine-tuning.

        Uses reduced augmentation strength to preserve learned features
        while still providing regularization during fine-tuning.

        Returns:
            AugmentationConfig with reduced augmentation probabilities.
        """
        return cls(
            enable_geometric=True,
            enable_color=True,
            enable_blur=True,
            enable_cutout=True,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.0,
            rotation_prob=0.2,
            rotation_angle_range=10.0,
            scale_jitter_prob=0.2,
            scale_range=(0.9, 1.1),
            shear_prob=0.1,
            shear_range=0.05,
            translate_prob=0.1,
            translate_range=0.05,
            brightness_prob=0.2,
            brightness_range=(0.9, 1.1),
            contrast_prob=0.2,
            contrast_range=(0.9, 1.1),
            saturation_prob=0.2,
            saturation_range=(0.9, 1.1),
            hue_prob=0.1,
            hue_range=10.0,
            gaussian_blur_prob=0.1,
            motion_blur_prob=0.0,
            grayscale_prob=0.05,
            cutout_prob=0.1,
        )

    @classmethod
    def aggressive(cls) -> AugmentationConfig:
        """Create aggressive augmentation configuration for challenging datasets.

        Uses strong augmentation to improve generalization on small or
        highly variable datasets. May require longer training.

        Returns:
            AugmentationConfig with increased augmentation strength.
        """
        return cls(
            enable_geometric=True,
            enable_color=True,
            enable_blur=True,
            enable_cutout=True,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.2,
            rotation_prob=0.5,
            rotation_angle_range=30.0,
            scale_jitter_prob=0.5,
            scale_range=(0.6, 1.4),
            shear_prob=0.3,
            shear_range=0.2,
            translate_prob=0.3,
            translate_range=0.15,
            brightness_prob=0.5,
            brightness_range=(0.6, 1.4),
            contrast_prob=0.5,
            contrast_range=(0.6, 1.4),
            saturation_prob=0.5,
            saturation_range=(0.6, 1.4),
            hue_prob=0.3,
            hue_range=30.0,
            gaussian_blur_prob=0.3,
            gaussian_sigma_range=(0.1, 3.0),
            motion_blur_prob=0.2,
            grayscale_prob=0.2,
            cutout_prob=0.3,
            cutout_num_regions=2,
            cutout_size_range=(0.05, 0.15),
        )

    @classmethod
    def hflip_only(cls) -> AugmentationConfig:
        """Create minimal augmentation configuration with horizontal flip only.

        Matches the default augmentation strategy used in the DIS paper.

        Returns:
            AugmentationConfig with only horizontal flip enabled.
        """
        return cls(
            horizontal_flip_prob=0.5,
        )


def get_im_gt_name_dict(datasets: list[dict], flag: str = "valid") -> list[dict]:
    """Build a list of dataset dictionaries with image and ground truth paths.

    For training, multiple datasets are merged into a single entry to enable
    unified shuffling. For validation/test, datasets remain separate to allow
    per-dataset evaluation.

    Args:
        datasets: List of dataset configuration dictionaries, each containing:
            - name: Dataset identifier.
            - im_dir: Directory containing images.
            - gt_dir: Directory containing ground truth masks (empty string if none).
            - im_ext: Image file extension (e.g., ".jpg").
            - gt_ext: Ground truth file extension (e.g., ".png").
        flag: Dataset split identifier ("train" or "valid"/"test").

    Returns:
        List of dictionaries with keys: dataset_name, im_path, gt_path,
        im_ext, gt_ext. Training returns a single merged entry; validation
        returns one entry per dataset.
    """
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []
    for i, dataset in enumerate(datasets):
        print(f"--->>> {flag} dataset {i}/{len(datasets)} {dataset['name']} <<<---")
        im_dir = Path(dataset["im_dir"])
        tmp_im_list = [str(p) for p in im_dir.glob(f"*{dataset['im_ext']}")]
        print(f"-im- {dataset['name']} {dataset['im_dir']}: {len(tmp_im_list)}")
        if dataset["gt_dir"] == "":
            print(f"-gt- {dataset['name']} {dataset['gt_dir']}: No Ground Truth Found")
            tmp_gt_list = []
        else:
            gt_dir = Path(dataset["gt_dir"])
            tmp_gt_list = [str(gt_dir / (Path(x).stem + dataset["gt_ext"])) for x in tmp_im_list]
            print(f"-gt- {dataset['name']} {dataset['gt_dir']}: {len(tmp_gt_list)}")
        match flag:
            case "train":
                # Merge training datasets for unified sampling
                if not name_im_gt_list:
                    name_im_gt_list.append({
                        "dataset_name": dataset["name"],
                        "im_path": tmp_im_list,
                        "gt_path": tmp_gt_list,
                        "im_ext": dataset["im_ext"],
                        "gt_ext": dataset["gt_ext"],
                    })
                else:
                    name_im_gt_list[0]["dataset_name"] += "_" + dataset["name"]
                    name_im_gt_list[0]["im_path"] += tmp_im_list
                    name_im_gt_list[0]["gt_path"] += tmp_gt_list
                    if dataset["im_ext"] != ".jpg" or dataset["gt_ext"] != ".png":
                        print(
                            "Error: Please make sure all you images and ground truth "
                            "masks are in jpg and png format respectively !!!"
                        )
                        import sys

                        sys.exit()
                    name_im_gt_list[0]["im_ext"] = ".jpg"
                    name_im_gt_list[0]["gt_ext"] = ".png"
            case _:
                # Keep validation/test datasets separate for per-dataset metrics
                name_im_gt_list.append({
                    "dataset_name": dataset["name"],
                    "im_path": tmp_im_list,
                    "gt_path": tmp_gt_list,
                    "im_ext": dataset["im_ext"],
                    "gt_ext": dataset["gt_ext"],
                })
    return name_im_gt_list


def create_dataloaders(
    name_im_gt_list: list[dict],
    image_size: list[int] | None = None,
    my_transforms: list | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int | None = None,
) -> tuple[list[DataLoader], list[Dataset]]:
    """Create DataLoaders for training or evaluation.

    Creates one DataLoader per dataset entry in name_im_gt_list. For training,
    augmentation is typically handled on GPU via KorniaAugmentationPipeline
    in the LightningModule; this function applies only CPU-side transforms
    like normalization.

    Args:
        name_im_gt_list: List of dataset info dictionaries from get_im_gt_name_dict.
        image_size: Target image size as [H, W]. If None, images retain original size.
        my_transforms: List of transforms to compose (e.g., [GOSNormalize()]).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Number of data loading workers. If None, determined automatically.

    Returns:
        Tuple of (dataloaders, datasets) where each list has one entry per
        dataset in name_im_gt_list.
    """
    if image_size is None:
        image_size = []

    gos_dataloaders = []
    gos_datasets = []

    if not name_im_gt_list:
        return gos_dataloaders, gos_datasets

    if my_transforms is not None and len(my_transforms) > 0:
        transform = transforms.Compose(my_transforms)
    else:
        transform = None

    for nameimgt in name_im_gt_list:
        gos_dataset = GOSDataset([nameimgt], image_size=image_size, transform=transform)
        gos_dataloaders.append(
            DataLoader(
                gos_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
        )
        gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


def im_reader(im_path: str) -> Tensor:
    """Read an image file and return as RGB tensor.

    Args:
        im_path: Path to the image file.

    Returns:
        Tensor with shape (3, H, W) and dtype uint8.
    """
    return decode_image(im_path, mode=ImageReadMode.RGB)


def im_preprocess(im: Tensor, size: list[int]) -> tuple[Tensor, tuple[int, int]]:
    """Resize image to target dimensions.

    Args:
        im: Input image tensor with shape (C, H, W).
        size: Target size as [H, W]. If empty, no resizing is performed.

    Returns:
        Tuple of (resized_image, original_size) where original_size is (H, W).
    """
    orig_size = (im.shape[1], im.shape[2])
    if len(size) < 2:
        return im, orig_size
    im = v2.functional.resize(
        im,
        size=size,
        interpolation=v2.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return im, orig_size


def gt_preprocess(gt: Tensor, size: list[int]) -> tuple[Tensor, tuple[int, int]]:
    """Preprocess ground truth mask with channel reduction and resizing.

    Multi-channel ground truth masks are reduced to single channel by
    taking the first channel. This handles RGB masks that should be binary.

    Args:
        gt: Input ground truth tensor with shape (C, H, W).
        size: Target size as [H, W]. If empty, no resizing is performed.

    Returns:
        Tuple of (processed_mask, original_size) where mask has shape (1, H, W).
    """
    if gt.shape[0] > 1:
        gt = gt[0:1, ...]
    orig_size = (gt.shape[1], gt.shape[2])
    if len(size) < 2:
        return gt, orig_size
    gt = v2.functional.resize(
        gt,
        size=size,
        interpolation=v2.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return gt, orig_size


class GOSNormalize:
    """ImageNet-style normalization transform for image tensors.

    Applies channel-wise mean subtraction and standard deviation division
    using torchvision's functional normalize. Default values are ImageNet
    statistics for compatibility with pretrained backbones.

    Attributes:
        mean: Per-channel mean values for normalization.
        std: Per-channel standard deviation values for normalization.
    """

    def __init__(self, mean: list[float] | None = None, std: list[float] | None = None) -> None:
        """Initialize normalization transform.

        Args:
            mean: RGB channel means. Defaults to ImageNet values.
            std: RGB channel standard deviations. Defaults to ImageNet values.
        """
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply normalization to the image in a sample dictionary.

        Args:
            sample: Dictionary containing at minimum an "image" key.

        Returns:
            Sample dictionary with normalized image tensor.
        """
        sample["image"] = normalize(sample["image"], self.mean, self.std)
        return sample


class GOSDataset(Dataset):
    """PyTorch Dataset for loading image-mask pairs with on-the-fly processing.

    Supports multiple input datasets that are internally merged into a single
    indexable collection. Images and masks are loaded and preprocessed on
    demand to minimize memory usage.

    Attributes:
        image_size: Target size for resizing images and masks.
        transform: Optional transform to apply to samples.
        dataset: Internal dictionary storing paths and metadata.
    """

    def __init__(
        self,
        name_im_gt_list: list[dict],
        image_size: list[int] | None = None,
        transform: Any = None,
    ) -> None:
        """Initialize dataset from list of dataset info dictionaries.

        Args:
            name_im_gt_list: List of dataset dictionaries from get_im_gt_name_dict.
            image_size: Target size as [H, W] for resizing. If None, keeps original.
            transform: Optional transform to apply to each sample dictionary.
        """
        if image_size is None:
            image_size = []
        self.image_size = image_size
        self.transform = transform
        self.dataset = {}

        dataset_names = []
        dt_name_list = []
        im_name_list = []
        im_path_list = []
        gt_path_list = []
        im_ext_list = []
        gt_ext_list = []
        for nameimgt in name_im_gt_list:
            dataset_names.append(nameimgt["dataset_name"])
            # Repeat dataset name for each image to enable per-sample dataset tracking
            dt_name_list.extend([nameimgt["dataset_name"] for _ in nameimgt["im_path"]])
            im_name_list.extend([Path(x).stem for x in nameimgt["im_path"]])
            im_path_list.extend(nameimgt["im_path"])
            gt_path_list.extend(nameimgt["gt_path"])
            im_ext_list.extend([nameimgt["im_ext"] for _ in nameimgt["im_path"]])
            gt_ext_list.extend([nameimgt["gt_ext"] for _ in nameimgt["gt_path"]])
        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.dataset["im_path"])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Load and preprocess a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - imidx: Sample index as tensor.
                - image: Preprocessed image tensor (C, H, W), float32 in [0, 1].
                - label: Ground truth mask tensor (1, H, W), float32 in [0, 1].
                - shape: Original image dimensions as tensor.
        """
        im_path = self.dataset["im_path"][idx]
        im = im_reader(im_path)
        im, im_shp = im_preprocess(im, self.image_size)

        # Use zero tensor as placeholder when ground truth is unavailable
        if len(self.dataset["gt_path"]) != 0 and self.dataset["gt_path"][idx]:
            gt = im_reader(self.dataset["gt_path"][idx])
            gt, _gt_shp = gt_preprocess(gt, self.image_size)
        else:
            gt = torch.zeros(1, im.shape[1], im.shape[2], dtype=im.dtype)

        im = to_dtype(im, dtype=torch.float32, scale=True)
        gt = to_dtype(gt, dtype=torch.float32, scale=True)

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "shape": torch.from_numpy(np.array(im_shp)),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
