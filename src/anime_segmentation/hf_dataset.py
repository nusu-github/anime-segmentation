"""HuggingFace Datasets integration for anime segmentation."""

import random
from pathlib import Path
from typing import Any

import torch
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Image,
    IterableDataset,
    IterableDatasetDict,
    Sequence,
    Value,
    interleave_datasets,
    load_dataset,
)
from PIL import Image as PILImage
from torchvision.transforms.v2 import functional as F


def load_real_dataset(
    data_dir: str | Path,
    img_dir: str = "imgs",
    mask_dir: str = "masks",
    img_ext: str = ".jpg",
    mask_ext: str = ".jpg",
    split_ratio: float = 0.95,
    seed: int = 42,
) -> DatasetDict:
    """Load real image dataset from local directory.

    Args:
        data_dir: Root directory containing image and mask folders.
        img_dir: Subdirectory name for images.
        mask_dir: Subdirectory name for masks.
        img_ext: Image file extension.
        mask_ext: Mask file extension.
        split_ratio: Train/validation split ratio.
        seed: Random seed for shuffling.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    data_root = Path(data_dir)
    imgs_path = data_root / img_dir
    masks_path = data_root / mask_dir

    # Collect image paths
    img_files = sorted(imgs_path.glob(f"*{img_ext}"))
    if not img_files:
        return DatasetDict({"train": Dataset.from_dict({}), "validation": Dataset.from_dict({})})

    # Build corresponding mask paths
    image_paths = []
    mask_paths = []
    for img_file in img_files:
        mask_file = masks_path / img_file.name.replace(img_ext, mask_ext)
        if mask_file.exists():
            image_paths.append(str(img_file))
            mask_paths.append(str(mask_file))

    # Shuffle with fixed seed
    rng = random.Random(seed)
    paired = list(zip(image_paths, mask_paths, strict=True))
    rng.shuffle(paired)
    image_paths, mask_paths = zip(*paired, strict=True) if paired else ([], [])

    # Split into train/val
    split_idx = int(len(image_paths) * split_ratio)
    train_images, val_images = list(image_paths[:split_idx]), list(image_paths[split_idx:])
    train_masks, val_masks = list(mask_paths[:split_idx]), list(mask_paths[split_idx:])

    # Create datasets with Image feature
    features = Features({"image": Image(), "mask": Image()})

    train_ds = Dataset.from_dict(
        {"image": train_images, "mask": train_masks},
        features=features,
    )
    val_ds = Dataset.from_dict(
        {"image": val_images, "mask": val_masks},
        features=features,
    )

    return DatasetDict({"train": train_ds, "validation": val_ds})


def load_foreground_dataset(
    data_dir: str | Path,
    fg_dir: str = "fg",
    fg_ext: str = ".png",
    split_ratio: float = 0.95,
    seed: int = 42,
) -> DatasetDict:
    """Load foreground images (RGBA) from local directory.

    Args:
        data_dir: Root directory containing foreground folder.
        fg_dir: Subdirectory name for foreground images.
        fg_ext: Foreground file extension.
        split_ratio: Train/validation split ratio.
        seed: Random seed for shuffling.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    data_root = Path(data_dir)
    fgs_path = data_root / fg_dir

    fg_files = sorted(fgs_path.glob(f"*{fg_ext}"))
    if not fg_files:
        return DatasetDict({"train": Dataset.from_dict({}), "validation": Dataset.from_dict({})})

    fg_paths = [str(f) for f in fg_files]

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(fg_paths)

    # Split into train/val
    split_idx = int(len(fg_paths) * split_ratio)
    train_fg = fg_paths[:split_idx]
    val_fg = fg_paths[split_idx:]

    features = Features({"image": Image()})

    train_ds = Dataset.from_dict({"image": train_fg}, features=features)
    val_ds = Dataset.from_dict({"image": val_fg}, features=features)

    return DatasetDict({"train": train_ds, "validation": val_ds})


def load_background_dataset(
    data_dir: str | Path,
    bg_dir: str = "bg",
    bg_ext: str = ".jpg",
    split_ratio: float = 0.95,
    seed: int = 42,
) -> DatasetDict:
    """Load background images from local directory.

    Args:
        data_dir: Root directory containing background folder.
        bg_dir: Subdirectory name for background images.
        bg_ext: Background file extension.
        split_ratio: Train/validation split ratio.
        seed: Random seed for shuffling.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    data_root = Path(data_dir)
    bgs_path = data_root / bg_dir

    bg_files = sorted(bgs_path.glob(f"*{bg_ext}"))
    if not bg_files:
        return DatasetDict({"train": Dataset.from_dict({}), "validation": Dataset.from_dict({})})

    bg_paths = [str(f) for f in bg_files]

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(bg_paths)

    # Split into train/val
    split_idx = int(len(bg_paths) * split_ratio)
    train_bg = bg_paths[:split_idx]
    val_bg = bg_paths[split_idx:]

    features = Features({"image": Image()})

    train_ds = Dataset.from_dict({"image": train_bg}, features=features)
    val_ds = Dataset.from_dict({"image": val_bg}, features=features)

    return DatasetDict({"train": train_ds, "validation": val_ds})


def create_synthetic_index_dataset(
    num_foregrounds: int,
    characters_range: tuple[int, int] = (0, 3),
    seed: int = 42,
) -> Dataset:
    """Create dataset of foreground indices for synthetic compositing.

    Groups foreground indices into character groups for compositing.

    Args:
        num_foregrounds: Total number of foreground images.
        characters_range: Range for number of characters per sample.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with 'fg_indices' column (list of fg indices per sample).
    """
    rng = random.Random(seed)
    groups: list[list[int]] = []
    total = 0

    while total < num_foregrounds:
        num_chars = rng.randint(*characters_range)
        group = [total + x for x in range(num_chars) if total + x < num_foregrounds]
        groups.append(group)
        total += num_chars

    features = Features({"fg_indices": Sequence(Value("int32"))})
    return Dataset.from_dict({"fg_indices": groups}, features=features)


class SyntheticCompositor:
    """Compositor for creating synthetic anime segmentation images.

    Handles foreground/background compositing with random augmentations.
    Designed to be used with HuggingFace Datasets set_transform.
    Uses lazy loading to avoid OOM with large datasets.
    """

    def __init__(
        self,
        foreground_paths: list[str],
        background_paths: list[str],
        output_size: tuple[int, int],
        seed: int | None = None,
        edge_blur_p: float = 0.2,
        edge_blur_kernel_size: int = 5,
    ) -> None:
        """Initialize compositor.

        Args:
            foreground_paths: List of paths to RGBA foreground images.
            background_paths: List of paths to RGB background images.
            output_size: Output image size (height, width).
            seed: Optional random seed for reproducibility.
            edge_blur_p: Probability of applying edge blur during blending.
            edge_blur_kernel_size: Kernel size for edge blur.
        """
        self.foreground_paths = foreground_paths
        self.background_paths = background_paths
        self.output_size = output_size
        self.rng = random.Random(seed)
        self.edge_blur_p = edge_blur_p
        self.edge_blur_kernel_size = edge_blur_kernel_size

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Transform function for set_transform.

        Args:
            examples: Batch of examples with 'fg_indices' column.

        Returns:
            Dict with 'image' and 'mask' columns as tensors.
        """
        images = []
        masks = []

        for fg_indices in examples["fg_indices"]:
            image, mask = self._composite(fg_indices)
            images.append(image)
            masks.append(mask)

        return {"image": images, "mask": masks}

    def _composite(self, fg_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Composite foregrounds onto a random background.

        Args:
            fg_indices: List of foreground image indices to composite.

        Returns:
            Tuple of (composited_image, mask) as tensors.
        """
        h, w = self.output_size

        # Select and prepare random background (load on demand)
        bg_path = self.rng.choice(self.background_paths)
        bg_pil = PILImage.open(bg_path).convert("RGB")
        bg = self._prepare_background(bg_pil, (h, w))

        # Initialize output
        image = bg.clone()
        label = torch.zeros(1, h, w, dtype=torch.float32)

        # Composite each foreground (load on demand)
        for fg_idx in fg_indices:
            if fg_idx < len(self.foreground_paths):
                fg_path = self.foreground_paths[fg_idx]
                fg_pil = PILImage.open(fg_path).convert("RGBA")
                fg = self._prepare_foreground(fg_pil, (h, w))
                image, label = self._blend(image, label, fg)

        # Binarize label
        label = (label > 0.5).float()

        return image, label

    def _prepare_background(
        self, bg_pil: PILImage.Image, output_size: tuple[int, int]
    ) -> torch.Tensor:
        """Prepare background image with random crop and resize."""
        bg = F.pil_to_tensor(bg_pil.convert("RGB")).float() / 255.0

        _, h, w = bg.shape
        out_h, out_w = output_size
        r = min(h / out_h, w / out_w)
        crop_h, crop_w = int(out_h * r), int(out_w * r)

        # Random crop position
        top = self.rng.randint(0, max(0, h - crop_h))
        left = self.rng.randint(0, max(0, w - crop_w))
        bg = F.crop(bg, top, left, crop_h, crop_w)

        return F.resize(bg, list(output_size), antialias=True)

    def _prepare_foreground(
        self, fg_pil: PILImage.Image, output_size: tuple[int, int]
    ) -> torch.Tensor:
        """Prepare foreground with random transform."""
        fg = F.pil_to_tensor(fg_pil.convert("RGBA")).float() / 255.0
        h, w = output_size

        # Resize to fit output
        _, fg_h, fg_w = fg.shape
        r = min(h / fg_h, w / fg_w)
        new_h, new_w = int(fg_h * r), int(fg_w * r)
        fg = F.resize(fg, [new_h, new_w], antialias=True)

        # Pad to output size (centered)
        pad_top = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left = (w - new_w) // 2
        pad_right = w - new_w - pad_left
        fg = F.pad(fg, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

        # Center foreground based on alpha center of mass
        alpha = fg[3]
        cy, cx = self._center_of_mass(alpha)

        # Translation to center the character
        dx = w / 2 - cx
        dy = h / 2 - cy
        fg = F.affine(fg, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0])

        # Random scale, rotation, and translation
        scale = self.rng.uniform(0.2, 0.8)
        angle = self.rng.randint(-90, 90)
        trans_dx = self.rng.randint(-w // 3, w // 3)
        trans_dy = self.rng.randint(-h // 3, h // 3)

        return F.affine(fg, angle=angle, translate=[trans_dx, trans_dy], scale=scale, shear=[0.0])

    def _center_of_mass(self, alpha: torch.Tensor) -> tuple[float, float]:
        """Compute center of mass for alpha channel."""
        h, w = alpha.shape
        total = alpha.sum()
        if total == 0:
            return h / 2, w / 2

        y_coords = torch.arange(h, dtype=alpha.dtype)
        x_coords = torch.arange(w, dtype=alpha.dtype)

        cy = (y_coords.unsqueeze(1) * alpha).sum() / total
        cx = (x_coords.unsqueeze(0) * alpha).sum() / total

        return cy.item(), cx.item()

    def _blend(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        fg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend foreground onto image and update label."""
        fg_rgb = fg[:3]
        fg_alpha = fg[3:4]

        # Optionally apply soft edge blending with gaussian blur
        if self.rng.random() < self.edge_blur_p:
            ks = self.edge_blur_kernel_size
            blurred_alpha = F.gaussian_blur(fg_alpha, kernel_size=[ks, ks])
            mask = fg_alpha * blurred_alpha
        else:
            mask = fg_alpha

        # Composite
        image = mask * fg_rgb + (1 - mask) * image
        label = torch.maximum(fg_alpha, label)

        return image, label


class RealImageTransform:
    """Transform for real image dataset.

    Converts PIL images to tensors with proper preprocessing.
    Designed to be used with HuggingFace Datasets set_transform.
    """

    def __init__(
        self,
        edge_crop: int = 10,
        mask_threshold: float = 0.3,
    ) -> None:
        """Initialize transform.

        Args:
            edge_crop: Number of pixels to crop from edges.
            mask_threshold: Threshold for binarizing mask.
        """
        self.edge_crop = edge_crop
        self.mask_threshold = mask_threshold

    def __call__(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Transform function for set_transform.

        Args:
            examples: Batch of examples with 'image' and 'mask' columns.

        Returns:
            Dict with 'image' and 'mask' as tensors.
        """
        images = []
        masks = []

        batch_images = examples["image"]
        batch_masks = examples["mask"]

        for img_pil, mask_pil in zip(batch_images, batch_masks, strict=True):
            # Convert to tensor
            image = F.pil_to_tensor(img_pil.convert("RGB")).float() / 255.0
            mask = F.pil_to_tensor(mask_pil.convert("L")).float() / 255.0

            # Binarize mask
            mask = (mask > self.mask_threshold).float()

            # Crop edges
            if self.edge_crop > 0:
                c = self.edge_crop
                image = image[:, c:-c, c:-c]
                mask = mask[:, c:-c, c:-c]

            images.append(image)
            masks.append(mask)

        return {"image": images, "mask": masks}


def get_image_paths_from_column(dataset: Dataset, column: str = "image") -> list[str]:
    """Get file paths from dataset by temporarily disabling image decoding.

    Uses batch processing for efficient extraction.

    Args:
        dataset: HuggingFace Dataset with image column.
        column: Name of the image column.

    Returns:
        List of image file paths.
    """
    # Cast to decode=False to get paths without loading images
    ds_no_decode = dataset.cast_column(column, Image(decode=False))

    # Use select_columns and batch access for efficiency
    ds_column_only = ds_no_decode.select_columns([column])
    all_data = ds_column_only[:]  # Get all data at once

    paths: list[str] = []
    for i, img_data in enumerate(all_data[column]):
        if isinstance(img_data, dict) and "path" in img_data:
            paths.append(img_data["path"])
        elif isinstance(img_data, str):
            paths.append(img_data)
        else:
            msg = f"Cannot extract path from dataset item {i}: {type(img_data)}"
            raise ValueError(msg)
    return paths


def create_interleaved_dataset(
    datasets: list[Dataset],
    probabilities: list[float] | None = None,
    seed: int = 42,
    num_shards: int = 64,
    *,
    use_iterable: bool = False,
) -> Dataset | IterableDataset:
    """Create an interleaved dataset from multiple datasets.

    Uses HuggingFace's native interleave_datasets for optimal performance.
    Optionally converts to IterableDataset with sharding for large datasets.

    Args:
        datasets: List of datasets to interleave.
        probabilities: Sampling probabilities for each dataset (must sum to 1).
        seed: Random seed for reproducibility.
        num_shards: Number of shards for IterableDataset (for parallel loading).
        use_iterable: If True, convert to IterableDataset with sharding.

    Returns:
        Interleaved Dataset or IterableDataset.
    """
    if not datasets:
        msg = "At least one dataset must be provided"
        raise ValueError(msg)

    if len(datasets) == 1:
        combined = datasets[0]
    else:
        # Calculate default probabilities based on dataset sizes
        if probabilities is None:
            total_len = sum(len(ds) for ds in datasets)
            probabilities = [len(ds) / total_len for ds in datasets]

        combined = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    if use_iterable:
        # Convert to IterableDataset with sharding for parallel DataLoader
        return combined.to_iterable_dataset(num_shards=num_shards)

    return combined


def load_anime_seg_dataset(
    data_dir: str | Path | None = None,
    dataset_name: str | None = None,
    *,
    fg_dir: str = "fg",
    bg_dir: str = "bg",
    img_dir: str = "imgs",
    mask_dir: str = "masks",
    fg_ext: str = ".png",
    bg_ext: str = ".jpg",
    img_ext: str = ".jpg",
    mask_ext: str = ".jpg",
    split_ratio: float = 0.95,
    seed: int = 42,
    streaming: bool = False,
) -> dict[str, DatasetDict | IterableDatasetDict]:
    """Load anime segmentation dataset from local directory or HuggingFace Hub.

    Args:
        data_dir: Local data directory (mutually exclusive with dataset_name).
        dataset_name: HuggingFace Hub dataset name (mutually exclusive with data_dir).
        fg_dir: Foreground subdirectory name.
        bg_dir: Background subdirectory name.
        img_dir: Image subdirectory name.
        mask_dir: Mask subdirectory name.
        fg_ext: Foreground file extension.
        bg_ext: Background file extension.
        img_ext: Image file extension.
        mask_ext: Mask file extension.
        split_ratio: Train/validation split ratio.
        seed: Random seed.
        streaming: Whether to use streaming mode (Hub only).

    Returns:
        Dict with 'real', 'foreground', 'background' DatasetDicts.
    """
    if dataset_name is not None:
        # Load from HuggingFace Hub
        real_ds = load_dataset(dataset_name, "real", streaming=streaming)
        fg_ds = load_dataset(dataset_name, "foreground", streaming=streaming)
        bg_ds = load_dataset(dataset_name, "background", streaming=streaming)
        return {"real": real_ds, "foreground": fg_ds, "background": bg_ds}

    if data_dir is None:
        msg = "Either data_dir or dataset_name must be provided"
        raise ValueError(msg)

    # Load from local directory
    real_ds = load_real_dataset(
        data_dir,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        split_ratio=split_ratio,
        seed=seed,
    )
    fg_ds = load_foreground_dataset(
        data_dir,
        fg_dir=fg_dir,
        fg_ext=fg_ext,
        split_ratio=split_ratio,
        seed=seed,
    )
    bg_ds = load_background_dataset(
        data_dir,
        bg_dir=bg_dir,
        bg_ext=bg_ext,
        split_ratio=split_ratio,
        seed=seed,
    )

    return {"real": real_ds, "foreground": fg_ds, "background": bg_ds}
