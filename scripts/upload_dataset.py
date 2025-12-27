"""Upload local anime segmentation dataset to HuggingFace Hub."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, HfHubHTTPError

from anime_segmentation.hf_dataset import (
    load_background_dataset,
    load_foreground_dataset,
    load_real_dataset,
)


def upload_dataset(
    data_dir: str,
    repo_id: str,
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
    private: bool = False,
) -> None:
    """Upload local dataset to HuggingFace Hub.

    Creates a dataset with three configs: real, foreground, background.

    Args:
        data_dir: Local data directory.
        repo_id: HuggingFace Hub repository ID (e.g., 'username/dataset-name').
        fg_dir: Foreground subdirectory name.
        bg_dir: Background subdirectory name.
        img_dir: Image subdirectory name.
        mask_dir: Mask subdirectory name.
        fg_ext: Foreground file extension.
        bg_ext: Background file extension.
        img_ext: Image file extension.
        mask_ext: Mask file extension.
        split_ratio: Train/validation split ratio.
        private: Whether to create a private repository.
    """
    data_path = Path(data_dir)
    print(f"Loading datasets from {data_path}")

    # Load real dataset
    print("Loading real images...")
    real_ds = load_real_dataset(
        data_dir,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        split_ratio=split_ratio,
    )
    print(f"  Train: {len(real_ds['train'])} samples")
    print(f"  Validation: {len(real_ds['validation'])} samples")

    # Load foreground dataset
    print("Loading foreground images...")
    fg_ds = load_foreground_dataset(
        data_dir,
        fg_dir=fg_dir,
        fg_ext=fg_ext,
        split_ratio=split_ratio,
    )
    print(f"  Train: {len(fg_ds['train'])} samples")
    print(f"  Validation: {len(fg_ds['validation'])} samples")

    # Load background dataset
    print("Loading background images...")
    bg_ds = load_background_dataset(
        data_dir,
        bg_dir=bg_dir,
        bg_ext=bg_ext,
        split_ratio=split_ratio,
    )
    print(f"  Train: {len(bg_ds['train'])} samples")
    print(f"  Validation: {len(bg_ds['validation'])} samples")

    # Create repository if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"Repository {repo_id} ready")
    except HfHubHTTPError as e:
        print(f"Note: {e}")

    # Upload each config
    print("\nUploading to HuggingFace Hub...")

    print("Uploading 'real' config...")
    real_ds.push_to_hub(repo_id, config_name="real", private=private)
    print("  Done!")

    print("Uploading 'foreground' config...")
    fg_ds.push_to_hub(repo_id, config_name="foreground", private=private)
    print("  Done!")

    print("Uploading 'background' config...")
    bg_ds.push_to_hub(repo_id, config_name="background", private=private)
    print("  Done!")

    print(f"\nDataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
    print("\nUsage:")
    print("  from datasets import load_dataset")
    print(f'  real_ds = load_dataset("{repo_id}", "real")')
    print(f'  fg_ds = load_dataset("{repo_id}", "foreground")')
    print(f'  bg_ds = load_dataset("{repo_id}", "background")')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload anime segmentation dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to local dataset directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID (e.g., 'username/anime-seg')",
    )
    parser.add_argument(
        "--fg-dir",
        type=str,
        default="fg",
        help="Foreground subdirectory name (default: fg)",
    )
    parser.add_argument(
        "--bg-dir",
        type=str,
        default="bg",
        help="Background subdirectory name (default: bg)",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="imgs",
        help="Image subdirectory name (default: imgs)",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="masks",
        help="Mask subdirectory name (default: masks)",
    )
    parser.add_argument(
        "--fg-ext",
        type=str,
        default=".png",
        help="Foreground file extension (default: .png)",
    )
    parser.add_argument(
        "--bg-ext",
        type=str,
        default=".jpg",
        help="Background file extension (default: .jpg)",
    )
    parser.add_argument(
        "--img-ext",
        type=str,
        default=".jpg",
        help="Image file extension (default: .jpg)",
    )
    parser.add_argument(
        "--mask-ext",
        type=str,
        default=".jpg",
        help="Mask file extension (default: .jpg)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.95,
        help="Train/validation split ratio (default: 0.95)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()

    upload_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        fg_dir=args.fg_dir,
        bg_dir=args.bg_dir,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        fg_ext=args.fg_ext,
        bg_ext=args.bg_ext,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        split_ratio=args.split_ratio,
        private=args.private,
    )


if __name__ == "__main__":
    main()
