#!/usr/bin/env python
"""Download and extract skytnt/anime-segmentation dataset from Hugging Face Hub.

Usage:
    python scripts/download_anime_segmentation.py
    python scripts/download_anime_segmentation.py --output datasets/anime-segmentation
    python scripts/download_anime_segmentation.py --imgs-masks-only  # Only download imgs and masks

This script downloads the dataset files from Hugging Face Hub and extracts them
to a local directory with the expected structure:

    {output}/
    ├── bg/       # Background images (8,057 JPG)
    ├── fg/       # Character cutouts (11,802 PNG with alpha)
    ├── imgs/     # Pre-composed images (1,111 JPG)
    └── masks/    # Segmentation masks (1,111 JPG)
"""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "skytnt/anime-segmentation"
REPO_TYPE = "dataset"

# Files in the repository
BG_FILES = [f"data/bg-0{i}.zip" for i in range(5)]  # bg-00.zip to bg-04.zip
FG_FILES = [f"data/fg-0{i}.zip" for i in range(6)]  # fg-00.zip to fg-05.zip
IMGS_MASKS_FILE = "data/imgs-masks.zip"


def download_and_extract(
    output_dir: Path,
    *,
    imgs_masks_only: bool = False,
    cache_dir: str | None = None,
) -> None:
    """Download and extract the dataset.

    Args:
        output_dir: Directory to extract files to.
        imgs_masks_only: If True, only download imgs and masks (skip bg and fg).
        cache_dir: Cache directory for downloads. Defaults to HF cache.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which files to download
    if imgs_masks_only:
        files_to_download = [IMGS_MASKS_FILE]
        print("Downloading imgs and masks only...")
    else:
        files_to_download = [IMGS_MASKS_FILE, *BG_FILES, *FG_FILES]
        print("Downloading full dataset (bg, fg, imgs, masks)...")

    # Download and extract each file
    for filename in files_to_download:
        print(f"\nDownloading {filename}...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            cache_dir=cache_dir,
        )

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(output_dir)

    # Verify extracted directories
    print("\nVerifying extracted files...")
    expected_dirs = ["imgs", "masks"]
    if not imgs_masks_only:
        expected_dirs.extend(["bg", "fg"])

    for dirname in expected_dirs:
        dir_path = output_dir / dirname
        if dir_path.exists():
            file_count = len(list(dir_path.iterdir()))
            print(f"  {dirname}/: {file_count} files")
        else:
            print(f"  {dirname}/: NOT FOUND (warning)")

    print(f"\nDataset extracted to: {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download skytnt/anime-segmentation dataset from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("datasets/anime-segmentation"),
        help="Output directory (default: datasets/anime-segmentation)",
    )
    parser.add_argument(
        "--imgs-masks-only",
        action="store_true",
        help="Only download imgs and masks (skip bg and fg)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads (default: HF cache)",
    )

    args = parser.parse_args()

    download_and_extract(
        args.output,
        imgs_masks_only=args.imgs_masks_only,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
