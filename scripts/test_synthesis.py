"""Test script for synthesis data pipeline.

Tests whether the Copy-Paste synthesis pipeline works correctly.

Usage:
    # Use existing data
    python scripts/test_synthesis.py --fg_dir data/fg --bg_dir data/bg --output_dir output/synthesis

    # Test with dummy data
    python scripts/test_synthesis.py --demo --output_dir output/synthesis_demo

    # Generate multiple samples
    python scripts/test_synthesis.py --fg_dir data/fg --bg_dir data/bg -n 10 --output_dir output
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from anime_segmentation.data.pools import BackgroundPool, ForegroundPool
from anime_segmentation.training.synthesis import (
    CompositorConfig,
    CopyPasteCompositor,
    InstanceTransform,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_demo_foreground(path: Path, size: int = 256) -> None:
    """Generate demo foreground image (RGBA)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw character-like shapes with random colors
    rng = np.random.default_rng()
    color = (*tuple(rng.integers(100, 255, 3).tolist()), 255)

    # Head (circle)
    head_center = (size // 2, size // 3)
    head_radius = size // 5
    draw.ellipse(
        [
            head_center[0] - head_radius,
            head_center[1] - head_radius,
            head_center[0] + head_radius,
            head_center[1] + head_radius,
        ],
        fill=color,
    )

    # Body (rectangle)
    body_top = head_center[1] + head_radius
    body_bottom = int(size * 0.85)
    body_width = size // 3
    draw.rectangle(
        [
            size // 2 - body_width // 2,
            body_top,
            size // 2 + body_width // 2,
            body_bottom,
        ],
        fill=color,
    )

    img.save(path, "PNG")


def create_demo_background(path: Path, size: int = 512) -> None:
    """Generate demo background image (RGB)."""
    rng = np.random.default_rng()

    # Gradient background
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    base_color = rng.integers(50, 200, 3)

    for i in range(size):
        factor = i / size
        arr[i, :] = (base_color * (1 - factor * 0.5)).astype(np.uint8)

    # Add noise
    noise = rng.integers(-20, 20, (size, size, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, "RGB")
    img.save(path)


def setup_demo_data(base_dir: Path, num_fg: int = 5, num_bg: int = 3) -> tuple[Path, Path]:
    """Create demo dataset."""
    fg_dir = base_dir / "fg"
    bg_dir = base_dir / "bg"
    fg_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating demo foreground images...")
    for i in range(num_fg):
        create_demo_foreground(fg_dir / f"char_{i:03d}.png", size=256 + i * 32)

    logger.info("Generating demo background images...")
    for i in range(num_bg):
        create_demo_background(bg_dir / f"bg_{i:03d}.png", size=512 + i * 64)

    return fg_dir, bg_dir


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def create_visualization(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> Image.Image:
    """Generate visualization image of composite result."""
    h, w = image.shape[1], image.shape[2]

    # Arrange 3 images horizontally: Composite | Mask | Overlay
    vis = Image.new("RGB", (w * 3, h))

    # Composite image
    img_pil = tensor_to_pil(image)
    vis.paste(img_pil, (0, 0))

    # Mask (grayscale -> RGB)
    mask_arr = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_arr).convert("RGB")
    vis.paste(mask_pil, (w, 0))

    # Overlay (highlight mask region in red)
    overlay = img_pil.copy().convert("RGBA")
    mask_rgba = Image.new("RGBA", (w, h), (255, 0, 0, 100))
    mask_binary = Image.fromarray((mask_arr > 127).astype(np.uint8) * 255, "L")
    overlay.paste(mask_rgba, mask=mask_binary)
    vis.paste(overlay.convert("RGB"), (w * 2, 0))

    return vis


def test_synthesis(
    fg_dir: Path,
    bg_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
    target_size: tuple[int, int] = (512, 512),
    seed: int | None = None,
) -> None:
    """Test synthesis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pools
    logger.info("ForegroundPool: %s", fg_dir)
    fg_pool = ForegroundPool(fg_dir)
    logger.info(f"  Image count: {len(fg_pool)}")

    logger.info("BackgroundPool: %s", bg_dir)
    bg_pool = BackgroundPool(bg_dir)
    logger.info(f"  Image count: {len(bg_pool)}")

    # Compositor configuration
    config = CompositorConfig(
        k_probs={0: 0.05, 1: 0.35, 2: 0.35, 3: 0.20, 4: 0.05},
        min_area_ratio=0.05,
        max_area_ratio=0.50,
        max_total_coverage=0.80,
        max_iou_overlap=0.25,
        blending_probs={"hard": 0.4, "feather": 0.6},
    )

    instance_transform = InstanceTransform(
        hflip_prob=0.5,
        rotation_range=(-15.0, 15.0),
        scale_range=(0.6, 1.4),
    )

    compositor = CopyPasteCompositor(
        fg_pool=fg_pool,
        bg_pool=bg_pool,
        config=config,
        instance_transform=instance_transform,
    )

    # Random generator
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.manual_seed(torch.initial_seed() % (2**31))

    # Generate samples
    logger.info("Generating %s samples...", num_samples)
    k_counts: dict[int, int] = {}

    for i in range(num_samples):
        image, mask, k = compositor.synthesize(target_size, rng)

        # k statistics
        k_counts[k] = k_counts.get(k, 0) + 1

        # Save visualization
        vis = create_visualization(image, mask)
        vis_path = output_dir / f"sample_{i:04d}_k{k}.png"
        vis.save(vis_path)

        # Also save individual composite images
        img_path = output_dir / f"sample_{i:04d}_image.png"
        mask_path = output_dir / f"sample_{i:04d}_mask.png"
        tensor_to_pil(image).save(img_path)
        mask_arr = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_arr, "L").save(mask_path)

        logger.info(f"  [{i + 1}/{num_samples}] k={k} -> {vis_path.name}")

    # Display statistics
    logger.info("=" * 50)
    logger.info("Generation complete!")
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Character count (k) distribution:")
    for k in sorted(k_counts.keys()):
        count = k_counts[k]
        pct = count / num_samples * 100
        bar = "█" * int(pct / 5)
        logger.info(f"    k={k}: {count:3d} ({pct:5.1f}%) {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Copy-Paste synthesis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fg_dir",
        type=Path,
        help="Directory of foreground images (RGBA PNG)",
    )
    parser.add_argument(
        "--bg_dir",
        type=Path,
        help="Directory of background images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/synthesis_test"),
        help="Output directory (default: output/synthesis_test)",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate (default: 5)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image size (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for reproducibility)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Test with demo dummy data",
    )
    args = parser.parse_args()

    if args.demo:
        # Demo mode: Create dummy data in temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fg_dir, bg_dir = setup_demo_data(tmp_path)

            test_synthesis(
                fg_dir=fg_dir,
                bg_dir=bg_dir,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                target_size=(args.size, args.size),
                seed=args.seed,
            )
    else:
        # Normal mode: Use specified directories
        if args.fg_dir is None or args.bg_dir is None:
            parser.error("--fg_dir and --bg_dir are required (or use --demo)")

        test_synthesis(
            fg_dir=args.fg_dir,
            bg_dir=args.bg_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            target_size=(args.size, args.size),
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
