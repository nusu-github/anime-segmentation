"""Detailed test script for synthesis processing.

Tests all blending methods and consistency operations individually,
allowing verification of each processing result.

Usage:
    python scripts/test_synthesis_detailed.py \
        --fg_dir datasets/anime-segmentation/fg \
        --bg_dir datasets/anime-segmentation/bg \
        --output_dir output/synthesis_detailed

    # Test specific blending methods only
    python scripts/test_synthesis_detailed.py \
        --fg_dir data/fg --bg_dir data/bg \
        --blending hard feather seamless

    # Also test consistency operations
    python scripts/test_synthesis_detailed.py \
        --fg_dir data/fg --bg_dir data/bg \
        --consistency
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from anime_segmentation.data.pools import BackgroundPool, ForegroundPool
from anime_segmentation.training.synthesis.blending import (
    FeatherBlending,
    HardPasteBlending,
)
from anime_segmentation.training.synthesis.consistency import (
    ColorToneMatching,
    LightWrap,
    NoiseGrainConsistency,
    SimpleShadow,
)
from anime_segmentation.training.synthesis.transforms import crop_to_content

if TYPE_CHECKING:
    from torch import Generator, Tensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case information."""

    name: str
    description: str
    blending: str
    consistency_ops: list[str]
    image: Tensor
    mask: Tensor
    fg_mask_original: Tensor  # Original foreground mask before compositing (for comparison)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_label_to_image(img: Image.Image, label: str, font_size: int = 16) -> Image.Image:
    """Add label to image."""
    draw = ImageDraw.Draw(img)
    # Try to load font (use default if system font not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Add background
    bbox = draw.textbbox((5, 5), label, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill="black")
    draw.text((5, 5), label, fill="white", font=font)
    return img


def scale_foreground_to_target_area(
    fg_rgb: Tensor,
    fg_mask: Tensor,
    target_area_ratio: float,
    canvas_size: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Scale foreground to specified area ratio."""
    canvas_h, canvas_w = canvas_size
    canvas_area = canvas_h * canvas_w

    current_area = (fg_mask > 0.5).sum().item()
    if current_area == 0:
        return fg_rgb, fg_mask

    target_area = target_area_ratio * canvas_area
    scale = (target_area / current_area) ** 0.5

    _, fh, fw = fg_rgb.shape
    new_h = max(1, int(fh * scale))
    new_w = max(1, int(fw * scale))

    fg_rgb = F.interpolate(
        fg_rgb.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    fg_mask = F.interpolate(
        fg_mask.unsqueeze(0),
        size=(new_h, new_w),
        mode="nearest",
    ).squeeze(0)

    return fg_rgb, fg_mask


def create_test_cases(
    fg_rgb: Tensor,
    fg_mask: Tensor,
    bg_rgb: Tensor,
    position: tuple[int, int],
    blending_methods: list[str],
    test_consistency: bool,
    rng: Generator | None = None,
) -> list[TestCase]:
    """Generate all test cases."""
    test_cases = []
    _, bh, bw = bg_rgb.shape

    # Define blending methods
    blenders = {
        "hard": (HardPasteBlending(), "Hard Paste: Simple alpha compositing (no edge processing)"),
        "feather": (
            FeatherBlending(feather_radius_range=(2, 8)),
            "Feather: Gaussian blur for edge softening",
        ),
    }

    # Define consistency operations
    consistency_ops = {
        "color_tone_weak": (
            ColorToneMatching(strength=0.3),
            "Color Tone (weak): Color matching 30%",
        ),
        "color_tone_strong": (
            ColorToneMatching(strength=0.7),
            "Color Tone (strong): Color matching 70%",
        ),
        "light_wrap": (
            LightWrap(wrap_radius=5, intensity=0.3),
            "Light Wrap: Background light edge bleed",
        ),
        "shadow": (
            SimpleShadow(offset=(8, 8), blur_radius=12, opacity=0.4),
            "Shadow: Drop shadow",
        ),
        "noise": (NoiseGrainConsistency(noise_std=0.03), "Noise: Add noise grain"),
    }

    # 1. Test each blending method
    for blend_name in blending_methods:
        if blend_name not in blenders:
            logger.warning("Unknown blending method: %s", blend_name)
            continue

        blender, desc = blenders[blend_name]
        result = blender.blend(fg_rgb, fg_mask, bg_rgb, position, rng)

        # Composite mask as well
        composite_mask = torch.zeros(1, bh, bw, device=fg_mask.device)
        y, x = position
        _, fh, fw = fg_mask.shape
        y1 = max(0, y)
        x1 = max(0, x)
        y2 = min(bh, y + fh)
        x2 = min(bw, x + fw)
        fy1 = y1 - y
        fx1 = x1 - x
        fy2 = fy1 + (y2 - y1)
        fx2 = fx1 + (x2 - x1)

        if y2 > y1 and x2 > x1:
            composite_mask[:, y1:y2, x1:x2] = fg_mask[:, fy1:fy2, fx1:fx2]

        composite_mask = (composite_mask > 0.5).float()

        test_cases.append(
            TestCase(
                name=f"blend_{blend_name}",
                description=desc,
                blending=blend_name,
                consistency_ops=[],
                image=result,
                mask=composite_mask,
                fg_mask_original=fg_mask,
            ),
        )

    # 2. Test consistency operations (optional)
    if test_consistency:
        # First composite with hard paste
        hard_blender = HardPasteBlending()
        base_image = hard_blender.blend(fg_rgb, fg_mask, bg_rgb, position, rng)

        # Create mask
        composite_mask = torch.zeros(1, bh, bw, device=fg_mask.device)
        y, x = position
        _, fh, fw = fg_mask.shape
        y1 = max(0, y)
        x1 = max(0, x)
        y2 = min(bh, y + fh)
        x2 = min(bw, x + fw)
        fy1 = y1 - y
        fx1 = x1 - x
        fy2 = fy1 + (y2 - y1)
        fx2 = fx1 + (x2 - x1)

        if y2 > y1 and x2 > x1:
            composite_mask[:, y1:y2, x1:x2] = fg_mask[:, fy1:fy2, fx1:fx2]

        composite_mask = (composite_mask > 0.5).float()

        # Expand foreground only onto canvas
        fg_on_canvas = torch.zeros_like(bg_rgb)
        if y2 > y1 and x2 > x1:
            fg_on_canvas[:, y1:y2, x1:x2] = fg_rgb[:, fy1:fy2, fx1:fx2]

        # Test each consistency operation
        for cons_name, (processor, desc) in consistency_ops.items():
            if cons_name.startswith("color_tone"):
                result = processor.apply(fg_on_canvas, bg_rgb, composite_mask)
                # Re-composite
                result = result * composite_mask + bg_rgb * (1 - composite_mask)
            elif cons_name == "light_wrap":
                result = processor.apply(base_image, bg_rgb, composite_mask)
            elif cons_name == "shadow":
                result = processor.apply(base_image, composite_mask)
            elif cons_name == "noise":
                result = processor.apply(base_image, composite_mask, rng=rng)
            else:
                result = base_image

            test_cases.append(
                TestCase(
                    name=f"consistency_{cons_name}",
                    description=f"Hard Paste + {desc}",
                    blending="hard",
                    consistency_ops=[cons_name],
                    image=result,
                    mask=composite_mask,
                    fg_mask_original=fg_mask,
                ),
            )

        # Test combined processing
        combined = base_image.clone()
        applied_ops = []

        # Apply in order: Color tone -> Light wrap -> Shadow -> Noise
        color_matcher = consistency_ops["color_tone_weak"][0]
        matched_fg = color_matcher.apply(fg_on_canvas, bg_rgb, composite_mask)
        combined = matched_fg * composite_mask + bg_rgb * (1 - composite_mask)
        applied_ops.append("color_tone_weak")

        light_wrap = consistency_ops["light_wrap"][0]
        combined = light_wrap.apply(combined, bg_rgb, composite_mask)
        applied_ops.append("light_wrap")

        shadow = consistency_ops["shadow"][0]
        combined = shadow.apply(combined, composite_mask)
        applied_ops.append("shadow")

        test_cases.append(
            TestCase(
                name="consistency_combined",
                description="Hard Paste + Color/LightWrap/Shadow combined",
                blending="hard",
                consistency_ops=applied_ops,
                image=combined,
                mask=composite_mask,
                fg_mask_original=fg_mask,
            ),
        )

    return test_cases


def create_comparison_grid(
    test_cases: list[TestCase],
    bg_rgb: Tensor,
    fg_rgb: Tensor,
    fg_mask: Tensor,
) -> Image.Image:
    """Generate comparison grid image."""
    if not test_cases:
        return Image.new("RGB", (100, 100), "gray")

    # Get size of each image
    _, h, w = test_cases[0].image.shape
    n = len(test_cases)

    # Calculate grid layout
    cols = min(4, n + 2)  # Original images + mask + test cases, max 4 columns
    rows = (n + 2 + cols - 1) // cols

    # Create grid image
    grid_w = cols * w
    grid_h = rows * h
    grid = Image.new("RGB", (grid_w, grid_h), "gray")

    # Place background and foreground first
    bg_pil = tensor_to_pil(bg_rgb)
    bg_pil = add_label_to_image(bg_pil, "Background")
    grid.paste(bg_pil, (0, 0))

    # Foreground (with mask applied)
    fg_with_mask = fg_rgb * fg_mask + torch.ones_like(fg_rgb) * 0.5 * (1 - fg_mask)
    fg_pil = tensor_to_pil(fg_with_mask)
    fg_pil = add_label_to_image(fg_pil, "Foreground")
    grid.paste(fg_pil, (w, 0))

    # Place each test case
    for i, tc in enumerate(test_cases):
        row = (i + 2) // cols
        col = (i + 2) % cols
        x = col * w
        y = row * h

        img_pil = tensor_to_pil(tc.image)
        img_pil = add_label_to_image(img_pil, tc.name)
        grid.paste(img_pil, (x, y))

    return grid


def save_detailed_results(
    output_dir: Path,
    test_cases: list[TestCase],
    bg_rgb: Tensor,
    fg_rgb: Tensor,
    fg_mask: Tensor,
    sample_idx: int,
) -> None:
    """Save detailed results."""
    sample_dir = output_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save original images
    tensor_to_pil(bg_rgb).save(sample_dir / "00_background.png")

    fg_with_alpha = torch.cat([fg_rgb, fg_mask], dim=0)
    fg_arr = (fg_with_alpha.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(fg_arr, "RGBA").save(sample_dir / "01_foreground_rgba.png")

    # Save test case results
    results_info = []
    for i, tc in enumerate(test_cases):
        idx = i + 2
        base_name = f"{idx:02d}_{tc.name}"

        # Save image
        tensor_to_pil(tc.image).save(sample_dir / f"{base_name}_image.png")

        # Save mask
        mask_arr = (tc.mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_arr, "L").save(sample_dir / f"{base_name}_mask.png")

        # Overlay (highlight mask boundary)
        img_pil = tensor_to_pil(tc.image).convert("RGBA")
        overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Detect mask edges and draw in red
        mask_np = tc.mask.squeeze(0).cpu().numpy()
        from scipy import ndimage

        edges = ndimage.binary_dilation(mask_np > 0.5) ^ (mask_np > 0.5)
        edge_points = np.argwhere(edges)
        for py, px in edge_points:
            draw.point((px, py), fill=(255, 0, 0, 200))

        img_pil = Image.alpha_composite(img_pil, overlay)
        img_pil.convert("RGB").save(sample_dir / f"{base_name}_overlay.png")

        # Metadata
        results_info.append(
            {
                "name": tc.name,
                "description": tc.description,
                "blending": tc.blending,
                "consistency_ops": tc.consistency_ops,
                "mask_coverage": float((tc.mask > 0.5).sum().item() / tc.mask.numel()),
            },
        )

    # Save comparison grid
    grid = create_comparison_grid(test_cases, bg_rgb, fg_rgb, fg_mask)
    grid.save(sample_dir / "comparison_grid.png")

    # Save metadata as JSON
    with Path(sample_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(results_info, f, indent=2, ensure_ascii=False)

    logger.info("  Saved to: %s", sample_dir)


def run_detailed_test(
    fg_dir: Path,
    bg_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
    target_size: tuple[int, int] = (512, 512),
    target_area_ratio: float = 0.25,
    blending_methods: list[str] | None = None,
    test_consistency: bool = True,
    seed: int | None = None,
) -> None:
    """Run detailed test."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default blending methods
    if blending_methods is None:
        blending_methods = ["hard", "feather"]

    # Initialize pools
    logger.info("ForegroundPool: %s", fg_dir)
    fg_pool = ForegroundPool(fg_dir)
    logger.info("  Image count: %d", len(fg_pool))

    logger.info("BackgroundPool: %s", bg_dir)
    bg_pool = BackgroundPool(bg_dir)
    logger.info("  Image count: %d", len(bg_pool))

    # Random generator
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    else:
        rng.manual_seed(42)  # Fixed seed for reproducibility

    logger.info("=" * 60)
    logger.info("Test settings:")
    logger.info("  Blending methods: %s", blending_methods)
    logger.info("  Consistency test: %s", test_consistency)
    logger.info("  Number of samples: %d", num_samples)
    logger.info("  Target size: %s", target_size)
    logger.info("  Foreground area ratio: %.1f%%", target_area_ratio * 100)
    logger.info("=" * 60)

    for i in range(num_samples):
        logger.info("[%d/%d] Generating sample...", i + 1, num_samples)

        # Sample foreground and background
        fg_rgb, fg_mask = fg_pool.sample(rng)
        bg_rgb = bg_pool.sample(target_size, rng)

        # Crop foreground to content
        fg_rgb, fg_mask = crop_to_content(fg_rgb, fg_mask, padding=5)

        # Scale foreground to target size
        fg_rgb, fg_mask = scale_foreground_to_target_area(
            fg_rgb,
            fg_mask,
            target_area_ratio,
            target_size,
        )

        # Place at center
        _, fh, fw = fg_rgb.shape
        h, w = target_size
        position = ((h - fh) // 2, (w - fw) // 2)

        # Generate test cases
        test_cases = create_test_cases(
            fg_rgb=fg_rgb,
            fg_mask=fg_mask,
            bg_rgb=bg_rgb,
            position=position,
            blending_methods=blending_methods,
            test_consistency=test_consistency,
            rng=rng,
        )

        # Save results
        save_detailed_results(
            output_dir=output_dir,
            test_cases=test_cases,
            bg_rgb=bg_rgb,
            fg_rgb=fg_rgb,
            fg_mask=fg_mask,
            sample_idx=i,
        )

    logger.info("=" * 60)
    logger.info("Complete!")
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Each sample directory contains:")
    logger.info("    - 00_background.png: Background image")
    logger.info("    - 01_foreground_rgba.png: Foreground image (RGBA)")
    logger.info("    - XX_<method>_image.png: Composite image after processing")
    logger.info("    - XX_<method>_mask.png: Mask")
    logger.info("    - XX_<method>_overlay.png: Mask boundary highlight")
    logger.info("    - comparison_grid.png: Comparison grid")
    logger.info("    - info.json: Processing information")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detailed test for synthesis processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fg_dir",
        type=Path,
        required=True,
        help="Directory of foreground images (RGBA PNG)",
    )
    parser.add_argument(
        "--bg_dir",
        type=Path,
        required=True,
        help="Directory of background images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/synthesis_detailed"),
        help="Output directory (default: output/synthesis_detailed)",
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
        "--area_ratio",
        type=float,
        default=0.25,
        help="Foreground area ratio (default: 0.25)",
    )
    parser.add_argument(
        "--blending",
        nargs="+",
        default=["hard", "feather"],
        help="Blending methods to test (default: all)",
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        default=True,
        help="Also test consistency operations (default: True)",
    )
    parser.add_argument(
        "--no-consistency",
        action="store_false",
        dest="consistency",
        help="Skip consistency operations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    run_detailed_test(
        fg_dir=args.fg_dir,
        bg_dir=args.bg_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_size=(args.size, args.size),
        target_area_ratio=args.area_ratio,
        blending_methods=args.blending,
        test_consistency=args.consistency,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
