"""Simple inference script for anime segmentation.

Usage:
    python scripts/inference.py --ckpt ckpts/anime_seg_epoch_28.ckpt --input image.png --output mask.png
    python scripts/inference.py --ckpt ckpts/anime_seg_epoch_28.ckpt --input_dir images/ --output_dir masks/
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from anime_segmentation.training.lightning_module import BiRefNetLightning


class AnimeSegmentationPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device

        # Load model from checkpoint
        self.model = BiRefNetLightning.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict_loading=False,
        )
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
        )

    def predict(
        self,
        image: Image.Image,
        target_size: tuple[int, int] = (1024, 1024),
        *,
        binarize: bool = True,
        threshold: float = 0.5,
    ) -> Image.Image:
        """Returns a grayscale PIL Image mask for the given image."""
        w, h = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        image_resized = image.resize(target_size, Image.Resampling.BILINEAR)
        input_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)

        # Inference
        with torch.inference_mode():
            preds = self.model(input_tensor)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
            pred = pred.sigmoid().cpu()
            pred = torch.nn.functional.interpolate(
                pred,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            )
            pred = pred.squeeze()
            if binarize:
                pred = (pred >= threshold).float()

        return transforms.ToPILImage()(pred)


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply mask to image, returning RGBA with transparent background."""
    image = image.convert("RGBA")
    mask = mask.convert("L")
    image.putalpha(mask)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Anime segmentation inference")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--input_dir", help="Input directory with images")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--size", type=int, default=1024, help="Inference size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold (default: 0.5)",
    )
    parser.add_argument(
        "--prob-map",
        action="store_true",
        help="Output probability map instead of binary mask",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Output mask only (default: output RGBA image)",
    )
    args = parser.parse_args()

    predictor = AnimeSegmentationPredictor(args.ckpt, device=args.device)

    if args.input:
        # Single image
        image = Image.open(args.input)
        mask = predictor.predict(
            image,
            target_size=(args.size, args.size),
            binarize=not args.prob_map,
            threshold=args.threshold,
        )

        if args.mask_only:
            output_path = args.output or Path(args.input).stem + "_mask.png"
            mask.save(output_path)
        else:
            output_path = args.output or Path(args.input).stem + "_rgba.png"
            result = apply_mask(image, mask)
            result.save(output_path)
        print(f"Saved: {output_path}")

    elif args.input_dir:
        # Directory
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or "output")
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        for img_path in images:
            image = Image.open(img_path)
            mask = predictor.predict(
                image,
                target_size=(args.size, args.size),
                binarize=not args.prob_map,
                threshold=args.threshold,
            )

            if args.mask_only:
                output_path = output_dir / f"{img_path.stem}_mask.png"
                mask.save(output_path)
            else:
                output_path = output_dir / f"{img_path.stem}_rgba.png"
                result = apply_mask(image, mask)
                result.save(output_path)
            print(f"Saved: {output_path}")

    else:
        parser.error("Either --input or --input_dir is required")


if __name__ == "__main__":
    main()
