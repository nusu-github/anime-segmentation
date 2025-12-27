"""Complex augmentation transforms extracted from DatasetGenerator."""

# ruff: noqa: ARG002

import math
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


def _to_numpy_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor to HWC numpy array."""
    return tensor.permute(1, 2, 0).numpy()


def _to_tensor_chw(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """Convert HWC numpy array to CHW tensor."""
    result = torch.from_numpy(array).permute(2, 0, 1)
    if isinstance(like, tv_tensors.TVTensor):
        return tv_tensors.wrap(result, like=like)  # type: ignore[call-arg]
    return result


class SharpBackground(v2.Transform):
    """Generate masked background with random polygon contours.

    Creates a random polygon mask and makes the area outside white.
    Optionally draws a colored edge on the polygon boundary.

    Note: This transform is designed to be applied BEFORE compositing fg onto bg.
    When used on a composited image, it affects the entire image.

    Args:
        p: Probability of applying the transform. Default is 0.5.
    """

    _transformed_types = (tv_tensors.Image,)

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if torch.rand(1).item() > self.p:
            return {"apply": False}

        # Find image dimensions
        for inpt in flat_inputs:
            if isinstance(inpt, (tv_tensors.Image, torch.Tensor)):
                h, w = inpt.shape[-2:]

                # Generate contour points (50 points in polar coordinates)
                d = 50
                ms = max(h, w)
                counts = []
                for i in range(d):
                    radius = random.randint(ms * 2 // 10, ms * 6 // 10)
                    x = int(w // 2 + radius * math.cos(math.radians(i / d * 360)))
                    y = int(h // 2 + radius * math.sin(math.radians(i / d * 360)))
                    counts.append([x, y])

                return {
                    "apply": True,
                    "contour_points": counts,
                    "output_size": (h, w),
                    "draw_edge": random.random() < 0.5,
                    "edge_color": (random.random(), random.random(), random.random()),
                    "edge_thickness": random.randint(max(1, ms // 600), max(2, ms // 400)),
                }

        return {"apply": False}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        h, w = params["output_size"]
        counts = [np.array(params["contour_points"], dtype=np.int32)]

        # Create mask
        bg_mask = cv2.drawContours(
            np.zeros((h, w, 1), dtype=np.float32), counts, 0, (1.0,), cv2.FILLED
        )

        # Convert image to numpy
        img_np = _to_numpy_hwc(inpt.float())

        # Apply mask: inside = image, outside = white
        result = img_np * bg_mask + (1 - bg_mask)

        # Optionally draw edge
        if params["draw_edge"]:
            result = cv2.drawContours(
                result, counts, 0, params["edge_color"], params["edge_thickness"]
            )

        return _to_tensor_chw(result.astype(np.float32), inpt)


class SketchConvert(v2.Transform):
    """Convert image to sketch style.

    Supports 3 modes:
    - Mode 0: Grayscale with edges
    - Mode 1: Quantized grayscale with edges
    - Mode 2: Pure edges

    Args:
        p: Probability of applying the transform. Default is 0.25.
    """

    def __init__(self, p: float = 0.25) -> None:
        super().__init__()
        self.p = p

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if torch.rand(1).item() > self.p:
            return {"apply": False}

        return {
            "apply": True,
            "mode": random.randint(0, 2),
            "use_gray_mask": random.random() < 0.5,
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if isinstance(inpt, tv_tensors.Mask):
            # Need mask for processing but don't modify it
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        img_np = _to_numpy_hwc(inpt.float())

        # Convert to grayscale
        image_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        image_gray = cv2.GaussianBlur(image_gray, (3, 3), sigmaX=0, sigmaY=0)[:, :, np.newaxis]

        # Edge detection
        image_edge = (
            cv2.adaptiveThreshold(
                (image_gray * 255).astype(np.uint8),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=5,
                C=5,
            ).astype(np.float32)
            / 255
        )
        image_edge = image_edge[:, :, np.newaxis]

        mode = params["mode"]
        if mode == 0:
            # Grayscale with edges
            result = (image_gray * image_edge).repeat(3, axis=2)
        elif mode == 1:
            # Quantized grayscale with edges
            threshold = image_gray.mean()
            image_gray_q = image_gray.copy()
            image_gray_q[image_gray_q > threshold] = 1
            image_gray_q = np.floor(image_gray_q * 3) / 3
            result = (image_gray_q * image_edge).repeat(3, axis=2)
        else:  # mode == 2
            # Pure edges
            result = image_edge.repeat(3, axis=2)

        return _to_tensor_chw(result.astype(np.float32), inpt)


class RandomColorBlocks(v2.Transform):
    """Overlay random colored rectangles and circles.

    Args:
        p: Probability of applying the transform. Default is 0.5.
        num_blocks_range: Range for number of blocks. Default is (1, 10).
    """

    _transformed_types = (tv_tensors.Image,)

    def __init__(self, p: float = 0.5, num_blocks_range: tuple[int, int] = (1, 10)) -> None:
        super().__init__()
        self.p = p
        self.num_blocks_range = num_blocks_range

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if torch.rand(1).item() > self.p:
            return {"apply": False}

        # Find image dimensions
        for inpt in flat_inputs:
            if isinstance(inpt, (tv_tensors.Image, torch.Tensor)):
                h, w = inpt.shape[-2:]

                # Generate block parameters
                num_blocks = random.randint(*self.num_blocks_range)
                blocks = []
                for _ in range(num_blocks):
                    is_rect = random.random() < 0.5
                    color = (
                        random.random(),
                        random.random(),
                        random.random(),
                        random.uniform(0.2, 0.3),  # alpha
                    )
                    if is_rect:
                        bw = random.randint(w // 20, w // 3)
                        bh = random.randint(h // 20, h // 3)
                        x = random.randint(0, w - bw)
                        y = random.randint(0, h - bh)
                        blocks.append(
                            {"type": "rect", "x": x, "y": y, "w": bw, "h": bh, "color": color}
                        )
                    else:
                        r = random.randint((h + w) // 40, (h + w) // 8)
                        x = random.randint(r, w - r)
                        y = random.randint(r, h - r)
                        blocks.append({"type": "circle", "x": x, "y": y, "r": r, "color": color})

                return {
                    "apply": True,
                    "output_size": (h, w),
                    "blocks": blocks,
                    "rotation_angle": random.randint(-90, 90),
                }

        return {"apply": False}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        h, w = params["output_size"]
        img_np = _to_numpy_hwc(inpt.float())

        # Create overlay with alpha channel
        temp_img = np.zeros((h, w, 4), dtype=np.float32)

        for block in params["blocks"]:
            color = block["color"]
            if block["type"] == "rect":
                cv2.rectangle(
                    temp_img,
                    (block["x"], block["y"]),
                    (block["x"] + block["w"], block["y"] + block["h"]),
                    color,
                    cv2.FILLED,
                )
            else:
                cv2.circle(temp_img, (block["x"], block["y"]), block["r"], color, cv2.FILLED)

        # Apply rotation
        angle = params["rotation_angle"]
        trans_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        temp_img = cv2.warpAffine(
            temp_img, trans_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # Alpha blend
        overlay, mask = temp_img[:, :, :3], temp_img[:, :, 3:]
        result = mask * overlay + (1 - mask) * img_np

        return _to_tensor_chw(result.astype(np.float32), inpt)


class RandomTextOverlay(v2.Transform):
    """Overlay random Japanese text (Hiragana/Katakana).

    Args:
        p: Probability of applying the transform. Default is 0.5.
        font_path: Path to font file. Default uses bundled font.
        num_texts_range: Range for number of text strings. Default is (1, 10).
    """

    _transformed_types = (tv_tensors.Image,)

    def __init__(
        self,
        p: float = 0.5,
        font_path: str | Path | None = None,
        num_texts_range: tuple[int, int] = (1, 10),
    ) -> None:
        super().__init__()
        self.p = p
        self.font_path = font_path or (Path(__file__).parent.parent / "assets" / "font.otf")
        self.num_texts_range = num_texts_range
        self.texts = [chr(x) for x in range(0x3040, 0x30FF + 1)]
        self._fonts: list[ImageFont.FreeTypeFont] | None = None

    def _get_fonts(self, min_size: int) -> list[ImageFont.FreeTypeFont]:
        if self._fonts is None:
            self._fonts = [
                ImageFont.truetype(str(self.font_path), x, encoding="utf-8")
                for x in range(min_size, min_size * 5, 2)
            ]
        return self._fonts

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if torch.rand(1).item() > self.p:
            return {"apply": False}

        # Find image dimensions
        for inpt in flat_inputs:
            if isinstance(inpt, (tv_tensors.Image, torch.Tensor)):
                h, w = inpt.shape[-2:]

                num_texts = random.randint(*self.num_texts_range)
                texts = []
                for _ in range(num_texts):
                    text = "".join(random.choice(self.texts) for _ in range(10))
                    font_idx = random.randint(0, 10)  # Will be clamped later
                    is_white = random.random() < 0.5
                    x = random.randint(0, w)
                    y = random.randint(0, h)
                    texts.append(
                        {
                            "text": text,
                            "font_idx": font_idx,
                            "is_white": is_white,
                            "x": x,
                            "y": y,
                        }
                    )

                return {
                    "apply": True,
                    "output_size": (h, w),
                    "texts": texts,
                }

        return {"apply": False}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        h, w = params["output_size"]
        min_size = min(h, w) // 100

        try:
            fonts = self._get_fonts(max(1, min_size))
        except OSError:
            # Font not found, skip transform
            return inpt

        # Convert to PIL
        img_np = _to_numpy_hwc(inpt.float())
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)

        for text_params in params["texts"]:
            font_idx = min(text_params["font_idx"], len(fonts) - 1)
            font = fonts[font_idx]
            color = (255, 255, 255) if text_params["is_white"] else (0, 0, 0)
            draw.text((text_params["x"], text_params["y"]), text_params["text"], color, font=font)

        result = np.asarray(pil_img).astype(np.float32) / 255
        return _to_tensor_chw(result, inpt)


def _vector_included_angle(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Calculate included angle between two vectors."""
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    a = a1 - a2
    if a > math.pi:
        a = a - math.pi * 2
    if a < -math.pi:
        a = a + math.pi * 2
    return a


class SimulateLight(v2.Transform):
    """Simulate light source with radial lines.

    Creates a lighting effect from a random position outside the image.

    Args:
        p: Probability of applying the transform. Default is 0.5.
        strength: Light intensity variation. Default is 0.2.
    """

    _transformed_types = (tv_tensors.Image,)

    def __init__(self, p: float = 0.5, strength: float = 0.2) -> None:
        super().__init__()
        self.p = p
        self.strength = strength

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if torch.rand(1).item() > self.p:
            return {"apply": False}

        # Find image dimensions
        for inpt in flat_inputs:
            if isinstance(inpt, (tv_tensors.Image, torch.Tensor)):
                h, w = inpt.shape[-2:]

                # Calculate light source position (outside image)
                a = int(np.linalg.norm([h, w]) / 2)
                r = random.randint(a * 11 // 10, a * 2)
                b = random.uniform(0, math.pi * 2)
                cx = int(w // 2 + r * math.cos(b))
                cy = int(h // 2 + r * math.sin(b))

                return {
                    "apply": True,
                    "output_size": (h, w),
                    "light_center": (cx, cy),
                    "use_inverse": random.random() < 0.5,
                    "color": (
                        random.uniform(1 - self.strength, 1),
                        random.uniform(1 - self.strength, 1),
                        random.uniform(1 - self.strength, 1),
                    ),
                }

        return {"apply": False}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        h, w = params["output_size"]
        cx, cy = params["light_center"]
        strength = self.strength

        c_v = (w // 2 - cx, h // 2 - cy)

        # Calculate angles to corners
        rs = [
            _vector_included_angle((-cx, -cy), c_v),
            _vector_included_angle((w - cx, -cy), c_v),
            _vector_included_angle((-cx, h - cy), c_v),
            _vector_included_angle((w - cx, h - cy), c_v),
        ]

        # Calculate distances to corners
        ds = [
            np.linalg.norm((-cx, -cy)),
            np.linalg.norm((w - cx, -cy)),
            np.linalg.norm((-cx, h - cy)),
            np.linalg.norm((w - cx, h - cy)),
        ]
        r2 = max(ds)

        cr = math.atan2(c_v[1], c_v[0])
        if cr < 0:
            cr = math.pi * 2 + cr

        sr = min(rs) + cr
        er = max(rs) + cr
        n = int(50 * (er - sr) * 2 / math.pi)

        color = params["color"]
        if params["use_inverse"]:
            light_mask = np.full(
                (h, w, 3), (1 + strength, 1 + strength, 1 + strength), dtype=np.float32
            )
            line_color = color
        else:
            light_mask = np.full((h, w, 3), color, dtype=np.float32)
            line_color = (1 + strength, 1 + strength, 1 + strength)

        # Draw radial lines
        for angle in np.linspace(sr, er, num=max(n, 1)):
            x2 = int(cx + r2 * math.cos(angle))
            y2 = int(cy + r2 * math.sin(angle))
            light_mask = cv2.line(light_mask, (cx, cy), (x2, y2), line_color, 10)

        img_np = _to_numpy_hwc(inpt.float())
        result = (img_np * light_mask).clip(0, 1)

        return _to_tensor_chw(result.astype(np.float32), inpt)


class ResizeBlur(v2.Transform):
    """Blur by downscaling then upscaling.

    Args:
        p: Probability of applying the transform. Default is 0.5.
        scale_factor: Factor to downscale by. Default is 2.
    """

    _transformed_types = (tv_tensors.Image,)

    def __init__(self, p: float = 0.5, scale_factor: int = 2) -> None:
        super().__init__()
        self.p = p
        self.scale_factor = scale_factor

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "apply": torch.rand(1).item() < self.p,
            "interpolation": random.choice(
                [
                    v2.InterpolationMode.BILINEAR,
                    v2.InterpolationMode.NEAREST,
                ]
            ),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        h, w = inpt.shape[-2:]
        small_size = (h // self.scale_factor, w // self.scale_factor)

        small = F.resize(inpt, list(small_size))
        result = F.resize(small, [h, w], interpolation=params["interpolation"])

        return tv_tensors.wrap(result, like=inpt)  # type: ignore[call-arg]


class JPEGCompression(v2.Transform):
    """Simulate JPEG compression artifacts.

    Args:
        p: Probability of applying the transform. Default is 0.5.
        quality_range: Range for JPEG quality (lower = more artifacts). Default is (20, 70).
    """

    _transformed_types = (tv_tensors.Image,)

    def __init__(self, p: float = 0.5, quality_range: tuple[int, int] = (20, 70)) -> None:
        super().__init__()
        self.p = p
        self.quality_range = quality_range

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "apply": torch.rand(1).item() < self.p,
            "quality": random.randint(*self.quality_range),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        # Convert to PIL
        img_np = _to_numpy_hwc(inpt.float())
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))

        # Compress via JPEG
        buffer = BytesIO()
        pil_img.save(buffer, "JPEG", quality=params["quality"], optimize=True)
        buffer.seek(0)

        # Reload
        result = np.asarray(Image.open(buffer)).astype(np.float32) / 255

        return _to_tensor_chw(result, inpt)


# Note: RandomRotation180 has been replaced by torchvision.transforms.v2.RandomRotation
# Use: v2.RandomRotation(degrees=(-180, 180), interpolation=v2.InterpolationMode.BILINEAR, fill=0.0)
