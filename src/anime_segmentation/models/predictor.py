import torch
from PIL import Image
from torchvision import transforms

from anime_segmentation.constants import IMAGENET_MEAN, IMAGENET_STD
from anime_segmentation.exceptions import InvalidInputError, InvalidTargetSizeError

from .birefnet import BiRefNet


def _validate_image(image: Image.Image | None) -> Image.Image:
    """Validate that image is a valid PIL Image."""
    if image is None:
        msg = "image cannot be None"
        raise InvalidInputError(msg)
    if not isinstance(image, Image.Image):
        msg = f"Expected PIL.Image.Image, got {type(image).__name__}"
        raise InvalidInputError(msg)
    return image


def _validate_target_size(target_size: tuple[int, int]) -> tuple[int, int]:
    """Validate that target_size is a valid (width, height) tuple."""
    if not isinstance(target_size, tuple):
        msg = f"target_size must be a tuple, got {type(target_size).__name__}"
        raise InvalidTargetSizeError(msg)
    if len(target_size) != 2:
        msg = f"target_size must be (width, height), got {target_size}"
        raise InvalidTargetSizeError(msg)
    width, height = target_size
    if not isinstance(width, int) or not isinstance(height, int):
        msg = f"target_size dimensions must be integers, got ({type(width).__name__}, {type(height).__name__})"
        raise InvalidTargetSizeError(msg)
    if width <= 0 or height <= 0:
        msg = f"target_size dimensions must be positive, got {target_size}"
        raise InvalidTargetSizeError(msg)
    return target_size


def _validate_threshold(threshold: float) -> float:
    """Validate that threshold is within [0, 1]."""
    if not isinstance(threshold, (float, int)):
        msg = f"threshold must be a float, got {type(threshold).__name__}"
        raise InvalidInputError(msg)
    threshold_f = float(threshold)
    if not 0.0 <= threshold_f <= 1.0:
        msg = f"threshold must be in [0, 1], got {threshold_f}"
        raise InvalidInputError(msg)
    return threshold_f


class BiRefNetPredictor:
    def __init__(
        self,
        model_name: str = "ZhengPeng7/BiRefNet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compile: bool = False,
        compile_mode: str = "default",
    ) -> None:
        self.device = device
        self.model = BiRefNet.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if compile:
            self.model = torch.compile(self.model, mode=compile_mode)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ],
        )

    def preprocess(
        self,
        image: Image.Image,
        target_size: tuple[int, int] = (1024, 1024),
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess image for model input.

        Args:
            image: PIL Image to preprocess.
            target_size: Target size (width, height) for model input.

        Returns:
            Tuple of (input_tensor, original_size).

        Raises:
            InvalidInputError: If image is not a valid PIL Image.
            InvalidTargetSizeError: If target_size is not a valid (w, h) tuple.
        """
        image = _validate_image(image)
        target_size = _validate_target_size(target_size)

        w, h = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_resized = image.resize(target_size, Image.Resampling.BILINEAR)
        input_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
        return input_tensor, (w, h)

    def predict(
        self,
        image: Image.Image,
        target_size: tuple[int, int] = (1024, 1024),
        *,
        binarize: bool = True,
        threshold: float = 0.5,
    ) -> Image.Image:
        """Predict segmentation mask for the given image.

        Args:
            image: PIL Image to segment.
            target_size: Target size (width, height) for model input.
            binarize: Whether to return a binary mask.
            threshold: Threshold for binarization (0-1).

        Returns:
            Grayscale PIL Image mask.

        Raises:
            InvalidInputError: If image is not a valid PIL Image.
            InvalidTargetSizeError: If target_size is not a valid (w, h) tuple.
        """
        input_tensor, original_size = self.preprocess(image, target_size)
        threshold = _validate_threshold(threshold)

        with torch.inference_mode():
            preds = self.model(input_tensor)
            if not isinstance(preds, (list, tuple)):
                msg = f"Model returned unexpected type: {type(preds).__name__}, expected list or tuple"
                raise InvalidInputError(msg)
            if len(preds) == 0:
                msg = "Model returned empty predictions"
                raise InvalidInputError(msg)
            pred = preds[-1]  # finest scale
            pred = pred.sigmoid().cpu()
            pred = torch.nn.functional.interpolate(
                pred,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=True,
            )
            pred = pred.squeeze()  # [H, W]
            if binarize:
                pred = (pred >= threshold).float()

        return transforms.ToPILImage()(pred)
