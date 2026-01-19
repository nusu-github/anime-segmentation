import torch
from PIL import Image
from torchvision import transforms

from anime_segmentation.constants import IMAGENET_MEAN, IMAGENET_STD
from anime_segmentation.exceptions import InvalidInputError, InvalidTargetSizeError

from .birefnet import BiRefNet


def _validate_image(image: Image.Image | None) -> Image.Image:
    """Validate that image is a valid PIL Image."""
    if image is None:
        raise InvalidInputError("image cannot be None")
    if not isinstance(image, Image.Image):
        raise InvalidInputError(
            f"Expected PIL.Image.Image, got {type(image).__name__}"
        )
    return image


def _validate_target_size(target_size: tuple[int, int]) -> tuple[int, int]:
    """Validate that target_size is a valid (width, height) tuple."""
    if not isinstance(target_size, tuple):
        raise InvalidTargetSizeError(
            f"target_size must be a tuple, got {type(target_size).__name__}"
        )
    if len(target_size) != 2:
        raise InvalidTargetSizeError(
            f"target_size must be (width, height), got {target_size}"
        )
    width, height = target_size
    if not isinstance(width, int) or not isinstance(height, int):
        raise InvalidTargetSizeError(
            f"target_size dimensions must be integers, got ({type(width).__name__}, {type(height).__name__})"
        )
    if width <= 0 or height <= 0:
        raise InvalidTargetSizeError(
            f"target_size dimensions must be positive, got {target_size}"
        )
    return target_size


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
        self, image: Image.Image, target_size: tuple[int, int] = (1024, 1024)
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
        self, image: Image.Image, target_size: tuple[int, int] = (1024, 1024)
    ) -> Image.Image:
        """Predict segmentation mask for the given image.

        Args:
            image: PIL Image to segment.
            target_size: Target size (width, height) for model input.

        Returns:
            Grayscale PIL Image mask.

        Raises:
            InvalidInputError: If image is not a valid PIL Image.
            InvalidTargetSizeError: If target_size is not a valid (w, h) tuple.
        """
        input_tensor, original_size = self.preprocess(image, target_size)

        with torch.inference_mode():
            preds = self.model(input_tensor)
            if not isinstance(preds, (list, tuple)):
                raise InvalidInputError(
                    f"Model returned unexpected type: {type(preds).__name__}, expected list or tuple"
                )
            if len(preds) == 0:
                raise InvalidInputError("Model returned empty predictions")
            pred = preds[-1]  # finest scale
            pred = pred.sigmoid().cpu()
            pred = torch.nn.functional.interpolate(
                pred,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=True,
            )
            pred = pred.squeeze()  # [H, W]

        return transforms.ToPILImage()(pred)
