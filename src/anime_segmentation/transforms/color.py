"""Color transforms for anime segmentation."""

# ruff: noqa: ARG002

import random
from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

# Note: GaussianNoise has been replaced by torchvision.transforms.v2.GaussianNoise
# Use: v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.05)], p=0.5)


class RandomColor(v2.Transform):
    """Random brightness and contrast adjustment with specific ranges.

    This replicates the original RandomColor behavior:
    - 50% probability of applying the transform
    - Brightness: either dim (0.4-0.5) or normal/bright (1.0-1.2)
    - Contrast: either low (0.4-0.5) or normal/high (1.0-1.5)

    Args:
        p: Probability of applying the transform. Default is 0.5.
    """

    _transformed_types = (tv_tensors.Image,)

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        if torch.rand(1).item() > self.p:
            return {"apply": False, "low_definition": False}

        brightness = random.choice(
            [
                random.uniform(0.4, 0.5),
                random.uniform(1.0, 1.2),
            ]
        )
        contrast = random.choice(
            [
                random.uniform(0.4, 0.5),
                random.uniform(1.0, 1.5),
            ]
        )
        return {
            "apply": True,
            "brightness": brightness,
            "contrast": contrast,
            "low_definition": brightness <= 0.5 and contrast <= 0.5,
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt

        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        result = F.adjust_brightness(inpt, params["brightness"])
        result = F.adjust_contrast(result, params["contrast"])
        return tv_tensors.wrap(result, like=inpt)  # type: ignore[call-arg]
