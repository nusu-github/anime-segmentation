"""Geometric transforms for anime segmentation."""

from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class RescalePad(v2.Transform):
    """Rescale image to fit output_size (longest edge), then pad to square.

    This transform:
    1. Resizes the image so the longest edge equals output_size
    2. Pads the shorter dimension to create a square output

    Args:
        output_size: Target size for the output (both height and width).
        fill: Fill value for padding. Default is 0.
    """

    def __init__(self, output_size: int, fill: float = 0) -> None:
        super().__init__()
        self.output_size = output_size
        self.fill = fill

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        # Find first image or mask to get dimensions
        for inpt in flat_inputs:
            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Mask, torch.Tensor)):
                h, w = inpt.shape[-2:]
                if h == self.output_size and w == self.output_size:
                    return {"skip": True}

                r = min(self.output_size / h, self.output_size / w)
                new_h, new_w = int(h * r), int(w * r)
                ph = self.output_size - new_h
                pw = self.output_size - new_w
                return {
                    "skip": False,
                    "new_size": (new_h, new_w),
                    "padding": (pw // 2, ph // 2, pw // 2 + pw % 2, ph // 2 + ph % 2),
                }
        return {"skip": True}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if params.get("skip", True):
            return inpt

        if isinstance(inpt, (tv_tensors.Image, tv_tensors.Mask, torch.Tensor)):
            resized = F.resize(inpt, list(params["new_size"]))
            padded = F.pad(resized, list(params["padding"]), fill=self.fill)
            if isinstance(inpt, tv_tensors.TVTensor):
                return tv_tensors.wrap(padded, like=inpt)  # type: ignore[call-arg]
            return padded
        return inpt
