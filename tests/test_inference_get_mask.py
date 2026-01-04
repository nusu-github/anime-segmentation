from __future__ import annotations

import numpy as np
import torch

from anime_segmentation.inference import get_mask


class _DummyModel:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Return a simple constant mask-like tensor: (N, C=1, H, W)
        n, _, h, w = x.shape
        return torch.full((n, 1, h, w), 0.75, device=x.device, dtype=torch.float32)


def test_get_mask_shape_and_range_cpu() -> None:
    model = _DummyModel(torch.device("cpu"))

    h, w = 37, 53
    rng = np.random.default_rng(0)
    img = (rng.random((h, w, 3)) * 255).astype(np.float32)

    mask = get_mask(model, img, use_amp=False, s=64)

    assert mask.shape == (h, w, 1)
    assert mask.dtype in {np.dtype("float32"), np.dtype("float64")}
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
