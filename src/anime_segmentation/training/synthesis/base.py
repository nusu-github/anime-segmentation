"""Abstract base classes for the synthesis pipeline components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Generator, Tensor


@dataclass
class ValidationResult:
    """Result of a validation pass."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class BaseCompositor(ABC):
    """Interface for copy-paste compositors."""

    @abstractmethod
    def synthesize(
        self,
        target_size: tuple[int, int],
        rng: Generator | None = None,
        *,
        return_background: bool = False,
    ) -> tuple[Tensor, Tensor, int] | tuple[Tensor, Tensor, int, Tensor]: ...

    def validate_output(self, image: Tensor, mask: Tensor) -> bool:
        if torch.isnan(image).any() or torch.isinf(image).any():
            return False
        if torch.isnan(mask).any():
            return False
        if image.min() < -0.1 or image.max() > 1.1:
            return False
        return not (mask.min() < 0 or mask.max() > 1)


class BaseConsistencyPipeline(ABC):
    """Interface for consistency processing."""

    @abstractmethod
    def apply(
        self,
        fg: Tensor,
        bg: Tensor,
        mask: Tensor,
        canvas: Tensor,
        rng: Generator | None = None,
    ) -> Tensor: ...

    def clamp_output(self, image: Tensor) -> Tensor:
        return torch.clamp(image, 0.0, 1.0)


class BaseDegradation(ABC):
    """Interface for degradation processing."""

    @abstractmethod
    def __call__(
        self,
        image: Tensor,
        mask: Tensor,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]: ...

    def should_apply(self, prob: float, rng: Generator | None = None) -> bool:
        if rng is not None:
            return torch.rand(1, generator=rng).item() < prob
        return torch.rand(1).item() < prob


class BaseValidator(ABC):
    """Interface for data validators."""

    @abstractmethod
    def validate(self, image: Tensor, mask: Tensor) -> ValidationResult: ...

    def check_shapes(self, image: Tensor, mask: Tensor) -> list[str]:
        errors: list[str] = []
        img = image[0] if image.ndim == 4 else image
        msk = mask[0] if mask.ndim == 4 else mask
        _, img_h, img_w = img.shape
        _, mask_h, mask_w = msk.shape
        if img_h != mask_h or img_w != mask_w:
            errors.append(f"Shape mismatch: image ({img_h}, {img_w}) vs mask ({mask_h}, {mask_w})")
        return errors


class BaseBlendingStrategy(ABC):
    """Interface for blending strategies."""

    @abstractmethod
    def blend(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        bg_rgb: Tensor,
        position: tuple[int, int],
        rng: Generator | None = None,
    ) -> Tensor: ...

    def compute_valid_region(
        self,
        fg_shape: tuple[int, int, int],
        bg_shape: tuple[int, int, int],
        position: tuple[int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        y, x = position
        _, fh, fw = fg_shape
        _, bh, bw = bg_shape
        y1 = max(0, y)
        x1 = max(0, x)
        y2 = min(bh, y + fh)
        x2 = min(bw, x + fw)
        fy1 = y1 - y
        fx1 = x1 - x
        fy2 = fy1 + (y2 - y1)
        fx2 = fx1 + (x2 - x1)
        return (y1, y2, x1, x2), (fy1, fy2, fx1, fx2)


class BaseInstanceTransform(ABC):
    """Interface for instance transforms."""

    @abstractmethod
    def __call__(
        self,
        fg_rgb: Tensor,
        fg_mask: Tensor,
        rng: Generator | None = None,
    ) -> tuple[Tensor, Tensor]: ...


class BaseForegroundPool(ABC):
    """Interface for foreground pools."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def sample(self, rng: Generator | None = None) -> tuple[Tensor, Tensor]: ...


class BaseBackgroundPool(ABC):
    """Interface for background pools."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def sample(
        self,
        target_size: tuple[int, int],
        rng: Generator | None = None,
    ) -> Tensor: ...
