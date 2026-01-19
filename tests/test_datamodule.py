"""Tests for AnimeSegmentationDataModule dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image

from anime_segmentation.training.config import SynthesisConfig
from anime_segmentation.training.datamodule import AnimeSegmentationDataModule
from anime_segmentation.training.synthesis.base import (
    BaseCompositor,
    BaseValidator,
    ValidationResult,
)

if TYPE_CHECKING:
    from pathlib import Path


class MockCompositor(BaseCompositor):
    """Simple mock compositor returning random tensors."""

    def __init__(self, fixed_k: int = 1) -> None:
        self.fixed_k = fixed_k
        self.call_count = 0

    def synthesize(
        self,
        target_size: tuple[int, int],
        rng: torch.Generator | None = None,
        *,
        return_background: bool = False,
    ):
        self.call_count += 1
        h, w = target_size
        image = torch.rand(3, h, w)
        mask = torch.ones(1, h, w)
        if return_background:
            bg = torch.rand(3, h, w)
            return image, mask, self.fixed_k, bg
        return image, mask, self.fixed_k


class MockValidator(BaseValidator):
    """Always-valid validator for injection."""

    def __init__(self, always_valid: bool = True) -> None:
        self.always_valid = always_valid
        self.validate_count = 0

    def validate(self, image: torch.Tensor, mask: torch.Tensor) -> ValidationResult:
        self.validate_count += 1
        return ValidationResult(
            is_valid=self.always_valid,
            errors=[] if self.always_valid else ["mock failure"],
            warnings=[],
            stats={"mock": True},
        )


@pytest.fixture
def real_data_path(tmp_path: Path) -> Path:
    data_root = tmp_path / "dataset"
    for folder in ("imgs", "masks", "fg", "bg"):
        (data_root / folder).mkdir(parents=True)

    # Create a matching image-mask pair
    Image.new("RGB", (16, 16), (255, 255, 255)).save(data_root / "imgs" / "sample.png")
    Image.new("L", (16, 16), 0).save(data_root / "masks" / "sample.png")

    # Create minimal FG/PNG and background images
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(data_root / "fg" / "char.png")
    Image.new("RGB", (32, 32), (0, 0, 255)).save(data_root / "bg" / "bg.png")

    return data_root


class TestDataModuleDI:
    def test_mock_compositor_injection(self, tmp_path: Path) -> None:
        mock = MockCompositor(fixed_k=2)
        dm = AnimeSegmentationDataModule(
            data_root=str(tmp_path),
            synthesis=SynthesisConfig(enabled=True),
            compositor=mock,
        )

        dm.setup("fit")

        assert dm._compositor is mock
        assert mock.call_count == 0

    def test_mock_validator_injection(self, tmp_path: Path) -> None:
        mock_validator = MockValidator(always_valid=True)
        dm = AnimeSegmentationDataModule(
            data_root=str(tmp_path),
            synthesis=SynthesisConfig(enabled=True),
            validator=mock_validator,
        )

        dm.setup("fit")

        assert dm._validator is mock_validator

    def test_default_components_when_no_injection(self, real_data_path: Path) -> None:
        dm = AnimeSegmentationDataModule(
            data_root=str(real_data_path),
            synthesis=SynthesisConfig(enabled=True, length=1),
        )

        dm.setup("fit")

        from anime_segmentation.training.synthesis.compositor import CopyPasteCompositor
        from anime_segmentation.training.synthesis.validation import DataValidator

        assert isinstance(dm._compositor, CopyPasteCompositor)
        assert isinstance(dm._validator, DataValidator)

    def test_partial_injection(self, tmp_path: Path) -> None:
        mock_compositor = MockCompositor()
        dm = AnimeSegmentationDataModule(
            data_root=str(tmp_path),
            synthesis=SynthesisConfig(enabled=True),
            compositor=mock_compositor,
        )

        dm.setup("fit")

        from anime_segmentation.training.synthesis.validation import DataValidator

        assert dm._compositor is mock_compositor
        assert isinstance(dm._validator, DataValidator)


class TestSynthesisIntegration:
    def test_synthesis_with_controlled_output(self, tmp_path: Path) -> None:
        mock = MockCompositor(fixed_k=2)
        dm = AnimeSegmentationDataModule(
            data_root=str(tmp_path),
            size=(256, 256),
            synthesis=SynthesisConfig(enabled=True, ratio=1.0, length=4),
            compositor=mock,
        )

        dm.setup("fit")

        image, mask, k = mock.synthesize((256, 256))

        assert k == 2
        assert image.shape == (3, 256, 256)
        assert mask.shape == (1, 256, 256)
