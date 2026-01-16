"""Characterization tests for backbone wrappers.

These tests verify the output shapes and channels of backbone wrappers,
serving as a safety net for refactoring.
"""

import pytest
import torch

from anime_segmentation.models.backbones.build_backbone import build_backbone

from .conftest import BACKBONE_CHANNELS, QUICK_TEST_BACKBONES


class TestBackboneOutputShape:
    """Verify backbone output shapes match expected channel counts."""

    @pytest.mark.parametrize("bb_name", QUICK_TEST_BACKBONES)
    def test_backbone_output_channels(self, bb_name: str, sample_input_small) -> None:
        """Each backbone must produce 4 feature maps with correct channel dims."""
        expected_channels = BACKBONE_CHANNELS[bb_name]
        backbone = build_backbone(bb_name, pretrained=False)
        outputs = backbone(sample_input_small)

        # Must have exactly 4 levels
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

        # Verify channels at each level (x1, x2, x3, x4)
        for i, (out, expected_ch) in enumerate(zip(outputs, expected_channels, strict=False)):
            assert out.shape[1] == expected_ch, (
                f"Level {i} ({bb_name}): expected {expected_ch} channels, got {out.shape[1]}"
            )

    @pytest.mark.parametrize("bb_name", QUICK_TEST_BACKBONES)
    def test_backbone_output_is_tuple(self, bb_name: str, sample_input_small) -> None:
        """All backbones must return tuple (not list)."""
        backbone = build_backbone(bb_name, pretrained=False)
        outputs = backbone(sample_input_small)
        assert isinstance(outputs, tuple), f"{bb_name} returned {type(outputs)}, expected tuple"

    @pytest.mark.parametrize("bb_name", QUICK_TEST_BACKBONES)
    def test_backbone_output_spatial_dims(self, bb_name: str, sample_input_small) -> None:
        """Output spatial dimensions should follow pyramid pattern."""
        backbone = build_backbone(bb_name, pretrained=False)
        outputs = backbone(sample_input_small)
        _input_h, _input_w = sample_input_small.shape[2:]

        # Typically: x1 = 1/4, x2 = 1/8, x3 = 1/16, x4 = 1/32
        # But exact ratios vary by backbone
        for i in range(len(outputs) - 1):
            # Each level should be same or smaller than previous
            assert outputs[i].shape[2] >= outputs[i + 1].shape[2], (
                f"Level {i} height should be >= level {i + 1}"
            )
            assert outputs[i].shape[3] >= outputs[i + 1].shape[3], (
                f"Level {i} width should be >= level {i + 1}"
            )

    def test_swin_output_is_nchw(self, sample_input_small) -> None:
        """Swin wrapper must convert NHWC to NCHW format."""
        backbone = build_backbone("swin_v1_t", pretrained=False)
        outputs = backbone(sample_input_small)

        # Verify NCHW format: channel dim should be expected value, not spatial
        expected_channel = BACKBONE_CHANNELS["swin_v1_t"][0]
        assert outputs[0].shape[1] == expected_channel, (
            f"Swin x1 should have {expected_channel} channels (NCHW), got {outputs[0].shape[1]}"
        )
        # Spatial dims should be smaller than channels
        assert outputs[0].shape[2] < outputs[0].shape[1], "Shape looks like NHWC not NCHW"

    @pytest.mark.parametrize("bb_name", QUICK_TEST_BACKBONES)
    def test_backbone_batch_dimension_preserved(self, bb_name: str) -> None:
        """Batch dimension should be preserved through backbone."""
        backbone = build_backbone(bb_name, pretrained=False)

        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 64, 64)
            outputs = backbone(x)
            for i, out in enumerate(outputs):
                assert out.shape[0] == batch_size, (
                    f"Level {i}: batch size {batch_size} not preserved"
                )


class TestBackboneRegistry:
    """Test that all expected backbones are registered and buildable."""

    @pytest.mark.parametrize("bb_name", QUICK_TEST_BACKBONES)
    def test_backbone_buildable(self, bb_name: str) -> None:
        """Quick test backbones should be constructable."""
        backbone = build_backbone(bb_name, pretrained=False)
        assert backbone is not None

    def test_unknown_backbone_raises(self) -> None:
        """Unknown backbone name should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            build_backbone("unknown_backbone_xyz", pretrained=False)


class TestBackboneGradients:
    """Test that backbones support gradient computation."""

    @pytest.mark.parametrize("bb_name", QUICK_TEST_BACKBONES)
    def test_backbone_gradients_flow(self, bb_name: str) -> None:
        """Gradients should flow through backbone."""
        backbone = build_backbone(bb_name, pretrained=False)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        outputs = backbone(x)

        # Sum all outputs and compute gradient
        loss = sum(out.sum() for out in outputs)
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert not torch.all(x.grad == 0), "Gradients should be non-zero"
