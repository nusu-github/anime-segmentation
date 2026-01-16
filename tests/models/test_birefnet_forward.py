"""Characterization tests for BiRefNet model forward pass.

These tests capture the current behavior of BiRefNet to ensure
refactoring does not alter external behavior.
"""

import pytest
import torch

from anime_segmentation.models.birefnet import BiRefNet

from .conftest import QUICK_TEST_BACKBONES


class TestBiRefNetTrainingMode:
    """Verify BiRefNet training mode output structure."""

    @pytest.fixture
    def model(self):
        """Create a minimal BiRefNet for testing."""
        return BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
            ms_supervision=True,
            out_ref=True,
        )

    def test_training_output_structure_with_out_ref(self, model, sample_input_training) -> None:
        """Training mode with out_ref should return nested structure."""
        model.train()
        outputs = model(sample_input_training)

        # Returns tuple of (scaled_preds, [class_preds])
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2

        scaled_preds, _class_preds = outputs
        # scaled_preds should be ([gdt_outputs], [predictions])
        assert isinstance(scaled_preds, tuple)
        assert len(scaled_preds) == 2

        gdt_outputs, predictions = scaled_preds
        # gdt_outputs: [gdt_pred, gdt_label]
        assert isinstance(gdt_outputs, list)
        assert len(gdt_outputs) == 2

        # predictions: list of prediction tensors
        assert isinstance(predictions, list)
        assert len(predictions) >= 1

    def test_training_output_without_out_ref(self, sample_input_training) -> None:
        """Training mode without out_ref returns simpler structure."""
        model = BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
            ms_supervision=True,
            out_ref=False,
        )
        model.train()
        outputs = model(sample_input_training)

        # Returns (scaled_preds_list, [class_preds])
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2

        scaled_preds, _class_preds = outputs
        assert isinstance(scaled_preds, list)

    def test_training_with_auxiliary_classification(self, sample_input_training) -> None:
        """Auxiliary classification should produce class predictions."""
        model = BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
            auxiliary_classification=True,
            num_classes=10,
            ms_supervision=False,
            out_ref=False,
        )
        model.train()
        outputs = model(sample_input_training)

        _, class_preds = outputs
        assert class_preds[0] is not None
        assert class_preds[0].shape == (2, 10)  # batch_size=2


class TestBiRefNetEvalMode:
    """Verify BiRefNet eval mode output structure."""

    @pytest.fixture
    def model(self):
        """Create a minimal BiRefNet for testing."""
        return BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
            ms_supervision=True,
            out_ref=True,
        )

    def test_eval_output_structure(self, model, sample_input_small) -> None:
        """Eval mode should return only predictions list."""
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input_small)

        # In eval mode, returns list of predictions
        assert isinstance(outputs, list)
        assert len(outputs) >= 1

    def test_eval_output_shape(self, model, sample_input_small) -> None:
        """Final prediction should match input spatial dims."""
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input_small)

        # Final prediction (last in list) should have:
        # - Batch size preserved
        # - 1 channel (binary mask)
        # - Same spatial dims as input
        final_pred = outputs[-1]
        assert final_pred.shape[0] == sample_input_small.shape[0]
        assert final_pred.shape[1] == 1
        assert final_pred.shape[2:] == sample_input_small.shape[2:]

    @pytest.mark.parametrize("bb_name", QUICK_TEST_BACKBONES)
    def test_different_backbones(self, bb_name: str, sample_input_small) -> None:
        """Model should work with all backbone types."""
        model = BiRefNet(bb_name=bb_name, bb_pretrained=False)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_input_small)

        assert outputs[-1].shape[0] == sample_input_small.shape[0]
        assert outputs[-1].shape[1] == 1


class TestBiRefNetGradients:
    """Test gradient flow through BiRefNet."""

    def test_gradients_flow_to_input(self, sample_input_training) -> None:
        """Gradients should flow from loss to input."""
        model = BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
            ms_supervision=False,
            out_ref=False,
        )
        model.train()

        x = sample_input_training.clone().requires_grad_(True)
        outputs = model(x)
        scaled_preds, _ = outputs

        loss = scaled_preds[-1].sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestBiRefNetConfiguration:
    """Test various configuration combinations."""

    @pytest.mark.parametrize(
        ("ms_supervision", "out_ref"),
        [
            (True, True),
            (True, False),
            # (False, True) is not supported: out_ref requires ms_supervision
            (False, False),
        ],
    )
    def test_supervision_combinations(
        self,
        ms_supervision: bool,
        out_ref: bool,
        sample_input_training,
    ) -> None:
        """Model should work with supported supervision combinations."""
        model = BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
            ms_supervision=ms_supervision,
            out_ref=out_ref,
        )

        # Should work in both modes
        model.train()
        train_out = model(sample_input_training)
        assert train_out is not None

        model.eval()
        with torch.no_grad():
            eval_out = model(sample_input_training)
        assert eval_out is not None
        assert isinstance(eval_out, list)

    @pytest.mark.parametrize("mul_scl_ipt", ["cat", "add", None])
    def test_multi_scale_input_modes(self, mul_scl_ipt, sample_input_small) -> None:
        """Model should work with different multi-scale input modes."""
        model = BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
            mul_scl_ipt=mul_scl_ipt,
            ms_supervision=False,
            out_ref=False,
        )
        model.eval()

        with torch.no_grad():
            outputs = model(sample_input_small)

        assert outputs[-1].shape[1] == 1

    def test_unknown_backbone_raises(self) -> None:
        """Unknown backbone should raise error."""
        # build_backbone raises NotImplementedError first
        with pytest.raises(NotImplementedError):
            BiRefNet(bb_name="nonexistent_backbone", bb_pretrained=False)


class TestBiRefNetDeterminism:
    """Test deterministic behavior for reproducibility."""

    def test_eval_deterministic(self, sample_input_small) -> None:
        """Same input should produce same output in eval mode."""
        model = BiRefNet(
            bb_name="convnext_atto",
            bb_pretrained=False,
        )
        model.eval()

        with torch.no_grad():
            out1 = model(sample_input_small)
            out2 = model(sample_input_small)

        torch.testing.assert_close(out1[-1], out2[-1])
