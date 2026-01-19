"""Inference wrapper providing consistent interface regardless of model configuration.

This module addresses the LSP violation where BiRefNet returns different types
in training vs eval mode by providing a unified inference interface.
"""

import torch
from torch import nn

from .birefnet import BiRefNet


class BiRefNetInference(nn.Module):
    """Wrapper providing consistent inference interface.

    Always returns the final prediction tensor with sigmoid applied,
    regardless of the underlying model configuration (out_ref, ms_supervision, etc.).
    Optionally binarizes the output using a fixed threshold.

    This wrapper:
    - Ensures the model is in eval mode
    - Uses inference_mode for optimal performance
    - Returns a consistent tensor type (not TrainingOutput)
    - Applies sigmoid activation to the final prediction

    Example:
        >>> model = BiRefNet.from_pretrained("model_name")
        >>> inference = BiRefNetInference(model)
        >>> mask = inference(input_tensor)  # Always returns [B, 1, H, W] tensor
    """

    def __init__(
        self,
        model: BiRefNet,
        *,
        binarize: bool = True,
        threshold: float = 0.5,
    ) -> None:
        """Initialize inference wrapper.

        Args:
            model: BiRefNet model instance.
        """
        super().__init__()
        self.model = model
        self.binarize = binarize
        if not 0.0 <= threshold <= 1.0:
            msg = f"threshold must be in [0, 1], got {threshold}"
            raise ValueError(msg)
        self.threshold = float(threshold)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only final prediction.

        Args:
            x: Input tensor [B, 3, H, W].

        Returns:
            Prediction tensor [B, 1, H, W] with sigmoid applied.
        """
        with torch.inference_mode():
            outputs = self.model(x)

            # Handle both training output format and eval output format
            if isinstance(outputs, (list, tuple)):
                # Training mode returns TrainingOutput(scaled_preds, class_preds)
                # or eval mode returns list of predictions
                if hasattr(outputs, "scaled_preds"):
                    # TrainingOutput namedtuple
                    scaled_preds = outputs.scaled_preds
                    final_pred = (
                        scaled_preds[-1]
                        if isinstance(scaled_preds, (list, tuple))
                        else scaled_preds
                    )
                else:
                    # Regular list/tuple
                    final_pred = outputs[-1]
            else:
                # Single tensor output
                final_pred = outputs

            pred = final_pred.sigmoid()
            if self.binarize:
                pred = (pred >= self.threshold).float()
            return pred

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        binarize: bool = True,
        threshold: float = 0.5,
        **kwargs,
    ) -> "BiRefNetInference":
        """Load model from pretrained weights and wrap for inference.

        Args:
            model_name: HuggingFace Hub model name or local path.
            device: Device to load model on.
            **kwargs: Additional kwargs passed to BiRefNet.from_pretrained.

        Returns:
            BiRefNetInference wrapper ready for inference.
        """
        model = BiRefNet.from_pretrained(model_name, **kwargs)
        model.to(device)
        wrapper = cls(model, binarize=binarize, threshold=threshold)
        wrapper.to(device)
        return wrapper
