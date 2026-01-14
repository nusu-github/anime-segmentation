"""Protocol definitions for type-safe callback-module interaction.

This module defines protocols that enable loose coupling between
callbacks and LightningModule implementations. By using protocols,
callbacks can work with any module that implements the required
interface without depending on concrete implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import nn


@runtime_checkable
class Finetunable(Protocol):
    """Protocol for modules supporting fine-tuning phase.

    Implement this protocol to enable loss weight adjustment
    during the final training epochs via FinetuneCallback.

    Example:
        class MyModule(L.LightningModule, Finetunable):
            def enter_finetune_phase(self) -> None:
                self.loss_weight *= 0.5

    """

    def enter_finetune_phase(self) -> None:
        """Enter the fine-tuning phase with adjusted loss weights.

        This method is called by FinetuneCallback when the training
        enters the final epochs. Implementations should adjust loss
        weights or other training parameters as needed.
        """
        ...


@runtime_checkable
class HasBackbone(Protocol):
    """Protocol for modules with a backbone network.

    Implement this protocol to enable backbone freezing/unfreezing
    via BackboneFreezeCallback.

    Example:
        class MyModule(L.LightningModule, HasBackbone):
            @property
            def backbone(self) -> nn.Module:
                return self.model.encoder

    """

    @property
    def backbone(self) -> nn.Module:
        """Return the backbone module for freezing/unfreezing.

        Returns:
            The backbone module (e.g., encoder, feature extractor).

        """
        ...
