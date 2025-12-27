"""Trimap generation transform for MODNet."""

import torch
from kornia.morphology import dilation, erosion
from torchvision.transforms import v2


class WithTrimap(v2.Transform):
    """Generate trimap from mask using morphological operations.

    Creates a trimap with three values:
    - 0: Definite background
    - 0.5: Uncertain region (boundary)
    - 1: Definite foreground

    The trimap is added as a third element to the output tuple.

    Input: (Image, Mask) or (Image, Mask, ...)
    Output: (Image, Mask, Trimap, ...)

    Args:
        boundary_ratio: Ratio of (h+w) to determine boundary size. Default is 0.025.
    """

    def __init__(self, boundary_ratio: float = 0.025) -> None:
        super().__init__()
        self.boundary_ratio = boundary_ratio

    def forward(self, *inputs):
        """Override forward to add trimap to output."""
        # Handle tuple input
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            inputs = inputs[0]

        if len(inputs) < 2:
            return inputs

        image, mask = inputs[0], inputs[1]
        rest = inputs[2:] if len(inputs) > 2 else ()

        # Generate trimap from mask
        mask_tensor = mask.float()

        # Ensure (B, C, H, W) format for kornia
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)

        _, _, h, w = mask_tensor.shape
        s = int((h + w) * self.boundary_ratio)
        s = max(s, 1)  # Ensure minimum size

        # Create kernel for morphological operations
        kernel = torch.ones(s, s, device=mask_tensor.device)

        dilated = dilation(mask_tensor, kernel)
        eroded = erosion(mask_tensor, kernel)

        # Create trimap: 0.5 for uncertain boundary region
        trimap = mask_tensor.clone()
        boundary = (dilated - eroded) > 0.5
        trimap[boundary] = 0.5

        # Return as (1, H, W) tensor
        trimap = trimap.squeeze(0)

        return (image, mask, trimap, *rest)
