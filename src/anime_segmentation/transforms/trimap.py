"""Trimap generation transform for MODNet."""

import torch
from scipy.ndimage import grey_dilation, grey_erosion
from torchvision import tv_tensors
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
        if isinstance(mask, tv_tensors.Mask):
            mask_np = mask[0].numpy()
        else:
            mask_np = mask[0].numpy() if mask.dim() == 3 else mask.numpy()

        h, w = mask_np.shape
        s = int((h + w) * self.boundary_ratio)
        s = max(s, 1)  # Ensure minimum size

        trimap = mask_np.copy()
        dilated = grey_dilation(trimap, size=(s, s))
        eroded = grey_erosion(trimap, size=(s, s))
        trimap[(dilated - eroded) > 0.5] = 0.5  # type: ignore[operator]

        trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).float()

        return (image, mask, trimap_tensor, *rest)
