"""ISNet model implementations for dichotomous image segmentation.

This module provides the ISNet architecture variants for high-accuracy
salient object detection and dichotomous image segmentation tasks.
"""

from .isnet import ISNetDIS, ISNetGTEncoder

__all__ = ["ISNetDIS", "ISNetGTEncoder"]
