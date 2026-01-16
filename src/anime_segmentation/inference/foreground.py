"""Foreground estimation utilities using blur fusion."""

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from einops import rearrange
from PIL import Image


def FB_blur_fusion_foreground_estimator_cpu(image, FG, B, alpha, r=90):
    """Foreground-Background blur fusion estimator (CPU).

    See: https://github.com/Photoroom/fast-foreground-estimation
    """
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FGA = cv2.blur(FG * alpha, (r, r))
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    FG = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG = np.clip(FG, 0, 1)
    return FG, blurred_B


def FB_blur_fusion_foreground_estimator_cpu_2(image, alpha, r=90):
    """Two-pass refinement for better foreground estimation."""
    alpha = alpha[:, :, None]
    FG, blur_B = FB_blur_fusion_foreground_estimator_cpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_cpu(image, FG, blur_B, alpha, r=6)[0]


def _mean_blur(x, kernel_size):
    """GPU-based mean blur equivalent to cv2.blur. Input shape: [B, C, H, W]."""
    if kernel_size % 2 == 0:
        pad_l = kernel_size // 2 - 1
        pad_r = kernel_size // 2
        pad_t = kernel_size // 2 - 1
        pad_b = kernel_size // 2
    else:
        pad_l = pad_r = pad_t = pad_b = kernel_size // 2

    x_padded = torch.nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")

    return torch.nn.functional.avg_pool2d(
        x_padded,
        kernel_size=(kernel_size, kernel_size),
        stride=1,
        count_include_pad=False,
    )


def FB_blur_fusion_foreground_estimator_gpu(image, FG, B, alpha, r=90):
    """Foreground-Background blur fusion estimator (GPU)."""

    def as_dtype(x, dtype):
        return x.to(dtype) if x.dtype != dtype else x

    input_dtype = image.dtype
    # convert image to float to avoid overflow
    image = as_dtype(image, torch.float32)
    FG = as_dtype(FG, torch.float32)
    B = as_dtype(B, torch.float32)
    alpha = as_dtype(alpha, torch.float32)

    blurred_alpha = _mean_blur(alpha, kernel_size=r)

    blurred_FGA = _mean_blur(FG * alpha, kernel_size=r)
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = _mean_blur(B * (1 - alpha), kernel_size=r)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)

    FG_output = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG_output = torch.clamp(FG_output, 0, 1)

    return as_dtype(FG_output, input_dtype), as_dtype(blurred_B, input_dtype)


def FB_blur_fusion_foreground_estimator_gpu_2(image, alpha, r=90):
    """Two-pass refinement for better foreground estimation.

    See: https://github.com/ZhengPeng7/BiRefNet/issues/226#issuecomment-3016433728
    """
    FG, blur_B = FB_blur_fusion_foreground_estimator_gpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_gpu(image, FG, blur_B, alpha, r=6)[0]


_compiled_FB_blur_fusion_foreground_estimator_gpu_2 = None


def refine_foreground(image, mask, r=90, device=None, compile=False):
    """Refine foreground using blur fusion.

    Args:
        image: PIL Image (RGB).
        mask: PIL Image (grayscale).
        r: Blur kernel radius.
        device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        compile: Whether to use torch.compile for GPU path.

    Returns:
        PIL Image with refined foreground.

    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if mask.size != image.size:
        mask = mask.resize(image.size)

    if device == "cuda":
        img_tensor = rearrange(F.to_tensor(image), "c h w -> 1 c h w").to(device)
        mask_tensor = rearrange(F.to_tensor(mask), "c h w -> 1 c h w").to(device)

        if compile:
            global _compiled_FB_blur_fusion_foreground_estimator_gpu_2
            if _compiled_FB_blur_fusion_foreground_estimator_gpu_2 is None:
                _compiled_FB_blur_fusion_foreground_estimator_gpu_2 = torch.compile(
                    FB_blur_fusion_foreground_estimator_gpu_2,
                )
            estimator = _compiled_FB_blur_fusion_foreground_estimator_gpu_2
        else:
            estimator = FB_blur_fusion_foreground_estimator_gpu_2

        estimated_foreground = estimator(
            img_tensor,
            mask_tensor,
            r=r,
        )

        estimated_foreground = estimated_foreground.squeeze(0).mul(255.0).to(torch.uint8)
        estimated_foreground = rearrange(estimated_foreground, "c h w -> h w c").cpu().numpy()
    else:
        img_np = np.array(image, dtype=np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        estimated_foreground = FB_blur_fusion_foreground_estimator_cpu_2(img_np, mask_np, r=r)
        estimated_foreground = (estimated_foreground * 255.0).astype(np.uint8)

    return Image.fromarray(estimated_foreground)
