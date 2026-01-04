from __future__ import annotations

import torch
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F

# NOTE: Some torchvision versions expose NEAREST_EXACT but don't support it
# for Tensor inputs in v2 geometry kernels. For masks, plain NEAREST is safe.
_NEAREST = InterpolationMode.NEAREST


class RescalePad:
    """Rescale image/mask to fit in square and pad to square.

    Keeps aspect ratio. Output is (output_size, output_size).
    """

    def __init__(self, output_size: int) -> None:
        self.output_size = int(output_size)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image, label = sample["image"], sample["label"]
        h, w = image.shape[-2:]
        if h == self.output_size and w == self.output_size:
            return sample

        r = min(self.output_size / h, self.output_size / w)
        new_h = max(1, round(h * r))
        new_w = max(1, round(w * r))

        image = F.resize(
            image,
            [new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        label = F.resize(
            label,
            [new_h, new_w],
            interpolation=_NEAREST,
            antialias=False,
        )

        ph = self.output_size - new_h
        pw = self.output_size - new_w
        padding = [pw // 2, ph // 2, pw // 2 + pw % 2, ph // 2 + ph % 2]

        image = F.pad(image, padding, fill=0)
        label = F.pad(label, padding, fill=0)

        sample["image"], sample["label"] = image, label
        return sample


class RandomCrop:
    def __init__(self, output_size: int) -> None:
        self.output_size = int(output_size)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image, label = sample["image"], sample["label"]
        h, w = image.shape[-2:]
        new_h = self.output_size
        new_w = self.output_size

        if h == new_h and w == new_w:
            return sample

        if h < new_h or w < new_w:
            msg = f"RandomCrop requires input >= crop size, got {(h, w)} vs {(new_h, new_w)}"
            raise ValueError(msg)

        top = int(torch.randint(0, h - new_h + 1, (1,)).item())
        left = int(torch.randint(0, w - new_w + 1, (1,)).item())
        sample["image"] = image[..., top : top + new_h, left : left + new_w]
        sample["label"] = label[..., top : top + new_h, left : left + new_w]
        return sample


class RandomRotate:
    def __init__(self, degrees: float = 90.0) -> None:
        self.degrees = float(degrees)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image, label = sample["image"], sample["label"]
        angle = float((torch.rand(()) * 2 - 1) * self.degrees)
        sample["image"] = F.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=[0.0, 0.0, 0.0],
        )
        sample["label"] = F.rotate(
            label,
            angle,
            interpolation=_NEAREST,
            fill=[0.0],
        )
        return sample


class RandomColor:
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if torch.rand(()) >= self.p:
            return sample

        image = sample["image"]

        # brightness
        if torch.rand(()) < 0.5:
            b = float(0.4 + torch.rand(()) * 0.1)  # [0.4, 0.5]
        else:
            b = float(1.0 + torch.rand(()) * 0.2)  # [1.0, 1.2]

        # contrast
        if torch.rand(()) < 0.5:
            c = float(0.4 + torch.rand(()) * 0.1)  # [0.4, 0.5]
        else:
            c = float(1.0 + torch.rand(()) * 0.5)  # [1.0, 1.5]

        image = F.adjust_brightness(image, b)
        image = F.adjust_contrast(image, c)
        sample["image"] = image
        return sample


class GaussianNoise:
    def __init__(self, mean: float = 0.0, sigma: float = 0.05, p: float = 0.5) -> None:
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.p = float(p)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if torch.rand(()) >= self.p:
            return sample

        image = sample["image"]
        noise = torch.randn_like(image) * self.sigma + self.mean
        sample["image"] = (image + noise).clamp(0, 1)
        return sample


class BinarizeMask:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        label = sample["label"]
        sample["label"] = (label > self.threshold).to(label.dtype)
        return sample


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_train_transforms_v2(image_size: int):
    pad_size = int(image_size + image_size // 4)

    train_real = Compose([
        RescalePad(pad_size),
        RandomRotate(90.0),
        RandomCrop(image_size),
        RandomColor(p=0.5),
        GaussianNoise(mean=0.0, sigma=0.05, p=0.5),
        BinarizeMask(0.5),
    ])

    train_synth = Compose([
        RandomColor(p=0.5),
        GaussianNoise(mean=0.0, sigma=0.05, p=0.5),
        BinarizeMask(0.5),
    ])

    return train_real, train_synth


def build_val_transforms_v2(image_size: int):
    return Compose([RescalePad(image_size), BinarizeMask(0.5)])
