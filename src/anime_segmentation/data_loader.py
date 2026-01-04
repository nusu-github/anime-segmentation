"""Dataset & dataloading utilities.

This module intentionally keeps the training loop API stable (returns dict with
`image` and `label`) while modernizing transforms and removing the old shared
memory cache/trimap logic.
"""

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_generator import DatasetGenerator
from .transforms import build_train_transforms_v2, build_val_transforms_v2


class AnimeSegDataset(Dataset):
    def __init__(
        self,
        real_img_list: list[str],
        real_mask_list: list[str],
        generator: DatasetGenerator | None = None,
        transform_real=None,
        transform_synth=None,
    ) -> None:
        self.dataset_generator = generator
        self.real_img_list = real_img_list
        self.real_mask_list = real_mask_list
        self.transform_real = transform_real
        self.transform_synth = transform_synth

        if generator is not None:
            assert generator.output_size_range_w[0] == generator.output_size_range_w[1]
            assert generator.output_size_range_h[0] == generator.output_size_range_h[1]

    def __len__(self) -> int:
        length = len(self.real_img_list)
        if self.dataset_generator is not None:
            length += len(self.dataset_generator)
        return length

    def __getitem__(self, idx):
        is_real = idx < len(self.real_img_list)
        if is_real:
            image = cv2.cvtColor(
                cv2.imread(self.real_img_list[idx], cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            label = cv2.imread(self.real_mask_list[idx], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
            image, label = image.astype(np.float32) / 255, label.astype(np.float32) / 255
            label = (label > 0.5).astype(np.float32)
            image = image[10:-10, 10:-10]
            label = label[
                10:-10,
                10:-10,
            ]  # in this dataset, there is a problem in the edge of some label
        else:
            image, label = self.dataset_generator[idx - len(self.real_img_list)]
        image, label = (
            torch.from_numpy(image).permute(2, 0, 1),
            torch.from_numpy(label).permute(2, 0, 1),
        )
        sample = {"image": image, "label": label}
        if is_real and self.transform_real is not None:
            sample = self.transform_real(sample)
        if (not is_real) and self.transform_synth is not None:
            sample = self.transform_synth(sample)

        # Always enforce binary mask (0.5 threshold fixed)
        sample["label"] = (sample["label"] > 0.5).to(sample["label"].dtype)
        return sample


def create_training_datasets(
    data_root,
    fgs_dir,
    bgs_dir,
    imgs_dir,
    masks_dir,
    fg_ext,
    bg_ext,
    img_ext,
    mask_ext,
    spilt_rate,
    image_size,
):
    def add_sep(path):
        return path if path.endswith(("/", "\\")) else path + os.sep

    data_root = add_sep(data_root)
    fgs_dir = add_sep(fgs_dir)
    bgs_dir = add_sep(bgs_dir)
    imgs_dir = add_sep(imgs_dir)
    masks_dir = add_sep(masks_dir)

    train_img_list = glob.glob(data_root + imgs_dir + "*" + img_ext)
    train_mask_list = [
        data_root + masks_dir + img_path.split(os.sep)[-1].replace(img_ext, mask_ext)
        for img_path in train_img_list
    ]
    train_fg_list = glob.glob(data_root + fgs_dir + "*" + fg_ext)
    train_bg_list = glob.glob(data_root + bgs_dir + "*" + bg_ext)
    random.Random(1).shuffle(train_fg_list)
    random.Random(1).shuffle(train_bg_list)
    random.Random(1).shuffle(train_img_list)
    random.Random(1).shuffle(train_mask_list)
    train_fg_list, val_fg_list = (
        train_fg_list[: int(len(train_fg_list) * spilt_rate)],
        train_fg_list[int(len(train_fg_list) * spilt_rate) :],
    )
    train_bg_list, val_bg_list = (
        train_bg_list[: int(len(train_bg_list) * spilt_rate)],
        train_bg_list[int(len(train_bg_list) * spilt_rate) :],
    )
    train_img_list, val_img_list = (
        train_img_list[: int(len(train_img_list) * spilt_rate)],
        train_img_list[int(len(train_img_list) * spilt_rate) :],
    )
    train_mask_list, val_mask_list = (
        train_mask_list[: int(len(train_mask_list) * spilt_rate)],
        train_mask_list[int(len(train_mask_list) * spilt_rate) :],
    )
    train_transform_real, train_transform_synth = build_train_transforms_v2(image_size)
    val_transform_real = build_val_transforms_v2(image_size)
    train_generator = DatasetGenerator(
        train_bg_list,
        train_fg_list,
        (image_size, image_size),
        (image_size, image_size),
    )
    train_dataset = AnimeSegDataset(
        train_img_list,
        train_mask_list,
        train_generator,
        transform_real=train_transform_real,
        transform_synth=train_transform_synth,
    )
    val_generator = DatasetGenerator(
        val_bg_list,
        val_fg_list,
        (image_size, image_size),
        (image_size, image_size),
    )
    val_dataset = AnimeSegDataset(
        val_img_list,
        val_mask_list,
        val_generator,
        transform_real=val_transform_real,
        transform_synth=None,
    )
    return train_dataset, val_dataset
