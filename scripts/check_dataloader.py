from __future__ import annotations

import cv2
import numpy as np
from torch.utils.data import DataLoader

from anime_segmentation.data_loader import create_training_datasets


def main() -> None:
    # NOTE: This is a manual visualization/debug script (not pytest).
    data_dir = "../../dataset/anime-seg/"
    tra_fg_dir = "fg/"
    tra_bg_dir = "bg/"
    tra_img_dir = "imgs/"
    tra_mask_dir = "masks/"
    fg_ext = ".png"
    bg_ext = ".*"
    img_ext = ".jpg"
    mask_ext = ".jpg"

    train_dataset, _ = create_training_datasets(
        data_dir,
        tra_fg_dir,
        tra_bg_dir,
        tra_img_dir,
        tra_mask_dir,
        fg_ext,
        bg_ext,
        img_ext,
        mask_ext,
        0.95,
        640,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    for data in dataloader:
        cv2.imshow(
            "sample",
            np.concatenate(
                [
                    data["image"][0].permute(1, 2, 0).numpy()[:, :, ::-1],
                    cv2.cvtColor(
                        data["label"][0].permute(1, 2, 0).numpy(),
                        cv2.COLOR_GRAY2RGB,
                    ),
                ],
                axis=1,
            ),
        )
        cv2.waitKey(1000)


if __name__ == "__main__":
    main()
