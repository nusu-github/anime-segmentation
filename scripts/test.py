import argparse
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.asyncio import tqdm

from anime_segmentation.data_loader import create_training_datasets
from anime_segmentation.train import AnimeSegmentation, net_names
from scripts.inference import get_mask

# warnings.filterwarnings("ignore")


def main(opt: Namespace):
    train_dataset, _ = create_training_datasets(
        opt.data_dir,
        opt.fg_dir,
        opt.bg_dir,
        opt.img_dir,
        opt.mask_dir,
        opt.fg_ext,
        opt.bg_ext,
        opt.img_ext,
        opt.mask_ext,
        1,
        opt.img_size,
    )
    salobj_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

    device = torch.device(opt.device)

    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, img_size=opt.img_size)
    model.eval()
    model.to(device)

    out_dir = Path(opt.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(tqdm(salobj_dataloader)):
        image, label = data["image"][0], data["label"][0]
        image = image.permute(1, 2, 0).numpy() * 255
        label = label.permute(1, 2, 0).numpy() * 255
        mask = get_mask(model, image, use_amp=not opt.fp32, s=opt.img_size)
        image = np.concatenate((image, mask.repeat(3, 2) * 255, label.repeat(3, 2)), axis=1).astype(
            np.uint8
        )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--net", type=str, default="isnet_is", choices=net_names, help="net name")
    parser.add_argument(
        "--ckpt", type=str, default="saved_models/isnetis.ckpt", help="resume training from ckpt"
    )
    parser.add_argument("--out", type=str, default="out", help="output dir")
    parser.add_argument("--img-size", type=int, default=1024, help="input image size")

    # dataset args
    parser.add_argument(
        "--data-dir", type=str, default="../../dataset/anime-seg", help="root dir of dataset"
    )
    parser.add_argument("--fg-dir", type=str, default="fg", help="relative dir of foreground")
    parser.add_argument("--bg-dir", type=str, default="bg", help="relative dir of background")
    parser.add_argument("--img-dir", type=str, default="imgs", help="relative dir of images")
    parser.add_argument("--mask-dir", type=str, default="masks", help="relative dir of masks")
    parser.add_argument("--fg-ext", type=str, default=".png", help="extension name of foreground")
    parser.add_argument("--bg-ext", type=str, default=".jpg", help="extension name of background")
    parser.add_argument("--img-ext", type=str, default=".jpg", help="extension name of images")
    parser.add_argument("--mask-ext", type=str, default=".jpg", help="extension name of masks")

    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda:0")
    parser.add_argument("--fp32", action="store_true", default=False, help="disable mix precision")

    opt = parser.parse_args()
    print(opt)

    main(opt)
