import glob
import pathlib
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .cli_parsers import build_inference_parser
from .lit_module import AnimeSegmentation


def get_mask(model, input_img, use_amp=True, s=640):
    input_img = (input_img / 255).astype(np.float32)
    h, w = h0, w0 = input_img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w] = cv2.resize(input_img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmpImg = torch.from_numpy(img_input).to(device=model.device, dtype=torch.float32)
    with torch.no_grad():
        device_type = model.device.type
        amp_ctx = (
            torch.autocast(device_type=device_type, dtype=torch.float16)
            if use_amp and device_type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            pred = model(tmpImg)
        pred = pred.to(dtype=torch.float32)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
        return cv2.resize(pred, (w0, h0))[:, :, np.newaxis]


if __name__ == "__main__":
    opt = build_inference_parser().parse_args()

    device = torch.device(opt.device)

    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, opt.device, img_size=opt.img_size)
    model.eval()
    model.to(device)

    if not pathlib.Path(opt.out).exists():
        pathlib.Path(opt.out).mkdir()

    for i, path in enumerate(tqdm(sorted(glob.glob(f"{opt.data}/*.*")))):
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = get_mask(model, img, use_amp=not opt.fp32, s=opt.img_size)
        if opt.only_matted and opt.bg_white:
            img = np.concatenate((mask * img + 255 * (1 - mask), mask * 255), axis=2).astype(
                np.uint8,
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{opt.out}/{i:06d}.png", img)
        elif opt.only_matted:
            img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(f"{opt.out}/{i:06d}.png", img)
        else:
            img = np.concatenate((img, mask * img, mask.repeat(3, 2) * 255), axis=1).astype(
                np.uint8,
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{opt.out}/{i:06d}.jpg", img)
