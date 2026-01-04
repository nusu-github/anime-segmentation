import random

import cv2
import numpy as np
from tqdm import tqdm


class DatasetGenerator:
    def __init__(
        self,
        bg_list,
        fg_list,
        output_size_range_h=(512, 1024),
        output_size_range_w=(512, 1024),
        characters_range=(0, 3),
        seed=1,
        load_all=False,
    ) -> None:
        self.bg_list = bg_list
        self.fg_list = fg_list
        self.output_size_range_h = output_size_range_h
        self.output_size_range_w = output_size_range_w
        self.load_all = load_all
        self.bgs = []
        self.fgs = []
        characters_idx = []
        characters_total = 0
        self.random = random.Random(seed)
        while not characters_total >= len(fg_list):
            num = self.random.randint(characters_range[0], characters_range[1])
            if num <= 0:
                num = 1
            characters_idx.append([
                characters_total + x for x in range(num) if characters_total + x < len(fg_list)
            ])
            characters_total += num
        self.characters_idx = characters_idx

        if load_all:
            for bg_path in tqdm(bg_list):
                bg = cv2.cvtColor(cv2.imread(bg_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                self.bgs.append(bg)
            for fg_path in tqdm(fg_list):
                fg = cv2.cvtColor(cv2.imread(fg_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                assert fg.shape[2] == 4
                self.fgs.append(fg)

    def random_corp(self, img, out_size=None):
        h, w = img.shape[:2]
        if out_size is None:
            min_s = min(h, w)
            out_size = (min_s, min_s)
        top = self.random.randint(0, h - out_size[0])
        left = self.random.randint(0, w - out_size[1])
        return img[top : top + out_size[0], left : left + out_size[1]]

    def process_fg(self, fg, output_size, scale):
        assert fg.shape[2] == 4
        h, w = fg.shape[:2]
        r = min(output_size[0] / h, output_size[1] / w)
        new_h, new_w = int(h * r), int(w * r)
        fg = cv2.resize(fg, (new_w, new_h))

        # fg random move
        h, w = output_size
        alpha = fg[:, :, 3]
        m = cv2.moments(alpha.astype(np.float32), binaryImage=False)
        if m["m00"] != 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            cx = w / 2
            cy = h / 2
        dx = w / 2 - cx
        dy = h / 2 - cy
        fg = cv2.warpAffine(
            fg,
            np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32),
            tuple(output_size[::-1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        dx = self.random.randint(-w // 3, w // 3)
        dy = self.random.randint(-h // 3, h // 3)
        angle = self.random.randint(-90, 90)
        trans_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        trans_mat[0][2] += dx
        trans_mat[1][2] += dy
        return cv2.warpAffine(
            fg,
            trans_mat,
            tuple(output_size[::-1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def __len__(self) -> int:
        return len(self.characters_idx)

    def __getitem__(self, idx):
        bg_idx = self.random.randint(0, len(self.bg_list) - 1)

        output_size = [
            self.random.randint(self.output_size_range_h[0], self.output_size_range_h[1]),
            self.random.randint(self.output_size_range_w[0], self.output_size_range_w[1]),
        ]

        if self.load_all:
            fgs = [self.fgs[x].astype(np.float32) / 255 for x in self.characters_idx[idx]]
            bg = self.bgs[bg_idx].astype(np.float32) / 255
        else:
            fgs = [
                cv2.cvtColor(
                    cv2.imread(self.fg_list[x], cv2.IMREAD_UNCHANGED),
                    cv2.COLOR_BGRA2RGBA,
                ).astype(np.float32)
                / 255
                for x in self.characters_idx[idx]
            ]
            bg = (
                cv2.cvtColor(
                    cv2.imread(self.bg_list[bg_idx], cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255
            )

        # resize to output_size

        h, w = bg.shape[:2]
        r = min(h / output_size[0], w / output_size[1])
        corp_size = (int(output_size[0] * r), int(output_size[1] * r))
        bg = self.random_corp(bg, corp_size)
        bg = cv2.resize(bg, tuple(output_size[::-1]))

        # mix fgs and bg
        image = bg
        label = np.zeros([*output_size, 1], dtype=np.float32)
        for fg_src in fgs:
            fg = fg_src
            if len(fgs) == 1 and self.random.randint(0, 1) == 0:
                h, w = fg.shape[:2]
                s = (int(output_size[0] * 1.25), int(output_size[1] * 1.25))
                r = min(s[0] / h, s[1] / w)
                new_h, new_w = int(h * r), int(w * r)
                ph = s[0] - new_h
                pw = s[1] - new_w
                fg0 = cv2.resize(fg, (new_w, new_h))
                fg = np.zeros([*s, 4], dtype=np.float32)
                fg[ph // 2 : ph // 2 + new_h, pw // 2 : pw // 2 + new_w] = fg0
                fg = self.random_corp(fg, output_size)
            else:
                scale = self.random.uniform(0.2, 0.8)
                fg = self.process_fg(fg, output_size, scale)
            image_i, label_i = fg[:, :, 0:3], fg[:, :, 3:]
            mask = label_i * cv2.blur(label_i, (5, 5))[:, :, np.newaxis]
            image = mask * image_i + (1 - mask) * image
            label = np.fmax(label_i, label)
        label = (label > 0.5).astype(np.float32)
        # Keep outputs simple & fast: only synth composition + binary mask.
        return image, label
