from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anime_segmentation.data_loader import create_training_datasets


def _write_rgb_jpg(path: Path, h: int, w: int) -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    # cv2 expects BGR
    cv2.imwrite(str(path), img[:, :, ::-1])


def _write_gray_jpg(path: Path, h: int, w: int) -> None:
    rng = np.random.default_rng(1)
    mask = (rng.random((h, w)) > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(path), mask)


def _write_rgba_png(path: Path, h: int, w: int) -> None:
    rng = np.random.default_rng(2)
    rgb = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    a = (rng.random((h, w)) * 255).astype(np.uint8)
    # cv2 expects BGRA
    bgra = np.dstack([rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0], a])
    cv2.imwrite(str(path), bgra)


def test_create_training_datasets_smoke(tmp_path: Path) -> None:
    # Use Path at runtime to satisfy Ruff TC003 (avoid type-only import).
    tmp_path = Path(tmp_path)
    (tmp_path / "fg").mkdir()
    (tmp_path / "bg").mkdir()
    (tmp_path / "imgs").mkdir()
    (tmp_path / "masks").mkdir()

    _write_rgba_png(tmp_path / "fg" / "fg0.png", 64, 64)
    _write_rgba_png(tmp_path / "fg" / "fg1.png", 64, 64)
    _write_rgb_jpg(tmp_path / "bg" / "bg0.jpg", 64, 64)
    _write_rgb_jpg(tmp_path / "bg" / "bg1.jpg", 64, 64)

    # Ensure mask filename matches image filename
    _write_rgb_jpg(tmp_path / "imgs" / "000001.jpg", 64, 64)
    _write_gray_jpg(tmp_path / "masks" / "000001.jpg", 64, 64)
    _write_rgb_jpg(tmp_path / "imgs" / "000002.jpg", 64, 64)
    _write_gray_jpg(tmp_path / "masks" / "000002.jpg", 64, 64)

    train_ds, val_ds = create_training_datasets(
        str(tmp_path),
        "fg",
        "bg",
        "imgs",
        "masks",
        ".png",
        ".jpg",
        ".jpg",
        ".jpg",
        0.5,
        32,
    )

    assert len(train_ds) > 0
    assert len(val_ds) > 0

    sample = train_ds[0]
    assert set(sample.keys()) == {"image", "label"}
    assert tuple(sample["image"].shape[:1]) == (3,)
    assert tuple(sample["label"].shape[:1]) == (1,)
