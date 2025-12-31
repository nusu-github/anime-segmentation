# Anime Segmentation

![Banner](./doc/banner.jpg)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/skytnt/anime-remove-background)

High-precision background removal tool specifically designed for anime characters. This project focuses on **IS-Net** and **IBIS-Net** (Intermediate Bilateral Integration for Segmentation Network), providing a complete pipeline for training, inference, and deployment.

## Features

- **IS-Net / IBIS-Net**: Focused on IS-Net with intermediate supervision and IBIS-Net (enhanced version for anime).
- **High Resolution**: Optimized for handling high-resolution anime artwork (1024x1024).
- **Web UI**: Integrated Gradio app for easy interactive usage.
- **Training Pipeline**: Full training support with PyTorch Lightning, including distributed training and mixed precision.
- **ONNX Export**: Tools to export models for efficient deployment.

## Installation

Ensure you have Python 3.12 or higher installed.

### Using `uv` (Recommended)

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

### Using pip

```bash
pip install .
```

## Usage

### Pre-trained Models

Download pre-trained models from [Hugging Face](https://huggingface.co/skytnt/anime-seg) and place them in the `checkpoints/` or `saved_models/` directory.

### Web UI (Gradio)

The easiest way to use the tool is via the web interface.

```bash
python scripts/app.py
```

This will launch a local server (usually at `http://127.0.0.1:6006`) where you can upload images and adjust settings interactively.

### Command Line Inference

Run inference on a single image or a directory of images.

```bash
python scripts/inference.py \
    --net isnet_is \
    --ckpt checkpoints/isnetis.ckpt \
    --data path/to/input_images \
    --out output_directory \
    --img-size 1024
```

**Options:**

- `--net`: Model architecture (`isnet_is`, `isnet`, `ibisnet_is`, `ibisnet`)
- `--ckpt`: Path to the model checkpoint.
- `--data`: Input directory or image path.
- `--out`: Output directory.
- `--only-matted`: Output only the matted image (RGBA).
- `--bg-white`: Replace background with white instead of transparent.

## Training

The project uses **PyTorch Lightning** and **LightningCLI** for training. Configuration is managed via YAML files.

### 1. Prepare Dataset

The training pipeline expects a dataset with the following structure:

```
dataset/
├── fg/          # Foreground images (RGBA png)
├── bg/          # Background images (jpg/png)
├── imgs/        # Real training images (jpg)
└── masks/       # Corresponding masks (jpg/png)
```

Or you can use the [Hugging Face Dataset](https://huggingface.co/datasets/skytnt/anime-segmentation).

### 2. Run Training

Train using the default configuration:

```bash
python -m anime_segmentation.train fit --config config/config.yaml
```

Override parameters from CLI:

```bash
python -m anime_segmentation.train fit \
    --config config/config.yaml \
    --model.net_name ibisnet_is \
    --data.batch_size_train 4
```

### 3. Configuration

Check `config/config.yaml` to modify:

- **Model**: Architecture, learning rate, optimizer, loss weights.
- **Data**: Batch size, image size, augmentation parameters.
- **Trainer**: Epochs, GPUs, precision, callbacks.

## Export to ONNX

Export trained models to ONNX for use in other applications.

```bash
python scripts/export.py \
    --net isnet_is \
    --ckpt checkpoints/isnetis.ckpt \
    --out exported_models/isnet.onnx \
    --img-size 1024
```

## Supported Models

| Model Code   | Description                                     | Recommended Size |
| ------------ | ----------------------------------------------- | ---------------- |
| `ibisnet_is` | IBIS-Net with intermediate supervision          | 1024             |
| `ibisnet`    | IBIS-Net without intermediate supervision       | 1024             |
| `isnet_is`   | IS-Net with intermediate supervision            | 1024             |
| `isnet`      | Standard IS-Net                                 | 1024             |
| `isnet_gt`   | IS-Net Ground Truth Encoder (for training only) | 1024             |

## Dataset

The official dataset is a combination of [AniSeg](https://github.com/jerryli27/AniSeg) and manual collections, cleaned and annotated for high-quality anime segmentation.

Download via Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/skytnt/anime-segmentation
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Custom Modifications

This fork introduces significant engineering improvements and modernization over the original [anime-segmentation](https://github.com/SkyTNT/anime-segmentation) repository:

### Architecture & Engineering

- **Package Structure**: Refactored into a proper `src/anime_segmentation` Python package.
- **Dependency Management**: Migrated to **uv** for deterministic and fast dependency resolution (`pyproject.toml`).
- **Code Quality**: Fully type-hinted codebase, formatted and linted with **Ruff**.
- **Configuration**: Replaced `argparse` with **LightningCLI**, allowing configuration via structured YAML files (`config/config.yaml`).

### Performance & Training

- **Data Pipeline**: Replaced custom data loaders with **Hugging Face Datasets** for efficient caching, streaming, and multiprocessing.
- **Augmentation**: Migrated all augmentations to **torchvision.transforms.v2** for GPU acceleration and modern API usage.
- **Optimization**: Added support for **TorchCompile** (`torch.compile`) and **Schedule-Free Optimizers** (`schedulefree`).
- **Metrics**: Implemented BiRefNet-style robust evaluation metrics (S-measure, E-measure, Weighted F-measure) via `torchmetrics`.

### Model Focus

- **IS-Net / IBIS-Net**: Concentrated development on IS-Net and IBIS-Net (Intermediate Bilateral Integration for Segmentation Network) for anime-specific segmentation.
- **IBIS-Net**: Enhanced IS-Net variant with OutRef (gradient-aware output refinement) and InRef (input reference fusion) modules.

## Acknowledgements

- [ISNet (DIS)](https://github.com/xuebinqin/DIS)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (for loss and metric implementations)
