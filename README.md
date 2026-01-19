# Anime Segmentation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/skytnt/anime-remove-background)

High-precision background removal for anime characters using **BiRefNet** (Bilateral Reference Network). This project provides a complete training and inference pipeline built on PyTorch Lightning with modern tooling and Hugging Face Hub integration.

## Features

- **BiRefNet Architecture**: State-of-the-art encoder-decoder with gradient-aware refinement (Out-Ref) and multi-scale supervision
- **16+ Backbone Options**: ConvNeXt, Swin Transformer, PVT v2, DINOv3
- **YAML Configuration**: Full training customization via LightningCLI
- **Hugging Face Integration**: Load/push models directly from/to the Hub
- **Production Ready**: Type-hinted codebase, torch.compile support, distributed training

## Installation

Requires Python 3.12+.

### Using uv (Recommended)

```bash
uv sync
```

### Using pip

```bash
pip install .
```

## Quick Start

### Training

```bash
python -m anime_segmentation.training.train fit --config configs/train.yaml
```

Override parameters from CLI:

```bash
python -m anime_segmentation.training.train fit \
    --config configs/train.yaml \
    --model.backbone.name swin_v1_t \
    --data.loader.batch_size 4
```

## Model Architecture

BiRefNet uses a hierarchical encoder-decoder structure with:

- **Backbone Encoder**: Extracts multi-scale features (16 variants available)
- **ASPP/ASPPDeformable**: Atrous Spatial Pyramid Pooling for context aggregation
- **Multi-scale Supervision**: Loss computed at multiple decoder levels
- **Out-Ref Module**: Gradient-aware output refinement for edge quality
- **In-Ref Module**: Input reference fusion for detail preservation

### Supported Backbones

| Family           | Variants                                                                                            |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| ConvNeXt         | `convnext_atto`, `convnext_femto`, `convnext_pico`, `convnext_nano`, `convnext_tiny`                |
| ConvNeXt V2      | `convnext_v2_atto`, `convnext_v2_femto`, `convnext_v2_pico`, `convnext_v2_nano`, `convnext_v2_tiny` |
| Swin Transformer | `swin_v1_t`, `swin_v1_s`, `swin_v1_b`, `swin_v1_l`                                                  |
| PVT v2           | `pvt_v2_b0`, `pvt_v2_b1`, `pvt_v2_b2`, `pvt_v2_b5`                                                  |
| DINOv3           | `dino_v3_s`, `dino_v3_b`, `dino_v3_l`, `dino_v3_h_plus`, `dino_v3_7b`                               |
| CAFormer         | `caformer_s18`, `caformer_s36`, `caformer_m36`, `caformer_b36`                                      |

## Dataset Format

Prepare your dataset in BiRefNet-style structure:

```text
dataset/
├── train/
│   ├── im/     # Input images (.png, .jpg)
│   └── gt/     # Ground truth masks (.png)
├── val/
│   ├── im/
│   └── gt/
└── test/
    ├── im/
    └── gt/
```

Update the config file:

```yaml
data:
  data_root: path/to/dataset
  training_sets:
    - train
  validation_sets:
    - val
  test_sets:
    - test
```

## Configuration

Training is configured via YAML files. See `configs/train.yaml` for all options.

### Key Configuration Sections

**Model**:

```yaml
model:
  backbone:
    name: convnext_atto     # Backbone selection
    pretrained: true

  decoder:
    out_ref: true
    ms_supervision: true
    dec_att: ASPPDeformable
```

**Optimizer & Scheduler**:

```yaml
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1e-4
    weight_decay: 0.01

lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 120
```

**Loss Weights**:

```yaml
model:
  loss:
    bce: 30.0
    iou: 0.5
    ssim: 10.0
```

**Data Augmentation**:

```yaml
data:
  loader:
    batch_size: 8
    num_workers: 4
  augmentation:
    enabled: true
    hflip_prob: 0.5
    rotation_degrees: 10.0
    color_jitter: true
```

**Trainer**:

```yaml
trainer:
  max_epochs: 120
  precision: 16-mixed
  accumulate_grad_batches: 4
```

## Evaluation Metrics

The training pipeline includes BiRefNet-style metrics:

| Metric    | Description                   |
| --------- | ----------------------------- |
| IoU       | Intersection over Union       |
| MAE       | Mean Absolute Error           |
| S-measure | Spatial/structural similarity |
| E-measure | Enhanced alignment measure    |
| F-measure | Weighted precision-recall     |

## Hugging Face Hub

### Load Pretrained Model

```python
from anime_segmentation import BiRefNet

model = BiRefNet.from_pretrained("ZhengPeng7/BiRefNet")
```

### Push to Hub

```python
model.push_to_hub("your-username/your-model")
```

## Project Structure

```text
anime-segmentation/
├── src/anime_segmentation/
│   ├── models/               # BiRefNet architecture
│   │   ├── birefnet.py       # Main model
│   │   ├── predictor.py      # Inference wrapper
│   │   ├── backbones/        # Encoder implementations
│   │   └── modules/          # Decoder components
│   ├── inference/
│   │   └── foreground.py     # Foreground refinement
│   └── training/
│       ├── train.py          # LightningCLI entry point
│       ├── lightning_module.py
│       ├── datamodule.py
│       ├── loss.py           # Multi-component loss
│       ├── metrics.py        # TorchMetrics implementations
│       └── callbacks.py      # Training callbacks
├── configs/
│   ├── default.yaml          # Base defaults
│   ├── train.yaml            # Full training config
│   └── custom_compositor.yaml # DI customization example
└── pyproject.toml
```

## Training Callbacks

| Callback                        | Description                                 |
| ------------------------------- | ------------------------------------------- |
| `FinetuneCallback`              | Adjusts loss weights in final epochs        |
| `BackboneFreezeCallback`        | Freezes backbone initially, unfreezes later |
| `GradientAccumulationScheduler` | Dynamic gradient accumulation               |
| `VisualizationCallback`         | Logs predictions during training            |
| `HubUploadCallback`             | Auto-uploads checkpoints to HF Hub          |

## Important Notes

This project integrates [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) with significant custom modifications:

- Added ConvNeXt backbone family (atto, femto, pico, nano, tiny)
- Added CAFormer backbone family
- Changed normalization from Batch Normalization to Group Normalization
- Refactored backbone interface and feature extraction
- Modified decoder architecture and loss computation
- Restructured training pipeline with LightningCLI

**These changes are NOT compatible with the original BiRefNet pretrained weights.** You will need to train from scratch or use weights specifically trained with this codebase.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

This project includes code from [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), which is licensed under the MIT License.

## Acknowledgements

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [DIS (ISNet)](https://github.com/xuebinqin/DIS)
- [Original anime-segmentation](https://github.com/SkyTNT/anime-segmentation)
