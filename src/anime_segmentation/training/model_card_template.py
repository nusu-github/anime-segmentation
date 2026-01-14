"""Custom model card template for anime-segmentation models."""

MODEL_CARD_TEMPLATE = """
---
{{ card_data }}
---

# {{ model_name | default("Anime Segmentation Model", true) }}

High-precision background removal for anime characters using **BiRefNet** (Bilateral Reference Network).

## Model Description

This model performs dichotomous image segmentation (foreground/background separation) optimized for anime-style artwork. It uses the BiRefNet architecture with:

- **Backbone**: {{ backbone_name | default("[Not specified]", true) }}
- **Multi-scale Supervision**: {{ "Enabled" if ms_supervision else "Disabled" }}
- **Output Refinement (Out-Ref)**: {{ "Enabled" if out_ref else "Disabled" }}

## Usage

### Quick Start

```python
from anime_segmentation.training import BiRefNetLightning
from PIL import Image
import torch

# Load model
model = BiRefNetLightning.from_pretrained("{{ repo_id | default('your-username/model-name', true) }}")
model.eval()

# Prepare image
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("input.png").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.inference_mode():
    output = model(input_tensor)
    mask = output[-1].sigmoid().squeeze()

# Convert to PIL Image
mask_pil = transforms.ToPILImage()(mask)
mask_pil.save("mask.png")
```

### Using BiRefNetPredictor (Recommended)

```python
from anime_segmentation.birefnet import BiRefNetPredictor
from PIL import Image

predictor = BiRefNetPredictor(
    model_name="{{ repo_id | default('your-username/model-name', true) }}",
    device="cuda"
)

image = Image.open("input.png")
mask = predictor.predict(image, target_size=(1024, 1024))
mask.save("mask.png")
```

## Training Configuration

{% if training_config %}
| Parameter | Value |
|-----------|-------|
{% for key, value in training_config.items() %}
| {{ key }} | {{ value }} |
{% endfor %}
{% endif %}

## Metrics

{% if metrics %}
| Metric | Value |
|--------|-------|
{% for key, value in metrics.items() %}
| {{ key }} | {{ value }} |
{% endfor %}
{% endif %}

## Architecture Details

BiRefNet uses a hierarchical encoder-decoder structure with:

- **Backbone encoder** for multi-scale feature extraction
- **ASPP/ASPPDeformable** for context aggregation
- **Multi-scale supervision** for training stability
- **Gradient-aware output refinement** for sharp edges

## Limitations

- Optimized for anime/illustration style images
- Best results at 1024x1024 resolution
- May require fine-tuning for photorealistic images

## Citation

If you use this model, please cite:

```bibtex
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laber, Jorma and Xie, Wenqi and Van Gool, Luc},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
```

## Links

- Code: {{ repo_url | default("[More Information Needed]", true) }}
- Paper: {{ paper_url | default("https://arxiv.org/abs/2401.03407", true) }}
{% if docs_url %}- Docs: {{ docs_url }}{% endif %}

---

*This model was trained using the [anime-segmentation](https://github.com/nusu-github/anime-segmentation) library.*
"""
