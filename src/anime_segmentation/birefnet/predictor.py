import torch
from PIL import Image
from torchvision import transforms

from .models.birefnet import BiRefNet


class BiRefNetPredictor:
    def __init__(
        self,
        model_name="ZhengPeng7/BiRefNet",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile: bool = False,
        compile_mode: str = "default",
    ) -> None:
        self.device = device
        self.model = BiRefNet.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if compile:
            self.model = torch.compile(self.model, mode=compile_mode)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
        )

    def preprocess(self, image: Image.Image, target_size=(1024, 1024)):
        w, h = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_resized = image.resize(target_size, Image.Resampling.BILINEAR)
        input_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
        return input_tensor, (w, h)

    def predict(self, image: Image.Image, target_size=(1024, 1024)) -> Image.Image:
        """Returns a grayscale PIL Image mask for the given image."""
        input_tensor, original_size = self.preprocess(image, target_size)

        with torch.inference_mode():
            preds = self.model(input_tensor)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds  # finest scale
            pred = pred.sigmoid().cpu()
            pred = torch.nn.functional.interpolate(
                pred,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=True,
            )
            pred = pred.squeeze()  # [H, W]

        return transforms.ToPILImage()(pred)
