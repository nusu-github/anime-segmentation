import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from torch.amp.autocast_mode import autocast

from anime_segmentation.train import NET_NAMES, AnimeSegmentation


def get_mask(model: AnimeSegmentation, input_img, use_amp=True, img_size=640):
    input_img = (input_img / 255).astype(np.float32)
    h, w = orig_h, orig_w = input_img.shape[:-1]
    h, w = (img_size, int(img_size * w / h)) if h > w else (int(img_size * h / w), img_size)
    ph, pw = img_size - h, img_size - w
    img_input = np.zeros([img_size, img_size, 3], dtype=np.float32)
    img_input[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w] = cv2.resize(input_img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmp_img = torch.from_numpy(img_input).float().to(model.device)
    with torch.inference_mode():
        with autocast(device_type=model.device.type, enabled=use_amp):
            pred = model(tmp_img)
            pred = pred.to(dtype=torch.float32)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
        return cv2.resize(pred, (orig_w, orig_h))[:, :, np.newaxis]


# global model state
class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.is_loaded = False
        self.current_device = None
        self.current_path = None


state = ModelState()


def rmbg_fn(img, img_size, white_bg_checkbox, only_matted_checkbox):
    if not state.is_loaded or state.model is None:
        msg = "Please load the model first!"
        raise gr.Error(msg)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = get_mask(state.model, img, False, int(img_size))
        match (white_bg_checkbox, only_matted_checkbox):
            case (True, True):
                img = np.concatenate((mask * img + 255 * (1 - mask), mask * 255), axis=2).astype(
                    np.uint8
                )
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            case (_, True):
                img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            case _:
                img = np.concatenate((img, mask * img, mask.repeat(3, 2) * 255), axis=1).astype(
                    np.uint8
                )
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = mask.repeat(3, axis=2)
        return mask, img
    except Exception as e:
        msg = f"Error processing image: {e!s}"
        raise gr.Error(msg) from e


def get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    return devices


def auto_load_model():
    global state
    if state.is_loaded:
        return gr.Info("Model already loaded successfully")

    try:
        project_root = Path(__file__).parent.parent
        model_paths = sorted(project_root.rglob("*.ckpt"))
        if not model_paths:
            msg = "No model files found"
            raise gr.Error(msg)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return load_model(str(model_paths[0]), "isnet_is", 1024, device)
    except Exception as e:
        return gr.Error(f"Failed to auto-load model: {e!s}")


def load_model(path: str, net_name: str, img_size: int, device: str = "cuda:0"):
    global state

    # check if model is already loaded
    if state.is_loaded and state.current_path == path and state.current_device == device:
        return gr.Info("Model already loaded successfully")

    try:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        new_model = AnimeSegmentation.try_load(
            net_name=net_name,
            img_size=img_size,
            ckpt_path=path,
            map_location=device,
            weights_only=True,
        )
        new_model.eval()
        new_model.to(device)

        # update state
        state.model = new_model
        state.is_loaded = True
        state.current_device = device
        state.current_path = path

        return gr.Info(f"Model loaded successfully on {device}")
    except Exception as e:
        state.is_loaded = False
        state.model = None
        return gr.Error(f"Failed to load model: {e!s}")


def get_model_path():
    # Search from script's parent directory (project root)
    project_root = Path(__file__).parent.parent
    if model_paths := sorted(project_root.rglob("*.ckpt")):
        model_paths = [str(p) for p in model_paths]
        return gr.Dropdown(choices=model_paths, value=model_paths[0])

    msg = "No model files found"
    raise gr.Error(msg)


def batch_inference(
    input_dir, output_dir, img_size, white_bg_checkbox, only_matted_checkbox
) -> str:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        msg = "Input directory does not exist"
        raise gr.Error(msg)

    img_paths = sorted(input_path.glob("*.*"))
    if not img_paths:
        msg = "No image files found"
        raise gr.Error(msg)

    progress = gr.Progress(track_tqdm=True)
    total_images = len(img_paths)

    try:
        output_path.mkdir(parents=True, exist_ok=True)

        for i, path in enumerate(progress.tqdm(img_paths, desc="Processing images")):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                msg = f"Failed to read image: {path}"
                raise gr.Error(msg)

            # no need mask for batch processing
            _, processed_img = rmbg_fn(img, img_size, white_bg_checkbox, only_matted_checkbox)

            cv2.imwrite(str(output_path / f"{i:06d}.png"), processed_img)

    except Exception as e:
        msg = f"Processing error: {e!s}"
        raise gr.Error(msg) from e

    return f"Batch processing completed: {total_images} images processed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6006, help="gradio server port")
    opt = parser.parse_args()

    app = gr.Blocks()
    with app:
        app.load(auto_load_model)

        with gr.Accordion("Model Settings", open=False):
            load_model_path_btn = gr.Button("Get Models")
            model_path_input = gr.Dropdown(label="Model Path")
            model_type = gr.Dropdown(label="Model Type", value="isnet_is", choices=NET_NAMES)
            model_image_size = gr.Slider(
                label="Image Size", value=1024, minimum=0, maximum=1280, step=32
            )
            device_dropdown = gr.Dropdown(
                label="Device",
                choices=get_available_devices(),
                value="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            load_model_path_btn.click(get_model_path, [], model_path_input)
            load_model_btn = gr.Button("Load")
            load_model_btn.click(
                load_model,
                inputs=[model_path_input, model_type, model_image_size, device_dropdown],
                outputs=[],
            )

        with gr.Tabs():
            with gr.Tab("Image Inference"):
                input_img = gr.Image(label="Input Image")

                white_bg_checkbox = gr.Checkbox(label="White Background", value=False)
                only_matted_checkbox = gr.Checkbox(label="Only Matted", value=True)

                run_btn = gr.Button("Process", variant="primary")

                with gr.Row():
                    output_mask = gr.Image(label="Mask")
                    output_img = gr.Image(label="Result", image_mode="RGBA")

                run_btn.click(
                    fn=rmbg_fn,
                    inputs=[input_img, model_image_size, white_bg_checkbox, only_matted_checkbox],
                    outputs=[output_mask, output_img],
                )

            with gr.Tab("Batch Processing"):
                input_dir = gr.Textbox(label="Input Directory")
                output_dir = gr.Textbox(label="Output Directory")

                batch_white_bg_checkbox = gr.Checkbox(label="White Background", value=False)
                batch_only_matted_checkbox = gr.Checkbox(label="Only Matted", value=True)
                status_text = gr.Textbox(label="Status", interactive=False)
                batch_run_btn = gr.Button("Start Processing", variant="primary")

                batch_run_btn.click(
                    batch_inference,
                    inputs=[
                        input_dir,
                        output_dir,
                        model_image_size,
                        batch_white_bg_checkbox,
                        batch_only_matted_checkbox,
                    ],
                    outputs=[status_text],
                )

    app.launch(server_port=opt.port)
