import argparse

import torch

from anime_segmentation.train import NET_NAMES, AnimeSegmentation


def export_onnx(model: AnimeSegmentation, img_size, path) -> None:
    import onnx
    from onnxslim import slim

    torch.onnx.export(
        model,  # model being run
        torch.randn(1, 3, img_size, img_size),  # model input (or a tuple for multiple inputs)
        path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["img"],  # the model's input names
        output_names=["mask"],  # the model's output names
        dynamic_axes={
            "img": {0: "batch_size"},  # variable length axes
            "mask": {0: "batch_size"},
        },
        verbose=True,
    )
    onnx_model = onnx.load(path)
    model_simp = slim(onnx_model)
    onnx.save(model_simp, path)
    print("finished exporting onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--net", type=str, default="isnet_is", choices=NET_NAMES, help="net name")
    parser.add_argument(
        "--ckpt", type=str, default="saved_models/isnetis.ckpt", help="model checkpoint path"
    )
    parser.add_argument("--out", type=str, default="saved_models/isnetis.onnx", help="output path")
    parser.add_argument(
        "--to",
        type=str,
        default="onnx",
        choices=["only_state_dict", "only_net_state_dict", "onnx"],
        help="export to ()",
    )
    parser.add_argument("--img-size", type=int, default=1024, help="input image size")
    opt = parser.parse_args()
    print(opt)

    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, "cpu", img_size=opt.img_size)
    model.eval()
    match opt.to:
        case "only_state_dict":
            torch.save(model.state_dict(), opt.out)
        case "only_net_state_dict":
            torch.save(model.net.state_dict(), opt.out)
        case "onnx":
            export_onnx(model, opt.img_size, opt.out)
