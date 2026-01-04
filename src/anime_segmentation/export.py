import torch

from .cli_parsers import build_export_parser
from .lit_module import AnimeSegmentation


def export_onnx(model, img_size, path) -> None:
    import onnx
    from onnxsim import simplify

    torch.onnx.export(
        model,  # model being run
        torch.randn(
            1,
            3,
            img_size,
            img_size,
        ),  # model input (or a tuple for multiple inputs)
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
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, path)


if __name__ == "__main__":
    opt = build_export_parser().parse_args()

    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, "cpu", img_size=opt.img_size)
    model.eval()
    if opt.to == "only_state_dict":
        torch.save(model.state_dict(), opt.out)
    elif opt.to == "only_net_state_dict":
        torch.save(model.net.state_dict(), opt.out)
    elif opt.to == "onnx":
        export_onnx(model, opt.img_size, opt.out)
