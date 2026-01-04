from __future__ import annotations

from jsonargparse import ArgumentParser

from .lit_module import net_names


def _base_parser(*, prog: str) -> ArgumentParser:
    parser = ArgumentParser(prog=prog)
    # Enable config files consistently across CLIs
    parser.add_argument("--config", action="config", help="path to a YAML/JSON config file")
    return parser


def build_inference_parser() -> ArgumentParser:
    parser = _base_parser(prog="anime_segmentation.inference")

    parser.add_argument("--net", type=str, default="isnet_is", choices=net_names, help="net name")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="saved_models/isnetis.ckpt",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../../dataset/anime-seg/test2",
        help="input data dir",
    )
    parser.add_argument("--out", type=str, default="out", help="output dir")
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="hyperparameter, input image size of the net",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda:0")
    parser.add_argument("--fp32", action="store_true", default=False, help="disable mix precision")
    parser.add_argument(
        "--only-matted",
        action="store_true",
        default=False,
        help="only output matted image",
    )
    parser.add_argument(
        "--bg-white",
        action="store_true",
        default=False,
        help="change transparent background to white",
    )

    return parser


def build_export_parser() -> ArgumentParser:
    parser = _base_parser(prog="anime_segmentation.export")

    parser.add_argument(
        "--net",
        type=str,
        default="isnet_is",
        choices=net_names,
        help="net name",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="saved_models/isnetis.ckpt",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="saved_models/isnetis.onnx",
        help="output path",
    )
    parser.add_argument(
        "--to",
        type=str,
        default="onnx",
        choices=["only_state_dict", "only_net_state_dict", "onnx"],
        help="export to ()",
    )
    parser.add_argument("--img-size", type=int, default=1024, help="input image size")

    return parser


def build_app_parser() -> ArgumentParser:
    parser = _base_parser(prog="anime_segmentation.app")
    parser.add_argument("--port", type=int, default=6006, help="gradio server port")
    return parser
