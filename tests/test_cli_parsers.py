from __future__ import annotations

from anime_segmentation.cli_parsers import (
    build_app_parser,
    build_export_parser,
    build_inference_parser,
)


def test_inference_parser_defaults() -> None:
    parser = build_inference_parser()
    opt = parser.parse_args([])

    assert opt.net in {"isnet_is", "isnet", "isnet_gt"}
    assert isinstance(opt.img_size, int)
    assert opt.img_size > 0
    assert opt.fp32 is False


def test_export_parser_defaults() -> None:
    parser = build_export_parser()
    opt = parser.parse_args([])

    assert opt.to in {"only_state_dict", "only_net_state_dict", "onnx"}
    assert isinstance(opt.img_size, int)
    assert opt.img_size > 0


def test_app_parser_defaults() -> None:
    parser = build_app_parser()
    opt = parser.parse_args([])

    assert isinstance(opt.port, int)
    assert opt.port > 0
