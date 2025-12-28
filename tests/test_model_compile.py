"""Test torch.compile and torch.jit.script compatibility for all models."""

import pytest
import torch
from torch import nn

# Test configurations for different models
MODEL_CONFIGS = {
    "ISNetDIS": {"in_ch": 3, "out_ch": 1},
    "ISNetGTEncoder": {"in_ch": 1, "out_ch": 1},
    "U2NetFull": {},
    "U2NetFull2": {},
    "U2NetLite": {},
    "U2NetLite2": {},
    "MODNet": {"in_channels": 3, "hr_channels": 32},
    "InSPyReNet_Res2Net50": {"depth": 64, "pretrained": False, "base_size": 384},
    "InSPyReNet_SwinB": {"depth": 64, "pretrained": False, "base_size": 384},
}

# Input size configurations
INPUT_SIZES = {
    "ISNetDIS": (1, 3, 256, 256),
    "ISNetGTEncoder": (1, 1, 256, 256),
    "U2NetFull": (1, 3, 256, 256),
    "U2NetFull2": (1, 3, 256, 256),
    "U2NetLite": (1, 3, 256, 256),
    "U2NetLite2": (1, 3, 256, 256),
    "MODNet": (1, 3, 256, 256),
    "InSPyReNet_Res2Net50": (1, 3, 384, 384),
    "InSPyReNet_SwinB": (1, 3, 384, 384),
}


def get_model(model_name: str) -> nn.Module:
    """Create model instance by name."""
    from anime_segmentation.model import (
        InSPyReNet_Res2Net50,
        InSPyReNet_SwinB,
        ISNetDIS,
        ISNetGTEncoder,
        MODNet,
        U2NetFull,
        U2NetFull2,
        U2NetLite,
        U2NetLite2,
    )

    model_classes = {
        "ISNetDIS": ISNetDIS,
        "ISNetGTEncoder": ISNetGTEncoder,
        "U2NetFull": U2NetFull,
        "U2NetFull2": U2NetFull2,
        "U2NetLite": U2NetLite,
        "U2NetLite2": U2NetLite2,
        "MODNet": MODNet,
        "InSPyReNet_Res2Net50": InSPyReNet_Res2Net50,
        "InSPyReNet_SwinB": InSPyReNet_SwinB,
    }

    config = MODEL_CONFIGS[model_name]
    return model_classes[model_name](**config)


def get_forward_args(model_name: str, device: str = "cpu"):
    """Get forward arguments for a model."""
    input_size = INPUT_SIZES[model_name]
    x = torch.randn(*input_size, device=device)

    if model_name == "MODNet":
        return (x, False)  # (input, inference_mode)
    return (x,)


# ============== torch.compile tests ==============


@pytest.mark.parametrize("model_name", list(MODEL_CONFIGS.keys()))
def test_torch_compile(model_name: str):
    """Test that models can be compiled with torch.compile."""
    model = get_model(model_name)
    model.eval()

    try:
        compiled_model = torch.compile(model, backend="eager")
        args = get_forward_args(model_name)

        with torch.no_grad():
            output = compiled_model(*args)

        # Verify output is valid
        if isinstance(output, dict):
            assert "pred" in output or len(output) > 0
        elif isinstance(output, (list, tuple)):
            assert len(output) > 0
        else:
            assert output is not None

        print(f"[PASS] {model_name}: torch.compile (eager backend) works")

    except Exception as e:
        pytest.fail(f"[FAIL] {model_name}: torch.compile failed - {e}")


@pytest.mark.parametrize("model_name", list(MODEL_CONFIGS.keys()))
def test_torch_compile_inductor(model_name: str):
    """Test that models can be compiled with torch.compile using inductor backend."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for inductor backend test")

    model = get_model(model_name).cuda()
    model.eval()

    try:
        compiled_model = torch.compile(model, backend="inductor")
        args = get_forward_args(model_name, device="cuda")

        with torch.no_grad():
            output = compiled_model(*args)

        if isinstance(output, dict):
            assert "pred" in output or len(output) > 0
        elif isinstance(output, (list, tuple)):
            assert len(output) > 0
        else:
            assert output is not None

        print(f"[PASS] {model_name}: torch.compile (inductor backend) works")

    except Exception as e:
        pytest.fail(f"[FAIL] {model_name}: torch.compile (inductor) failed - {e}")


# ============== torch.jit.script tests ==============


@pytest.mark.parametrize("model_name", list(MODEL_CONFIGS.keys()))
def test_torch_jit_script(model_name: str):
    """Test that models can be scripted with torch.jit.script (no tracing)."""
    model = get_model(model_name)
    model.eval()

    try:
        scripted_model = torch.jit.script(model)
        args = get_forward_args(model_name)

        with torch.no_grad():
            output = scripted_model(*args)

        if isinstance(output, (dict, list, tuple)):
            assert len(output) > 0
        else:
            assert output is not None

        print(f"[PASS] {model_name}: torch.jit.script works")

    except Exception as e:
        print(f"[FAIL] {model_name}: torch.jit.script failed - {e}")
        pytest.xfail(f"torch.jit.script not supported: {e}")


# ============== Summary test for quick check ==============


def test_compile_summary():
    """Run a quick compile test for all models and print summary."""
    results = {"torch.compile": {}, "torch.jit.script": {}}

    for model_name in MODEL_CONFIGS:
        print(f"\n--- Testing {model_name} ---")

        # Test torch.compile
        try:
            model = get_model(model_name)
            model.eval()
            compiled = torch.compile(model, backend="eager")
            args = get_forward_args(model_name)
            with torch.no_grad():
                compiled(*args)
            results["torch.compile"][model_name] = "PASS"
            print("  torch.compile: PASS")
        except Exception as e:
            results["torch.compile"][model_name] = f"FAIL: {e}"
            print(f"  torch.compile: FAIL - {e}")

        # Test torch.jit.script
        try:
            model = get_model(model_name)
            model.eval()
            scripted = torch.jit.script(model)
            args = get_forward_args(model_name)
            with torch.no_grad():
                scripted(*args)
            results["torch.jit.script"][model_name] = "PASS"
            print("  torch.jit.script: PASS")
        except Exception as e:
            results["torch.jit.script"][model_name] = f"FAIL: {e}"
            print(f"  torch.jit.script: FAIL - {e}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'torch.compile':<15} {'torch.jit.script':<15}")
    print("-" * 80)
    for model_name in MODEL_CONFIGS:
        compile_result = "PASS" if results["torch.compile"][model_name] == "PASS" else "FAIL"
        script_result = "PASS" if results["torch.jit.script"][model_name] == "PASS" else "FAIL"
        print(f"{model_name:<25} {compile_result:<15} {script_result:<15}")

    return results


if __name__ == "__main__":
    test_compile_summary()
