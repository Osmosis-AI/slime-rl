"""Smoke test for LoRA + MLflow + Qwen3.5 dependencies.

Run inside the Docker container (no GPUs required):
    python tests/test_lora_deps.py
"""

import sys


def test_core_imports():
    """Test that all core slime + LoRA modules import cleanly."""
    print("Testing core imports...")
    import os
    os.environ.setdefault("PYTHONPATH", "/root/Megatron-LM")
    import sys
    if "/root/Megatron-LM" not in sys.path:
        sys.path.insert(0, "/root/Megatron-LM")
    import slime
    from slime.utils.arguments import parse_args
    from slime.backends.megatron_utils.lora_utils import (
        is_lora_enabled,
        is_lora_model,
        LORA_ADAPTER_NAME,
        build_lora_adapter_config,
    )
    from slime.backends.megatron_utils.bridge_lora_helpers import (
        _setup_lora_model_via_bridge,
    )
    from slime.backends.megatron_utils.checkpoint import save_checkpoint_with_lora
    print("  OK: slime core + LoRA modules")


def test_bridge_imports():
    """Test megatron-bridge with Qwen3.5 support."""
    print("Testing bridge imports...")
    from megatron.bridge import AutoBridge
    # Check Qwen3.5 is registered
    try:
        from megatron.bridge.models.registry import BRIDGE_REGISTRY
        qwen35_found = any("qwen3" in k.lower() or "qwen3_5" in k.lower() for k in BRIDGE_REGISTRY)
    except ImportError:
        qwen35_found = False
    if not qwen35_found:
        try:
            from megatron.bridge.models.qwen3_5 import Qwen35Bridge
            qwen35_found = True
        except ImportError:
            pass
    print(f"  Qwen3.5 bridge registered: {qwen35_found}")
    if not qwen35_found:
        print("  WARNING: Qwen3.5 bridge not found — make sure coding-famer fork is installed")
    print("  OK: megatron-bridge")


def test_mlflow():
    """Test MLflow import and basic functionality."""
    print("Testing MLflow...")
    import mlflow
    from slime.utils.tracking.mlflow_utils import init_mlflow, log_metrics
    print(f"  MLflow version: {mlflow.__version__}")
    print("  OK: MLflow")


def test_sglang_lora():
    """Test SGLang LoRA support exists."""
    print("Testing SGLang LoRA...")
    try:
        from sglang.srt.lora.lora_manager import LoRAManager
        print("  OK: SGLang LoRA manager")
    except ImportError as e:
        print(f"  WARNING: SGLang LoRA manager not available: {e}")

    try:
        from sglang.srt.entrypoints.engine import Engine
        print("  OK: SGLang engine")
    except ImportError as e:
        print(f"  WARNING: SGLang engine import failed: {e}")


def test_mbridge():
    """Test mbridge (lightweight weight conversion)."""
    print("Testing mbridge...")
    try:
        import mbridge
        print("  OK: mbridge")
    except ImportError as e:
        print(f"  WARNING: mbridge not available: {e}")


def test_qwen35_model_spec():
    """Test Qwen3.5 model spec loads."""
    print("Testing Qwen3.5 model spec...")
    from slime_plugins.models.qwen3_5 import get_qwen3_5_spec
    print("  OK: qwen3_5 model spec")

    from slime_plugins.mbridge.qwen3_5 import Qwen3_5Bridge
    print("  OK: qwen3_5 mbridge")


def test_numpy_version():
    """Megatron requires numpy < 2."""
    print("Testing numpy version...")
    import numpy as np
    major = int(np.__version__.split(".")[0])
    assert major < 2, f"numpy {np.__version__} >= 2.0 — Megatron requires numpy < 2"
    print(f"  OK: numpy {np.__version__}")


def test_cudnn_version():
    """CuDNN must be >= 9.15 for SGLang compatibility check."""
    print("Testing CuDNN version...")
    try:
        import nvidia.cudnn as cudnn
        version = getattr(cudnn, "__version__", None)
        if version is None:
            # Newer cudnn packages use version() or _version
            version = getattr(cudnn, "version", lambda: "unknown")
            if callable(version):
                version = version()
        print(f"  CuDNN version: {version}")
        # Try pip package version as fallback
        import importlib.metadata
        pip_version = importlib.metadata.version("nvidia-cudnn-cu12")
        major, minor = int(pip_version.split(".")[0]), int(pip_version.split(".")[1])
        if major > 9 or (major == 9 and minor >= 15):
            print(f"  OK: nvidia-cudnn-cu12=={pip_version}")
        else:
            print(f"  WARNING: nvidia-cudnn-cu12=={pip_version} < 9.15 — SGLang may fail")
    except ImportError:
        print("  SKIP: nvidia.cudnn not importable (expected on non-GPU host)")


def test_torch_memory_saver():
    """Test torch_memory_saver for offload support."""
    print("Testing torch_memory_saver...")
    import torch_memory_saver
    print("  OK: torch_memory_saver")


def main():
    failures = []
    tests = [
        test_core_imports,
        test_bridge_imports,
        test_mlflow,
        test_sglang_lora,
        test_mbridge,
        test_qwen35_model_spec,
        test_numpy_version,
        test_cudnn_version,
        test_torch_memory_saver,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAIL: {e}")
            failures.append((test.__name__, str(e)))

    print()
    if failures:
        print(f"FAILED ({len(failures)}/{len(tests)}):")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print(f"ALL PASSED ({len(tests)}/{len(tests)})")
        sys.exit(0)


if __name__ == "__main__":
    main()
