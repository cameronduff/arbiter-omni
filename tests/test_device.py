"""
Unit tests for ArbiterOmni hardware device resolution and diagnostics.
"""

import torch
import pytest

from arbiter_omni.device import (
    resolve_device,
    configure_cpu_threads,
    get_device_telemetry,
    is_directml_available,
)


def test_resolve_device_default():
    dev = resolve_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "cpu", "mps", "privateuseone")


def test_resolve_device_explicit_cpu():
    dev = resolve_device(preference="cpu")
    assert dev.type == "cpu"


def test_resolve_device_fallback_on_invalid_or_missing_cuda():
    # If cuda is not available on this test host, requesting cuda should fall back to cpu or return cuda if available
    dev = resolve_device(preference="cuda")
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_configure_cpu_threads():
    configured = configure_cpu_threads(num_threads=2)
    assert configured == 2
    assert torch.get_num_threads() == 2


def test_device_telemetry_keys():
    telemetry = get_device_telemetry()
    assert "device" in telemetry
    assert "device_type" in telemetry
    assert "torch_version" in telemetry
    assert "cuda_available" in telemetry
    assert "rocm_active" in telemetry
    assert "directml_available" in telemetry
    assert "cpu_threads" in telemetry
