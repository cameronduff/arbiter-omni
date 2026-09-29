"""
Unit tests for Microsoft DirectML hardware dispatch and telemetry in ArbiterOmni.
Supports AMD Radeon RX 480 acceleration on Windows and WSL2.
"""

import sys
from unittest.mock import MagicMock
import pytest
import torch

from arbiter_omni.device import (
    get_device_telemetry,
    get_directml_device,
    is_directml_available,
    resolve_device,
)


def test_directml_availability_boolean():
    # Function must return boolean without throwing exceptions
    avail = is_directml_available()
    assert isinstance(avail, bool)


def test_resolve_device_directml_fallback():
    # If directml is not installed in the current environment, it should fall back to CPU or CUDA
    dev = resolve_device(preference="directml")
    assert dev is not None
    assert isinstance(dev, torch.device)


def test_directml_mocked_dispatch(monkeypatch):
    # Simulate presence of torch_directml
    mock_dml = MagicMock()
    mock_dml.is_available.return_value = True
    mock_device = torch.device("cpu")  # proxy device
    mock_dml.device.return_value = mock_device
    mock_dml.device_name.return_value = "Radeon RX 480 Graphics"

    monkeypatch.setitem(sys.modules, "torch_directml", mock_dml)

    assert is_directml_available() is True
    dev = get_directml_device()
    assert dev == mock_device

    resolved = resolve_device(preference="directml")
    assert resolved == mock_device


def test_device_telemetry_includes_directml():
    telemetry = get_device_telemetry()
    assert "directml_available" in telemetry
    assert "device" in telemetry
    assert "torch_version" in telemetry
    assert isinstance(telemetry["directml_available"], bool)
