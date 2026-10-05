"""
Hardware Agnostic Device Dispatcher and System Telemetry for ArbiterOmni.
Supports CUDA, AMD ROCm, DirectML (WSL2 / Windows), Apple MPS, and multi-core CPU.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger(__name__)


def is_directml_available() -> bool:
    """Checks whether Microsoft DirectML is available (e.g. on Windows or WSL2)."""
    try:
        import torch_directml  # type: ignore

        return bool(torch_directml.is_available())
    except (ImportError, Exception):
        return False


def get_directml_device() -> torch.device:
    """Returns the primary DirectML device."""
    import torch_directml  # type: ignore

    return torch_directml.device()


def resolve_device(preference: Optional[str] = None) -> torch.device:
    """
    Dynamically resolves the optimal compute device.
    
    Order of preference when preference is None:
    1. CUDA / AMD ROCm (via torch.cuda)
    2. Microsoft DirectML (via torch_directml on WSL2 / Windows)
    3. Apple Silicon MPS (via torch.backends.mps)
    4. Multi-threaded CPU fallback
    """
    if preference:
        pref = preference.lower().strip()
        if pref in ("cuda", "rocm"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA/ROCm requested but not available. Falling back.")
        elif pref in ("dml", "directml"):
            if is_directml_available():
                return get_directml_device()
            logger.warning("DirectML requested but not available. Falling back.")
        elif pref == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but not available. Falling back.")
        elif pref == "cpu":
            return torch.device("cpu")
        else:
            return torch.device(pref)

    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")

    if is_directml_available():
        return get_directml_device()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Optimal CPU setup
    configure_cpu_threads()
    return torch.device("cpu")


def configure_cpu_threads(num_threads: Optional[int] = None) -> int:
    """Configures OpenMP / MKL thread count for maximum CPU throughput."""
    if num_threads is None:
        cores = os.cpu_count() or 4
        num_threads = cores
    try:
        torch.set_num_threads(num_threads)
    except Exception as e:
        logger.debug(f"Could not set PyTorch thread count: {e}")
    try:
        torch.set_num_interop_threads(num_threads)
    except Exception as e:
        logger.debug(f"Could not set PyTorch interop thread count: {e}")
    return num_threads


def get_device_telemetry(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Inspects compute hardware and returns structured telemetry."""
    dev = device or resolve_device()
    rocm_active = hasattr(torch.version, "hip") and torch.version.hip is not None
    dml_active = is_directml_available()

    info: Dict[str, Any] = {
        "device": str(dev),
        "device_type": dev.type,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "rocm_active": rocm_active,
        "directml_available": dml_active,
        "cpu_count": os.cpu_count(),
        "cpu_threads": torch.get_num_threads(),
        "os_platform": platform.platform(),
    }

    if dev.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_vram_mb"] = round(props.total_memory / (1024**2), 1)
        info["allocated_vram_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 1)
        info["reserved_vram_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 1)
    elif dml_active and dev.type == "privateuseone":
        try:
            import torch_directml
            info["gpu_name"] = torch_directml.device_name(0)
        except Exception:
            info["gpu_name"] = "DirectML GPU"
        info["total_vram_mb"] = 8192.0  # RX 480 8GB
        info["allocated_vram_mb"] = None
        info["reserved_vram_mb"] = None
    else:
        info["gpu_name"] = "CPU / Emulated"
        info["total_vram_mb"] = None
        info["allocated_vram_mb"] = None

    return info


def print_device_diagnostics() -> None:
    """Pretty prints the device diagnostics and telemetry table to stdout."""
    telemetry = get_device_telemetry()
    print("=" * 70)
    print("           ARBITER-OMNI HARDWARE & ACCELERATION TELEMETRY")
    print("=" * 70)
    print(f" Resolved Compute Device : {telemetry['device']}")
    print(f" Hardware Identifier     : {telemetry['gpu_name']}")
    print(f" PyTorch Version         : {telemetry['torch_version']}")
    print(f" CUDA Enabled            : {telemetry['cuda_available']}")
    print(f" AMD ROCm (HIP) Active   : {telemetry['rocm_active']}")
    print(f" DirectML (WSL2/Win)     : {telemetry['directml_available']}")
    print(f" Host OS / Kernel        : {telemetry['os_platform']}")
    print(f" Active CPU Cores/Threads: {telemetry['cpu_count']} cores / {telemetry['cpu_threads']} worker threads")

    if telemetry["total_vram_mb"] is not None:
        print(f" Dedicated VRAM          : {telemetry['total_vram_mb']:,.1f} MB")
        if telemetry.get("allocated_vram_mb") is not None:
            print(f" Current Allocated VRAM  : {telemetry['allocated_vram_mb']:,.1f} MB")
    else:
        print(" Dedicated VRAM          : N/A (Host RAM / Shared Memory)")
    print("=" * 70)


if __name__ == "__main__":
    print_device_diagnostics()
