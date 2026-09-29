#!/usr/bin/env bash
# ==============================================================================
# ArbiterOmni: DirectML Setup Script for AMD Radeon RX 480 on WSL2 / Linux
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "⚡ ArbiterOmni DirectML Setup (AMD Radeon RX 480 / WSL2 GPU Pipeline)"
echo "======================================================================"

# 1. Verify WSL DirectX Kernel Driver
echo "[1/4] Checking WSL2 DirectX Graphics Driver (/dev/dxg)..."
if [ -e "/dev/dxg" ]; then
    echo "  ✓ /dev/dxg detected (DirectX GPU kernel interface active)"
else
    echo "  ⚠️ /dev/dxg not found. Ensure WSL2 GPU paravirtualization is enabled in Windows."
fi

if [ -d "/usr/lib/wsl/lib" ]; then
    echo "  ✓ /usr/lib/wsl/lib detected (libd3d12.so available)"
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

# 2. Check Python 3.12 availability (torch-directml requires Python <= 3.12)
echo "[2/4] Initializing Python 3.12 virtual environment for torch-directml..."
uv venv --python 3.12 .venv-directml
echo "  ✓ Virtual environment created at .venv-directml (Python 3.12)"

# 3. Install torch-directml and project requirements
echo "[3/4] Installing torch-directml and dependencies..."
uv pip install --python .venv-directml/bin/python torch-directml
uv pip install --python .venv-directml/bin/python -e .

# 4. Run hardware verification
echo "[4/4] Verifying DirectML hardware tensor execution..."
.venv-directml/bin/python -c "
import torch
import torch_directml

print('=' * 65)
print('  DirectML Available  :', torch_directml.is_available())
if torch_directml.is_available():
    device = torch_directml.device()
    print('  DirectML Device     :', device)
    print('  Device Name         :', torch_directml.device_name(0))
    x = torch.randn(1024, 1024, device=device)
    y = torch.matmul(x, x)
    print('  GEMM Check (1024x1024): OK (device:', y.device, ')')
print('=' * 65)
"

echo "✅ DirectML setup complete! Activate with: source .venv-directml/bin/activate"
