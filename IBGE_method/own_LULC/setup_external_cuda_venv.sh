#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

cd "${ROOT_DIR}"

if [[ ! -d ".venv" ]]; then
  echo "[lulc-setup] creating .venv with ${PYTHON_CMD}"
  "${PYTHON_CMD}" -m venv .venv
fi

echo "[lulc-setup] upgrading installer tooling"
.venv/bin/python -m pip install --upgrade pip setuptools wheel

echo "[lulc-setup] installing CUDA PyTorch from ${PYTORCH_INDEX_URL}"
.venv/bin/pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision

echo "[lulc-setup] installing project requirements"
.venv/bin/pip install -r requirements.txt

echo "[lulc-setup] verifying torch import"
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

echo "[lulc-setup] done"
