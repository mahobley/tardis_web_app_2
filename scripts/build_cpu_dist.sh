#!/usr/bin/env bash

conda activate tardis-app


# Build a CPU-only EchoSeg one-folder bundle (no NVIDIA/CUDA/triton in dist).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Installing CPU-only dependencies into active environment…"
pip install -r requirements-cpu.txt
pip install pyinstaller
# Align system libexpat with the Python that builds pyexpat (conda).
if command -v conda >/dev/null 2>&1; then
  conda install -y -q expat 2>/dev/null || true
fi

echo "==> Verifying torch is CPU build…"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    raise SystemExit(
        "CUDA torch is still installed. Use a clean venv and:\n"
        "  pip install -r requirements-cpu.txt"
    )
PY

echo "==> Running PyInstaller (CPU-only spec)…"
pyinstaller --noconfirm --clean echo_seg_app.spec

DIST="$ROOT/dist/EchoSeg"
if [[ -d "$DIST" ]]; then
  install -m 755 "$ROOT/packaging/run_EchoSeg.sh" "$DIST/run_EchoSeg.sh"
  echo "==> Installed launcher: dist/EchoSeg/run_EchoSeg.sh"
fi

echo "==> Done. Output: dist/EchoSeg/"
echo "    Run:  ./dist/EchoSeg/run_EchoSeg.sh   (or ./dist/EchoSeg/EchoSeg from that directory)"
echo "    Copy weights/*.pt to dist/EchoSeg/weights/ before distributing."
echo "    If the window does not appear, check ~/.local/share/echo-seg/crash.log"
