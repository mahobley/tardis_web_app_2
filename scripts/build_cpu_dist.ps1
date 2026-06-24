# Build a CPU-only EchoSeg one-folder bundle on Windows (no NVIDIA/CUDA/triton).
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Installing CPU-only dependencies into active environment..."
pip install -r requirements-cpu.txt
pip install pyinstaller

Write-Host "==> Verifying torch is CPU build..."
python -c @"
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    raise SystemExit(
        'CUDA torch is still installed. Use a clean venv and:\n'
        '  pip install -r requirements-cpu.txt'
    )
"@

Write-Host "==> Running PyInstaller (CPU-only spec)..."
pyinstaller --noconfirm --clean echo_seg_app.spec

$Dist = Join-Path $Root "dist\EchoSeg"
$Launcher = Join-Path $Root "packaging\run_EchoSeg.bat"
if (Test-Path $Dist) {
    if (Test-Path $Launcher) {
        Copy-Item -Force $Launcher (Join-Path $Dist "run_EchoSeg.bat")
        Write-Host "==> Installed launcher: dist\EchoSeg\run_EchoSeg.bat"
    }
}

Write-Host "==> Done. Output: dist\EchoSeg\"
Write-Host "    Run:  dist\EchoSeg\EchoSeg.exe   (or run_EchoSeg.bat)"
Write-Host "    Copy weights\*.pt to dist\EchoSeg\weights\ before distributing."
Write-Host "    If the window does not appear, check %LOCALAPPDATA%\echo-seg\crash.log"
