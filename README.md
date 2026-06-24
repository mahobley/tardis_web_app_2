# Echo Seg User Pipeline

Load ARIS/DIDSON echograms, run YOLO segmentation, and export prediction plots and CSV detections.

## Browser app

A browser-first local app now lives under [`web/`](./web) and runs the sonar decode path in JavaScript, then executes the exported ONNX model locally in the browser with `onnxruntime-web`.

Start it with:

```bash
npm install
npm run dev
```

Build a static bundle with:

```bash
npm run build
```

The bundle is written to `dist-web/` and includes the current `weights/noklamath.onnx` model.

Current scope:

- local single-file `.aris` / `.ddf` decode in JavaScript
- local ONNX inference with `webgpu` or `wasm`
- in-browser decoded echogram and overlay preview
- upstream-direction aware detections table
- PNG / CSV / FC / Echotastic downloads from the browser UI

It does **not** yet port the desktop batch planner or multi-file workflow.

### Browser decoder parity check

The browser decoder is verified against the existing Python `fisheye_loading` path on `example.aris`:

```bash
npm run verify:decoder
```

That check is byte-for-byte, not just visual.

## Examples

### Desktop UI

![EchoSeg desktop UI](figures/UI_visual.png)

### Example output

![Example prediction output](figures/kenai-rightbank-stratum1__2018-06-10-JD161_RightNear_Stratum1_Set1_RN_2018-06-10_220003_0_-1_predictions.png)

## Desktop app (PySide6)

Activate your conda env first (the shell must **not** show `(base)` unless that env has the packages):

```bash
conda activate tardis-app-cpu
cd /path/to/echo-seg-user-pipeline
pip install -r requirements.txt
pip install -e .
python -m echo_seg_app
```

Or create/update the env from `environment.yml`:

```bash
conda env create -f environment.yml   # first time
# conda env update -f environment.yml --prune
conda activate tardis-app-cpu
pip install -e .
python -m echo_seg_app
```

Confirm you are using the right Python:

```bash
which python   # should be .../envs/tardis-app-cpu/bin/python
python -c "import matplotlib; print('ok')"
```

Or after `pip install -e .`:

```bash
echo-seg-gui
```

### GUI options

- **Checkpoint** — `.pt` YOLO weights (browse or pick from bundled `weights/`).
- **ARIS input** — add multiple `.aris` / `.ddf` files, or a directory (all files A–Z, or first N).
- **Skip already processed** — skip files whose enabled export outputs already exist in the output directory.
- **Output directory** — required for multiple files; defaults to `<folder>/outputs` when adding a directory.
- **Start / end frame** — end frame is exclusive; `-1` means through end of file.
- **Export PNG**, **Export CSV**, **Export FC files**, **Export Echotastic**.
- **Background subtraction** and **raw third channel** are always enabled; **filter submasks** remains under Advanced.
- **Inference FPS**, **Inference bins**, **device** (Advanced).

Logs are written under `logs/` in development. Frozen builds use `~/.local/share/echo-seg/` (Linux) or `%LOCALAPPDATA%\echo-seg\` (Windows).

## CLI pipeline

```bash
python -m user_pipeline.aris_to_detections
```

Edit paths in the `if __name__ == "__main__"` block, or use the API:

```python
from pathlib import Path
from user_pipeline.pipeline import PipelineConfig, run_pipeline

result = run_pipeline(
    PipelineConfig(
        aris_path=Path("file.aris"),
        checkpoint=Path("weights/noklamath.pt"),
    )
)
```

## Headless batch script

Run one or more files or directories without launching the GUI:

```bash
python -m user_pipeline.batch_cli \
  --checkpoint weights/noklamath.pt \
  --output-dir outputs \
  --skip-already-processed \
  data/run_a/file1.aris data/run_a/file2.aris
```

Or after `pip install -e .`:

```bash
echo-seg-batch --checkpoint weights/noklamath.pt --output-dir outputs data/run_a/
```

This writes the normal pipeline outputs plus `batch_timings.csv`, with one row per
input file and columns for status, wall-clock time, pipeline load/inference time,
exported output paths, and any error message.

It also supports split-driven runs over a super-directory of location subfolders:

```bash
echo-seg-batch \
  --checkpoint weights/noklamath.pt \
  --output-dir outputs \
  --split test \
  --split-path /mnt/data/CFC26_MAH/splits.json \
  --aris-super-dir /mnt/data/CFC26_MAH/ARIS
```

Use `--locations loc_a loc_b` to restrict the split JSON to specific locations. You
can still pass direct file or directory inputs in the same command; the CLI unions
both sources and de-duplicates files.

## PyInstaller (one-folder, CPU-only)

The GUI defaults to CPU inference. For a smaller distributable, build from a **CPU-only** PyTorch install (avoids bundling `nvidia/`, `triton/`, and CUDA torch libs).

**PyInstaller must run on the same OS as the target machine** (build on Windows for Windows, Linux for Linux).

Before building, confirm CUDA is not available in that environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# cuda available should be False
```

Copy `weights/*.pt` into `dist/EchoSeg/weights/` next to the `EchoSeg` executable (`EchoSeg.exe` on Windows).

If `dist/EchoSeg/_internal/nvidia` still appears, the build used a CUDA torch install — recreate the venv and install only `requirements-cpu.txt`.

### Linux build

```bash
# Recommended: clean venv, then
pip install -r requirements-cpu.txt
pip install pyinstaller
pyinstaller --clean echo_seg_app.spec
```

Or:

```bash
chmod +x scripts/build_cpu_dist.sh
./scripts/build_cpu_dist.sh
```

Launch:

```bash
./dist/EchoSeg/run_EchoSeg.sh   # if packaging/run_EchoSeg.sh was installed
./dist/EchoSeg/EchoSeg
```

Startup crashes are logged to `~/.local/share/echo-seg/crash.log`.

### Windows build

On **Windows 10/11 x64**, install [Python 3.10 or 3.11 (64-bit)](https://www.python.org/downloads/windows/) and check **“Add python.exe to PATH”** during setup.

From PowerShell in the repo root, confirm Python works **before** creating a venv:

```powershell
py -3.11 --version
# or:  python --version
```

If `python` fails but `py` works, use the launcher for the venv:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-cpu.txt
pip install pyinstaller
.\scripts\build_cpu_dist.ps1
```

**“running scripts is disabled on this system”** — either allow scripts for your user (once):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Or skip `.ps1` activation and use **cmd** instead:

```cmd
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r requirements-cpu.txt
pip install pyinstaller
pyinstaller --noconfirm --clean echo_seg_app.spec
```

**`python.exe` / “system cannot find the path specified” when creating `.venv`**

- **Windows Store alias** — Settings → Apps → Advanced app settings → **App execution aliases** → turn **off** “python.exe” and “python3.exe”, then open a **new** PowerShell window and use `py -3.11` or a full path (below).
- **`python` not on PATH** — Use the launcher: `py -3.11 -m venv .venv`, or the full path, e.g.  
  `& "$env:LocalAppData\Programs\Python\Python311\python.exe" -m venv .venv`
- **Wrong folder** — `cd` to the repo root (where `requirements-cpu.txt` lives) before `venv`.
- **Broken prior `.venv`** — Delete the `.venv` folder and run `py -3.11 -m venv .venv` again.

Or run the same `pip` / `pyinstaller` steps manually, then:

```powershell
pyinstaller --noconfirm --clean echo_seg_app.spec
```

**Distribute** the entire `dist\EchoSeg\` folder (zip it). Users run `EchoSeg.exe` or `run_EchoSeg.bat`. Expect roughly 1–2 GB.

- Logs and crash details: `%LOCALAPPDATA%\echo-seg\`
- Unsigned PyInstaller builds may trigger Windows Defender — allowlist internally if needed
- Build with **Python 3.10 or 3.11 x64**

## Layout

| Path               | Role                                  |
| ------------------ | ------------------------------------- |
| `echo_seg_app/`    | PySide6 GUI                           |
| `user_pipeline/`   | Echogram loading and `run_pipeline()` |
| `fisheye_loading/` | ARIS/DIDSON reader                    |
| `beam_widths/`     | Beam-width CSVs (bundled in releases) |
| `analysis/`        | YOLO prediction plotting              |
| `weights/`         | Model checkpoints (not in git)        |
