# PyInstaller spec — CPU-only one-folder build (no CUDA / nvidia / triton).
#
# Use a CPU-only torch install before building:
#   pip install -r requirements-cpu.txt
#   pyinstaller echo_seg_app.spec
#
# Or:  ./scripts/build_cpu_dist.sh   (Linux)
#      .\scripts\build_cpu_dist.ps1  (Windows)
#
# Copy weights/ beside dist/EchoSeg/EchoSeg(.exe) after building.

# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).resolve()


def _collect_expat_binaries():
    """Bundle expat matching the build Python (fixes pyexpat symbol errors)."""
    import sys

    binaries = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        binaries.append((resolved, "."))
        print(f"[echo_seg_app.spec] Bundling {path}")

    if sys.platform == "win32":
        search_dirs = [
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.prefix) / "DLLs",
            Path(sys.base_prefix) / "DLLs",
        ]
        patterns = ("libexpat*.dll", "expat.dll")
    else:
        search_dirs = [
            Path(sys.prefix) / "lib",
            Path(sys.base_prefix) / "lib",
        ]
        patterns = ("libexpat.so*",)

    for lib_dir in search_dirs:
        if not lib_dir.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(lib_dir.glob(pattern)):
                _add(path)

    try:
        from PyInstaller.utils.hooks import collect_dynamic_libs

        for entry in collect_dynamic_libs("pyexpat"):
            src = entry[0] if isinstance(entry, tuple) else entry
            if src not in seen:
                seen.add(src)
                binaries.append(entry if isinstance(entry, tuple) else (entry, "."))
    except Exception as exc:
        print(f"[echo_seg_app.spec] collect_dynamic_libs(pyexpat): {exc}")

    if not binaries:
        print("[echo_seg_app.spec] WARNING: expat not found — pyexpat may fail at runtime")
    return binaries


def _matplotlib_datas():
    try:
        from PyInstaller.utils.hooks import collect_data_files

        return collect_data_files("matplotlib")
    except Exception as exc:
        print(f"[echo_seg_app.spec] WARNING: matplotlib data collection failed: {exc}")
        return []


def _pyside6_plugin_datas():
    """Bundle Qt platform plugins (required when double-clicking the binary)."""
    import PySide6

    plugins = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    if not plugins.is_dir():
        print(f"[echo_seg_app.spec] WARNING: PySide6 plugins not found at {plugins}")
        return []
    print(f"[echo_seg_app.spec] Bundling Qt plugins from {plugins}")
    return [(str(plugins), "PySide6/Qt/plugins")]


_binaries = _collect_expat_binaries()

datas = [
    (str(ROOT / "beam_widths"), "beam_widths"),
    *_pyside6_plugin_datas(),
    *_matplotlib_datas(),
]

# Drop heavy GPU vendor packages only. Do not exclude torch.* submodules — ultralytics
# imports many of them on CPU builds; EXCLUDES caused repeated "No module named …" errors.
EXCLUDES = [
    "triton",
    "nvidia",
    "cupy",
    "pycuda",
    "tensorflow",
    "tensorboard",
    "tkinter",
]


def _torch_hiddenimports():
    """Collect torch submodules PyInstaller's static analysis often misses."""
    names = [
        "torch.cuda",
        "torch.backends.cuda",
        "torch.backends.cudnn",
        "torch.distributed",
        "torch.nn",
        "torch.jit",
        "torch._inductor",
        "torch._inductor.test_operators",
    ]
    try:
        from PyInstaller.utils.hooks import collect_submodules

        names.extend(collect_submodules("torch._inductor"))
    except Exception as exc:
        print(f"[echo_seg_app.spec] torch._inductor collect_submodules: {exc}")
    return list(dict.fromkeys(names))


hiddenimports = [
    "ultralytics",
    "torch",
    *_torch_hiddenimports(),
    "cv2",
    "PIL",
    "matplotlib.backends.backend_agg",
    "shiboken6",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "fisheye_loading.pyDIDSON",
    "fisheye_loading.pyARIS",
    "fisheye_loading.echogram",
    "analysis.plot_echogram_predictions",
    "analysis.prediction_exports",
    "user_pipeline.config",
    "user_pipeline.pipeline",
    "user_pipeline.aris_to_detections",
    "user_pipeline.batch",
    "xml.parsers.expat",
    "pyexpat",
    "echo_seg_app.frozen_preload",
]

# Substrings matched against binary basenames and source paths (lowercase).
_CUDA_BINARY_SKIP = (
    "nvidia",
    "triton",
    "cudart",
    "cublas",
    "cudnn",
    "cufft",
    "cupti",
    "nccl",
    "cusparse",
    "curand",
    "cusolver",
    "cufile",
    "libtorch_cuda",
    "torch_cuda",
    "/cuda/",
)


def _skip_artifact(name: str, src: str) -> bool:
    hay = f"{name} {src}".lower().replace("\\", "/")
    return any(token in hay for token in _CUDA_BINARY_SKIP)


def _filter_binaries(binaries):
    kept = []
    dropped = 0
    for entry in binaries:
        name = entry[0]
        src = entry[1] if len(entry) > 1 else ""
        if _skip_artifact(name, src):
            dropped += 1
            continue
        kept.append(entry)
    if dropped:
        print(f"[echo_seg_app.spec] Excluded {dropped} CUDA/GPU binary artifact(s)")
    return kept


def _filter_datas(datas):
    kept = []
    dropped = 0
    for entry in datas:
        src = entry[0] if entry else ""
        dest = entry[1] if len(entry) > 1 else ""
        if _skip_artifact(dest, src):
            dropped += 1
            continue
        kept.append(entry)
    if dropped:
        print(f"[echo_seg_app.spec] Excluded {dropped} CUDA/GPU data artifact(s)")
    return kept


a = Analysis(
    [str(ROOT / "echo_seg_app" / "main.py")],
    pathex=[str(ROOT), str(ROOT / "echo_seg_app")],
    binaries=_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "echo_seg_app" / "pyinstaller_cpu_runtime.py")],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a.binaries = _filter_binaries(a.binaries)
a.datas = _filter_datas(a.datas)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EchoSeg",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EchoSeg",
)
