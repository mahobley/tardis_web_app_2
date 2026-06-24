"""Load ARIS/DIDSON echograms and optionally plot model predictions."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# ``python user_pipeline/aris_to_detections.py`` — repo root is not on sys.path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fisheye_loading.pyDIDSON import DIDSON

logger = logging.getLogger(__name__)


def generate_echogram_from_aris(
    aris_path: str | Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    *,
    bgs: bool = True,
    return_raw_echogram_as_third_channel: bool = False,
    debug_plot: bool = False,
    return_as_bgr: bool = False,
    beam_width_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Load an echogram as an RGB ``uint8`` array from a DIDSON/ARIS file.

    Args:
        aris_path: Path to a ``.ddf`` or ``.aris`` file.
        start_frame: First frame index; default is ``0``.
        end_frame: Exclusive end frame index; default is ``-1`` (through end of file).
        bgs: When True, use background-subtracted echogram from the loader.
        return_raw_echogram_as_third_channel: Passed through to ``DIDSON.load_echogram``.
        debug_plot: When True, show per-channel previews with matplotlib.
        beam_width_dir: Optional override for ARIS beam-width CSV directory.

    Returns:
        ``(rgb, metadata)`` — RGB array of shape ``(height, num_frames, 3)`` (``uint8``)
        and DIDSON/ARIS header ``info`` dict (meters where applicable).
    """
    path = Path(aris_path)
    suf = path.suffix.lower()
    if suf not in (".ddf", ".aris"):
        raise ValueError(f"Expected .ddf or .aris file, got {path}")

    logger.info("Opening ARIS/DIDSON file: %s", path.name)

    didson_kwargs: dict = {}
    if beam_width_dir is not None:
        didson_kwargs["beam_width_dir"] = Path(beam_width_dir)
    logger.info("Reading file header…")
    didson = DIDSON(str(path), **didson_kwargs)
    frame_start = 0 if start_frame is None else start_frame
    frame_end = -1 if end_frame is None else end_frame

    logger.info(
        "Loading echogram frames %s–%s (this may take a while)…", frame_start, frame_end
    )

    echogram, metadata = didson.load_echogram(
        start_frame=frame_start,
        end_frame=frame_end,
        num_frames_bg_subtract=100000,
        use_blur=False,
        return_echogram_with_bg_subtracted=bgs,
        return_raw_echogram_as_third_channel=return_raw_echogram_as_third_channel,
    )

    echogram = echogram.transpose(1, 0, 2)
    echogram = np.nan_to_num(echogram, nan=0.0, posinf=0.0, neginf=0.0)
    echogram[:, 0, :] = 0
    echogram[:, -1, :] = 0
    echogram[0, :, :] = 0
    echogram[-1, :, :] = 0

    height, num_frames, n_ch = echogram.shape
    if n_ch > 3:
        logger.warning(
            "Echogram has more than 3 channels; only the first 3 will be used"
        )

    if debug_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, min(n_ch, 3), figsize=(15, 5))
        axes_list = axes if isinstance(axes, np.ndarray) else [axes]
        for c, ax in enumerate(axes_list):
            ax.imshow(echogram[:, :, c])
            ax.set_title(f"Channel {c}")
        plt.show()

    rgb = np.zeros((height, num_frames, 3), dtype=np.float32)
    rgb[:, :, 1] = (echogram[:, :, 1] + 0.5) * 255

    if return_as_bgr:
        rgb[:, :, 2] = echogram[:, :, 0] * 255
        if n_ch > 2:
            rgb[:, :, 0] = echogram[:, :, 2] * 255
    else:
        rgb[:, :, 0] = echogram[:, :, 0] * 255
        if n_ch > 2:
            rgb[:, :, 2] = echogram[:, :, 2] * 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, metadata


if __name__ == "__main__":
    # Prefer from repo root:  python -m user_pipeline.aris_to_detections
    from user_pipeline.config import PipelineConfig
    from user_pipeline.pipeline import run_pipeline

    cfg = PipelineConfig(
        aris_path=Path("./example.aris"),
        checkpoint=Path("./weights/noklamath.pt"),
        start_frame=0,
        end_frame=-1,
        bgs=True,
        raw_third_channel=True,
        output_dir=Path("./outputs"),
        device="cpu",
        fp16=False,
    )
    t0 = time.perf_counter()
    result = run_pipeline(cfg)
    if result.prediction_png is not None:
        print(f"Saved {result.prediction_png}")
    elif result.exported_paths:
        print(f"Saved {result.exported_paths[0]}")
    print(f"Total time: {time.perf_counter() - t0:.3f}s")
