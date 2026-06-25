#!/usr/bin/env python3
"""Export a Python-decoded echogram for browser decoder parity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisheye_loading.pyDIDSON import DIDSON


def generate_echogram_from_aris(
    aris_path: Path,
    *,
    start_frame: int,
    end_frame: int,
) -> tuple[np.ndarray, dict]:
    didson = DIDSON(str(aris_path))
    echogram, metadata = didson.load_echogram(
        start_frame=start_frame,
        end_frame=end_frame,
        num_frames_bg_subtract=100000,
        use_blur=False,
        return_echogram_with_bg_subtracted=True,
        return_raw_echogram_as_third_channel=True,
    )

    echogram = echogram.transpose(1, 0, 2)
    echogram = np.nan_to_num(echogram, nan=0.0, posinf=0.0, neginf=0.0)
    echogram[:, 0, :] = 0
    echogram[:, -1, :] = 0
    echogram[0, :, :] = 0
    echogram[-1, :, :] = 0

    height, num_frames, n_channels = echogram.shape
    rgb = np.zeros((height, num_frames, 3), dtype=np.float32)
    rgb[:, :, 1] = (echogram[:, :, 1] + 0.5) * 255
    rgb[:, :, 2] = echogram[:, :, 0] * 255
    if n_channels > 2:
        rgb[:, :, 0] = echogram[:, :, 2] * 255
    return np.clip(rgb, 0, 255).astype(np.uint8), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-bin", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    args = parser.parse_args()

    rgb, metadata = generate_echogram_from_aris(
        args.input,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    args.output_bin.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_bin.write_bytes(rgb.tobytes())
    args.output_json.write_text(
        json.dumps(
            {
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
                "metadata": {
                    key: metadata.get(key)
                    for key in (
                        "version_id",
                        "numframes",
                        "numbeams",
                        "samplesperchannel",
                        "frameheadersize",
                        "fileheadersize",
                        "framesize",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
