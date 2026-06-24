#!/usr/bin/env python3
"""Export the current Python echogram image as raw bytes plus JSON metadata."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_pipeline.aris_to_detections import generate_echogram_from_aris


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-bin", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    rgb, metadata = generate_echogram_from_aris(
        args.input,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        bgs=True,
        return_raw_echogram_as_third_channel=True,
        return_as_bgr=True,
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
