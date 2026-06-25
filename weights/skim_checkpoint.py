#!/usr/bin/env python3
"""
Strip unnecessary data from a YOLO/PyTorch checkpoint.

Usage:
  python shrink_yolo_ckpt.py path/to/best.pt
  python shrink_yolo_ckpt.py path/to/best.pt --out path/to/best_stripped.pt
"""

import argparse
from pathlib import Path
import torch

DROP_KEYS = {
    "optimizer",
    "ema",
    "updates",
    "best_fitness",
    "wandb_id",
    "date",
    "train_args",
    "epoch",
    "metrics",
    "fitness",
    "loss",
    "git",
    "version",
    "args",
}


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def strip_checkpoint(input_path: Path, output_path: Path) -> None:
    # PyTorch 2.6+ defaults to weights_only=True, which often fails on YOLO ckpts.
    # Only use weights_only=False for checkpoints you trust.
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        torch.save(ckpt, output_path)
        return

    print("Original checkpoint keys:")
    for k in ckpt.keys():
        print(f"  - {k}")

    stripped = dict(ckpt)

    # Prefer EMA weights for inference if present.
    if "ema" in stripped and stripped["ema"] is not None:
        stripped["model"] = stripped["ema"]

    drop_keys = {
        "optimizer",
        "ema",
        "updates",
        "best_fitness",
        "wandb_id",
        "date",
        "train_args",
        "epoch",
        "metrics",
        "fitness",
        "loss",
        "git",
        "version",
        "args",
    }

    for key in drop_keys:
        stripped.pop(key, None)

    # Convert model object to fp32 for compatibility.
    if "model" in stripped and hasattr(stripped["model"], "float"):
        stripped["model"] = stripped["model"].float()

    torch.save(stripped, output_path)

    print()
    print(f"Saved stripped checkpoint: {output_path}")
    print(f"Before: {file_size_mb(input_path):.2f} MB")
    print(f"After:  {file_size_mb(output_path):.2f} MB")
    print(f"Saved:  {file_size_mb(input_path) - file_size_mb(output_path):.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.checkpoint

    if args.out is None:
        output_path = input_path.with_name(input_path.stem + "_stripped.pt")
    else:
        output_path = args.out

    strip_checkpoint(input_path, output_path)


if __name__ == "__main__":
    main()
