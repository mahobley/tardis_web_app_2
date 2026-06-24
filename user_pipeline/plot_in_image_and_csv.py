#!/usr/bin/env python3
"""Overlay prediction centers from a `_predictions.csv` file on an echogram image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

csv_path = "/home/mahobley/Code/echo-seg/2018-06-10-JD161_RightNear_Stratum1_Set1_RN_2018-06-10_030003_0_-1_predictions.csv"
image_path = "/home/mahobley/Code/echo-seg/2018-06-10-JD161_RightNear_Stratum1_Set1_RN_2018-06-10_030003_0_-1_predictions.png"


df = pd.read_csv(csv_path)
required = {"center_frame", "center_frame_bin"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"CSV missing columns {sorted(missing)}; got {list(df.columns)}")

plot_df = df.dropna(subset=["center_frame", "center_frame_bin"]).copy()
if plot_df.empty:
    raise SystemExit("No rows with valid center_frame / center_frame_bin.")

img = np.asarray(Image.open(image_path).convert("RGB"))
h, w = img.shape[:2]

fig, ax = plt.subplots(figsize=(max(12, w / 120), max(6, h / 120)))
ax.imshow(img)
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

label_col = "class_name" if "class_name" in plot_df.columns else None
if label_col is not None:
    for name, group in plot_df.groupby(label_col, sort=False):
        ax.scatter(
            group["center_frame"],
            group["center_frame_bin"],
            s=36,
            label=str(name),
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
        )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
else:
    ax.scatter(
        plot_df["center_frame"],
        plot_df["center_frame_bin"],
        s=36,
        c="cyan",
        edgecolors="white",
        linewidths=0.6,
        alpha=0.95,
    )

span = {"enter_frame", "exit_frame"}.issubset(plot_df.columns)
if span:
    for _, row in plot_df.iterrows():
        y = row["center_frame_bin"]
        ax.plot(
            [row["enter_frame"], row["exit_frame"]],
            [y, y],
            color="yellow",
            alpha=0.35,
            linewidth=1.2,
        )

ax.axis("off")
fig.tight_layout()


plt.show()
