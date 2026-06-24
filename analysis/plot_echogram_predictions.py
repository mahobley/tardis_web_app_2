"""
Plot YOLO segmentation predictions for a single echogram.

The figure contains:
- Left: rendered echogram
- Middle: the same echogram with predicted masks overlaid and colored by class
- Right: optional ground-truth instance masks from a YOLO polygon label file
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils.make_echogram_image import make_echogram_image
from utils.submask_utils import filter_out_submasks
from utils.yolo_segmentation import (
    get_prediction_class_ids,
    get_prediction_confidences,
    get_prediction_masks_in_image_space,
    predict_segmentation,
    resize_mask,
)

DEFAULT_MODEL = str(
    Path(__file__).parent.parent
    / "runs"
    / "segment_and_classify"
    / "yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6"
    / "weights"
    / "best.pt"
)


CLASS_COLOUR_ANCHORS = {
    0: (
        (178, 235, 242),
        (0, 188, 212),
        (3, 169, 244),
        (30, 136, 229),
        (13, 71, 161),
    ),
    1: (
        (255, 224, 130),
        (255, 193, 7),
        (255, 167, 38),
        (245, 124, 0),
        (230, 81, 0),
    ),
    2: (
        (248, 187, 208),
        (236, 64, 122),
        (171, 71, 188),
        (126, 87, 194),
        (74, 20, 140),
    ),
}

CLASS_COUNT_LABELS = {
    0: "left",
    1: "right",
    2: "no cross",
}

DEFAULT_HEADER_FONT_SIZE = 18
HEADER_TITLE_HEIGHT_FRACTION = 0.045
HEADER_SUBTITLE_HEIGHT_FRACTION = 0.04
HEADER_VERTICAL_PADDING_FRACTION = 0.18
HEADER_HORIZONTAL_PADDING_FRACTION = 0.5
HEADER_SUPERSAMPLE_SCALE = 8


def _framerate_from_metadata(metadata: dict | None) -> float | None:
    if metadata is None:
        return None
    for key in ("framerate", "FrameRate", "frame_rate", "fps"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _choose_frame_keep_ratio(
    original_frame_rate: float, goal_frame_rate: float, max_cycle_length: int = 60
) -> tuple[int, int]:
    target_keep_ratio = goal_frame_rate / original_frame_rate
    best_keep_count = 1
    best_cycle_length = 1
    best_error = abs(original_frame_rate - goal_frame_rate)

    for cycle_length in range(1, max_cycle_length + 1):
        for keep_count in range(1, cycle_length + 1):
            achieved_frame_rate = original_frame_rate * keep_count / cycle_length
            error = abs(achieved_frame_rate - goal_frame_rate)

            if error < best_error or (
                error == best_error
                and abs(keep_count / cycle_length - target_keep_ratio)
                < abs(best_keep_count / best_cycle_length - target_keep_ratio)
            ):
                best_keep_count = keep_count
                best_cycle_length = cycle_length
                best_error = error

    return best_keep_count, best_cycle_length


def _build_keep_indices(
    frame_count: int, keep_count: int, cycle_length: int
) -> np.ndarray:
    positions = np.arange(frame_count)
    cycle_positions = positions % cycle_length
    cycle_mask = np.ceil((cycle_positions + 1) * keep_count / cycle_length) > np.ceil(
        cycle_positions * keep_count / cycle_length
    )
    return positions[cycle_mask]


def _downsample_echogram_width(
    echogram_image: np.ndarray,
    goal_frame_rate: float,
    original_frame_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if goal_frame_rate <= 0 or original_frame_rate <= goal_frame_rate:
        return (
            echogram_image,
            np.arange(echogram_image.shape[1], dtype=int),
            original_frame_rate,
        )

    keep_count, cycle_length = _choose_frame_keep_ratio(
        original_frame_rate, goal_frame_rate
    )
    keep_indices = _build_keep_indices(
        echogram_image.shape[1], keep_count, cycle_length
    )
    downsampled = echogram_image[:, keep_indices, ...]
    achieved_frame_rate = original_frame_rate * keep_count / cycle_length
    return downsampled, keep_indices, achieved_frame_rate


def _map_frame_coordinate_to_original(coord: float, frame_indices: np.ndarray) -> float:
    if frame_indices.size == 0:
        return float("nan")
    if coord <= 0:
        return float(frame_indices[0])
    if coord >= frame_indices.size - 1:
        return float(frame_indices[-1])
    lo = int(math.floor(coord))
    hi = int(math.ceil(coord))
    if lo == hi:
        return float(frame_indices[lo])
    frac = coord - lo
    return float(frame_indices[lo]) * (1.0 - frac) + float(frame_indices[hi]) * frac


def _resize_height_to_bins(image: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0 or image.shape[0] == bins:
        return image
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((image.shape[1], bins), Image.Resampling.BILINEAR)
    return np.asarray(resized)


def _map_prediction_extents_to_original_frames(
    enter_f: float,
    exit_f: float,
    center_fr: float,
    frame_indices: np.ndarray,
) -> tuple[float, float, float, float]:
    mapped_enter = _map_frame_coordinate_to_original(enter_f, frame_indices)
    mapped_exit = _map_frame_coordinate_to_original(exit_f, frame_indices)
    mapped_center = _map_frame_coordinate_to_original(center_fr, frame_indices)
    mapped_duration = float(int(round(mapped_exit)) - int(round(mapped_enter)) + 1)
    return mapped_enter, mapped_exit, mapped_center, mapped_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an echogram image, run a YOLO segmentation model on it, "
            "and save a side-by-side prediction plot."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(DEFAULT_MODEL),
        help="Path to YOLO segmentation weights.",
    )
    parser.add_argument(
        "--echogram_filepath",
        type=Path,
        required=False,
        help="Path to a single echogram image file, typically PNGs.",
    )
    parser.add_argument(
        "--echogram_dir",
        type=Path,
        required=False,
        help="Path to a directory containing input echogram image files, typically PNGs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory for the prediction images. Defaults to <echogram_dir>/predictions.",
    )
    parser.add_argument(
        "--label_filepath",
        type=Path,
        required=False,
        help="Path to a single label file, typically TXT.",
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing YOLO segmentation label files (.txt). If provided, a third panel "
            "with ground-truth instance overlays is added on the right."
        ),
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional filename to use for the output image.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for YOLO inference.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO inference.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size for YOLO inference.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Mask overlay alpha in the right-hand panel.",
    )
    parser.add_argument(
        "--header-font-size",
        type=int,
        default=18,
        help=(
            "Relative header scale. 18 is the default; larger values make the "
            "title and subtitle rows occupy a larger fraction of the tile height."
        ),
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=1000,
        help=(
            "Final output width in pixels. Images narrower than this are first "
            "upscaled by an integer factor, then resized to this exact width."
        ),
    )
    parser.add_argument(
        "--spacer-width",
        type=float,
        default=0.0,
        help=(
            "Spacer width between columns as a fraction of a single column image width."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional YOLO device override, e.g. cpu or 0.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )

    parser.add_argument(
        "--crop_around_gt",
        action="store_true",
        help="Crop the echogram around the ground truth bounding boxes.",
    )

    parser.add_argument(
        "--crop_ignore_no_cross",
        action="store_true",
        help="Ignore no_cross masks when cropping around the ground truth.",
    )

    parser.add_argument(
        "--filter-submasks",
        action="store_true",
        help="Filter out submasks.",
    )
    parser.add_argument(
        "--no-end2end",
        action="store_true",
        help="Disable YOLO end-to-end inference to match evaluation when needed.",
    )

    return parser.parse_args()


def get_class_name(names: dict[int, str] | list[str] | None, class_id: int) -> str:
    if class_id < 0:
        return "unknown"
    if class_id >= 2:
        return "no-cross"
    if names is None:
        return str(class_id)
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def mask_horizontal_extent_and_centroid(
    mask: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Frame / bin statistics for a mask in image space (x = frame column, y = bin / row).

    Returns ``enter_frame``, ``exit_frame`` (horizontal extent), ``center_frame``
    (rounded mean column with nearest-column fallback), and ``center_frame_bin``
    (mean row on pixels in that column only). Row stats over **all** mask pixels:
    ``minimum_bin_y``, ``maximum_bin_y``, ``average_bin_y``. ``duration`` is the
    inclusive horizontal span in frames, ``floor(enter_frame)`` .. ``floor(exit_frame)``
    as ints: ``exit_x - enter_x + 1``.
    """
    m = np.asarray(mask)
    if np.issubdtype(m.dtype, np.bool_) or m.dtype == bool:
        binary = m.astype(bool, copy=False)
    else:
        binary = m > 0.5
    if not np.any(binary):
        return (float("nan"),) * 10
    ys, xs = np.nonzero(binary)
    left = float(xs.min())
    right = float(xs.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    minimum_bin_y = float(ys.min())
    maximum_bin_y = float(ys.max())
    average_bin_y = float(ys.mean())
    duration = float(xmax - xmin + 1)
    center_x_target = int(np.round(float(xs.mean())))
    cols_with_pixels = set(np.unique(xs).tolist())

    center_x_i = center_x_target
    if center_x_i not in cols_with_pixels:
        max_delta = max(center_x_target - xmin, xmax - center_x_target)
        chosen: int | None = None
        for delta in range(max_delta + 1):
            if delta == 0:
                candidates = [center_x_target]
            else:
                candidates = []
                lo = center_x_target - delta
                hi = center_x_target + delta
                if lo >= xmin:
                    candidates.append(lo)
                if hi <= xmax and hi != lo:
                    candidates.append(hi)
            for col in candidates:
                if col in cols_with_pixels:
                    chosen = col
                    break
            if chosen is not None:
                break
        center_x_i = chosen if chosen is not None else xmin
    slice_y = ys[xs == center_x_i]
    center_frame_bin = float(slice_y.mean())
    start_bin_y = float(ys[xs == min(xs)].mean())
    end_bin_y = float(ys[xs == max(xs)].mean())
    return (
        left,
        right,
        float(center_x_i),
        center_frame_bin,
        minimum_bin_y,
        maximum_bin_y,
        average_bin_y,
        duration,
        start_bin_y,
        end_bin_y,
    )


def prediction_csv_round_int(value: float | int | np.integer) -> int | float:
    """Nearest int for CSV; NaN preserved."""
    x = float(value)
    if math.isnan(x):
        return float("nan")
    return int(round(x))


def prediction_csv_round_decimal(
    value: float | int | np.integer, decimal_places: int = 2
) -> int | float:
    """Nearest dp for CSV; NaN preserved."""
    x = float(value)
    if math.isnan(x):
        return float("nan")
    return round(x, decimal_places)


def prediction_csv_confidence_cell(value: float) -> str:
    """Confidence formatted for CSV (exactly 4 decimal places); NaN as ``nan``."""
    x = float(value)
    if math.isnan(x):
        return "nan"
    return f"{round(x, 4):.4f}"


def interpolate_colour(
    anchors: tuple[tuple[int, int, int], ...],
    position: float,
) -> np.ndarray:
    import numpy as np

    if len(anchors) == 1:
        return np.asarray(anchors[0], dtype=np.float32)

    scaled_position = np.clip(position, 0.0, 1.0) * (len(anchors) - 1)
    left_idx = int(np.floor(scaled_position))
    right_idx = min(left_idx + 1, len(anchors) - 1)
    mix = scaled_position - left_idx

    left = np.asarray(anchors[left_idx], dtype=np.float32)
    right = np.asarray(anchors[right_idx], dtype=np.float32)
    return left * (1.0 - mix) + right * mix


def build_instance_colours(class_id: int, count: int) -> list[np.ndarray]:
    import numpy as np

    anchors = CLASS_COLOUR_ANCHORS.get(class_id, CLASS_COLOUR_ANCHORS[2])
    if count <= 0:
        return []
    if count == 1:
        return [interpolate_colour(anchors, 0.5)]

    positions = np.linspace(0.0, 1.0, count)
    return [interpolate_colour(anchors, position) for position in positions]


def clamp_class_id(class_id: int) -> int:
    return class_id if class_id in {0, 1} else 2


def get_class_counts(class_ids: list[int] | np.ndarray) -> dict[int, int]:
    counts = {class_id: 0 for class_id in CLASS_COUNT_LABELS}
    for class_id in class_ids:
        counts[clamp_class_id(int(class_id))] += 1
    return counts


def format_class_count_summary(class_ids: list[int] | np.ndarray) -> str:
    counts = get_class_counts(class_ids)
    return " | ".join(
        f"{CLASS_COUNT_LABELS[class_id]}: {counts[class_id]}"
        for class_id in sorted(CLASS_COUNT_LABELS)
    )


def get_class_header_colour(class_id: int) -> tuple[int, int, int]:
    colour = build_instance_colours(class_id, 1)[0]
    return tuple(int(channel) for channel in colour)


def get_pil_resample(name: str) -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def load_header_font(font_size: int, bold: bool = False):
    from PIL import ImageFont

    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", max(1, font_size))
        return ImageFont.truetype("DejaVuSans.ttf", max(1, font_size))
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", max(1, font_size))
        except OSError:
            return ImageFont.load_default()


def measure_text_layout(
    text: str,
    font_size: int,
    *,
    bold: bool = False,
) -> dict[str, int]:
    from PIL import Image, ImageDraw

    font = load_header_font(font_size, bold=bold)
    probe_image = Image.new("RGB", (1, 1), color=(20, 20, 20))
    probe_draw = ImageDraw.Draw(probe_image)
    text_bbox = probe_draw.textbbox((0, 0), text, font=font)
    return {
        "width": text_bbox[2] - text_bbox[0],
        "height": text_bbox[3] - text_bbox[1],
        "top_offset": text_bbox[1],
    }


def measure_subtitle_layout(
    subtitle_chunks: list[tuple[str, tuple[int, int, int]]] | None,
    font_size: int,
) -> dict[str, int | list[int]]:
    from PIL import Image, ImageDraw

    if not subtitle_chunks:
        return {"width": 0, "height": 0, "top_offset": 0, "chunk_widths": []}

    subtitle_font = load_header_font(font_size)
    probe_image = Image.new("RGB", (1, 1), color=(20, 20, 20))
    probe_draw = ImageDraw.Draw(probe_image)
    subtitle_bboxes = [
        probe_draw.textbbox((0, 0), chunk_text, font=subtitle_font)
        for chunk_text, _ in subtitle_chunks
    ]
    subtitle_widths = [bbox[2] - bbox[0] for bbox in subtitle_bboxes]
    return {
        "width": sum(subtitle_widths),
        "height": max((bbox[3] - bbox[1]) for bbox in subtitle_bboxes),
        "top_offset": min(bbox[1] for bbox in subtitle_bboxes),
        "chunk_widths": subtitle_widths,
    }


def get_header_scale(font_size: int) -> float:
    return max(0.1, font_size / DEFAULT_HEADER_FONT_SIZE)


def get_header_row_height(
    image_height: int,
    row_fraction: float,
    total_fraction: float,
) -> int:
    if image_height <= 0:
        return 1
    body_fraction = max(1e-6, 1.0 - total_fraction)
    return max(1, int(round(image_height * row_fraction / body_fraction)))


def find_fitting_font_size(
    *,
    max_width: int,
    max_height: int,
    measure_layout,
) -> int:
    low = 1
    high = max(1, max_height * 2)
    best_size = 1
    while low <= high:
        mid = (low + high) // 2
        layout = measure_layout(mid)
        if int(layout["width"]) <= max_width and int(layout["height"]) <= max_height:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1
    return best_size


def get_title_row_spec(
    image_width: int,
    image_height: int,
    font_size: int,
    title_texts: list[str],
) -> dict[str, int | float]:
    supersample_scale = HEADER_SUPERSAMPLE_SCALE
    header_scale = get_header_scale(font_size)
    title_fraction = HEADER_TITLE_HEIGHT_FRACTION * header_scale
    subtitle_fraction = HEADER_SUBTITLE_HEIGHT_FRACTION * header_scale
    total_fraction = min(0.95, title_fraction + subtitle_fraction)
    row_height = get_header_row_height(
        image_height=image_height,
        row_fraction=min(title_fraction, max(1e-6, total_fraction - 1e-6)),
        total_fraction=total_fraction,
    )
    render_row_width = max(1, image_width * supersample_scale)
    render_row_height = max(1, row_height * supersample_scale)
    horizontal_padding = max(
        1,
        int(round(render_row_height * HEADER_HORIZONTAL_PADDING_FRACTION)),
    )
    vertical_padding = max(
        1,
        int(round(render_row_height * HEADER_VERTICAL_PADDING_FRACTION)),
    )
    max_text_width = max(1, render_row_width - 2 * horizontal_padding)
    max_text_height = max(1, render_row_height - 2 * vertical_padding)
    render_font_size = find_fitting_font_size(
        max_width=max_text_width,
        max_height=max_text_height,
        measure_layout=lambda candidate_size: {
            "width": max(
                int(
                    measure_text_layout(
                        title_text,
                        candidate_size,
                        bold=True,
                    )["width"]
                )
                for title_text in title_texts
            ),
            "height": max(
                int(
                    measure_text_layout(
                        title_text,
                        candidate_size,
                        bold=True,
                    )["height"]
                )
                for title_text in title_texts
            ),
        },
    )
    return {
        "render_font_size": render_font_size,
        "render_row_width": render_row_width,
        "render_row_height": render_row_height,
        "row_height": row_height,
    }


def get_subtitle_row_spec(
    image_width: int,
    image_height: int,
    font_size: int,
    subtitle_rows: list[list[tuple[str, tuple[int, int, int]]] | None],
) -> dict[str, int | float]:
    supersample_scale = HEADER_SUPERSAMPLE_SCALE
    header_scale = get_header_scale(font_size)
    title_fraction = HEADER_TITLE_HEIGHT_FRACTION * header_scale
    subtitle_fraction = HEADER_SUBTITLE_HEIGHT_FRACTION * header_scale
    total_fraction = min(0.95, title_fraction + subtitle_fraction)
    row_height = get_header_row_height(
        image_height=image_height,
        row_fraction=min(subtitle_fraction, max(1e-6, total_fraction - 1e-6)),
        total_fraction=total_fraction,
    )
    render_row_width = max(1, image_width * supersample_scale)
    render_row_height = max(1, row_height * supersample_scale)
    horizontal_padding = max(
        1,
        int(round(render_row_height * HEADER_HORIZONTAL_PADDING_FRACTION)),
    )
    vertical_padding = max(
        1,
        int(round(render_row_height * HEADER_VERTICAL_PADDING_FRACTION)),
    )
    max_text_width = max(1, render_row_width - 2 * horizontal_padding)
    max_text_height = max(1, render_row_height - 2 * vertical_padding)
    render_font_size = find_fitting_font_size(
        max_width=max_text_width,
        max_height=max_text_height,
        measure_layout=lambda candidate_size: {
            "width": max(
                int(measure_subtitle_layout(subtitle_chunks, candidate_size)["width"])
                for subtitle_chunks in subtitle_rows
            ),
            "height": max(
                int(measure_subtitle_layout(subtitle_chunks, candidate_size)["height"])
                for subtitle_chunks in subtitle_rows
            ),
        },
    )
    return {
        "render_font_size": render_font_size,
        "render_row_width": render_row_width,
        "render_row_height": render_row_height,
        "row_height": row_height,
    }


def upscale_image_integer(image: np.ndarray | None, scale: int) -> np.ndarray | None:
    if image is None or scale <= 1:
        return image
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def render_title_row(
    image_width: int,
    title_text: str,
    title_row_spec: dict[str, int | float],
    bg_colour: tuple[int, int, int],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    title_layout = measure_text_layout(
        title_text,
        int(title_row_spec["render_font_size"]),
        bold=True,
    )
    title_font = load_header_font(int(title_row_spec["render_font_size"]), bold=True)
    row_image = Image.new(
        "RGB",
        (
            int(title_row_spec["render_row_width"]),
            int(title_row_spec["render_row_height"]),
        ),
        color=bg_colour,
    )
    draw = ImageDraw.Draw(row_image)
    title_x = max(0, (row_image.width - int(title_layout["width"])) // 2)
    title_y = max(0, (row_image.height - int(title_layout["height"])) // 2) - int(
        title_layout["top_offset"]
    )
    draw.text((title_x, title_y), title_text, fill=(245, 245, 245), font=title_font)
    row_image = row_image.resize(
        (image_width, int(title_row_spec["row_height"])),
        resample=get_pil_resample("LANCZOS"),
    )
    return np.asarray(row_image, dtype=np.uint8)


def render_subtitle_row(
    image_width: int,
    subtitle_chunks: list[tuple[str, tuple[int, int, int]]] | None,
    subtitle_row_spec: dict[str, int | float],
    bg_colour: tuple[int, int, int],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    subtitle_layout = measure_subtitle_layout(
        subtitle_chunks,
        int(subtitle_row_spec["render_font_size"]),
    )
    subtitle_font = load_header_font(int(subtitle_row_spec["render_font_size"]))
    row_image = Image.new(
        "RGB",
        (
            int(subtitle_row_spec["render_row_width"]),
            int(subtitle_row_spec["render_row_height"]),
        ),
        color=bg_colour,
    )
    if subtitle_chunks:
        draw = ImageDraw.Draw(row_image)
        subtitle_x = max(0, (row_image.width - int(subtitle_layout["width"])) // 2)
        subtitle_y = max(
            0, (row_image.height - int(subtitle_layout["height"])) // 2
        ) - int(subtitle_layout["top_offset"])
        current_x = subtitle_x
        for (chunk_text, chunk_colour), chunk_width in zip(
            subtitle_chunks, subtitle_layout["chunk_widths"]
        ):
            draw.text(
                (current_x, subtitle_y),
                chunk_text,
                fill=chunk_colour,
                font=subtitle_font,
            )
            current_x += int(chunk_width)
    row_image = row_image.resize(
        (image_width, int(subtitle_row_spec["row_height"])),
        resample=get_pil_resample("LANCZOS"),
    )
    return np.asarray(row_image, dtype=np.uint8)


def add_tile_header(
    image: np.ndarray,
    title_text: str,
    font_size: int,
    subtitle_chunks: list[tuple[str, tuple[int, int, int]]] | None = None,
    title_row_spec: dict[str, int | float] | None = None,
    subtitle_row_spec: dict[str, int | float] | None = None,
    bg_colour: tuple[int, int, int] | None = None,
) -> np.ndarray:
    header_bg_colour = bg_colour if bg_colour is not None else (20, 20, 20)
    if title_row_spec is None:
        title_row_spec = get_title_row_spec(
            image_width=image.shape[1],
            image_height=image.shape[0],
            font_size=font_size,
            title_texts=[title_text],
        )
    if subtitle_row_spec is None:
        subtitle_row_spec = get_subtitle_row_spec(
            image_width=image.shape[1],
            image_height=image.shape[0],
            font_size=font_size,
            subtitle_rows=[subtitle_chunks],
        )

    title_row = render_title_row(
        image_width=image.shape[1],
        title_text=title_text,
        title_row_spec=title_row_spec,
        bg_colour=header_bg_colour,
    )
    subtitle_row = render_subtitle_row(
        image_width=image.shape[1],
        subtitle_chunks=subtitle_chunks,
        subtitle_row_spec=subtitle_row_spec,
        bg_colour=header_bg_colour,
    )
    header_image = np.concatenate([title_row, subtitle_row], axis=0)
    image_rgb = image.astype(np.uint8)
    return np.concatenate([header_image, image_rgb], axis=0)


def resize_combined_image_to_width(
    image: np.ndarray,
    target_width: int,
) -> np.ndarray:
    if target_width <= 0:
        return image

    current_height, current_width = image.shape[:2]
    if current_width == target_width:
        return image

    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")

    if current_width < target_width:
        integer_scale = max(1, (target_width + current_width - 1) // current_width)
        if integer_scale > 1:
            pil_image = pil_image.resize(
                (current_width * integer_scale, current_height * integer_scale),
                resample=get_pil_resample("NEAREST"),
            )
    resized_height = max(1, round(pil_image.height * target_width / pil_image.width))
    pil_image = pil_image.resize(
        (target_width, resized_height),
        resample=get_pil_resample("LANCZOS"),
    )
    return np.asarray(pil_image, dtype=np.uint8)


def load_yolo_segmentation_masks(
    label_path: Path,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[list[int], list[np.ndarray]]:
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = image_shape[:2]
    class_ids: list[int] = []
    masks: list[np.ndarray] = []

    with open(label_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            try:
                class_id = clamp_class_id(int(float(parts[0])))
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                print(f"Skipping invalid label row {line_number} in {label_path}")
                continue

            if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                print(f"Skipping malformed polygon row {line_number} in {label_path}")
                continue

            polygon = []
            for x_norm, y_norm in zip(coordinates[0::2], coordinates[1::2]):
                x = min(max(x_norm, 0.0), 1.0) * (width - 1)
                y = min(max(y_norm, 0.0), 1.0) * (height - 1)
                polygon.append((x, y))

            if len(polygon) < 3:
                continue

            mask_image = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask_image).polygon(polygon, outline=255, fill=255)
            masks.append(np.asarray(mask_image, dtype=np.uint8) / 255.0)
            class_ids.append(class_id)

    return class_ids, masks


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    import numpy as np

    if mask_a.shape != mask_b.shape:
        target_height = max(mask_a.shape[0], mask_b.shape[0])
        target_width = max(mask_a.shape[1], mask_b.shape[1])
        mask_a = resize_mask(mask_a, target_height, target_width)
        mask_b = resize_mask(mask_b, target_height, target_width)

    mask_a_bool = mask_a > 0.5
    mask_b_bool = mask_b > 0.5
    intersection = np.logical_and(mask_a_bool, mask_b_bool).sum()
    union = np.logical_or(mask_a_bool, mask_b_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def assign_matched_instance_colours(
    pred_class_ids: list[int],
    pred_masks: list[np.ndarray] | np.ndarray,
    gt_class_ids: list[int],
    gt_masks: list[np.ndarray] | np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    pred_class_ids = [clamp_class_id(int(class_id)) for class_id in pred_class_ids]
    gt_class_ids = [clamp_class_id(int(class_id)) for class_id in gt_class_ids]

    pred_colours: list[np.ndarray | None] = [None] * len(pred_class_ids)
    gt_colours: list[np.ndarray | None] = [None] * len(gt_class_ids)
    matched_count = 0

    all_class_ids = sorted(set(pred_class_ids) | set(gt_class_ids))
    for class_id in all_class_ids:
        pred_indices = [
            idx
            for idx, predicted_class_id in enumerate(pred_class_ids)
            if predicted_class_id == class_id
        ]
        gt_indices = [
            idx
            for idx, ground_truth_class_id in enumerate(gt_class_ids)
            if ground_truth_class_id == class_id
        ]

        candidate_pairs: list[tuple[float, int, int]] = []
        for pred_idx in pred_indices:
            for gt_idx in gt_indices:
                iou = compute_mask_iou(pred_masks[pred_idx], gt_masks[gt_idx])
                if iou > 0.0:
                    candidate_pairs.append((iou, pred_idx, gt_idx))

        candidate_pairs.sort(reverse=True)
        matched_pred_indices: set[int] = set()
        matched_gt_indices: set[int] = set()
        matched_pairs: list[tuple[int, int]] = []

        for iou, pred_idx, gt_idx in candidate_pairs:
            if pred_idx in matched_pred_indices or gt_idx in matched_gt_indices:
                continue
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)
            matched_pairs.append((pred_idx, gt_idx))

        matched_pairs.sort()
        unmatched_pred_indices = sorted(
            idx for idx in pred_indices if idx not in matched_pred_indices
        )
        unmatched_gt_indices = sorted(
            idx for idx in gt_indices if idx not in matched_gt_indices
        )

        class_colour_count = (
            len(matched_pairs) + len(unmatched_pred_indices) + len(unmatched_gt_indices)
        )
        class_colours = build_instance_colours(class_id, class_colour_count)

        colour_offset = 0
        for pred_idx, gt_idx in matched_pairs:
            colour = class_colours[colour_offset]
            pred_colours[pred_idx] = colour
            gt_colours[gt_idx] = colour
            colour_offset += 1
            matched_count += 1

        for pred_idx in unmatched_pred_indices:
            pred_colours[pred_idx] = class_colours[colour_offset]
            colour_offset += 1

        for gt_idx in unmatched_gt_indices:
            gt_colours[gt_idx] = class_colours[colour_offset]
            colour_offset += 1

    return pred_colours, gt_colours, matched_count


def overlay_instance_masks(
    base_image: np.ndarray,
    class_ids: list[int] | np.ndarray,
    mask_data: list[np.ndarray] | np.ndarray,
    alpha: float,
    class_names: dict[int, str] | list[str] | None = None,
    instance_colours: list[np.ndarray] | None = None,
    given_bgr: bool = False,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    import numpy as np

    if given_bgr:
        greyscale_image = base_image[:, :, 2].astype(np.float32).copy()
    else:
        greyscale_image = base_image[:, :, 0].astype(np.float32).copy()
    overlay = np.stack([greyscale_image, greyscale_image, greyscale_image], axis=-1)
    height, width = overlay.shape[:2]
    legend_items: list[tuple[str, np.ndarray]] = []

    if len(class_ids) == 0 or len(mask_data) == 0:
        return overlay.astype(np.uint8), legend_items

    class_ids = [clamp_class_id(int(class_id)) for class_id in class_ids]
    unique_class_ids = sorted(set(class_ids))
    colour_by_instance: list[np.ndarray] | None = None
    colour_by_class: dict[int, list[np.ndarray]] = {}

    if instance_colours is not None:
        if len(instance_colours) != len(class_ids):
            raise ValueError("instance_colours must match number of masks")
        colour_by_instance = instance_colours

    if colour_by_instance is None:
        for class_id in unique_class_ids:
            class_count = sum(
                predicted_class_id == class_id for predicted_class_id in class_ids
            )
            colour_by_class[class_id] = build_instance_colours(class_id, class_count)

    for class_id in unique_class_ids:
        class_name = (
            get_class_name(class_names, class_id)
            if class_names is not None
            else str(class_id)
        )
        # print(f"class_id: {class_id}, class_name: {class_name}")
        color = build_instance_colours(class_id, 1)[0]
        legend_items.append((f"{class_id}: {class_name}", color))

    class_offsets = {class_id: 0 for class_id in unique_class_ids}
    for idx, class_id in enumerate(class_ids):
        mask = resize_mask(mask_data[idx], height=height, width=width) > 0.5
        if colour_by_instance is not None:
            color = colour_by_instance[idx]
        else:
            color = colour_by_class[class_id][class_offsets[class_id]]
            class_offsets[class_id] += 1
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8), legend_items


def overlay_predictions(
    base_image: np.ndarray,
    result_masks: list[np.ndarray],
    result_class_names: list[int],
    result_class_ids: list[int],
    alpha: float,
    instance_colours: list[np.ndarray] | None = None,
    given_bgr: bool = False,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    if result_masks is None or result_class_names is None or len(result_masks) == 0:
        return overlay_instance_masks(
            base_image=base_image,
            class_ids=[],
            mask_data=[],
            alpha=alpha,
            class_names=result_class_names,
            instance_colours=instance_colours,
            given_bgr=given_bgr,
        )

    # class_ids = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    # mask_data = get_prediction_masks_in_image_space(result)
    mask_data = result_masks
    return overlay_instance_masks(
        base_image=base_image,
        class_ids=result_class_ids,
        mask_data=mask_data,
        alpha=alpha,
        class_names=result_class_names,
        instance_colours=instance_colours,
        given_bgr=given_bgr,
    )


def convert_bin_to_m(bin_value: float, metadata: dict | None) -> float:
    """Convert an echogram row/bin index to range in meters using file metadata."""
    if metadata is None:
        return float("nan")
    x = float(bin_value)
    if math.isnan(x):
        return float("nan")

    if "windowstart" in metadata and "windowlength" in metadata:
        num_bins = metadata.get("samplesperbeam") or metadata.get("samplesperchannel")
        if not num_bins:
            return float("nan")
        bin_size = float(metadata["windowlength"]) / float(num_bins)
        return (
            float(metadata["windowlength"])
            - x * bin_size
            + float(metadata["windowstart"])
        )

    sample_length = metadata.get("sample_length")
    if sample_length is not None:
        win_start = metadata.get("WinStart")
        if win_start is None:
            delay = metadata.get("samplestartdelay")
            speed = metadata.get("soundspeed")
            if delay is not None and speed is not None:
                win_start = float(delay) * 1e-6 * float(speed) / 2.0
        if win_start is not None:
            return float(win_start) + x * float(sample_length)

    return float("nan")


def get_and_plot_echogram_predictions(
    model_path: Path | None = None,
    model: YOLO | None = None,
    echogram_image: np.ndarray | None = None,
    echogram_metadata: dict | None = None,
    echogram_filepath: Path | None = None,
    echogram_dir: Path | None = None,
    filename: str | None = None,
    aris_stem: str | None = None,
    aris_path: Path | None = None,
    label_filepath: Path | None = None,
    label_dir: Path | None = None,
    output_dir: Path | None = None,
    imgsz: int | float = 1280,
    infer_bins: int = -1,
    infer_fps: float = -1.0,
    filter_submasks: bool = False,
    conf: float = 0.1,
    iou: float = 0.5,
    device: str | None = None,
    half: bool = False,
    no_end2end: bool = False,
    save_png: bool = True,
    save_csv: bool = False,
    save_fc: bool = False,
    save_echotastic: bool = False,
    upstream_direction: str = "left",
    mask_alpha: float = 0.75,
    header_font_size: int = 18,
    crop_around_gt: bool = False,
    crop_around_pred: bool = False,
    crop_ignore_no_cross: bool = False,
    show: bool = False,
    horizontal_stretch: int = 1,
    min_width: int = None,
    spacer_width: float = 0,
    spacer_colour: tuple[int, int, int] = (20, 20, 20),
    header_bg_colour: tuple[int, int, int] = (20, 20, 20),
    make_gt_random_colours: bool = False,
    include_input_echogram_in_png: bool = True,
    include_summary_bar: bool = True,
) -> dict[int, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if filename is None:
        filename = echogram_filepath.stem
        print(f"Using filename: {filename} from echogram_filepath")

    # --- Helper: load display image from various inputs ---
    def _load_display_image(echogram_image, echogram_filepath, echogram_dir, filename):
        if echogram_image is not None:
            return echogram_image, filename

        if echogram_filepath is not None:
            if echogram_dir is not None:
                raise ValueError(
                    "Cannot provide both echogram_filepath and echogram_dir"
                )
            echogram_path = echogram_filepath
        elif echogram_dir is not None:
            if filename is not None:
                echogram_path = echogram_dir / f"{filename}.png"
            else:
                raise ValueError("Must provide filename when providing echogram_dir")
        else:
            raise ValueError("Must provide either echogram_filepath or echogram_dir")

        if not echogram_path.exists():
            raise ValueError(f"Echogram path does not exist: {echogram_path}")
        img = np.asarray(Image.open(echogram_path).convert("RGB"))
        return img, filename

    display_image, filename = _load_display_image(
        echogram_image, echogram_filepath, echogram_dir, filename
    )

    # --- Helper: ensure model is loaded or return None ---
    def _load_model_if_needed(model, model_path):
        if model is None:
            if model_path is None:
                print("No model provided")
                return None
            else:
                print(f"Loading model: {model_path}")
                return YOLO(str(model_path))
        # pre-loaded model takes precedence
        return model

    model = _load_model_if_needed(model, model_path)

    # --- Helper: prepare image for inference (downsample/resize) ---
    def _prepare_inference_image(
        display_image, echogram_metadata, infer_fps, infer_bins, imgsz
    ):
        frame_indices = None
        post_processed_image = display_image
        achieved_fps = None
        if infer_fps > 0.0:
            original_frame_rate = _framerate_from_metadata(echogram_metadata)
            if original_frame_rate is not None:
                post_processed_image, frame_indices, achieved_fps = (
                    _downsample_echogram_width(
                        display_image, infer_fps, original_frame_rate
                    )
                )
                print(
                    f"Downsampled echogram width from {display_image.shape[1]} to {post_processed_image.shape[1]} columns (target {infer_fps} fps, achieved {achieved_fps:.3f} fps)"
                )
            else:
                post_processed_image = display_image
                frame_indices = np.arange(display_image.shape[1], dtype=int)
                print(
                    "Inference FPS specified but unable to read original frame rate from metadata; using native width"
                )
        else:
            frame_indices = np.arange(display_image.shape[1], dtype=int)

        if infer_bins > 0:
            post_processed_image = _resize_height_to_bins(
                post_processed_image, infer_bins
            )

        effective_imgsz = -1 if infer_bins > 0 or infer_fps > 0.0 else imgsz
        return post_processed_image, frame_indices, effective_imgsz, achieved_fps

    post_processed_image, frame_indices, effective_imgsz, achieved_fps = (
        _prepare_inference_image(
            display_image, echogram_metadata, infer_fps, infer_bins, imgsz
        )
    )

    result = None
    if model is not None:
        predict_source = post_processed_image
        result = predict_segmentation(
            model=model,
            source=predict_source,
            conf=conf,
            iou=iou,
            imgsz=effective_imgsz,
            device=device,
            end2end=not no_end2end,
            half=half,
        )
        if (
            result.masks is None
            or len(result.masks) == 0
            or result.boxes is None
            or len(result.boxes) == 0
        ):
            print(f"No boxes or masks predicted {result.masks=} {result.boxes=}")

        elif (
            result.masks is not None
            and len(result.masks) > 0
            and result.boxes is not None
            and len(result.boxes) > 0
        ):
            if len(result.masks) != len(result.boxes):
                print(
                    f"Number of masks {len(result.masks)} does not match number of boxes {len(result.boxes)}"
                )
            else:
                print(
                    f"Predicted {len(result.masks)} masks and {len(result.boxes)} boxes"
                )
        else:
            print("No boxes predicted")
        if result.names is not None and len(result.names) > 0:
            pass
            # print(
            #     f"Predicted {len(result.names)} names: {list(result.names.values())}"
            # )
        else:
            print("No names predicted")
    else:
        print("No model provided")
        result = None

    def _postprocess_predictions(result, filter_submasks):
        pred_class_ids: list[int] = []
        pred_masks: list[np.ndarray] = []
        pred_confidences_list: list[float] = []
        bool_filtered = None
        if result is not None:
            if result.masks is not None and len(result.masks) > 0:
                pred_masks, kept_indices = get_prediction_masks_in_image_space(
                    result, return_indices=True
                )
                pred_class_ids = get_prediction_class_ids(
                    result, kept_indices=kept_indices, fill_value=-1
                )
                pred_confidences_list = get_prediction_confidences(
                    result, kept_indices=kept_indices
                )
                if any(class_id < 0 for class_id in pred_class_ids):
                    print(
                        "Some predicted masks did not have class IDs; plotting them as unknown/no-cross"
                    )
                if filter_submasks:
                    print(f"{filter_submasks=}")
                    pre_len = len(pred_masks)

                    pred_masks, pred_class_ids, bool_filtered = filter_out_submasks(
                        pred_masks,
                        pred_class_ids,
                        pred_confidences_list,
                        remove_super=False,
                    )
                    pred_confidences_list = [
                        pred_confidences_list[i]
                        for i, kept in enumerate(bool_filtered)
                        if kept
                    ]
                    if pre_len > len(pred_masks):
                        print(
                            f"Filtered masks: {pre_len} -> {len(pred_masks)} [{pred_class_ids}]"
                        )
                else:
                    print(f"{filter_submasks=} False")
        return pred_masks, pred_class_ids, pred_confidences_list, bool_filtered

    pred_masks, pred_class_ids, pred_confidences_list, bool_filtered = (
        _postprocess_predictions(result, filter_submasks)
    )

    def _export_predictions_if_requested(
        save_csv,
        save_fc,
        save_echotastic,
        output_dir,
        filename,
        pred_masks,
        pred_class_ids,
        pred_confidences_list,
        result,
        echogram_metadata,
        frame_indices,
        aris_stem,
        aris_path,
        upstream_direction,
    ):
        if not (save_csv or save_fc or save_echotastic):
            return None, None, None
        from analysis.prediction_exports import (
            export_predictions_csv,
            export_predictions_echotastic,
            export_predictions_fc,
        )

        class_names = result.names if result is not None else None
        csv_path = None
        echotastic_path = None
        if save_csv:
            csv_path = export_predictions_csv(
                output_dir=output_dir,
                filename=filename,
                pred_masks=pred_masks,
                pred_class_ids=pred_class_ids,
                pred_confidences_list=pred_confidences_list,
                class_names=class_names,
                echogram_metadata=echogram_metadata,
                frame_indices=frame_indices,
            )
            print(f"Saved predictions CSV to: {csv_path}")
        if save_fc:
            fc_stem = aris_stem or filename
            if not fc_stem:
                raise ValueError("aris_stem or filename required for FC export")
            export_predictions_fc(
                output_dir=output_dir,
                aris_stem=fc_stem,
                pred_masks=pred_masks,
                pred_class_ids=pred_class_ids,
                pred_confidences_list=pred_confidences_list,
                class_names=class_names,
                echogram_metadata=echogram_metadata,
                frame_indices=frame_indices,
                upstream_direction=upstream_direction,
                date_source=filename or fc_stem,
            )
        if save_echotastic:
            echotastic_stem = aris_stem or filename
            if not echotastic_stem:
                raise ValueError("aris_stem or filename required for Echotastic export")
            echotastic_path = export_predictions_echotastic(
                output_dir=output_dir,
                aris_stem=echotastic_stem,
                pred_masks=pred_masks,
                pred_class_ids=pred_class_ids,
                pred_confidences_list=pred_confidences_list,
                class_names=class_names,
                echogram_metadata=echogram_metadata,
                frame_indices=frame_indices,
                upstream_direction=upstream_direction,
                aris_path=aris_path,
            )
            print(f"Saved Echotastic export to: {echotastic_path}")
        return csv_path, echotastic_path, None

    csv_path, echotastic_path, _ = _export_predictions_if_requested(
        save_csv,
        save_fc,
        save_echotastic,
        output_dir,
        filename,
        pred_masks,
        pred_class_ids,
        pred_confidences_list,
        result,
        echogram_metadata,
        frame_indices,
        aris_stem,
        aris_path,
        upstream_direction,
    )

    # make the image pretty
    display_image_visual = display_image.copy().astype(np.float32)

    display_image_visual[:, :, 1] = display_image_visual[:, :, 1] / 255.0 - 0.5
    display_image_visual = make_echogram_image(
        display_image_visual,
        colour_power=2,
        colour_intensity_cutoff=0.0,
        colour_mask_power=2,
        do_bgs=False,
        given_bgr=echogram_image is not None,
    )

    display_image_visual = 0.5 + 1.5 * (display_image_visual - 0.5)
    display_image_visual = np.clip(display_image_visual, 0, 1)

    gt_class_ids: list[int] = []
    gt_masks: list[np.ndarray] = []
    pred_instance_colours = None
    gt_instance_colours = None

    if label_filepath is not None:
        if label_dir is not None:
            raise ValueError("Cannot provide both label_filepath and label_dir")
        label_path = label_filepath
    elif label_dir is not None:
        if filename is not None:
            label_path = label_dir / f"{filename}.txt"
        else:
            raise ValueError("Must provide filename when providing label_dir")
    else:
        label_path = None

    if label_path is not None:
        gt_class_ids, gt_masks = load_yolo_segmentation_masks(
            label_path,
            display_image.shape,
        )
        if result is not None:
            pred_instance_colours, gt_instance_colours, matched_pairs = (
                assign_matched_instance_colours(
                    pred_class_ids=pred_class_ids,
                    pred_masks=pred_masks,
                    gt_class_ids=gt_class_ids,
                    gt_masks=gt_masks,
                )
            )
            print(
                f"Matched {matched_pairs} prediction/ground-truth pairs by class and overlap"
            )

    if result is not None:
        overlay_image, _ = overlay_predictions(
            base_image=display_image,
            result_masks=pred_masks,
            result_class_ids=pred_class_ids,
            result_class_names=result.names if result is not None else None,
            alpha=mask_alpha,
            instance_colours=pred_instance_colours,
            given_bgr=echogram_image is not None,
        )
    else:
        overlay_image = np.zeros_like(display_image)

    ground_truth_overlay = None
    if label_path is not None:
        if make_gt_random_colours and len(gt_masks) > 0:
            gt_colourmap = plt.get_cmap("tab20", 20)

            # Exclude tab20 blue/orange entries:
            # 0,1 = blue shades
            # 2,3 = orange shades
            allowed_indices = [i for i in range(20) if i not in {0, 1, 2, 3}]

            rng = np.random.default_rng()
            chosen_indices = rng.choice(
                allowed_indices, size=len(gt_masks), replace=True
            )

            gt_instance_colours = [
                np.array(
                    [channel * 255 for channel in gt_colourmap(colour_index)[:3]],
                    dtype=np.float32,
                )
                for colour_index in chosen_indices
            ]

        ground_truth_overlay, _ = overlay_instance_masks(
            base_image=display_image,
            class_ids=gt_class_ids,
            mask_data=gt_masks,
            alpha=mask_alpha,
            class_names=result.names if result is not None else None,
            instance_colours=gt_instance_colours,
        )
        print(f"Loaded {len(gt_masks)} ground-truth instances from: {label_path}")

    pred_counts = get_class_counts(pred_class_ids)
    pred_subtitle_chunks = [
        (f"left: {pred_counts[0]}", get_class_header_colour(0)),
        (", ", (245, 245, 245)),
        (f"right: {pred_counts[1]}", get_class_header_colour(1)),
        (", ", (245, 245, 245)),
        (f"no cross: {pred_counts[2]}", get_class_header_colour(2)),
    ]
    gt_subtitle_chunks = None
    if ground_truth_overlay is not None:
        gt_counts = get_class_counts(gt_class_ids)
        gt_subtitle_chunks = [
            (f"left: {gt_counts[0]}", get_class_header_colour(0)),
            (", ", (245, 245, 245)),
            (f"right: {gt_counts[1]}", get_class_header_colour(1)),
            (", ", (245, 245, 245)),
            (f"no cross: {gt_counts[2]}", get_class_header_colour(2)),
        ]

    sorted_pred_class_ids = sorted(pred_class_ids)
    sorted_gt_class_ids = sorted(gt_class_ids)
    if sorted_pred_class_ids == sorted_gt_class_ids:
        print(f"pred_class_ids = gt_class_ids {pred_class_ids=} {gt_class_ids=}")
    else:
        print(f"pred_class_ids:{sorted_pred_class_ids}")
        print(f"gt_class_ids  :{sorted_gt_class_ids}")

    if crop_around_pred or crop_around_gt:
        crop_bounds = None

        if crop_around_gt and ground_truth_overlay is not None and len(gt_masks) > 0:
            gt_masks_for_crop = gt_masks

            if crop_ignore_no_cross:
                gt_masks_for_crop = [
                    mask
                    for class_id, mask in zip(gt_class_ids, gt_masks)
                    if class_id < 2
                ]
                print(
                    f"Filtered out no_cross GT masks for crop; remaining={len(gt_masks_for_crop)}"
                )

            if len(gt_masks_for_crop) > 0:
                gt_masks_np = np.stack(gt_masks_for_crop, axis=0)
                gt_masks_np_sum = np.sum(gt_masks_np, axis=0)
                non_zero_indices = np.where(gt_masks_np_sum != 0)

                if len(non_zero_indices[0]) > 0:
                    xmin = np.min(non_zero_indices[1])
                    xmax = np.max(non_zero_indices[1])
                    ymin = np.min(non_zero_indices[0])
                    ymax = np.max(non_zero_indices[0])

                    width = xmax - xmin
                    height = ymax - ymin
                    pad_amount = 0.1

                    xmin = max(0, int(xmin - pad_amount * width))
                    xmax = min(
                        display_image_visual.shape[1], int(xmax + pad_amount * width)
                    )
                    ymin = max(0, int(ymin - pad_amount * height))
                    ymax = min(
                        display_image_visual.shape[0], int(ymax + pad_amount * height)
                    )

                    crop_bounds = (xmin, xmax, ymin, ymax)
                    print(f" crop_around_gt {xmin=}, {xmax=}, {ymin=}, {ymax=}")

        if crop_around_pred:
            pred_masks_for_crop = pred_masks
            pred_class_ids_for_crop = pred_class_ids

            if crop_ignore_no_cross:
                pred_masks_for_crop = [
                    mask
                    for class_id, mask in zip(
                        pred_class_ids_for_crop, pred_masks_for_crop
                    )
                    if class_id < 2
                ]
                print(
                    f"Filtered out no_cross prediction masks for crop; remaining={len(pred_masks_for_crop)}"
                )

            if len(pred_masks_for_crop) > 0:
                pred_masks_np = np.stack(pred_masks_for_crop, axis=0)
                pred_masks_np_sum = np.sum(pred_masks_np, axis=0)
                non_zero_indices = np.where(pred_masks_np_sum != 0)

                if len(non_zero_indices[0]) > 0:
                    x_min_pred = np.min(non_zero_indices[1])
                    x_max_pred = np.max(non_zero_indices[1])
                    y_min_pred = np.min(non_zero_indices[0])
                    y_max_pred = np.max(non_zero_indices[0])

                    width_pred = x_max_pred - x_min_pred
                    height_pred = y_max_pred - y_min_pred
                    pad_amount = 0.1

                    x_min_pred = max(0, int(x_min_pred - pad_amount * width_pred))
                    x_max_pred = min(
                        display_image_visual.shape[1],
                        int(x_max_pred + pad_amount * width_pred),
                    )
                    y_min_pred = max(0, int(y_min_pred - pad_amount * height_pred))
                    y_max_pred = min(
                        display_image_visual.shape[0],
                        int(y_max_pred + pad_amount * height_pred),
                    )

                    print(
                        f" crop_around_pred {x_min_pred=}, {x_max_pred=}, {y_min_pred=}, {y_max_pred=}"
                    )

                    if crop_bounds is None:
                        crop_bounds = (
                            x_min_pred,
                            x_max_pred,
                            y_min_pred,
                            y_max_pred,
                        )
                    else:
                        xmin, xmax, ymin, ymax = crop_bounds
                        crop_bounds = (
                            min(xmin, x_min_pred),
                            max(xmax, x_max_pred),
                            min(ymin, y_min_pred),
                            max(ymax, y_max_pred),
                        )

        if crop_bounds is not None:
            # print(f"{'X'*100} MAH HACKED CROPPING FOR 1 IMAGE")
            # crop_bounds = (29, 108, 2246, 2500)
            # crop_bounds = (29, 108, 2400, 2600)
            # crop_bounds = (29, 108, 2246, 2346)
            # crop_bounds = (29, 108, 1820, 1980)
            xmin, xmax, ymin, ymax = crop_bounds

            display_image_visual = display_image_visual[ymin:ymax, xmin:xmax, :]
            overlay_image = overlay_image[ymin:ymax, xmin:xmax, :]

            if ground_truth_overlay is not None:
                ground_truth_overlay = ground_truth_overlay[ymin:ymax, xmin:xmax, :]
        else:
            print(
                "Crop requested but no valid crop bounds were found; using full image"
            )

    if horizontal_stretch != 1:
        display_image_visual = np.repeat(
            display_image_visual, horizontal_stretch, axis=1
        )
        overlay_image = np.repeat(overlay_image, horizontal_stretch, axis=1)
        if ground_truth_overlay is not None:
            ground_truth_overlay = np.repeat(
                ground_truth_overlay, horizontal_stretch, axis=1
            )

    spacer_pixel_height = int(round(spacer_width * display_image_visual.shape[0]))
    combined_width_before_scaling = display_image_visual.shape[1]
    integer_output_scale = 1
    if min_width is not None and min_width > combined_width_before_scaling:
        integer_output_scale = int(np.ceil(min_width / combined_width_before_scaling))
        display_image_visual = upscale_image_integer(
            display_image_visual, integer_output_scale
        )
        overlay_image = upscale_image_integer(overlay_image, integer_output_scale)
        ground_truth_overlay = upscale_image_integer(
            ground_truth_overlay, integer_output_scale
        )

    title_row_spec = get_title_row_spec(
        image_width=display_image_visual.shape[1],
        image_height=display_image_visual.shape[0],
        font_size=header_font_size,
        title_texts=["Echogram", "Prediction", "Ground Truth"],
    )
    subtitle_row_spec = get_subtitle_row_spec(
        image_width=display_image_visual.shape[1],
        image_height=display_image_visual.shape[0],
        font_size=header_font_size,
        subtitle_rows=[None, pred_subtitle_chunks, gt_subtitle_chunks],
    )

    def _maybe_add_tile_header(
        image: np.ndarray,
        title_text: str,
        subtitle_chunks: list[tuple[str, tuple[int, int, int]]] | None = None,
    ) -> np.ndarray:
        if not include_summary_bar:
            return image
        return add_tile_header(
            image,
            title_text,
            header_font_size,
            subtitle_chunks=subtitle_chunks,
            title_row_spec=title_row_spec,
            subtitle_row_spec=subtitle_row_spec,
            bg_colour=header_bg_colour,
        )

    image_tiles = []
    if include_input_echogram_in_png and display_image_visual is not None:
        image_tiles.append(
            _maybe_add_tile_header(
                (display_image_visual * 255).astype(np.uint8),
                "Echogram",
            )
        )
    if result is not None:
        image_tiles.append(
            _maybe_add_tile_header(
                overlay_image,
                "Prediction",
                subtitle_chunks=pred_subtitle_chunks,
            )
        )
    if ground_truth_overlay is not None:
        image_tiles.append(
            _maybe_add_tile_header(
                ground_truth_overlay,
                "Ground Truth",
                subtitle_chunks=gt_subtitle_chunks,
            )
        )

    spacer_pixel_height = int(round(spacer_width * display_image_visual.shape[0]))
    combined_parts: list[np.ndarray] = []
    for tile_index, tile in enumerate(image_tiles):
        print(f"tile_index={tile_index} {type(tile)} {tile.shape}")
        combined_parts.append(tile)
        if spacer_pixel_height > 0 and tile_index < len(image_tiles) - 1:
            combined_parts.append(
                np.full(
                    (spacer_pixel_height, tile.shape[1], 3),
                    spacer_colour,
                    dtype=np.uint8,
                )
            )
    print(f"combined_parts={len(combined_parts)}")
    print(f"sizes: {[type(part) for part in combined_parts]}")
    print(f"sizes: {[part.shape for part in combined_parts]}")
    combined_image = np.concatenate(combined_parts, axis=0)
    if min_width is not None:
        combined_image = resize_combined_image_to_width(combined_image, min_width)

    output_path = (
        Path(
            f"{output_dir}/{filename}_predictions{'_cropped' if crop_around_gt else ''}.png"
        )
        if output_dir is not None
        else Path(f"{filename}_predictions.png")
    )
    if save_png:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(combined_image).save(output_path.with_suffix(".png"))
        print(f"Saved prediction plot to: {output_path.with_suffix('.png')}")
    if result is not None and result.boxes is not None and len(result.boxes) > 0:
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        if not filter_submasks:
            bool_filtered = [1] * len(class_ids)
        for idx, (class_id, confidence, filtered) in enumerate(
            zip(class_ids, confidences, bool_filtered)
        ):
            class_name = get_class_name(result.names, class_id)
            area = np.sum(result.masks[idx].data.cpu().numpy())
            print(
                f"  mask_{idx}: class_id={class_id} ({class_name}), confidence={confidence:.3f} filtered:{filtered} {area=}"
            )
    else:
        print("  No detections")

    if show:
        fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
        ax.imshow(combined_image)
        ax.axis("off")
        plt.show()

    return pred_counts


def main() -> None:
    args = parse_args()

    get_and_plot_echogram_predictions(
        model_path=args.model,
        echogram_filepath=args.echogram_filepath,
        echogram_dir=args.echogram_dir,
        filename=args.filename,
        label_filepath=args.label_filepath,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        filter_submasks=args.filter_submasks,
        conf=args.conf,
        iou=args.iou,
        mask_alpha=args.alpha,
        header_font_size=args.header_font_size,
        crop_around_gt=args.crop_around_gt,
        crop_ignore_no_cross=args.crop_ignore_no_cross,
        show=args.show,
        device=args.device,
        no_end2end=args.no_end2end,
        min_width=args.min_width,
        spacer_width=args.spacer_width,
    )


if __name__ == "__main__":
    main()


"""
python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename RB_Nusagak_Sonar_Files_2018_RB_2018-08-06_171000_6300_6600


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename Elwha_2018_OM_ARIS_2018_07_12_2018-07-12_220000_2592_3043


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-03-JD154_LeftNear_Stratum1_Set1_LN_2018-06-03_090000_778_1319


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-04-JD155_LeftFar_Stratum2_Set1_LO_2018-06-04_161004_2007_2207



python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename Elwha_2018_OM_ARIS_2018_07_11_2018-07-11_010000_3200_3651



ALL THREE CLASSES

python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-03-JD154_LeftFar_Stratum2_Set1_LO_2018-06-03_121003_1310_1790
 

python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-01-JD152_LeftFar_Stratum2_Set1_LO_2018-06-01_181004_2092_2572

python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-05-26-JD146_LeftFar_Stratum1_Set1_LO_2018-05-26_180004_1253_1488

python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-05-JD156_RightFar_Stratum2_Set1_RO_2018-06-05_071003_4961_5161

python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-02-JD153_LeftFar_Stratum1_Set1_LO_2018-06-02_060004_2288_2888

# BOTH UP AND DOWN ONLY



python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-03-JD154_LeftFar_Stratum1_Set1_LO_2018-06-03_030004_1934_2534


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-05-27-JD147_LeftFar_Stratum2_Set1_LO_2018-05-27_141003_1994_2474


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-02-JD153_LeftFar_Stratum1_Set1_LO_2018-06-02_190004_3716_4316


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-06-01-JD152_LeftNear_Stratum1_Set1_LN_2018-06-01_010000_3186_3727





python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename Elwha_2018_OM_ARIS_2018_07_24_2018-07-24_050003_1976_2427


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename Elwha_2018_OM_ARIS_2018_07_25_2018-07-25_070000_6028_6479


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-05-30-JD150_RightFar_Stratum1_Set1_RO_2018-05-30_050004_1009_1550\
  --crop_around_gt


python -m analysis.plot_echogram_predictions   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
  --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
  --imgsz 1920\
  --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
  --filename 2018-05-27-JD147_LeftFar_Stratum2_Set1_LO_2018-05-27_141003_1994_2474\
  --crop_around_gt

python -m analysis.plot_echogram_predictions   --model "/home/mahobley/Code/echo-seg/runs/cfc26/noklamath_bgsanglenobgs_myolo_1280imgsz_5class_1000epnostop_6bs_valbyriver_nmae_removesubs_100mosaic_96seed_0worker/weights/best_avg_crossing_nmae.pt" \
  --echogram_dir /home/mahobley/Code/echo-seg/data/CFC26/yolo_echograms_bgs_angle_nobgs_nanaimo/images/test/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results/nan\
  --imgsz 1280\
  --label_dir /home/mahobley/Code/echo-seg/data/CFC26/yolo_echograms_bgs_angle_nobgs_nanaimo/labels/test/\
  --filename nanaimo__EASY_2024-05-22_Nanaimo_RightBank_2024-05-22-013000_00643_01054\
  --device cpu\
  --filter-submasks\
  --crop_around_gt

python -m analysis.plot_echogram_predictions   --model "/home/mahobley/Code/echo-seg/runs/cfc26/noklamath_bgsanglenobgs_myolo_1280imgsz_5class_1000epnostop_6bs_valbyriver_nmae_removesubs_100mosaic_96seed_0worker/weights/best_avg_crossing_nmae.pt" \
  --echogram_dir /home/mahobley/Code/echo-seg/data/CFC26/yolo_echograms_bgs_angle_nobgs_nanaimo/images/test/\
  --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results/nan\
  --imgsz 1280\
  --label_dir /home/mahobley/Code/echo-seg/data/CFC26/yolo_echograms_bgs_angle_nobgs_nanaimo/labels/test/\
  --filename nanaimo__HARD_2023-10-19_Nanaimo_173000_2060_4120_02000_02060\
  --device cpu\
  --filter-submasks\
  --crop_around_gt
"""


# clips = ("2018-05-27-JD147_LeftFar_Stratum1_Set1_LO_2018-05-27_160004_1980_2580" \
# "2018-05-27-JD147_LeftFar_Stratum2_Set1_LO_2018-05-27_141003_1994_2474" \
# "Elwha_2018_OM_ARIS_2018_07_11_2018-07-11_040000_2285_2736" \
# "Elwha_2018_OM_ARIS_2018_07_12_2018-07-12_220000_2592_3043" \
# "2018-05-29-JD149_RightFar_Stratum1_Set1_RO_2018-05-29_230004_5034_5195" \
# "2018-05-27-JD147_RightFar_Stratum1_Set1_RO_2018-05-27_150004_3218_3418" \
# "2018-05-27-JD147_RightFar_Stratum1_Set1_RO_2018-05-27_220004_3347_3547" \
# "2018-06-03-JD154_LeftNear_Stratum1_Set1_LN_2018-06-03_090000_778_1319" \
# "2018-05-27-JD147_LeftFar_Stratum1_Set1_LO_2018-05-27_020004_2005_2205" \
# "2018-05-27-JD147_RightFar_Stratum2_Set1_RO_2018-05-27_091004_3049_3249" \
# "2018-06-01-JD152_RightFar_Stratum2_Set1_RO_2018-06-01_201004_3230_3771" \
# "2018-05-29-JD149_RightFar_Stratum1_Set1_RO_2018-05-29_100004_1086_1627" \
# "Elwha_2018_OM_ARIS_2018_07_26_2018-07-26_060000_6459_6910" \
# "2018-05-30-JD150_RightFar_Stratum1_Set1_RO_2018-05-30_230004_868_1409" \
# "RB_Nusagak_Sonar_Files_2018_RB_2018-07-30_201000_1500_1800" \
# "RB_Nusagak_Sonar_Files_2018_RB_2018-08-06_171000_6300_6600" \
# "Elwha_2018_OM_ARIS_2018_07_11_2018-07-11_010000_3200_3651")

# for clip in "${clips[@]}"; do
#   python -m analysis.plot_echogram_predictions   \
#   --model /home/mahobley/Code/echo-seg/runs/segment_and_classify/yolo_m_pos_neg_no-cross-center-pos-neg_1280imgsz_bs6/weights/best.pt \
#   --echogram_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/images/\
#   --output_dir /home/mahobley/Code/echo-seg/echogram_vis/results\
#   --imgsz 1920\
#   --label_dir /home/mahobley/Code/echo-seg/data/yolo_data_pos_neg_no-cross-center-pos-neg/val/labels/\
#   --filename "${clip}"

# done
