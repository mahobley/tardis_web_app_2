"""Export prediction detections to CSV, FC, Echotastic, and other formats."""

from __future__ import annotations

import csv
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from analysis.plot_echogram_predictions import (
    _map_frame_coordinate_to_original,
    convert_bin_to_m,
    get_class_name,
    mask_horizontal_extent_and_centroid,
    prediction_csv_confidence_cell,
    prediction_csv_round_decimal,
    prediction_csv_round_int,
)

logger = logging.getLogger(__name__)

FC_TITLE = "*** Manual Marking (Manual Sizing: Q = Quality, N = Repeat Count) ***"
FC_COMMENT = "TaRDIS"


class FCColumn(NamedTuple):
    width: int
    default: Any


FC_SCHEMA: dict[str, FCColumn] = {
    "File": FCColumn(4, 1),
    "Total": FCColumn(7, 0),
    "Frame#": FCColumn(8, 0),
    "Dir": FCColumn(5, ""),
    "R (m)": FCColumn(8, 0.0),
    "Theta": FCColumn(8, 99.0),
    "L(cm)": FCColumn(8, 0.0),
    "dR(cm)": FCColumn(8, 0.0),
    "L/dR": FCColumn(8, 0.0),
    "Aspect": FCColumn(8, 0.0),
    "Time": FCColumn(10, "00:00:00"),
    "Date": FCColumn(12, ""),
    "Latitude": FCColumn(19, "N 00 d  0.00000 m"),
    "Longitude": FCColumn(20, "E 000 d  0.00000 m"),
    "Pan": FCColumn(9, 0.0),
    "Tilt": FCColumn(9, 0.0),
    "Roll": FCColumn(9, 0.0),
    "Species": FCColumn(10, "Unknown"),
    "Motion": FCColumn(39, "Running <-->"),
    "Q": FCColumn(7, 5),
    "N": FCColumn(8, 1),
    "Comment": FCColumn(15, ""),
}

FC_HEADERS = list(FC_SCHEMA.keys())

UpstreamDirection = str  # "left" | "right"

PREDICTION_CSV_FIELDNAMES = [
    "instance_index",
    "confidence",
    "class_id",
    "class_name",
    "enter_frame",
    "exit_frame",
    "center_frame",
    "center_frame_bin",
    "center_frame_distance",
    "duration",
    "minimum_bin_y",
    "maximum_bin_y",
    "average_bin_y",
    "start_bin_y",
    "end_bin_y",
    "minimum_distance",
    "maximum_distance",
    "average_distance",
    "start_distance",
    "end_distance",
]


def predictions_csv_path(output_dir: Path | None, filename: str) -> Path:
    if output_dir is not None:
        return Path(output_dir) / f"{filename}_predictions.csv"
    return Path(f"{filename}_predictions.csv")


def predictions_fc_path(output_dir: Path | None, aris_stem: str) -> Path:
    """FishClass manual marking file: ``FCe_<arisname>_ID_.txt`` (no frame range suffix)."""
    name = f"FCe_{aris_stem}_ID_.txt"
    if output_dir is not None:
        return Path(output_dir) / name
    return Path(name)


def predictions_echotastic_path(output_dir: Path | None, aris_stem: str) -> Path:
    """Echotastic export: ``<arisname>.aris.txt`` (no frame range suffix)."""
    name = f"{aris_stem}.aris.txt"
    if output_dir is not None:
        return Path(output_dir) / name
    return Path(name)


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        if value.lower() == "nan":
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _date_from_filename(filename: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")


# Trailing ``_{text}_yyyy-mm-dd_hhmmss`` e.g. ``..._2018-06-02_170004``
_ARIS_FILENAME_DATETIME_RE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})_(\d{6})$")


def _echotastic_header_datetime(stem: str) -> tuple[str, str]:
    """Parse trailing ``_yyyy-mm-dd_hhmmss`` from ARIS stem → (Date, Start Time) or blanks."""
    match = _ARIS_FILENAME_DATETIME_RE.search(stem)
    if not match:
        return "", ""
    year, month, day, hmmss = match.groups()
    date_str = f"{int(month):02d}/{int(day):02d}/{year}"
    if len(hmmss) != 6 or not hmmss.isdigit():
        return date_str, ""
    start_str = f"{hmmss[0:2]}:{hmmss[2:4]}:{hmmss[4:6]}"
    return date_str, start_str


def aris_duration_minutes(meta: dict) -> float | None:
    """
    ARIS file duration in minutes for the Echotastic ``Total Time`` header.

    Echotastic derives recording length from hardware timing in the file header,
    not from ``numframes / framerate``. The frame interval is taken as
    ``cycleperiod * sampleperiod`` (microseconds). Elapsed time from the first
    frame to the last uses ``numframes - 1`` intervals (one fewer than the frame
    count).
    """
    try:
        numframes = int(meta["numframes"])
        cycleperiod = float(meta["cycleperiod"])
        sampleperiod = float(meta["sampleperiod"])
    except (KeyError, TypeError, ValueError):
        return None
    if numframes < 1:
        return None
    # Echotastic timing: (n - 1) frame intervals, not numframes / header framerate.
    duration_seconds = (numframes - 1) * cycleperiod * sampleperiod / 1_000_000
    return duration_seconds / 60.0


def _echotastic_total_time_header(metadata: dict | None) -> str:
    """Echotastic ``Total Time = … minutes`` using :func:`aris_duration_minutes`."""
    if not metadata:
        return ""
    minutes = aris_duration_minutes(metadata)
    if minutes is None:
        return ""
    return f"{minutes:.3f} minutes"


def _normalize_class_label(class_name: str, class_id: int) -> str:
    name = str(class_name).lower().strip().replace("_", "-")
    if name in ("-1", "nan", "unknown"):
        if class_id == 0:
            return "left"
        if class_id == 1:
            return "right"
        if class_id == 2:
            return "no-cross"
        return name
    return name


def is_no_cross_detection(class_name: str, class_id: int) -> bool:
    """True for no-cross classes (excluded from FC export)."""
    label = _normalize_class_label(class_name, class_id)
    if class_id == 2:
        return True
    return label in ("no-cross", "no cross", "nocross") or "no-cross" in label


def crossing_side(class_name: str, class_id: int) -> str | None:
    """Return ``pos`` or ``neg`` for FC direction; ``None`` if not a crossing."""
    if is_no_cross_detection(class_name, class_id):
        return None
    label = _normalize_class_label(class_name, class_id)
    if "pos" in label:
        return "pos"
    if "neg" in label:
        return "neg"
    if label in ("left", "0"):
        return "pos"
    if label in ("right", "1"):
        return "neg"
    if class_id == 0:
        return "pos"
    if class_id == 1:
        return "neg"
    return None


def fc_direction_for_crossing(
    class_name: str, class_id: int, upstream_direction: UpstreamDirection
) -> str:
    """Map pos/neg to FishClass ``Dir`` (Up/Down) from upstream travel direction."""
    side = crossing_side(class_name, class_id)
    if side is None:
        return ""
    upstream = upstream_direction.lower().strip()
    if upstream == "left":
        return "Down" if side == "pos" else "Up"
    if upstream == "right":
        return "Up" if side == "pos" else "Down"
    raise ValueError(
        f"upstream_direction must be 'left' or 'right', got {upstream_direction!r}"
    )


def echotastic_direction(
    class_name: str, class_id: int, upstream_direction: UpstreamDirection
) -> int | None:
    """Map pos/neg to Echotastic Direction (1 = upstream, -1 = downstream)."""
    side = crossing_side(class_name, class_id)
    if side is None:
        return None
    upstream = upstream_direction.lower().strip()
    if upstream == "left":
        return 1 if side == "pos" else -1
    if upstream == "right":
        return -1 if side == "pos" else 1
    raise ValueError(
        f"upstream_direction must be 'left' or 'right', got {upstream_direction!r}"
    )


ECHOTASTIC_COLUMNS = [
    "Sample",
    "Ping",
    "Time",
    "Range",
    "Amplitude",
    "XAngle",
    "YAngle",
    "Direction",
    "Length",
    "Area",
    "Operator",
]

ECHOTASTIC_VERSION = "2.0"
ECHOTASTIC_OPERATOR = "AUT"


def _framerate_from_metadata(metadata: dict | None) -> float:
    """Recorded frame rate from ARIS/DIDSON header (Hz)."""
    if not metadata:
        return 15.0
    for key in ("framerate", "FrameRate", "frame_rate", "fps"):
        if key in metadata:
            rate = _as_float(metadata[key])
            if not math.isnan(rate) and rate > 0:
                return rate
    return 15.0


def _echotastic_time_seconds(ping: int, framerate: float) -> float:
    """Time in seconds: ``ping / framerate / 60`` per Echotastic spec."""
    if framerate <= 0:
        return 0.0
    return ping / framerate / 60.0


def _num_bins_from_metadata(metadata: dict | None) -> int | None:
    if not metadata:
        return None
    for key in ("samplesperbeam", "samplesperchannel", "ydim"):
        if key in metadata:
            try:
                n = int(metadata[key])
            except (TypeError, ValueError):
                continue
            if n > 0:
                return n
    return None


def _echotastic_sample(bin_index: float | int, metadata: dict | None) -> int:
    """Echotastic Sample column: ``num_bins - bin`` (flip image row index)."""
    try:
        b = int(round(float(bin_index)))
    except (TypeError, ValueError):
        b = 0
    num_bins = _num_bins_from_metadata(metadata)
    if num_bins is not None:
        return num_bins - b
    return b


def build_prediction_rows(
    pred_masks: list[np.ndarray],
    pred_class_ids: list[int],
    pred_confidences_list: list[float],
    class_names: dict | list | None,
    echogram_metadata: dict | None,
    frame_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build one dict per instance matching :data:`PREDICTION_CSV_FIELDNAMES`."""
    rows: list[dict[str, Any]] = []
    frame_indices_array: np.ndarray | None = (
        np.asarray(frame_indices, dtype=int) if frame_indices is not None else None
    )
    for idx, mask in enumerate(pred_masks):
        class_id = pred_class_ids[idx] if idx < len(pred_class_ids) else -1
        conf = (
            pred_confidences_list[idx]
            if idx < len(pred_confidences_list)
            else float("nan")
        )
        (
            enter_f,
            exit_f,
            center_fr,
            center_fr_bin,
            min_by,
            max_by,
            avg_by,
            duration,
            start_by,
            end_by,
        ) = mask_horizontal_extent_and_centroid(mask)

        if frame_indices_array is not None and frame_indices_array.size > 0:
            enter_f = _map_frame_coordinate_to_original(enter_f, frame_indices_array)
            exit_f = _map_frame_coordinate_to_original(exit_f, frame_indices_array)
            center_fr = _map_frame_coordinate_to_original(
                center_fr, frame_indices_array
            )
            duration = float(int(round(exit_f)) - int(round(enter_f)) + 1)

        center_frame_distance = convert_bin_to_m(center_fr_bin, echogram_metadata)
        start_distance = convert_bin_to_m(start_by, echogram_metadata)
        end_distance = convert_bin_to_m(end_by, echogram_metadata)
        minimum_distance = convert_bin_to_m(min_by, echogram_metadata)
        maximum_distance = convert_bin_to_m(max_by, echogram_metadata)
        average_distance = convert_bin_to_m(avg_by, echogram_metadata)

        rows.append(
            {
                "instance_index": prediction_csv_round_int(idx),
                "confidence": prediction_csv_confidence_cell(conf),
                "class_id": prediction_csv_round_int(class_id),
                "class_name": get_class_name(class_names, class_id),
                "enter_frame": prediction_csv_round_int(enter_f),
                "exit_frame": prediction_csv_round_int(exit_f),
                "center_frame": prediction_csv_round_int(center_fr),
                "center_frame_bin": prediction_csv_round_int(center_fr_bin),
                "center_frame_distance": prediction_csv_round_decimal(
                    center_frame_distance, 2
                ),
                "duration": prediction_csv_round_int(duration),
                "minimum_bin_y": prediction_csv_round_int(min_by),
                "maximum_bin_y": prediction_csv_round_int(max_by),
                "average_bin_y": prediction_csv_round_int(avg_by),
                "start_bin_y": prediction_csv_round_int(start_by),
                "end_bin_y": prediction_csv_round_int(end_by),
                "minimum_distance": prediction_csv_round_decimal(minimum_distance, 2),
                "maximum_distance": prediction_csv_round_decimal(maximum_distance, 2),
                "average_distance": prediction_csv_round_decimal(average_distance, 2),
                "start_distance": prediction_csv_round_decimal(start_distance, 2),
                "end_distance": prediction_csv_round_decimal(end_distance, 2),
            }
        )
    return rows


def _prediction_rows_to_fc_records(
    rows: list[dict[str, Any]],
    *,
    filename: str,
    upstream_direction: UpstreamDirection = "left",
) -> list[dict[str, Any]]:
    """Map pipeline prediction rows to FishClass manual-marking columns."""
    date = _date_from_filename(filename)
    defaults = {name: col.default for name, col in FC_SCHEMA.items()}
    fc_rows: list[dict[str, Any]] = []

    for row in rows:
        class_name = str(row.get("class_name", ""))
        class_id_raw = row.get("class_id", -1)
        try:
            class_id = int(class_id_raw)
        except (TypeError, ValueError):
            class_id = -1

        if is_no_cross_detection(class_name, class_id):
            continue

        r_m = _as_float(row.get("center_frame_distance"))
        if math.isnan(r_m):
            continue

        min_d = _as_float(row.get("minimum_distance"))
        max_d = _as_float(row.get("maximum_distance"))
        start_d = _as_float(row.get("start_distance"))
        end_d = _as_float(row.get("end_distance"))

        dr_cm = (
            (max_d - min_d) * 100.0
            if not (math.isnan(min_d) or math.isnan(max_d))
            else 0.0
        )
        dr_cm = max(dr_cm, 0.0)

        if not (math.isnan(start_d) or math.isnan(end_d)):
            l_cm = abs(end_d - start_d) * 100.0
        else:
            l_cm = 0.0

        l_dr = (l_cm / dr_cm) if dr_cm > 0 else 0.0
        frame = row.get("center_frame", 0)
        try:
            frame_num = (
                int(frame)
                if not isinstance(frame, float) or not math.isnan(frame)
                else 0
            )
        except (TypeError, ValueError):
            frame_num = 0

        side = crossing_side(class_name, class_id)
        if side is None:
            continue

        species = str(defaults["Species"])

        direction = fc_direction_for_crossing(class_name, class_id, upstream_direction)

        record = dict(defaults)
        record.update(
            {
                "File": 1,
                "Frame#": frame_num,
                "Dir": direction[: FC_SCHEMA["Dir"].width],
                "R (m)": round(r_m, 2),
                "Theta": defaults["Theta"],
                "L(cm)": round(l_cm, 2),
                "dR(cm)": round(dr_cm, 2),
                "L/dR": round(l_dr, 2),
                "Aspect": round(l_dr, 2),
                "Date": date,
                "Species": species[: FC_SCHEMA["Species"].width],
                "Comment": FC_COMMENT[: FC_SCHEMA["Comment"].width],
            }
        )
        fc_rows.append(record)

    fc_rows.sort(key=lambda r: int(r["Frame#"]))
    for total, record in enumerate(fc_rows, start=1):
        record["Total"] = total
    return fc_rows


def _format_fc_lines(fc_rows: list[dict[str, Any]]) -> list[str]:
    """Build fixed-width FishClass text lines (title, header, separator, data)."""
    column_widths = {name: col.width for name, col in FC_SCHEMA.items()}
    header_line = "".join(f"{h:>{column_widths[h]}}" for h in FC_HEADERS)
    separator_line = "-" * len(header_line)
    lines = [FC_TITLE + "\n\n", header_line + "\n", separator_line + "\n"]

    if not fc_rows:
        return lines

    for record in fc_rows:
        cells = []
        for header in FC_HEADERS:
            val = record.get(header, FC_SCHEMA[header].default)
            cells.append(f"{str(val):>{column_widths[header]}}")
        lines.append("".join(cells) + "\n")
    return lines


def write_fc_file(fc_path: Path, fc_rows: list[dict[str, Any]]) -> Path:
    """Write FishClass manual-marking file to ``fc_path``."""
    fc_path = Path(fc_path)
    fc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _format_fc_lines(fc_rows)
    fc_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Saved predictions FC file to: %s", fc_path)
    return fc_path


def export_predictions_csv(
    *,
    output_dir: Path | None,
    filename: str,
    pred_masks: list[np.ndarray],
    pred_class_ids: list[int],
    pred_confidences_list: list[float],
    class_names: dict | list | None,
    echogram_metadata: dict | None,
    frame_indices: list[int] | None = None,
) -> Path:
    """Write prediction instances to a CSV file."""
    csv_path = predictions_csv_path(output_dir, filename)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_prediction_rows(
        pred_masks,
        pred_class_ids,
        pred_confidences_list,
        class_names,
        echogram_metadata,
        frame_indices=frame_indices,
    )
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=PREDICTION_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved predictions CSV to: %s", csv_path)
    return csv_path


def export_predictions_fc(
    *,
    output_dir: Path | None,
    aris_stem: str,
    pred_masks: list[np.ndarray],
    pred_class_ids: list[int],
    pred_confidences_list: list[float],
    class_names: dict | list | None,
    echogram_metadata: dict | None,
    frame_indices: list[int] | None = None,
    upstream_direction: UpstreamDirection = "left",
    date_source: str | None = None,
) -> Path:
    """Export prediction instances as a FishClass manual-marking text file."""
    fc_path = predictions_fc_path(output_dir, aris_stem)
    rows = build_prediction_rows(
        pred_masks,
        pred_class_ids,
        pred_confidences_list,
        class_names,
        echogram_metadata,
        frame_indices=frame_indices,
    )
    fc_rows = _prediction_rows_to_fc_records(
        rows,
        filename=date_source or aris_stem,
        upstream_direction=upstream_direction,
    )
    return write_fc_file(fc_path, fc_rows)


def _prediction_rows_to_echotastic(
    rows: list[dict[str, Any]],
    *,
    upstream_direction: UpstreamDirection,
    echogram_metadata: dict | None,
) -> list[dict[str, Any]]:
    framerate = _framerate_from_metadata(echogram_metadata)
    echotastic_rows: list[dict[str, Any]] = []

    for row in rows:
        class_name = str(row.get("class_name", ""))
        try:
            class_id = int(row.get("class_id", -1))
        except (TypeError, ValueError):
            class_id = -1

        if is_no_cross_detection(class_name, class_id):
            continue

        direction = echotastic_direction(class_name, class_id, upstream_direction)
        if direction is None:
            continue

        r_m = _as_float(row.get("center_frame_distance"))
        if math.isnan(r_m):
            continue

        try:
            ping = int(row.get("center_frame", 0))
        except (TypeError, ValueError):
            ping = 0
        sample = _echotastic_sample(row.get("center_frame_bin", 0), echogram_metadata)

        echotastic_rows.append(
            {
                "Sample": sample,
                "Ping": ping,
                "Time": _echotastic_time_seconds(ping, framerate),
                "Range": r_m,
                "Amplitude": 0.0,
                "XAngle": 0.0,
                "YAngle": 0.0,
                "Direction": direction,
                "Length": 0.0,
                "Area": 0.0,
                "Operator": ECHOTASTIC_OPERATOR,
            }
        )

    echotastic_rows.sort(key=lambda r: (r["Ping"], r["Sample"]))
    return echotastic_rows


def _format_echotastic_file(
    *,
    aris_path: Path | None,
    aris_stem: str,
    echogram_metadata: dict | None,
    data_rows: list[dict[str, Any]],
) -> str:
    file_name = str(aris_path.resolve()) if aris_path is not None else ""
    date_str, start_str = _echotastic_header_datetime(aris_stem)
    total_time_str = _echotastic_total_time_header(echogram_metadata)
    header_lines = [
        f"Version = {ECHOTASTIC_VERSION}",
        f"File Name = {file_name}",
        f"Total Number Of Marks = {len(data_rows)}",
        f"Total Time = {total_time_str}",
        f"Date = {date_str}",
        f"Start Time = {start_str}",
        "",
        "\t".join(ECHOTASTIC_COLUMNS),
    ]
    body_lines: list[str] = []
    for record in data_rows:
        cells = [
            str(record["Sample"]),
            str(record["Ping"]),
            f"{record['Time']:.2f}",
            f"{record['Range']:.2f}",
            f"{record['Amplitude']:.2f}",
            f"{record['XAngle']:.2f}",
            f"{record['YAngle']:.2f}",
            str(record["Direction"]),
            f"{record['Length']:.2f}",
            f"{record['Area']:.2f}",
            str(record["Operator"]),
        ]
        body_lines.append("\t".join(cells))
    return "\n".join(header_lines + body_lines) + "\n"


def write_echotastic_file(
    out_path: Path,
    *,
    aris_path: Path | None,
    aris_stem: str,
    echogram_metadata: dict | None,
    data_rows: list[dict[str, Any]],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = _format_echotastic_file(
        aris_path=aris_path,
        aris_stem=aris_stem,
        echogram_metadata=echogram_metadata,
        data_rows=data_rows,
    )
    out_path.write_text(content, encoding="utf-8")
    logger.info("Saved Echotastic export to: %s", out_path)
    return out_path


def export_predictions_echotastic(
    *,
    output_dir: Path | None,
    aris_stem: str,
    pred_masks: list[np.ndarray],
    pred_class_ids: list[int],
    pred_confidences_list: list[float],
    class_names: dict | list | None,
    echogram_metadata: dict | None,
    frame_indices: list[int] | None = None,
    upstream_direction: UpstreamDirection = "left",
    aris_path: Path | None = None,
) -> Path:
    """Export predictions as tab-delimited Echotastic ``<arisname>.aris.txt``."""
    out_path = predictions_echotastic_path(output_dir, aris_stem)
    rows = build_prediction_rows(
        pred_masks,
        pred_class_ids,
        pred_confidences_list,
        class_names,
        echogram_metadata,
        frame_indices=frame_indices,
    )
    echotastic_rows = _prediction_rows_to_echotastic(
        rows,
        upstream_direction=upstream_direction,
        echogram_metadata=echogram_metadata,
    )
    if aris_path is None and output_dir is not None:
        aris_path = Path(output_dir) / f"{aris_stem}.aris"
    return write_echotastic_file(
        out_path,
        aris_path=aris_path,
        aris_stem=aris_stem,
        echogram_metadata=echogram_metadata,
        data_rows=echotastic_rows,
    )
