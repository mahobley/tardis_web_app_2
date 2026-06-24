"""CLI batch runner for the ARIS detection pipeline with per-file timings."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

# ``python user_pipeline/batch_cli.py`` — repo root is not on sys.path.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo_seg_app.paths import beam_widths_dir

if TYPE_CHECKING:
    from user_pipeline.config import PipelineConfig, PipelineResult

logger = logging.getLogger(__name__)
ARIS_EXTENSIONS = {".aris", ".ddf"}
_TRAILING_FRAME_RANGE_RE = re.compile(r"_(\d+)_(\d+)$")

TIMING_FIELDNAMES = [
    "aris_path",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "image_width",
    "image_height",
    "predicted_pos_cross",
    "predicted_neg_cross",
    "predicted_no_cross",
    "wall_seconds",
    "pipeline_seconds",
    "load_seconds",
    "predict_seconds",
    "output_dir",
    "prediction_png",
    "prediction_csv",
    "prediction_fc",
    "prediction_echotastic",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ARIS detection pipeline without the GUI and save per-file timings."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional .aris/.ddf files or directories containing them.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to YOLO checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Required when processing more than one file.",
    )
    parser.add_argument(
        "--timings-csv",
        type=Path,
        help="Where to write the timing CSV. Defaults to <output-dir>/batch_timings.csv.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to process.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Exclusive end frame index; -1 means end of file.",
    )
    parser.add_argument(
        "--infer-bins",
        type=int,
        default=-1,
        help="Inference image height in bins; -1 keeps native height.",
    )
    parser.add_argument(
        "--infer-fps",
        type=float,
        default=-1.0,
        help="Inference frame rate to use for width rescaling; -1 means native width.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--upstream-direction",
        choices=("left", "right"),
        default="left",
        help="Maps pos/neg crossings to FC/Echotastic direction outputs.",
    )
    parser.add_argument(
        "--beam-width-dir",
        type=Path,
        default=beam_widths_dir(),
        help="Directory containing beam-width CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N discovered files after expansion and sorting.",
    )
    parser.add_argument(
        "--split",
        help="Split name inside the split JSON, e.g. train, val, or test.",
    )
    parser.add_argument(
        "--split-path",
        "--split_filepath",
        dest="split_path",
        type=Path,
        help="Path to a split JSON file mapping locations to split stem lists.",
    )
    parser.add_argument(
        "--aris-super-dir",
        "--aris-directory",
        dest="aris_super_dir",
        type=Path,
        help="Root directory containing ARIS files, optionally in location subdirectories.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        help="Optional subset of locations to use from the split JSON.",
    )
    parser.add_argument(
        "--skip-already-processed",
        action="store_true",
        help="Skip files whose enabled output artifacts already exist.",
    )
    parser.add_argument(
        "--no-filter-submasks",
        action="store_true",
        help="Disable submask filtering.",
    )
    parser.add_argument(
        "--fc",
        action="store_true",
        help="Export FishClass files.",
    )
    parser.add_argument(
        "--echotastic",
        action="store_true",
        help="Export Echotastic text files.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Disable PNG export.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable prediction CSV export.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _is_aris_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ARIS_EXTENSIONS


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _strip_trailing_frame_range(stem: str) -> str:
    """Remove trailing ``_<start>_<end>`` clip window from split stems."""
    return _TRAILING_FRAME_RANGE_RE.sub("", stem)


def resolve_inputs(inputs: Sequence[str], limit: int | None) -> list[Path]:
    from user_pipeline.batch import discover_aris_files

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            candidates = discover_aris_files(path)
        elif _is_aris_path(path):
            candidates = [path]
        else:
            raise FileNotFoundError(
                f"Expected an ARIS/DIDSON file or directory, got {path}"
            )

        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)

    resolved.sort(key=lambda p: (p.parent.as_posix().lower(), p.name.lower()))
    if limit is not None:
        return resolved[: max(0, limit)]
    return resolved


def discover_aris_files_recursive(directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = [path for path in directory.rglob("*") if _is_aris_path(path)]
    return sorted(files, key=lambda p: (p.parent.as_posix().lower(), p.name.lower()))


def _matching_location_candidates(
    candidates: list[Path], location_name: str
) -> list[Path]:
    target = _normalize_name(location_name)
    if not target:
        return candidates

    matched = []
    for path in candidates:
        parts = [_normalize_name(part) for part in path.parts]
        if target in parts:
            matched.append(path)
    return matched


def resolve_split_inputs(
    *,
    split_path: Path,
    split_name: str,
    aris_super_dir: Path,
    locations: Sequence[str] | None,
) -> list[Path]:
    split_path = split_path.expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split_data, dict):
        raise ValueError(f"Expected top-level object in split file: {split_path}")
    selected_locations: list[str]
    if locations:
        requested = set(locations)
        missing_locations = sorted(requested - set(split_data))
        if missing_locations:
            raise ValueError(
                "Locations not found in split file: " + ", ".join(missing_locations)
            )
        selected_locations = [loc for loc in split_data if loc in requested]
    else:
        selected_locations = list(split_data)

    all_files = discover_aris_files_recursive(aris_super_dir)
    stem_index: dict[str, list[Path]] = {}
    for path in all_files:
        stem_index.setdefault(path.stem, []).append(path)

    resolved: list[Path] = []
    seen: set[Path] = set()
    missing_stems: list[str] = []
    ambiguous_stems: list[str] = []

    for location in selected_locations:
        location_splits = split_data.get(location)
        if not isinstance(location_splits, dict):
            raise ValueError(
                f"Expected object for location {location!r} in {split_path}"
            )
        stems = location_splits.get(split_name)
        if stems is None:
            raise ValueError(
                f"Split {split_name!r} not found for location {location!r} in {split_path}"
            )
        if not isinstance(stems, list):
            raise ValueError(
                f"Expected list for location {location!r}, split {split_name!r}"
            )

        for stem in stems:
            if not isinstance(stem, str):
                raise ValueError(
                    f"Expected string stem for location {location!r}, split {split_name!r}"
                )
            candidates = stem_index.get(stem, [])
            if not candidates:
                clipped_stem = _strip_trailing_frame_range(stem)
                if clipped_stem != stem:
                    candidates = stem_index.get(clipped_stem, [])
            if not candidates:
                missing_stems.append(f"{location}:{stem}")
                continue

            location_candidates = _matching_location_candidates(candidates, location)
            if len(location_candidates) == 1:
                chosen = location_candidates[0]
            elif len(location_candidates) > 1:
                ambiguous_stems.append(
                    f"{location}:{stem} -> "
                    + ", ".join(str(path) for path in location_candidates[:5])
                )
                continue
            elif len(candidates) == 1:
                chosen = candidates[0]
            else:
                ambiguous_stems.append(
                    f"{location}:{stem} -> "
                    + ", ".join(str(path) for path in candidates[:5])
                )
                continue

            if chosen not in seen:
                seen.add(chosen)
                resolved.append(chosen)

    if missing_stems:
        preview = ", ".join(missing_stems[:10])
        more = f" (+{len(missing_stems) - 10} more)" if len(missing_stems) > 10 else ""
        raise FileNotFoundError(
            f"Could not resolve {len(missing_stems)} split entries under {aris_super_dir}: "
            f"{preview}{more}"
        )

    if ambiguous_stems:
        preview = "; ".join(ambiguous_stems[:5])
        more = (
            f" (+{len(ambiguous_stems) - 5} more)" if len(ambiguous_stems) > 5 else ""
        )
        raise ValueError(
            f"Ambiguous split entries under {aris_super_dir}: {preview}{more}"
        )

    return resolved


def combined_input_paths(args: argparse.Namespace) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    duplicate_count = 0

    direct_inputs = resolve_inputs(args.inputs, None) if args.inputs else []
    for path in direct_inputs:
        if path not in seen:
            seen.add(path)
            resolved.append(path)
        else:
            duplicate_count += 1

    split_mode_requested = any(
        value is not None
        for value in (args.split, args.split_path, args.aris_super_dir)
    )
    if split_mode_requested:
        missing = [
            name
            for name, value in (
                ("--split", args.split),
                ("--split-path", args.split_path),
                ("--aris-super-dir", args.aris_super_dir),
            )
            if value is None
        ]
        if missing:
            raise ValueError("Split mode requires " + ", ".join(missing))

        split_inputs = resolve_split_inputs(
            split_path=args.split_path,
            split_name=args.split,
            aris_super_dir=args.aris_super_dir,
            locations=args.locations,
        )
        for path in split_inputs:
            if path not in seen:
                seen.add(path)
                resolved.append(path)
            else:
                duplicate_count += 1

    if not resolved:
        raise ValueError(
            "Provide at least one input path or use split mode with --split, --split-path, and --aris-super-dir"
        )

    if duplicate_count:
        logger.info(
            "Removed %d duplicate ARIS file(s) from the input set", duplicate_count
        )

    if args.limit is not None:
        return resolved[: max(0, args.limit)]
    return resolved


def build_base_config(
    args: argparse.Namespace, first_aris_path: Path
) -> "PipelineConfig":
    from user_pipeline.config import PipelineConfig

    return PipelineConfig(
        aris_path=first_aris_path,
        checkpoint=args.checkpoint,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        bgs=True,
        raw_third_channel=True,
        output_dir=args.output_dir,
        device=args.device,
        fp16=False,
        filter_submasks=not args.no_filter_submasks,
        crop_around_gt=False,
        infer_bins=args.infer_bins,
        infer_fps=args.infer_fps,
        conf=args.conf,
        iou=args.iou,
        beam_width_dir=args.beam_width_dir,
        export_png=not args.no_png,
        export_csv=not args.no_csv,
        export_fc=args.fc,
        export_echotastic=args.echotastic,
        upstream_direction=args.upstream_direction,
    )


def timing_csv_path(args: argparse.Namespace, base_config: PipelineConfig) -> Path:
    if args.timings_csv is not None:
        return args.timings_csv.expanduser().resolve()
    if base_config.output_dir is not None:
        return base_config.output_dir.expanduser().resolve() / "batch_timings.csv"
    return base_config.aris_path.expanduser().resolve().parent / "batch_timings.csv"


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def timing_row(
    *,
    aris_path: Path,
    status: str,
    started_at: float | None = None,
    finished_at: float | None = None,
    wall_seconds: float | None = None,
    result: "PipelineResult" | None = None,
    error: str = "",
) -> dict[str, str]:
    def path_str(path: Path | None) -> str:
        return str(path) if path is not None else ""

    return {
        "aris_path": str(aris_path),
        "status": status,
        "started_at_utc": iso_utc(started_at) if started_at is not None else "",
        "finished_at_utc": iso_utc(finished_at) if finished_at is not None else "",
        "image_width": (
            str(result.image_width) if result and result.image_width is not None else ""
        ),
        "image_height": (
            str(result.image_height)
            if result and result.image_height is not None
            else ""
        ),
        "predicted_pos_cross": (
            str(result.predicted_pos_cross)
            if result and result.predicted_pos_cross is not None
            else ""
        ),
        "predicted_neg_cross": (
            str(result.predicted_neg_cross)
            if result and result.predicted_neg_cross is not None
            else ""
        ),
        "predicted_no_cross": (
            str(result.predicted_no_cross)
            if result and result.predicted_no_cross is not None
            else ""
        ),
        "wall_seconds": f"{wall_seconds:.6f}" if wall_seconds is not None else "",
        "pipeline_seconds": (
            f"{result.total_seconds:.6f}" if result is not None else ""
        ),
        "load_seconds": f"{result.load_seconds:.6f}" if result is not None else "",
        "predict_seconds": (
            f"{result.predict_seconds:.6f}" if result is not None else ""
        ),
        "output_dir": path_str(result.output_dir if result is not None else None),
        "prediction_png": path_str(
            result.prediction_png if result is not None else None
        ),
        "prediction_csv": path_str(
            result.prediction_csv if result is not None else None
        ),
        "prediction_fc": path_str(result.prediction_fc if result is not None else None),
        "prediction_echotastic": path_str(
            result.prediction_echotastic if result is not None else None
        ),
        "error": error,
    }


def write_timing_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TIMING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    from user_pipeline.batch import plan_batch_run
    from user_pipeline.pipeline import run_pipeline

    aris_paths = combined_input_paths(args)
    if not aris_paths:
        raise ValueError("No ARIS/DIDSON files found")

    base_config = build_base_config(args, aris_paths[0])
    base_config.validate_shared()
    if base_config.output_dir is None and len(aris_paths) > 1:
        raise ValueError("Output directory is required when processing multiple files")

    plan = plan_batch_run(
        base_config,
        aris_paths,
        skip_already_processed=args.skip_already_processed,
    )
    timings_path = timing_csv_path(args, base_config)
    logger.info(
        "Batch plan: %d to process, %d skipped, timing CSV -> %s",
        plan.process_count,
        plan.skip_count,
        timings_path,
    )

    timing_rows: list[dict[str, str]] = [
        timing_row(aris_path=path, status="skipped") for path in plan.to_skip
    ]
    write_timing_csv(timings_path, timing_rows)

    shared_model = None
    model_load_started = time.perf_counter()
    if plan.to_process:
        checkpoint = base_config.checkpoint.expanduser().resolve()
        logger.info("Loading model once for batch: %s", checkpoint)
        from ultralytics import YOLO

        shared_model = YOLO(str(checkpoint))
        logger.info(
            "Model loaded in %.3fs",
            time.perf_counter() - model_load_started,
        )

    failures = 0
    for index, aris_path in enumerate(plan.to_process, start=1):
        file_config = base_config.with_aris_path(aris_path)
        logger.info("[%d/%d] Processing %s", index, plan.process_count, aris_path.name)
        started_at = time.time()
        wall_start = time.perf_counter()
        try:
            result = run_pipeline(file_config, model=shared_model)
        except Exception as exc:
            failures += 1
            finished_at = time.time()
            wall_seconds = time.perf_counter() - wall_start
            logger.exception("Failed on %s", aris_path)
            timing_rows.append(
                timing_row(
                    aris_path=aris_path,
                    status="failed",
                    started_at=started_at,
                    finished_at=finished_at,
                    wall_seconds=wall_seconds,
                    error=str(exc),
                )
            )
            write_timing_csv(timings_path, timing_rows)
            continue

        finished_at = time.time()
        wall_seconds = time.perf_counter() - wall_start
        timing_rows.append(
            timing_row(
                aris_path=aris_path,
                status="processed",
                started_at=started_at,
                finished_at=finished_at,
                wall_seconds=wall_seconds,
                result=result,
            )
        )
        write_timing_csv(timings_path, timing_rows)

    logger.info(
        "Finished batch: %d processed, %d skipped, %d failed",
        plan.process_count - failures,
        plan.skip_count,
        failures,
    )
    logger.info("Saved timing CSV to %s", timings_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
