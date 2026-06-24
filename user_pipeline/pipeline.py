"""Configurable ARIS → echogram → detection pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from user_pipeline.batch import (
    BatchPipelineResult,
    BatchPlan,
    plan_batch_run,
    prediction_output_paths,
)
from user_pipeline.config import PipelineConfig, PipelineResult

logger = logging.getLogger(__name__)


def _ensure_matplotlib_agg() -> None:
    import sys

    if getattr(sys, "frozen", False):
        from echo_seg_app.frozen_preload import preload_shared_libraries

        preload_shared_libraries()

    import matplotlib

    matplotlib.use("Agg")


def run_pipeline(
    config: PipelineConfig,
    *,
    model: object | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Load ARIS echogram, run YOLO segmentation, save plot and CSV."""
    from user_pipeline.aris_to_detections import generate_echogram_from_aris

    config.validate()

    def progress(msg: str) -> None:
        logger.info(msg)
        if on_progress is not None:
            on_progress(msg)

    aris_path = config.aris_path.expanduser().resolve()
    checkpoint = config.checkpoint.expanduser().resolve()
    output_dir = config.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(
        f"Loading echogram from {aris_path.name} (frames {config.start_frame}–{config.end_frame})…"
    )
    t0 = time.perf_counter()
    echogram, metadata = generate_echogram_from_aris(
        aris_path,
        start_frame=config.start_frame,
        end_frame=config.end_frame,
        bgs=config.bgs,
        return_raw_echogram_as_third_channel=config.raw_third_channel,
        debug_plot=False,
        return_as_bgr=True,
        beam_width_dir=config.beam_width_dir,
    )
    load_seconds = time.perf_counter() - t0
    progress(f"Echogram loaded: shape={echogram.shape}, {load_seconds:.1f}s")

    _ensure_matplotlib_agg()
    t1 = time.perf_counter()
    from analysis.plot_echogram_predictions import get_and_plot_echogram_predictions

    if model is None:
        progress(f"Loading model ({checkpoint.name})…")
    progress("Running inference…")
    pred_counts = get_and_plot_echogram_predictions(
        model_path=None if model is not None else checkpoint,
        model=model,
        echogram_image=echogram,
        echogram_metadata=metadata,
        output_dir=output_dir,
        filename=config.png_output_basename,
        aris_stem=aris_path.stem,
        aris_path=aris_path,
        label_filepath=None,
        imgsz=-1,
        infer_bins=config.infer_bins,
        infer_fps=config.infer_fps,
        crop_around_gt=config.crop_around_gt,
        device=config.device,
        conf=config.conf,
        iou=config.iou,
        filter_submasks=config.filter_submasks,
        horizontal_stretch=1,
        spacer_width=0,
        spacer_colour=(255, 255, 255),
        include_input_echogram_in_png=config.include_input_echogram_in_png,
        include_summary_bar=config.include_summary_bar,
        half=config.fp16,
        save_png=config.export_png,
        save_csv=config.export_csv,
        save_fc=config.export_fc,
        save_echotastic=config.export_echotastic,
        upstream_direction=config.upstream_direction,
        show=False,
    )
    predict_seconds = time.perf_counter() - t1

    prediction_png, prediction_csv, prediction_fc, prediction_echotastic = (
        prediction_output_paths(config, aris_path)
    )

    saved_outputs = []
    if config.export_png:
        saved_outputs.append(prediction_png.name)
    if config.export_csv:
        saved_outputs.append(prediction_csv.name)
    if config.export_fc:
        saved_outputs.append(prediction_fc.name)
    if config.export_echotastic:
        saved_outputs.append(prediction_echotastic.name)
    progress(
        f"Done in {load_seconds + predict_seconds:.1f}s — saved "
        f"{', '.join(saved_outputs)}"
    )
    image_height, image_width = echogram.shape[:2]
    return PipelineResult(
        output_dir=output_dir,
        image_width=image_width,
        image_height=image_height,
        predicted_pos_cross=pred_counts.get(0, 0),
        predicted_neg_cross=pred_counts.get(1, 0),
        predicted_no_cross=pred_counts.get(2, 0),
        prediction_png=prediction_png if prediction_png.is_file() else None,
        prediction_csv=prediction_csv if prediction_csv.is_file() else None,
        prediction_fc=prediction_fc if prediction_fc.is_file() else None,
        prediction_echotastic=(
            prediction_echotastic if prediction_echotastic.is_file() else None
        ),
        load_seconds=load_seconds,
        predict_seconds=predict_seconds,
    )


def run_batch_pipeline(
    base_config: PipelineConfig,
    aris_paths: list[Path],
    *,
    skip_already_processed: bool = False,
    plan: BatchPlan | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_file_progress: Callable[[int, int, int, int], None] | None = None,
) -> BatchPipelineResult:
    """Run the pipeline on multiple ARIS files."""
    if not aris_paths:
        raise ValueError("No ARIS/DIDSON files to process")

    base_config.validate_shared()
    if base_config.output_dir is None and len(aris_paths) > 1:
        raise ValueError("Output directory is required when processing multiple files")

    if base_config.output_dir is not None:
        output_dir = base_config.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = aris_paths[0].expanduser().resolve().parent

    def progress(msg: str) -> None:
        logger.info(msg)
        if on_progress is not None:
            on_progress(msg)

    if plan is None:
        plan = plan_batch_run(
            base_config,
            aris_paths,
            skip_already_processed=skip_already_processed,
        )

    results: list[PipelineResult] = []
    failures: list[tuple[Path, str]] = []

    def emit_file_progress() -> None:
        if on_file_progress is not None:
            on_file_progress(
                plan.skip_count,
                len(results),
                len(failures),
                plan.total,
            )

    progress(
        f"Pre-scan: {plan.process_count} to process, {plan.skip_count} already done"
    )
    if plan.to_skip:
        preview = plan.to_skip[:15]
        for path in preview:
            progress(f"  skip: {path.name}")
        if len(plan.to_skip) > len(preview):
            progress(f"  … and {len(plan.to_skip) - len(preview)} more skipped")

    emit_file_progress()

    shared_model = None
    if plan.to_process:
        checkpoint = base_config.checkpoint.expanduser().resolve()
        progress(f"Loading model ({checkpoint.name})…")
        from ultralytics import YOLO

        shared_model = YOLO(str(checkpoint))

    for index, aris_path in enumerate(plan.to_process, start=1):
        file_config = base_config.with_aris_path(aris_path)
        overall = plan.skip_count + index
        progress(
            f"[{index}/{plan.process_count}] Processing {aris_path.name} "
            f"(overall {overall}/{plan.total})…"
        )
        try:
            results.append(
                run_pipeline(
                    file_config,
                    model=shared_model,
                    on_progress=progress,
                )
            )
        except Exception as exc:
            logger.exception("Failed on %s", aris_path)
            failures.append((aris_path, str(exc)))
            progress(f"[{index}/{plan.process_count}] Failed {aris_path.name}: {exc}")

        emit_file_progress()

    progress(
        f"Batch finished: {len(results)} processed, {plan.skip_count} skipped, "
        f"{len(failures)} failed"
    )
    return BatchPipelineResult(
        results=results,
        skipped=list(plan.to_skip),
        failures=failures,
        output_dir=output_dir,
        plan=plan,
    )
