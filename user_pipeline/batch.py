"""Batch ARIS processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from analysis.prediction_exports import (
    predictions_csv_path,
    predictions_echotastic_path,
    predictions_fc_path,
)
from user_pipeline.config import PipelineConfig, PipelineResult

ARIS_EXTENSIONS = {".aris", ".ddf"}


def is_aris_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ARIS_EXTENSIONS


def discover_aris_files(directory: Path) -> list[Path]:
    """Return ``.aris`` / ``.ddf`` files in ``directory``, sorted alphabetically."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = [p for p in directory.iterdir() if is_aris_file(p)]
    return sorted(files, key=lambda p: p.name.lower())


def select_directory_files(
    directory: Path,
    *,
    run_all: bool = True,
    limit: int | None = None,
) -> list[Path]:
    files = discover_aris_files(directory)
    if not run_all and limit is not None:
        return files[: max(0, limit)]
    return files


def prediction_output_paths(
    config, aris_path: Path | None = None
) -> tuple[Path, Path, Path, Path]:
    """Expected PNG, CSV, FC, and Echotastic paths for a run."""
    path = aris_path or config.aris_path
    if config.output_dir is not None:
        output_dir = config.output_dir.expanduser().resolve()
    else:
        output_dir = path.expanduser().resolve().parent
    basename = f"{path.stem}_{config.start_frame}_{config.end_frame}"
    river = path.expanduser().resolve().parent.name
    png_basename = f"{river}__{basename}"
    suffix = "_cropped" if config.crop_around_gt else ""
    png = output_dir / f"{png_basename}_predictions{suffix}.png"
    csv = predictions_csv_path(output_dir, basename)
    fc = predictions_fc_path(output_dir, path.stem)
    echotastic = predictions_echotastic_path(output_dir, path.stem)
    return png, csv, fc, echotastic


def is_already_processed(config, aris_path: Path) -> bool:
    png, csv, fc, echotastic = prediction_output_paths(config, aris_path)
    expected_outputs: list[Path] = []
    if config.export_png:
        expected_outputs.append(png)
    if config.export_csv:
        expected_outputs.append(csv)
    if config.export_fc:
        expected_outputs.append(fc)
    if config.export_echotastic:
        expected_outputs.append(echotastic)
    return bool(expected_outputs) and all(path.is_file() for path in expected_outputs)


@dataclass
class BatchPlan:
    """Pre-run breakdown of which files will be processed vs skipped."""

    total: int
    to_process: list[Path]
    to_skip: list[Path]

    @property
    def process_count(self) -> int:
        return len(self.to_process)

    @property
    def skip_count(self) -> int:
        return len(self.to_skip)


def plan_batch_run(
    base_config: PipelineConfig,
    aris_paths: list[Path],
    *,
    skip_already_processed: bool,
) -> BatchPlan:
    """Scan inputs and classify files before starting the batch."""
    to_skip: list[Path] = []
    to_process: list[Path] = []
    for aris_path in aris_paths:
        file_config = base_config.with_aris_path(aris_path)
        if skip_already_processed and is_already_processed(file_config, aris_path):
            to_skip.append(aris_path)
        else:
            to_process.append(aris_path)
    return BatchPlan(
        total=len(aris_paths),
        to_process=to_process,
        to_skip=to_skip,
    )


@dataclass
class BatchPipelineResult:
    results: list[PipelineResult]
    skipped: list[Path]
    failures: list[tuple[Path, str]]
    output_dir: Path
    plan: BatchPlan | None = None

    @property
    def processed_count(self) -> int:
        return len(self.results)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failures)

    @property
    def total_seconds(self) -> float:
        return sum(r.total_seconds for r in self.results)
