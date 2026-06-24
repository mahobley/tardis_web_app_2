"""Pipeline configuration types (no heavy imports — safe for GUI startup)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class PipelineConfig:
    aris_path: Path
    checkpoint: Path
    start_frame: int = 0
    end_frame: int = -1
    bgs: bool = True
    raw_third_channel: bool = True
    output_dir: Path | None = None
    device: str = "cpu"
    fp16: bool = False
    filter_submasks: bool = True
    crop_around_gt: bool = False
    infer_bins: int = -1
    infer_fps: float = -1.0
    conf: float = 0.1
    iou: float = 0.5
    beam_width_dir: Path | None = None
    export_png: bool = True
    export_csv: bool = True
    export_fc: bool = False
    export_echotastic: bool = False
    include_input_echogram_in_png: bool = True
    include_summary_bar: bool = True
    upstream_direction: str = (
        "left"  # "left" | "right" — maps pos/neg to FC Dir Up/Down
    )

    def validate(self) -> None:
        self.validate_shared()
        aris = self.aris_path.expanduser().resolve()
        if aris.suffix.lower() not in (".ddf", ".aris"):
            raise ValueError(f"Expected .ddf or .aris file, got {aris}")
        if not aris.is_file():
            raise FileNotFoundError(f"ARIS file not found: {aris}")

    def validate_shared(self) -> None:
        ckpt = self.checkpoint.expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        if self.fp16 and "cuda" not in self.device:
            raise ValueError("FP16 requires a CUDA device")

        # if not any(
        #     (self.export_png, self.export_csv, self.export_fc, self.export_echotastic)
        # ):
        #     raise ValueError("Enable at least one export format")

        if self.output_dir is not None:
            self.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    def with_aris_path(self, aris_path: Path) -> PipelineConfig:
        return replace(self, aris_path=aris_path)

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir.expanduser().resolve()
        return self.aris_path.expanduser().resolve().parent

    @property
    def output_basename(self) -> str:
        return f"{self.aris_path.stem}_{self.start_frame}_{self.end_frame}"

    @property
    def png_output_basename(self) -> str:
        river = self.aris_path.expanduser().resolve().parent.name
        return f"{river}__{self.output_basename}"


@dataclass
class PipelineResult:
    output_dir: Path
    load_seconds: float
    predict_seconds: float
    image_width: int | None = None
    image_height: int | None = None
    predicted_pos_cross: int | None = None
    predicted_neg_cross: int | None = None
    predicted_no_cross: int | None = None
    prediction_png: Path | None = None
    prediction_csv: Path | None = None
    prediction_fc: Path | None = None
    prediction_echotastic: Path | None = None

    @property
    def total_seconds(self) -> float:
        return self.load_seconds + self.predict_seconds

    @property
    def exported_paths(self) -> list[Path]:
        return [
            path
            for path in (
                self.prediction_png,
                self.prediction_csv,
                self.prediction_fc,
                self.prediction_echotastic,
            )
            if path is not None
        ]
