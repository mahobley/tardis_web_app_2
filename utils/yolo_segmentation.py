from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import math
import cv2

T = TypeVar("T")


def predict_segmentation(
    model: YOLO,
    source: str | Path | np.ndarray,
    conf: float,
    iou: float,
    imgsz: int | float,
    device: str | None = None,
    *,
    end2end: bool | None = None,
    half: bool | None = False,
) -> Any:

    if imgsz == -1:
        if isinstance(source, np.ndarray):
            img = source
        else:
            img = cv2.imread(source)
        h, w = img.shape[:2]

        # YOLO inputs usually need dimensions divisible by model stride, commonly 32.
        stride = int(model.model.stride.max())
        imgsz = (math.ceil(h / stride) * stride, math.ceil(w / stride) * stride)
        print(f"Using imgsz={imgsz} for img.shape={img.shape}")
    elif imgsz > 0 and imgsz < 1:
        if isinstance(source, np.ndarray):
            img = source
        else:
            img = cv2.imread(source)
        print(f"scaling img down by {imgsz} factor")
        h, w = img.shape[:2]

        scale = float(imgsz)

        stride = int(model.model.stride.max())

        target_h = max(stride, int(round(h * scale)))
        target_w = max(stride, int(round(w * scale)))

        imgsz = (
            math.ceil(target_h / stride) * stride,
            math.ceil(target_w / stride) * stride,
        )
        print(f"Using imgsz={imgsz} for img.shape={img.shape}")

    predict_kwargs: dict[str, Any] = {
        "source": str(source) if isinstance(source, Path) else source,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "save": False,
        "verbose": False,
        "half": half,
    }
    if device is not None:
        predict_kwargs["device"] = device
    if end2end is not None:
        predict_kwargs["end2end"] = end2end

    results = model.predict(**predict_kwargs)
    if len(results) == 0:
        raise RuntimeError("YOLO returned no prediction results.")
    return results[0]


def rasterize_polygon_mask(
    polygon_xy: np.ndarray,
    image_height: int,
    image_width: int,
    *,
    as_bool: bool = False,
) -> np.ndarray:
    mask_image = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask_image)
    points = [(float(x), float(y)) for x, y in polygon_xy]
    if len(points) >= 3:
        draw.polygon(points, outline=255, fill=255)

    mask = np.asarray(mask_image, dtype=np.uint8)
    if as_bool:
        return mask > 0
    return mask / 255.0


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask

    mask_dtype = mask.dtype
    mask_image = Image.fromarray((mask.astype(np.float32) * 255).astype(np.uint8))
    resized = np.asarray(
        mask_image.resize((width, height), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    if np.issubdtype(mask_dtype, np.bool_):
        return resized > 0
    return resized / 255.0


def get_prediction_masks_in_image_space(
    result: Any,
    *,
    as_bool: bool = False,
    return_indices: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[int]]:
    if result.masks is None or len(result.masks) == 0:
        return ([], []) if return_indices else []

    image_height, image_width = result.orig_shape
    polygons = getattr(result.masks, "xy", None)
    if polygons is not None:
        masks: list[np.ndarray] = []
        kept_indices: list[int] = []
        for index, polygon in enumerate(polygons):
            if polygon is None or len(polygon) == 0:
                continue
            masks.append(
                rasterize_polygon_mask(
                    polygon_xy=polygon,
                    image_height=image_height,
                    image_width=image_width,
                    as_bool=as_bool,
                )
            )
            kept_indices.append(index)
        if len(masks) > 0:
            return (masks, kept_indices) if return_indices else masks

    masks = []
    kept_indices = []
    for index, mask in enumerate(result.masks.data.detach().cpu().numpy()):
        resized_mask = resize_mask(mask, image_height, image_width)
        masks.append(resized_mask > 0.5 if as_bool else resized_mask)
        kept_indices.append(index)
    return (masks, kept_indices) if return_indices else masks


def get_prediction_class_ids(
    result: Any,
    *,
    kept_indices: list[int] | None = None,
    fill_value: int = -1,
) -> list[int]:
    class_ids: list[int] = []
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = [int(c) for c in result.boxes.cls.detach().cpu().tolist()]

    if kept_indices is None:
        return class_ids
    return [
        class_ids[idx] if idx < len(class_ids) else fill_value for idx in kept_indices
    ]


def get_prediction_confidences(
    result: Any,
    *,
    kept_indices: list[int] | None = None,
    fill_value: float = 0.0,
) -> list[float]:
    confidences: list[float] = []
    if result.boxes is not None and result.boxes.conf is not None:
        confidences = [float(c) for c in result.boxes.conf.detach().cpu().tolist()]

    if kept_indices is None:
        return confidences
    return [
        confidences[idx] if idx < len(confidences) else fill_value
        for idx in kept_indices
    ]


def get_class_name_from_index(class_index: int, names: Any) -> str:
    if isinstance(names, dict):
        if class_index in names:
            return str(names[class_index])
        if str(class_index) in names:
            return str(names[str(class_index)])
    elif isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)
