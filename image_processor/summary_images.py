"""Summary image generation for TIFF stack QC and segmentation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

try:
    from .tiff_backend import iter_tiff_frames
except ImportError:  # pragma: no cover - supports direct script-style imports.
    from tiff_backend import iter_tiff_frames


SummaryProgress = Callable[[int, int | None], None]
EXPORT_PNG_MIN_LONG_EDGE = 1024
EXPORT_PNG_MAX_SCALE = 64


@dataclass(frozen=True)
class SummaryImageSet:
    source_tiff: Path
    frame_count: int
    source_shape: tuple[int, int]
    source_dtype: str
    mean: np.ndarray
    std: np.ndarray
    max_projection: np.ndarray


def finite_min_max(frame: np.ndarray) -> tuple[float, float]:
    values = np.asarray(frame)
    if values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def auto_display_range(
    frame: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[float, float]:
    values = np.asarray(frame)
    if values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, lower_percentile))
    high = float(np.percentile(finite, upper_percentile))
    if high <= low:
        low, high = finite_min_max(values)
    if high <= low:
        high = low + 1.0
    return low, high


def scale_to_uint8(frame: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    if white_level <= black_level:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - float(black_level)) * (255.0 / (float(white_level) - float(black_level)))
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def export_png_scale_factor(
    width: int,
    height: int,
    *,
    min_long_edge: int = EXPORT_PNG_MIN_LONG_EDGE,
    max_scale: int = EXPORT_PNG_MAX_SCALE,
) -> int:
    long_edge = max(int(width), int(height))
    if long_edge <= 0 or long_edge >= int(min_long_edge):
        return 1
    return max(1, min(int(max_scale), int(math.ceil(float(min_long_edge) / float(long_edge)))))


def upscale_export_image(
    image: Image.Image,
    *,
    min_long_edge: int = EXPORT_PNG_MIN_LONG_EDGE,
    max_scale: int = EXPORT_PNG_MAX_SCALE,
    resample: Image.Resampling = Image.Resampling.NEAREST,
) -> Image.Image:
    scale = export_png_scale_factor(
        image.width,
        image.height,
        min_long_edge=min_long_edge,
        max_scale=max_scale,
    )
    if scale <= 1:
        return image
    return image.resize((image.width * scale, image.height * scale), resample=resample)


def save_png(frame: np.ndarray, output_path: Path) -> Path:
    black_level, white_level = auto_display_range(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(scale_to_uint8(frame, black_level, white_level))
    upscale_export_image(image).save(output_path)
    return output_path


def compute_summary_images(
    tiff_path: Path,
    progress: SummaryProgress | None = None,
) -> SummaryImageSet:
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    max_projection: np.ndarray | None = None
    source_shape: tuple[int, int] | None = None
    source_dtype = ""
    frame_count = 0

    for frame_index, raw_frame in enumerate(iter_tiff_frames(tiff_path)):
        frame = np.asarray(raw_frame)
        if frame.ndim != 2:
            raise ValueError(f"Summary images expect grayscale 2D frames, got shape {frame.shape}.")
        if mean is None:
            source_shape = (int(frame.shape[0]), int(frame.shape[1]))
            source_dtype = str(frame.dtype)
            mean = np.zeros(frame.shape, dtype=np.float64)
            m2 = np.zeros(frame.shape, dtype=np.float64)
            max_projection = frame.astype(np.float64, copy=True)
        elif frame.shape != source_shape:
            raise ValueError(f"Frame {frame_index} shape {frame.shape} does not match first frame {source_shape}.")
        else:
            max_projection = np.maximum(max_projection, frame)

        frame_count += 1
        values = frame.astype(np.float64, copy=False)
        delta = values - mean
        mean += delta / float(frame_count)
        delta2 = values - mean
        m2 += delta * delta2

        if progress is not None and (
            frame_index == 0
            or (frame_index + 1) % 500 == 0
        ):
            progress(frame_index + 1, None)

    if mean is None or m2 is None or max_projection is None or source_shape is None:
        raise ValueError(f"TIFF contains no frames: {tiff_path}")

    variance = m2 / float(max(frame_count - 1, 1))
    summary = SummaryImageSet(
        source_tiff=Path(tiff_path),
        frame_count=int(frame_count),
        source_shape=source_shape,
        source_dtype=source_dtype,
        mean=mean.astype(np.float32),
        std=np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
        max_projection=max_projection.astype(np.float32),
    )
    if progress is not None:
        progress(frame_count, frame_count)
    return summary


def write_summary_pngs(
    summary: SummaryImageSet,
    output_dir: Path,
    tiff_stem: str | None = None,
) -> dict[str, Path]:
    stem = tiff_stem or summary.source_tiff.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "summary_mean_png": save_png(summary.mean, output_dir / f"{stem}_summary_mean.png"),
        "summary_std_png": save_png(summary.std, output_dir / f"{stem}_summary_std.png"),
        "summary_max_png": save_png(summary.max_projection, output_dir / f"{stem}_summary_max.png"),
    }


def write_summary_manifest(
    summary: SummaryImageSet,
    output_dir: Path,
    image_paths: dict[str, Path],
    tiff_stem: str | None = None,
) -> Path:
    stem = tiff_stem or summary.source_tiff.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{stem}_summary_manifest.json"
    manifest = {
        "ok": True,
        "source_tiff": str(summary.source_tiff),
        "frame_count": summary.frame_count,
        "source_shape": [summary.source_shape[0], summary.source_shape[1]],
        "source_dtype": summary.source_dtype,
        "images": {key: str(path) for key, path in image_paths.items()},
        "display_ranges": {
            "mean": list(auto_display_range(summary.mean)),
            "std": list(auto_display_range(summary.std)),
            "max": list(auto_display_range(summary.max_projection)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_summary_outputs(summary: SummaryImageSet, output_dir: Path, tiff_stem: str | None = None) -> dict[str, Path]:
    image_paths = write_summary_pngs(summary, output_dir, tiff_stem=tiff_stem)
    manifest_path = write_summary_manifest(summary, output_dir, image_paths, tiff_stem=tiff_stem)
    return {**image_paths, "summary_manifest_path": manifest_path}
