"""Pixel-correlation motion QC for TIFF stacks."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .summary_images import compute_summary_images, save_png
    from .tiff_backend import iter_tiff_frames
except ImportError:  # pragma: no cover - supports direct script-style imports.
    from summary_images import compute_summary_images, save_png
    from tiff_backend import iter_tiff_frames


DEFAULT_THRESHOLD_MAD = 3.0


@dataclass(frozen=True)
class PixelCorrelationRow:
    frame_index: int
    corr_to_mean: float
    motion_score: float
    mean_abs_difference: float
    is_heavy_motion: bool = False
    rank_lowest_correlation: int = 0


@dataclass(frozen=True)
class PixelCorrelationResult:
    source_tiff: Path
    frame_count: int
    source_shape: tuple[int, int]
    source_dtype: str
    mean_image: np.ndarray
    threshold_correlation: float
    threshold_method: str
    threshold_mad: float
    rows: list[PixelCorrelationRow]

    @property
    def heavy_motion_count(self) -> int:
        return sum(1 for row in self.rows if row.is_heavy_motion)

    @property
    def min_correlation_row(self) -> PixelCorrelationRow:
        return min(self.rows, key=lambda row: row.corr_to_mean)


def _default_output_dir(tiff_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return tiff_path.parent / "exports" / tiff_path.stem


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), int(size))
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return max(0, int(box[2] - box[0])), max(0, int(box[3] - box[1]))


def pearson_correlation(reference: np.ndarray, frame: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    values = np.asarray(frame, dtype=np.float64)
    if ref.shape != values.shape:
        raise ValueError(f"Frame shape {values.shape} does not match reference shape {ref.shape}.")

    ref_flat = ref.reshape(-1)
    values_flat = values.reshape(-1)
    finite = np.isfinite(ref_flat) & np.isfinite(values_flat)
    if not np.any(finite):
        return 0.0

    ref_flat = ref_flat[finite]
    values_flat = values_flat[finite]
    ref_centered = ref_flat - float(np.mean(ref_flat))
    values_centered = values_flat - float(np.mean(values_flat))
    ref_power = float(np.dot(ref_centered, ref_centered))
    values_power = float(np.dot(values_centered, values_centered))
    denominator = float(np.sqrt(ref_power * values_power))
    if denominator <= 1e-12:
        difference = float(np.mean(np.abs(values_flat - ref_flat)))
        return 1.0 if difference <= 1e-12 else 0.0
    corr = float(np.dot(ref_centered, values_centered) / denominator)
    return float(np.clip(corr, -1.0, 1.0))


def _robust_low_correlation_threshold(values: np.ndarray, threshold_mad: float) -> tuple[float, str]:
    if values.size == 0:
        return 0.0, "median_minus_mad"
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1e-12:
        return median, "median"
    return median - (float(threshold_mad) * robust_sigma), "median_minus_mad"


def compute_pixel_correlation_qc(
    tiff_path: Path,
    *,
    threshold_mad: float = DEFAULT_THRESHOLD_MAD,
    correlation_threshold: float | None = None,
) -> PixelCorrelationResult:
    summary = compute_summary_images(tiff_path)
    reference = summary.mean.astype(np.float64, copy=False)
    rows: list[PixelCorrelationRow] = []

    for frame_index, raw_frame in enumerate(iter_tiff_frames(tiff_path)):
        frame = np.asarray(raw_frame)
        if frame.ndim != 2:
            raise ValueError(f"Pixel correlation expects grayscale 2D frames, got shape {frame.shape}.")
        if frame.shape != reference.shape:
            raise ValueError(f"Frame {frame_index} shape {frame.shape} does not match reference {reference.shape}.")
        corr = pearson_correlation(reference, frame)
        difference = float(np.mean(np.abs(frame.astype(np.float64, copy=False) - reference)))
        rows.append(
            PixelCorrelationRow(
                frame_index=int(frame_index),
                corr_to_mean=corr,
                motion_score=1.0 - corr,
                mean_abs_difference=difference,
            )
        )

    if not rows:
        raise ValueError(f"TIFF contains no frames: {tiff_path}")

    correlations = np.asarray([row.corr_to_mean for row in rows], dtype=np.float64)
    if correlation_threshold is None:
        threshold, threshold_method = _robust_low_correlation_threshold(correlations, threshold_mad)
    else:
        threshold = float(correlation_threshold)
        threshold_method = "fixed_correlation"

    ranked_frame_indices = {
        row.frame_index: rank
        for rank, row in enumerate(sorted(rows, key=lambda item: item.corr_to_mean), start=1)
    }
    scored_rows = [
        PixelCorrelationRow(
            frame_index=row.frame_index,
            corr_to_mean=row.corr_to_mean,
            motion_score=row.motion_score,
            mean_abs_difference=row.mean_abs_difference,
            is_heavy_motion=bool(row.corr_to_mean < threshold),
            rank_lowest_correlation=ranked_frame_indices[row.frame_index],
        )
        for row in rows
    ]

    return PixelCorrelationResult(
        source_tiff=Path(tiff_path),
        frame_count=summary.frame_count,
        source_shape=summary.source_shape,
        source_dtype=summary.source_dtype,
        mean_image=summary.mean,
        threshold_correlation=float(threshold),
        threshold_method=threshold_method,
        threshold_mad=float(threshold_mad),
        rows=scored_rows,
    )


def write_correlation_scores_csv(result: PixelCorrelationResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "corr_to_mean",
                "motion_score",
                "mean_abs_difference",
                "threshold_correlation",
                "is_heavy_motion",
                "rank_lowest_correlation",
            ],
        )
        writer.writeheader()
        for row in result.rows:
            writer.writerow(
                {
                    "frame_index": row.frame_index,
                    "corr_to_mean": f"{row.corr_to_mean:.10g}",
                    "motion_score": f"{row.motion_score:.10g}",
                    "mean_abs_difference": f"{row.mean_abs_difference:.10g}",
                    "threshold_correlation": f"{result.threshold_correlation:.10g}",
                    "is_heavy_motion": "true" if row.is_heavy_motion else "false",
                    "rank_lowest_correlation": row.rank_lowest_correlation,
                }
            )
    return output_path


def _plot_bounds(correlations: np.ndarray, threshold: float) -> tuple[float, float]:
    lower = float(np.min(correlations))
    upper = float(np.max(correlations))
    lower = min(lower, float(threshold))
    span = max(upper - lower, 1e-6)
    padding = max(span * 0.12, 0.002)
    y_min = max(-1.0, lower - padding)
    y_max = min(1.0, upper + padding)
    if y_max - y_min < 0.01:
        center = (y_min + y_max) / 2.0
        y_min = max(-1.0, center - 0.005)
        y_max = min(1.0, center + 0.005)
    if y_max <= y_min:
        y_min, y_max = -1.0, 1.0
    return y_min, y_max


def _map_x(frame_index: int, frame_count: int, left: int, width: int) -> int:
    if frame_count <= 1:
        return left
    return int(round(left + (float(frame_index) / float(frame_count - 1)) * width))


def _map_y(value: float, y_min: float, y_max: float, top: int, height: int) -> int:
    fraction = (float(value) - y_min) / max(y_max - y_min, 1e-12)
    return int(round(top + (1.0 - fraction) * height))


def _draw_dashed_horizontal(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    left: int,
    right: int,
    fill: tuple[int, int, int],
) -> None:
    dash = 8
    gap = 6
    x = left
    while x < right:
        draw.line([(x, y), (min(x + dash, right), y)], fill=fill, width=2)
        x += dash + gap


def save_correlation_plot_png(
    result: PixelCorrelationResult,
    output_path: Path,
    *,
    width: int = 1200,
    height: int = 680,
) -> Path:
    width = max(720, int(width))
    height = max(420, int(height))
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = _font(22, bold=True)
    label_font = _font(15)
    tick_font = _font(13)
    caption_font = _font(14, bold=True)

    left = 92
    right_margin = 34
    top = 78
    bottom = 88
    plot_width = width - left - right_margin
    plot_height = height - top - bottom
    plot_right = left + plot_width
    plot_bottom = top + plot_height

    correlations = np.asarray([row.corr_to_mean for row in result.rows], dtype=np.float64)
    y_min, y_max = _plot_bounds(correlations, result.threshold_correlation)

    title = "Frame correlation to mean image"
    subtitle = (
        f"Lowest correlation: frame {result.min_correlation_row.frame_index} = "
        f"{result.min_correlation_row.corr_to_mean:.4f}; "
        f"heavy frames: {result.heavy_motion_count}/{result.frame_count}"
    )
    draw.text((left, 26), title, fill=(17, 24, 39), font=title_font)
    draw.text((left, 54), subtitle, fill=(75, 85, 99), font=label_font)

    draw.rectangle([left, top, plot_right, plot_bottom], outline=(31, 41, 55), width=1)
    for tick_index in range(6):
        value = y_min + ((y_max - y_min) * tick_index / 5.0)
        y = _map_y(value, y_min, y_max, top, plot_height)
        draw.line([(left, y), (plot_right, y)], fill=(229, 231, 235), width=1)
        text = f"{value:.4f}" if y_max - y_min < 0.1 else f"{value:.2f}"
        text_w, text_h = _text_size(draw, text, tick_font)
        draw.text((left - text_w - 10, y - text_h // 2), text, fill=(55, 65, 81), font=tick_font)

    threshold_y = _map_y(result.threshold_correlation, y_min, y_max, top, plot_height)
    if top <= threshold_y <= plot_bottom:
        _draw_dashed_horizontal(
            draw,
            y=threshold_y,
            left=left,
            right=plot_right,
            fill=(220, 38, 38),
        )
        draw.text(
            (plot_right - 186, max(top + 4, threshold_y - 24)),
            f"threshold {result.threshold_correlation:.4f}",
            fill=(153, 27, 27),
            font=tick_font,
        )

    x_tick_count = min(6, max(2, result.frame_count))
    for tick_index in range(x_tick_count):
        if x_tick_count == 1:
            frame_index = 0
        else:
            frame_index = int(round((result.frame_count - 1) * tick_index / float(x_tick_count - 1)))
        x = _map_x(frame_index, result.frame_count, left, plot_width)
        draw.line([(x, plot_bottom), (x, plot_bottom + 6)], fill=(55, 65, 81), width=1)
        text = str(frame_index)
        text_w, _text_h = _text_size(draw, text, tick_font)
        draw.text((x - text_w // 2, plot_bottom + 12), text, fill=(55, 65, 81), font=tick_font)

    points = [
        (
            _map_x(row.frame_index, result.frame_count, left, plot_width),
            _map_y(row.corr_to_mean, y_min, y_max, top, plot_height),
        )
        for row in result.rows
    ]
    if len(points) >= 2:
        draw.line(points, fill=(37, 99, 235), width=2)
    elif points:
        x, y = points[0]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(37, 99, 235))

    for row in result.rows:
        if not row.is_heavy_motion:
            continue
        x = _map_x(row.frame_index, result.frame_count, left, plot_width)
        y = _map_y(row.corr_to_mean, y_min, y_max, top, plot_height)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(220, 38, 38), outline=(127, 29, 29))

    x_label = "Frame"
    x_w, x_h = _text_size(draw, x_label, caption_font)
    draw.text((left + (plot_width - x_w) // 2, height - x_h - 24), x_label, fill=(17, 24, 39), font=caption_font)
    y_label = "Pearson correlation coefficient"
    draw.text((left, top - 24), y_label, fill=(17, 24, 39), font=caption_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def write_pixel_correlation_qc_json(
    result: PixelCorrelationResult,
    output_path: Path,
    *,
    output_paths: dict[str, Path],
) -> Path:
    min_row = result.min_correlation_row
    payload = {
        "ok": True,
        "source_tiff": str(result.source_tiff),
        "frame_count": result.frame_count,
        "source_shape": [result.source_shape[0], result.source_shape[1]],
        "source_dtype": result.source_dtype,
        "reference": "mean_image",
        "metric": "pearson_correlation_to_mean_image",
        "threshold_method": result.threshold_method,
        "threshold_mad": result.threshold_mad,
        "threshold_correlation": result.threshold_correlation,
        "heavy_motion_count": result.heavy_motion_count,
        "min_correlation_frame": min_row.frame_index,
        "min_correlation": min_row.corr_to_mean,
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_pixel_correlation_outputs(
    tiff_path: Path,
    output_dir: Path | None = None,
    *,
    threshold_mad: float = DEFAULT_THRESHOLD_MAD,
    correlation_threshold: float | None = None,
    plot_width: int = 1200,
    plot_height: int = 680,
) -> dict[str, Any]:
    source = Path(tiff_path)
    target_dir = _default_output_dir(source, output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    result = compute_pixel_correlation_qc(
        source,
        threshold_mad=threshold_mad,
        correlation_threshold=correlation_threshold,
    )

    scores_csv = target_dir / f"{source.stem}_pixel_correlation_scores.csv"
    plot_png = target_dir / f"{source.stem}_pixel_correlation_plot.png"
    mean_png = target_dir / f"{source.stem}_pixel_correlation_mean_image.png"
    qc_json = target_dir / f"{source.stem}_pixel_correlation_qc.json"

    write_correlation_scores_csv(result, scores_csv)
    save_correlation_plot_png(result, plot_png, width=plot_width, height=plot_height)
    save_png(result.mean_image, mean_png)
    write_pixel_correlation_qc_json(
        result,
        qc_json,
        output_paths={
            "correlation_scores_csv": scores_csv,
            "correlation_plot_png": plot_png,
            "mean_image_png": mean_png,
        },
    )

    min_row = result.min_correlation_row
    return {
        "ok": True,
        "frame_count": result.frame_count,
        "frame_width": result.source_shape[1],
        "frame_height": result.source_shape[0],
        "source_dtype": result.source_dtype,
        "reference": "mean_image",
        "metric": "pearson_correlation_to_mean_image",
        "threshold_method": result.threshold_method,
        "threshold_mad": result.threshold_mad,
        "threshold_correlation": result.threshold_correlation,
        "heavy_motion_count": result.heavy_motion_count,
        "min_correlation_frame": min_row.frame_index,
        "min_correlation": min_row.corr_to_mean,
        "correlation_scores_csv": str(scores_csv),
        "correlation_plot_png": str(plot_png),
        "mean_image_png": str(mean_png),
        "correlation_qc_json": str(qc_json),
    }


def motion_hook(
    tiff_path: Path,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    threshold_mad = float(kwargs.get("threshold_mad", DEFAULT_THRESHOLD_MAD))
    raw_threshold = kwargs.get("correlation_threshold")
    correlation_threshold = None if raw_threshold is None or str(raw_threshold).strip() == "" else float(raw_threshold)
    plot_width = int(kwargs.get("plot_width", 1200))
    plot_height = int(kwargs.get("plot_height", 680))
    return write_pixel_correlation_outputs(
        Path(tiff_path),
        Path(output_dir) if output_dir is not None else None,
        threshold_mad=threshold_mad,
        correlation_threshold=correlation_threshold,
        plot_width=plot_width,
        plot_height=plot_height,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pixel-correlation motion QC for a TIFF stack.")
    parser.add_argument("tiff", type=Path, help="Input TIFF stack.")
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Defaults to exports/<tiff_stem>/.")
    parser.add_argument(
        "--threshold-mad",
        type=float,
        default=DEFAULT_THRESHOLD_MAD,
        help="Low-correlation threshold in robust MAD units. Default: 3.0.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=None,
        help="Optional fixed correlation threshold for heavy-motion frames.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_pixel_correlation_outputs(
        args.tiff,
        args.output,
        threshold_mad=args.threshold_mad,
        correlation_threshold=args.correlation_threshold,
    )
    print(f"[OK] pixel correlation plot: {result['correlation_plot_png']}")
    print(f"[OK] pixel correlation scores: {result['correlation_scores_csv']}")
    print(f"[OK] pixel correlation QC: {result['correlation_qc_json']}")
    return 0


run = motion_hook


if __name__ == "__main__":
    raise SystemExit(main())
