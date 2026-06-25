"""Built-in motion-correction hook for Femtonics TIFF stacks."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .roi_core import (
        RoiBox,
        compute_roi_layout,
        load_tiff_info,
        metadata_path_for_tiff,
        parse_metadata,
        save_tiff_array_stack,
        validate_tiff_matches_metadata,
    )
    from .tiff_backend import iter_tiff_frames
except ImportError:  # pragma: no cover - supports direct script-style imports.
    from roi_core import (
        RoiBox,
        compute_roi_layout,
        load_tiff_info,
        metadata_path_for_tiff,
        parse_metadata,
        save_tiff_array_stack,
        validate_tiff_matches_metadata,
    )
    from tiff_backend import iter_tiff_frames


_TILE_FIRST_METHODS = {"tile_first", "per_tile", "legacy", "integer_tile"}
_SHARED_TEMPLATE_METHODS = {
    "shared_template",
    "shared_mean",
    "shared_mean_tile",
    "mean_template",
    "normcorre_shared",
}
_SHARED_TEMPLATE_RESIDUAL_METHODS = {
    "shared_template_residual",
    "shared_mean_residual",
    "shared_then_tile_residual",
    "shared_template_local_residual",
}


def _default_output_dir(tiff_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return tiff_path.parent / "exports" / tiff_path.stem


def _normalize_method(value: Any) -> str:
    method = str(value or "tile_first").strip().lower()
    if method in _TILE_FIRST_METHODS:
        return "tile_first"
    if method in _SHARED_TEMPLATE_METHODS:
        return "shared_template"
    if method in _SHARED_TEMPLATE_RESIDUAL_METHODS:
        return "shared_template_residual"
    raise ValueError(
        "Unknown motion-correction method "
        f"{value!r}. Expected one of "
        f"{sorted(_TILE_FIRST_METHODS | _SHARED_TEMPLATE_METHODS | _SHARED_TEMPLATE_RESIDUAL_METHODS)}."
    )


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "off", "false"}:
        return None
    return float(value)


def _fill_value_for(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values.reshape(-1)
    return float(np.median(finite)) if finite.size else 0.0


def _load_stack(tiff_path: Path) -> np.ndarray:
    frames = [np.asarray(frame) for frame in iter_tiff_frames(tiff_path)]
    if not frames:
        raise ValueError(f"TIFF contains no frames: {tiff_path}")
    first_shape = frames[0].shape
    first_dtype = frames[0].dtype
    if len(first_shape) != 2:
        raise ValueError(f"Motion correction expects grayscale 2D frames, got shape {first_shape}.")
    for index, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(f"Frame {index} shape {frame.shape} does not match first frame {first_shape}.")
        if frame.dtype != first_dtype:
            raise ValueError(f"Frame {index} dtype {frame.dtype} does not match first frame {first_dtype}.")
    return np.stack(frames, axis=0)


def _layout_boxes_for_tiff(tiff_path: Path, frame_shape: tuple[int, int]) -> tuple[list[RoiBox], str]:
    metadata_path = metadata_path_for_tiff(tiff_path)
    if metadata_path.exists():
        metadata = parse_metadata(metadata_path)
        info = load_tiff_info(tiff_path)
        validate_tiff_matches_metadata(info, metadata)
        return compute_roi_layout(metadata).boxes, "metadata_tiles"

    height, width = frame_shape
    return [
        RoiBox(
            ordinal=1,
            roi_index=1,
            row=0,
            column=0,
            left=0,
            upper=0,
            right=width,
            lower=height,
        )
    ], "whole_frame"


def estimate_integer_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    max_shift_px: int | None = None,
) -> tuple[int, int, float]:
    """Return the integer y/x shift to apply to moving so it aligns to reference."""
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape:
        raise ValueError(f"Shape mismatch: reference {ref.shape}, moving {mov.shape}.")
    if ref.ndim != 2:
        raise ValueError(f"Phase correlation expects 2D arrays, got {ref.ndim}D.")

    ref = ref - float(np.mean(ref))
    mov = mov - float(np.mean(mov))
    if not np.any(ref) or not np.any(mov):
        return 0, 0, 0.0

    cross_power = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    magnitude = np.abs(cross_power)
    cross_power = cross_power / np.maximum(magnitude, 1e-12)
    correlation = np.abs(np.fft.ifft2(cross_power))

    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    height, width = correlation.shape
    dy = int(peak_y - height if peak_y > height // 2 else peak_y)
    dx = int(peak_x - width if peak_x > width // 2 else peak_x)
    peak = float(correlation[peak_y, peak_x])

    if max_shift_px is not None and (abs(dy) > max_shift_px or abs(dx) > max_shift_px):
        return 0, 0, peak
    return dy, dx, peak


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(float(sigma) * 3)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * float(sigma) ** 2))
    return kernel / np.sum(kernel)


def _convolve_reflect(values: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    padded = np.pad(values, [(pad, pad) if index == axis else (0, 0) for index in range(values.ndim)], mode="reflect")
    return np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis, padded)


def _prepare_for_registration(values: np.ndarray, *, high_pass_sigma: float | None = None) -> np.ndarray:
    prepared = np.asarray(values, dtype=np.float64)
    if high_pass_sigma is not None and high_pass_sigma > 0:
        kernel = _gaussian_kernel1d(high_pass_sigma)
        baseline = _convolve_reflect(_convolve_reflect(prepared, kernel, axis=0), kernel, axis=1)
        prepared = prepared - baseline
    prepared = prepared - float(np.mean(prepared))
    return prepared


def _candidate_coordinates(shape: tuple[int, int], max_shift_px: int | None) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    if max_shift_px is None:
        return np.indices(shape)

    radius = max(0, int(max_shift_px))
    if radius >= height // 2 and radius >= width // 2:
        return np.indices(shape)

    y_allowed = np.concatenate((np.arange(0, min(radius + 1, height)), np.arange(max(height - radius, 0), height)))
    x_allowed = np.concatenate((np.arange(0, min(radius + 1, width)), np.arange(max(width - radius, 0), width)))
    coords_y, coords_x = np.meshgrid(np.unique(y_allowed), np.unique(x_allowed), indexing="ij")
    return coords_y, coords_x


def _signed_shift_coordinate(coordinate: float, size: int) -> float:
    return coordinate - size if coordinate > size / 2 else coordinate


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denominator = left - (2 * center) + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def estimate_phase_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    max_shift_px: int | None = None,
    subpixel: bool = False,
    high_pass_sigma: float | None = None,
    min_peak_ratio: float | None = None,
) -> dict[str, Any]:
    """Return the y/x shift to apply to moving so it aligns to reference."""
    ref = _prepare_for_registration(reference, high_pass_sigma=high_pass_sigma)
    mov = _prepare_for_registration(moving, high_pass_sigma=high_pass_sigma)
    if ref.shape != mov.shape:
        raise ValueError(f"Shape mismatch: reference {ref.shape}, moving {mov.shape}.")
    if ref.ndim != 2:
        raise ValueError(f"Phase correlation expects 2D arrays, got {ref.ndim}D.")
    if not np.any(ref) or not np.any(mov):
        return {
            "estimated_dy": 0.0,
            "estimated_dx": 0.0,
            "dy": 0.0,
            "dx": 0.0,
            "peak": 0.0,
            "peak_ratio": 0.0,
            "accepted": False,
        }

    cross_power = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    magnitude = np.abs(cross_power)
    cross_power = cross_power / np.maximum(magnitude, 1e-12)
    correlation = np.abs(np.fft.ifft2(cross_power))

    coords_y, coords_x = _candidate_coordinates(correlation.shape, max_shift_px)
    candidate_values = correlation[coords_y, coords_x]
    candidate_flat_index = int(np.argmax(candidate_values))
    peak_y = int(coords_y.reshape(-1)[candidate_flat_index])
    peak_x = int(coords_x.reshape(-1)[candidate_flat_index])
    peak = float(correlation[peak_y, peak_x])

    if candidate_values.size > 1:
        second_peak = float(np.partition(candidate_values.reshape(-1), -2)[-2])
        peak_ratio = peak / max(second_peak, 1e-12)
    else:
        peak_ratio = float("inf")

    height, width = correlation.shape
    offset_y = 0.0
    offset_x = 0.0
    if subpixel:
        offset_y = _parabolic_offset(
            float(correlation[(peak_y - 1) % height, peak_x]),
            peak,
            float(correlation[(peak_y + 1) % height, peak_x]),
        )
        offset_x = _parabolic_offset(
            float(correlation[peak_y, (peak_x - 1) % width]),
            peak,
            float(correlation[peak_y, (peak_x + 1) % width]),
        )

    estimated_dy = _signed_shift_coordinate(float(peak_y) + offset_y, height)
    estimated_dx = _signed_shift_coordinate(float(peak_x) + offset_x, width)
    if max_shift_px is not None:
        radius = float(max_shift_px)
        estimated_dy = float(np.clip(estimated_dy, -radius, radius))
        estimated_dx = float(np.clip(estimated_dx, -radius, radius))

    accepted = True
    if min_peak_ratio is not None and min_peak_ratio > 0:
        accepted = peak_ratio >= min_peak_ratio
    applied_dy = estimated_dy if accepted else 0.0
    applied_dx = estimated_dx if accepted else 0.0
    return {
        "estimated_dy": float(estimated_dy),
        "estimated_dx": float(estimated_dx),
        "dy": float(applied_dy),
        "dx": float(applied_dx),
        "peak": peak,
        "peak_ratio": float(peak_ratio),
        "accepted": bool(accepted),
    }


def apply_integer_shift(frame: np.ndarray, dy: int, dx: int, fill_value: float | int | None = None) -> np.ndarray:
    """Shift a 2D frame without wrap-around."""
    values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"Integer shift expects a 2D frame, got shape {values.shape}.")
    if dy == 0 and dx == 0:
        return values.copy()

    if fill_value is None:
        finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values.reshape(-1)
        fill_value = float(np.median(finite)) if finite.size else 0.0

    output = np.full(values.shape, fill_value, dtype=values.dtype)
    height, width = values.shape
    src_y0 = max(0, -dy)
    src_y1 = height - max(0, dy)
    dst_y0 = max(0, dy)
    dst_y1 = height - max(0, -dy)
    src_x0 = max(0, -dx)
    src_x1 = width - max(0, dx)
    dst_x0 = max(0, dx)
    dst_x1 = width - max(0, -dx)
    if src_y0 < src_y1 and src_x0 < src_x1:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return output


def apply_subpixel_shift(frame: np.ndarray, dy: float, dx: float, fill_value: float | int | None = None) -> np.ndarray:
    """Shift a 2D frame with bilinear interpolation and no wrap-around."""
    values = np.asarray(frame)
    if values.ndim != 2:
        raise ValueError(f"Subpixel shift expects a 2D frame, got shape {values.shape}.")

    rounded_dy = int(round(float(dy)))
    rounded_dx = int(round(float(dx)))
    if abs(float(dy) - rounded_dy) < 1e-8 and abs(float(dx) - rounded_dx) < 1e-8:
        return apply_integer_shift(values, rounded_dy, rounded_dx, fill_value)

    if fill_value is None:
        fill_value = _fill_value_for(values)

    height, width = values.shape
    yy, xx = np.indices(values.shape, dtype=np.float64)
    src_y = yy - float(dy)
    src_x = xx - float(dx)
    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    valid = (y0 >= 0) & (y1 < height) & (x0 >= 0) & (x1 < width)

    output = np.full(values.shape, fill_value, dtype=np.float64)
    if np.any(valid):
        wy = src_y[valid] - y0[valid]
        wx = src_x[valid] - x0[valid]
        top_left = values[y0[valid], x0[valid]].astype(np.float64)
        top_right = values[y0[valid], x1[valid]].astype(np.float64)
        bottom_left = values[y1[valid], x0[valid]].astype(np.float64)
        bottom_right = values[y1[valid], x1[valid]].astype(np.float64)
        output[valid] = (
            ((1 - wy) * (1 - wx) * top_left)
            + ((1 - wy) * wx * top_right)
            + (wy * (1 - wx) * bottom_left)
            + (wy * wx * bottom_right)
        )

    if np.issubdtype(values.dtype, np.integer):
        dtype_info = np.iinfo(values.dtype)
        output = np.clip(output, dtype_info.min, dtype_info.max)
    return output.astype(values.dtype, copy=False)


def auto_max_shift_px(boxes: list[RoiBox]) -> int:
    if not boxes:
        return 1
    smallest_tile_edge = min(min(box.width, box.height) for box in boxes)
    return max(1, min(5, int(smallest_tile_edge) // 8))


def _correct_tile_first(
    stack: np.ndarray,
    boxes: list[RoiBox],
    *,
    max_shift_px: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    corrected = np.empty_like(stack)
    corrected[0] = stack[0]
    references = {
        box.ordinal: np.asarray(stack[0, box.upper : box.lower, box.left : box.right])
        for box in boxes
    }
    shift_rows: list[dict[str, Any]] = []

    for frame_index in range(stack.shape[0]):
        corrected_frame = stack[frame_index].copy()
        for box in boxes:
            moving = np.asarray(stack[frame_index, box.upper : box.lower, box.left : box.right])
            dy, dx, peak = estimate_integer_shift(
                references[box.ordinal],
                moving,
                max_shift_px=max_shift_px,
            )
            fill = _fill_value_for(moving)
            corrected_frame[box.upper : box.lower, box.left : box.right] = apply_integer_shift(moving, dy, dx, fill)
            shift_rows.append(
                {
                    "method": "tile_first",
                    "frame": frame_index,
                    "roi_index": box.roi_index,
                    "dy": dy,
                    "dx": dx,
                    "peak": peak,
                }
            )
        corrected[frame_index] = corrected_frame
    return corrected, shift_rows


def _correct_shared_template(
    stack: np.ndarray,
    boxes: list[RoiBox],
    *,
    max_shift_px: int,
    subpixel: bool,
    min_peak_ratio: float | None,
    high_pass_sigma: float | None,
    template_iterations: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    working = stack.copy()
    shift_rows: list[dict[str, Any]] = []
    iterations = max(1, int(template_iterations))

    for iteration in range(iterations):
        reference = np.mean(working.astype(np.float64, copy=False), axis=0)
        corrected = np.empty_like(stack)
        for frame_index in range(stack.shape[0]):
            frame = working[frame_index]
            shift = estimate_phase_shift(
                reference,
                frame,
                max_shift_px=max_shift_px,
                subpixel=subpixel,
                high_pass_sigma=high_pass_sigma,
                min_peak_ratio=min_peak_ratio,
            )
            dy = float(shift["dy"])
            dx = float(shift["dx"])
            if not subpixel:
                dy = float(int(round(dy)))
                dx = float(int(round(dx)))
            corrected_frame = frame.copy()
            for box in boxes:
                moving = np.asarray(frame[box.upper : box.lower, box.left : box.right])
                fill = _fill_value_for(moving)
                if subpixel:
                    shifted = apply_subpixel_shift(moving, dy, dx, fill)
                else:
                    shifted = apply_integer_shift(moving, int(round(dy)), int(round(dx)), fill)
                corrected_frame[box.upper : box.lower, box.left : box.right] = shifted
            corrected[frame_index] = corrected_frame
            shift_rows.append(
                {
                    "method": "shared_template",
                    "iteration": iteration,
                    "frame": frame_index,
                    "roi_index": 0,
                    "dy": dy,
                    "dx": dx,
                    "estimated_dy": shift["estimated_dy"],
                    "estimated_dx": shift["estimated_dx"],
                    "peak": shift["peak"],
                    "peak_ratio": shift["peak_ratio"],
                    "accepted": shift["accepted"],
                }
            )
        working = corrected
    return working, shift_rows


def _correct_shared_template_residual(
    stack: np.ndarray,
    boxes: list[RoiBox],
    *,
    max_shift_px: int,
    subpixel: bool,
    min_peak_ratio: float | None,
    high_pass_sigma: float | None,
    template_iterations: int,
    residual_max_shift_px: int,
    residual_min_peak_ratio: float | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    shared_corrected, shared_rows = _correct_shared_template(
        stack,
        boxes,
        max_shift_px=max_shift_px,
        subpixel=subpixel,
        min_peak_ratio=min_peak_ratio,
        high_pass_sigma=high_pass_sigma,
        template_iterations=template_iterations,
    )
    shift_rows = []
    for row in shared_rows:
        staged_row = dict(row)
        staged_row["stage"] = "shared"
        shift_rows.append(staged_row)

    reference = np.mean(shared_corrected.astype(np.float64, copy=False), axis=0)
    corrected = np.empty_like(stack)
    gate = 1.10 if residual_min_peak_ratio is None else float(residual_min_peak_ratio)
    residual_radius = max(0, int(residual_max_shift_px))

    for frame_index in range(shared_corrected.shape[0]):
        frame = shared_corrected[frame_index]
        corrected_frame = frame.copy()
        for box in boxes:
            reference_tile = np.asarray(reference[box.upper : box.lower, box.left : box.right])
            moving = np.asarray(frame[box.upper : box.lower, box.left : box.right])
            shift = estimate_phase_shift(
                reference_tile,
                moving,
                max_shift_px=residual_radius,
                subpixel=False,
                high_pass_sigma=high_pass_sigma,
                min_peak_ratio=gate,
            )
            dy = int(round(float(shift["dy"])))
            dx = int(round(float(shift["dx"])))
            fill = _fill_value_for(moving)
            corrected_frame[box.upper : box.lower, box.left : box.right] = apply_integer_shift(moving, dy, dx, fill)
            shift_rows.append(
                {
                    "method": "shared_template_residual",
                    "stage": "residual",
                    "frame": frame_index,
                    "roi_index": box.roi_index,
                    "dy": dy,
                    "dx": dx,
                    "estimated_dy": shift["estimated_dy"],
                    "estimated_dx": shift["estimated_dx"],
                    "peak": shift["peak"],
                    "peak_ratio": shift["peak_ratio"],
                    "accepted": shift["accepted"],
                    "residual_max_shift_px": residual_radius,
                    "residual_min_peak_ratio": gate,
                }
            )
        corrected[frame_index] = corrected_frame

    return corrected, shift_rows


def correct_tiff_stack(
    tiff_path: Path,
    output_path: Path,
    *,
    max_shift_px: int | None = None,
    method: str = "tile_first",
    subpixel: bool = False,
    min_peak_ratio: float | None = None,
    high_pass_sigma: float | None = None,
    template_iterations: int = 1,
    residual_max_shift_px: int = 1,
    residual_min_peak_ratio: float | None = None,
) -> dict[str, Any]:
    stack = _load_stack(tiff_path)
    frame_count, height, width = stack.shape
    boxes, layout_source = _layout_boxes_for_tiff(tiff_path, (height, width))
    effective_max_shift_px = auto_max_shift_px(boxes) if max_shift_px is None else int(max_shift_px)
    normalized_method = _normalize_method(method)

    if frame_count <= 1:
        return {
            "ok": True,
            "motion_skipped": True,
            "reason": "Single-frame TIFF; motion correction is not meaningful.",
            "frame_count": frame_count,
            "layout_source": layout_source,
            "tile_count": len(boxes),
            "method": normalized_method,
        }

    if normalized_method == "tile_first":
        corrected, shift_rows = _correct_tile_first(
            stack,
            boxes,
            max_shift_px=effective_max_shift_px,
        )
    elif normalized_method == "shared_template":
        corrected, shift_rows = _correct_shared_template(
            stack,
            boxes,
            max_shift_px=effective_max_shift_px,
            subpixel=subpixel,
            min_peak_ratio=min_peak_ratio,
            high_pass_sigma=high_pass_sigma,
            template_iterations=template_iterations,
        )
    else:
        corrected, shift_rows = _correct_shared_template_residual(
            stack,
            boxes,
            max_shift_px=effective_max_shift_px,
            subpixel=subpixel,
            min_peak_ratio=min_peak_ratio,
            high_pass_sigma=high_pass_sigma,
            template_iterations=template_iterations,
            residual_max_shift_px=residual_max_shift_px,
            residual_min_peak_ratio=residual_min_peak_ratio,
        )

    moved_shift_count = sum(
        1
        for row in shift_rows
        if abs(float(row.get("dy", 0.0))) > 1e-8 or abs(float(row.get("dx", 0.0))) > 1e-8
    )
    accepted_shift_count = sum(1 for row in shift_rows if bool(row.get("accepted", True)))
    residual_rows = [row for row in shift_rows if row.get("stage") == "residual"]
    shared_rows = [row for row in shift_rows if row.get("stage") in {None, "shared"}]
    residual_moved_shift_count = sum(
        1
        for row in residual_rows
        if abs(float(row.get("dy", 0.0))) > 1e-8 or abs(float(row.get("dx", 0.0))) > 1e-8
    )
    residual_accepted_shift_count = sum(1 for row in residual_rows if bool(row.get("accepted", True)))
    shared_moved_shift_count = sum(
        1
        for row in shared_rows
        if abs(float(row.get("dy", 0.0))) > 1e-8 or abs(float(row.get("dx", 0.0))) > 1e-8
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_tiff_array_stack(output_path, corrected)
    source_metadata = metadata_path_for_tiff(tiff_path)
    if source_metadata.exists():
        shutil.copy2(source_metadata, metadata_path_for_tiff(output_path))

    return {
        "ok": True,
        "corrected_tiff": str(output_path),
        "frame_count": frame_count,
        "frame_width": width,
        "frame_height": height,
        "layout_source": layout_source,
        "tile_count": len(boxes),
        "max_shift_px": int(effective_max_shift_px),
        "requested_max_shift_px": "auto" if max_shift_px is None else int(max_shift_px),
        "method": normalized_method,
        "subpixel": bool(subpixel),
        "min_peak_ratio": min_peak_ratio,
        "high_pass_sigma": high_pass_sigma,
        "template_iterations": int(template_iterations),
        "residual_max_shift_px": int(residual_max_shift_px),
        "residual_min_peak_ratio": residual_min_peak_ratio,
        "accepted_shift_count": int(accepted_shift_count),
        "moved_shift_count": int(moved_shift_count),
        "shared_moved_shift_count": int(shared_moved_shift_count),
        "residual_accepted_shift_count": int(residual_accepted_shift_count),
        "residual_moved_shift_count": int(residual_moved_shift_count),
        "shifts": shift_rows,
    }


def write_motion_qc(result: dict[str, Any], output_dir: Path, tiff_stem: str) -> tuple[Path, Path | None]:
    qc_path = output_dir / f"{tiff_stem}_motion_qc.json"
    serializable = dict(result)
    shifts = serializable.pop("shifts", [])
    qc_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    shifts_path = None
    if shifts:
        shifts_path = output_dir / f"{tiff_stem}_motion_shifts.csv"
        preferred_fields = [
            "method",
            "stage",
            "iteration",
            "frame",
            "roi_index",
            "dy",
            "dx",
            "estimated_dy",
            "estimated_dx",
            "peak",
            "peak_ratio",
            "accepted",
            "residual_max_shift_px",
            "residual_min_peak_ratio",
        ]
        all_fields = {field for row in shifts for field in row}
        fieldnames = [field for field in preferred_fields if field in all_fields]
        fieldnames.extend(sorted(all_fields - set(fieldnames)))
        with shifts_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shifts)
    return qc_path, shifts_path


def motion_hook(
    tiff_path: Path,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    source = Path(tiff_path)
    target_dir = _default_output_dir(source, Path(output_dir) if output_dir is not None else None)
    target_dir.mkdir(parents=True, exist_ok=True)
    corrected_tiff = target_dir / f"{source.stem}_motion_corrected.tiff"
    requested_max_shift = kwargs.get("max_shift_px")
    if requested_max_shift is None or str(requested_max_shift).strip().lower() == "auto":
        max_shift_px = None
    else:
        max_shift_px = int(requested_max_shift)
    method = kwargs.get("method", kwargs.get("mode", "tile_first"))
    subpixel = _as_bool(kwargs.get("subpixel"), default=False)
    min_peak_ratio = _as_optional_float(kwargs.get("min_peak_ratio"))
    high_pass_sigma = _as_optional_float(kwargs.get("high_pass_sigma"))
    template_iterations = int(kwargs.get("template_iterations", kwargs.get("iterations", 1)))
    residual_max_shift_px = int(kwargs.get("residual_max_shift_px", 1))
    residual_min_peak_ratio = _as_optional_float(kwargs.get("residual_min_peak_ratio", 1.10))

    result = correct_tiff_stack(
        source,
        corrected_tiff,
        max_shift_px=max_shift_px,
        method=str(method),
        subpixel=subpixel,
        min_peak_ratio=min_peak_ratio,
        high_pass_sigma=high_pass_sigma,
        template_iterations=template_iterations,
        residual_max_shift_px=residual_max_shift_px,
        residual_min_peak_ratio=residual_min_peak_ratio,
    )
    qc_path, shifts_path = write_motion_qc(result, target_dir, source.stem)

    hook_result = dict(result)
    hook_result.pop("shifts", None)
    hook_result["qc_json"] = str(qc_path)
    if shifts_path is not None:
        hook_result["shifts_csv"] = str(shifts_path)
    return hook_result


run = motion_hook
