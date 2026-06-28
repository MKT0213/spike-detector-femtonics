from __future__ import annotations

from dataclasses import dataclass, replace
import io
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from detector.core import gui_main
from detector.core.models import SpikeRecord, TraceCuration
from detector.core.render_normalization import (
    NORMALIZATION_PERCENT_DFF,
    PERCENT_DFF_SCALE_LABEL,
    finite_median,
    normalize_values_with_f0,
)
from detector.core.trace_identity import split_trace_identity

from .render_options import CHANNEL_LABELS, MARKER_STYLE_BAR, RenderOptions


CHANNEL_COLORS: Dict[str, str] = {
    "raw": "#455a64",
    "corrected_source": "#607d8b",
    "corrected": "#2979ff",
    "baseline": "#ff6d00",
    "correction_baseline": "#f59e0b",
    "highpass": "#00acc1",
    "lowpass": "#ef6c00",
    "denoised": "#8e24aa",
    "denoise_residual": "#d81b60",
}

PLATEAU_MARKER_COLORS: Dict[str, str] = {
    "plateau": "#c62828",
    "long_plateau": "#c62828",
    "short_plateau": "#1565c0",
    "burst_plateau": "#7c3aed",
}

BAR_MARKER_MAX_SIGNAL_HEIGHT: float = 18.0
_BAR_MARKER_SPIKE_HEIGHT_FRACTION: float = 0.12
PLATEAU_BAR_LINE_WIDTH: float = 4.2
ALIGNED_PANEL_PLATEAU_BAR_LINE_WIDTH: float = 3.0
ALIGNED_PANEL_MARKER_LANE_FRACTION: float = 0.22
SPIKE_HOLLOW_MARKER_SIZE: float = 8.0
ALIGNED_PANEL_SPIKE_HOLLOW_MARKER_SIZE: float = 24.0


@dataclass
class PreparedTrace:
    trace_name: str
    time: np.ndarray
    arrays: Dict[str, Optional[np.ndarray]]
    masks: Dict[str, Optional[np.ndarray]]
    spikes: List[SpikeRecord]
    startup_idx: int = 0
    window_offset: int = 0

    def array(self, channel: str) -> Optional[np.ndarray]:
        value = self.arrays.get(channel)
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        n = min(arr.size, self.time.size)
        if n <= 0:
            return None
        return arr[:n]

    def mask(self, name: str) -> Optional[np.ndarray]:
        value = self.masks.get(name)
        if value is None:
            return None
        arr = np.asarray(value, dtype=bool)
        n = min(arr.size, self.time.size)
        if n <= 0:
            return None
        return arr[:n]


def available_channel_ids(state: TraceCuration) -> List[str]:
    channels: List[str] = []
    for channel, value in [
        ("raw", state.raw),
        ("corrected_source", state.corrected_source),
        ("corrected", state.corrected),
        ("baseline", state.baseline),
        ("correction_baseline", state.correction_baseline),
        ("highpass", state.highpass_trace),
        ("lowpass", state.lowpass_trace),
        ("denoised", state.hybrid_denoised),
    ]:
        if value is not None and np.asarray(value).size > 0:
            channels.append(channel)
    if (
        state.hybrid_denoised is not None
        and np.asarray(state.hybrid_denoised).shape == np.asarray(state.corrected).shape
    ):
        channels.append("denoise_residual")
    if "corrected" not in channels:
        channels.append("corrected")
    return channels


def prepare_trace_for_render(
    state: TraceCuration,
    detection_params: object,
    render_options: RenderOptions,
    *,
    apply_trace_window: bool,
) -> PreparedTrace:
    time = np.asarray(state.time, dtype=float)
    corrected = np.asarray(state.corrected, dtype=float)
    n = min(time.size, corrected.size)
    time = time[:n]
    startup_idx = gui_main._startup_exclusion_index(time, float(detection_params.startup_exclusion_ms))

    def _trim(values: object, dtype: object = float) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim != 1:
            return arr
        source = arr[:n]
        return source[startup_idx:]

    denoised_trim = _trim(state.hybrid_denoised)
    corrected_trim = corrected[startup_idx:]
    residual_trim = None
    if denoised_trim is not None and denoised_trim.shape == corrected_trim.shape:
        residual_trim = corrected_trim - denoised_trim

    prepared = PreparedTrace(
        trace_name=state.trace_name,
        time=time[startup_idx:],
        arrays={
            "raw": _trim(state.raw),
            "corrected_source": _trim(state.corrected_source),
            "corrected": corrected_trim,
            "baseline": _trim(state.baseline),
            "correction_baseline": _trim(state.correction_baseline),
            "highpass": _trim(state.highpass_trace),
            "lowpass": _trim(state.lowpass_trace),
            "denoised": denoised_trim,
            "denoise_residual": residual_trim,
            "debug_threshold": _trim(state.debug_threshold),
        },
        masks={
            "plateau": _trim(state.plateau_mask, bool),
            "long_plateau": _trim(state.long_plateau_mask, bool),
            "short_plateau": _trim(state.short_plateau_mask, bool),
            "burst_plateau": _trim(state.burst_plateau_mask, bool),
            "artifact": _trim(state.artifact_mask, bool),
        },
        spikes=gui_main._adjust_spikes_for_startup(state.spikes, startup_idx, n),
        startup_idx=startup_idx,
    )
    if apply_trace_window:
        return crop_prepared_trace(prepared, render_options)
    return prepared


def crop_prepared_trace(data: PreparedTrace, render_options: RenderOptions) -> PreparedTrace:
    window = render_options.trace_window
    if not window.enabled or data.time.size == 0:
        return data
    start = float(window.start_time) if window.start_time is not None else float(data.time[0])
    end = float(window.end_time) if window.end_time is not None else float(data.time[-1])
    if end < start:
        start, end = end, start

    keep = np.flatnonzero((data.time >= start) & (data.time <= end))
    if keep.size == 0:
        return PreparedTrace(data.trace_name, np.array([], dtype=float), {}, {}, [], data.startup_idx, data.window_offset)
    lo = int(keep[0])
    hi = int(keep[-1]) + 1

    def _crop_arr(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values)
        return arr[lo:min(hi, arr.size)]

    spikes: List[SpikeRecord] = []
    for spike in data.spikes:
        idx = int(spike.spike_index)
        if lo <= idx < hi and start <= float(spike.spike_time) <= end:
            spikes.append(
                replace(
                    spike,
                    spike_index=idx - lo,
                    candidate_index=max(0, int(spike.candidate_index) - lo),
                )
            )

    return PreparedTrace(
        trace_name=data.trace_name,
        time=data.time[lo:hi],
        arrays={key: _crop_arr(value) for key, value in data.arrays.items()},
        masks={key: _crop_arr(value) for key, value in data.masks.items()},
        spikes=spikes,
        startup_idx=data.startup_idx,
        window_offset=data.window_offset + lo,
    )


def filtered_spikes(data: PreparedTrace, render_options: RenderOptions) -> List[SpikeRecord]:
    if data.time.size == 0:
        return []
    if not render_options.trace_window.enabled:
        return list(data.spikes)
    start = float(render_options.trace_window.start_time) if render_options.trace_window.start_time is not None else float(data.time[0])
    end = float(render_options.trace_window.end_time) if render_options.trace_window.end_time is not None else float(data.time[-1])
    if end < start:
        start, end = end, start
    return [spike for spike in data.spikes if start <= float(spike.spike_time) <= end]


def plateau_regions(data: PreparedTrace, render_options: RenderOptions, *, filtered: bool = True) -> List[Tuple[int, int]]:
    mask = data.mask("plateau")
    if mask is None or data.time.size == 0:
        return []
    regions = gui_main._contiguous_true_regions(mask)
    if not filtered or not render_options.trace_window.enabled:
        return regions
    start = float(render_options.trace_window.start_time) if render_options.trace_window.start_time is not None else float(data.time[0])
    end = float(render_options.trace_window.end_time) if render_options.trace_window.end_time is not None else float(data.time[-1])
    if end < start:
        start, end = end, start
    selected: List[Tuple[int, int]] = []
    for region_start, region_end in regions:
        idx = max(0, min(int(region_start), data.time.size - 1))
        if start <= float(data.time[idx]) <= end:
            selected.append((region_start, region_end))
    return selected


def _ensure_app() -> None:
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication([])


def _finalize_widget(container: QtWidgets.QWidget, width: int, height: int) -> QtGui.QPixmap:
    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _set_y_range(plot: object, arrays: Iterable[np.ndarray]) -> None:
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    for arr in arrays:
        data = np.asarray(arr, dtype=float)
        if data.size == 0:
            continue
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        arr_min = float(np.min(finite))
        arr_max = float(np.max(finite))
        y_min = arr_min if y_min is None else min(y_min, arr_min)
        y_max = arr_max if y_max is None else max(y_max, arr_max)
    if y_min is None or y_max is None:
        return
    pad = max(1e-3, (y_max - y_min) * 0.10) if y_max > y_min else 1e-3
    plot.setYRange(y_min - pad, y_max + pad, padding=0)


def _nice_floor(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(numeric) or numeric <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(numeric))
    scale = 10.0 ** exponent
    fraction = numeric / scale
    if fraction >= 5.0:
        nice = 5.0
    elif fraction >= 2.0:
        nice = 2.0
    else:
        nice = 1.0
    return float(nice * scale)


def _format_number(value: float) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1.0e-9:
        return str(int(round(numeric)))
    if abs(numeric) >= 1.0:
        return f"{numeric:.1f}".rstrip("0").rstrip(".")
    return f"{numeric:.2g}"


def _format_time_label(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1.0:
        ms = seconds * 1000.0
        return f"{_format_number(ms)} ms"
    return f"{_format_number(seconds)} s"


def _recording_time_scale(time_values: Sequence[np.ndarray]) -> Tuple[float, float, str, float, float, bool]:
    finite_times: List[np.ndarray] = []
    for time in time_values:
        arr = np.asarray(time, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_times.append(arr)
    if not finite_times:
        return 1.0, 1.0, "1 s", 0.0, 1.0, False
    joined = np.concatenate(finite_times)
    x_min = float(np.min(joined))
    x_max = float(np.max(joined))
    span_native = max(0.0, x_max - x_min)
    diffs = np.diff(np.sort(joined))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    median_dt = float(np.median(diffs)) if diffs.size else float("nan")
    direct_hz = 1.0 / median_dt if np.isfinite(median_dt) and median_dt > 0.0 else float("inf")
    is_ms = gui_main._time_axis_is_ms(np.asarray([x_min, x_max], dtype=float)) or (
        np.isfinite(median_dt) and median_dt >= 0.001 and direct_hz < 5.0
    )
    span_seconds = span_native / 1000.0 if is_ms else span_native
    if not np.isfinite(span_seconds) or span_seconds <= 0.0:
        span_seconds = 1.0
    scale_seconds = _nice_floor(span_seconds * 0.25)
    scale_native = scale_seconds * 1000.0 if is_ms else scale_seconds
    return scale_seconds, scale_native, _format_time_label(scale_seconds), x_min, x_max, is_ms


def robust_percent_scale(values: object, scale_label: str = PERCENT_DFF_SCALE_LABEL) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        q5 = -1.0
        q95 = 1.0
    else:
        q5, q95 = [float(value) for value in np.percentile(finite, [5.0, 95.0])]
    positive = max(0.0, float(q95))
    negative = max(0.0, -float(q5))
    polarity = "negative" if negative > positive else "positive"
    dominant = max(positive, negative, 1.0e-9)
    bar_value = _nice_floor(dominant)
    label_value = _format_number(bar_value)
    sign = "−" if polarity == "negative" else ""
    return {
        "value": float(bar_value),
        "label": f"{sign}{label_value}{scale_label}",
        "polarity": polarity,
        "robust_p05": float(q5),
        "robust_p95": float(q95),
    }


def _prepared_percent_dff(
    data: PreparedTrace,
    state: TraceCuration,
    mode: str,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    corrected = data.array("corrected")
    if corrected is None:
        return None, {}
    if str(mode) != NORMALIZATION_PERCENT_DFF:
        return corrected.copy(), {"normalization_mode": str(mode), "unit": "native", "f0_source": "", "f0_median": float("nan")}

    baseline = data.array("correction_baseline")
    if baseline is not None and baseline.shape == corrected.shape:
        finite = baseline[np.isfinite(baseline) & (np.abs(baseline) > 1.0e-12)]
        if finite.size:
            f0 = baseline.copy()
            f0_source = "correction_baseline"
            f0_median = float(np.median(finite))
            return normalize_values_with_f0(corrected, f0), {
                "normalization_mode": NORMALIZATION_PERCENT_DFF,
                "unit": PERCENT_DFF_SCALE_LABEL,
                "f0_source": f0_source,
                "f0_median": f0_median,
            }

    for source_name in ("raw", "corrected_source"):
        median = finite_median(getattr(state, source_name, None))
        if np.isfinite(median) and abs(float(median)) > 1.0e-12:
            f0 = np.full(corrected.size, float(median), dtype=float)
            return normalize_values_with_f0(corrected, f0), {
                "normalization_mode": NORMALIZATION_PERCENT_DFF,
                "unit": PERCENT_DFF_SCALE_LABEL,
                "f0_source": source_name,
                "f0_median": float(median),
            }
    return normalize_values_with_f0(corrected, np.ones(corrected.size, dtype=float)), {
        "normalization_mode": NORMALIZATION_PERCENT_DFF,
        "unit": PERCENT_DFF_SCALE_LABEL,
        "f0_source": "unit_fallback",
        "f0_median": 1.0,
    }


def _aligned_panel_f0(
    data: PreparedTrace,
    state: TraceCuration,
    size: int,
) -> Tuple[np.ndarray, str, float]:
    baseline = data.array("correction_baseline")
    if baseline is not None and baseline.size >= size:
        values = np.asarray(baseline[:size], dtype=float)
        finite = values[np.isfinite(values) & (np.abs(values) > 1.0e-12)]
        if finite.size:
            return values, "correction_baseline", float(np.median(finite))

    for source_name in ("raw", "corrected_source"):
        median = finite_median(getattr(state, source_name, None))
        if np.isfinite(median) and abs(float(median)) > 1.0e-12:
            return np.full(size, float(median), dtype=float), source_name, float(median)

    return np.ones(size, dtype=float), "unit_fallback", 1.0


def _prepared_aligned_panel_channel(
    data: PreparedTrace,
    state: TraceCuration,
    mode: str,
    preferred_channel: str,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    preferred = str(preferred_channel or "denoised")
    if str(mode) != NORMALIZATION_PERCENT_DFF and preferred == "raw":
        candidates = ["raw"]
    else:
        candidates = [preferred, "denoised", "corrected", "raw"]
    channel = ""
    values: Optional[np.ndarray] = None
    for candidate in candidates:
        if candidate in CHANNEL_LABELS:
            arr = data.array(candidate)
            if arr is not None and arr.size:
                channel = candidate
                values = np.asarray(arr, dtype=float)
                break
    if values is None:
        return None, {}

    if str(mode) != NORMALIZATION_PERCENT_DFF:
        return values.copy(), {
            "normalization_mode": str(mode),
            "unit": "native",
            "source_channel": channel,
            "source_label": CHANNEL_LABELS.get(channel, channel),
            "f0_source": "",
            "f0_median": float("nan"),
        }

    f0, f0_source, f0_median = _aligned_panel_f0(data, state, int(values.size))
    panel_values = values.copy()
    if channel in {"raw", "corrected_source", "correction_baseline", "baseline"}:
        panel_values = panel_values - f0[: panel_values.size]
    return normalize_values_with_f0(panel_values, f0), {
        "normalization_mode": NORMALIZATION_PERCENT_DFF,
        "unit": "dF/F0 (%)",
        "source_channel": channel,
        "source_label": CHANNEL_LABELS.get(channel, channel),
        "f0_source": f0_source,
        "f0_median": f0_median,
    }


PanelEntry = Tuple[TraceCuration, PreparedTrace, np.ndarray, Dict[str, object]]


def _nice_panel_y_range(
    values: Sequence[np.ndarray],
    fallback: Tuple[float, float],
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    padding_fraction: float = 0.15,
    min_span: float = 0.0,
    include_zero: bool = True,
) -> Tuple[float, float]:
    chunks = [np.asarray(value, dtype=float).reshape(-1) for value in values if np.asarray(value).size]
    if not chunks:
        return fallback
    joined = np.concatenate(chunks)
    finite = joined[np.isfinite(joined)]
    if finite.size <= 0:
        return fallback
    low_pct = min(max(float(low_percentile), 0.0), 99.0)
    high_pct = min(max(float(high_percentile), low_pct + 0.1), 100.0)
    low = float(np.percentile(finite, low_pct))
    high = float(np.percentile(finite, high_pct))
    tail_low = float(np.percentile(finite, 0.1))
    tail_high = float(np.percentile(finite, 99.9))
    low = min(low, tail_low)
    high = max(high, tail_high)
    min_event_run = 3
    for chunk in chunks:
        arr = np.asarray(chunk, dtype=float).reshape(-1)
        finite_chunk = np.isfinite(arr)
        if not np.any(finite_chunk):
            continue
        high_mask = finite_chunk & (arr > tail_high)
        low_mask = finite_chunk & (arr < tail_low)
        for mask, include_high in ((high_mask, True), (low_mask, False)):
            if not np.any(mask):
                continue
            padded = np.concatenate(([False], mask, [False]))
            starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
            ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
            for start, end in zip(starts, ends):
                if int(end - start) < min_event_run:
                    continue
                region = arr[start:end]
                region = region[np.isfinite(region)]
                if region.size <= 0:
                    continue
                if include_high:
                    high = max(high, float(np.max(region)))
                else:
                    low = min(low, float(np.min(region)))
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.median(finite))
        spread = float(np.std(finite))
        if not np.isfinite(spread) or spread <= 1.0e-9:
            spread = max(abs(center) * 0.1, 1.0)
        low = center - spread
        high = center + spread
    span = max(1.0e-9, high - low)
    padding = min(max(float(padding_fraction), 0.0), 1.0)
    low -= padding * span
    high += padding * span
    target_span = max(0.0, float(min_span))
    if target_span > 0.0 and (high - low) < target_span:
        center = (high + low) * 0.5
        half_span = target_span * 0.5
        low = center - half_span
        high = center + half_span
    step = _nice_floor((high - low) / 4.0)
    if step <= 0.0:
        return fallback
    y_min = math.floor(low / step) * step
    y_max = math.ceil(high / step) * step
    if not y_min < y_max:
        return fallback
    return float(y_min), float(y_max)


def _panel_ticks(y_min: float, y_max: float) -> np.ndarray:
    span = max(1.0e-9, float(y_max) - float(y_min))
    step = _nice_floor(span / 4.0)
    if step <= 0.0:
        return np.array([y_min, y_max], dtype=float)
    start = math.ceil(float(y_min) / step) * step
    ticks = np.arange(start, float(y_max) + (0.5 * step), step, dtype=float)
    while ticks.size > 7:
        step *= 2.0
        start = math.ceil(float(y_min) / step) * step
        ticks = np.arange(start, float(y_max) + (0.5 * step), step, dtype=float)
    return ticks


def _aligned_panel_marker_lane(signal_span: float) -> float:
    return max(1.0e-9, float(signal_span)) * ALIGNED_PANEL_MARKER_LANE_FRACTION


def _aligned_panel_marker_levels(marker_base_max: float, signal_span: float) -> Dict[str, float]:
    lane = _aligned_panel_marker_lane(signal_span)
    base = float(marker_base_max)
    spike_bottom = base + 0.66 * lane
    spike_top = base + 0.90 * lane
    return {
        "plateau_y": base + 0.25 * lane,
        "spike_bottom": spike_bottom,
        "spike_top": spike_top,
        "plateau_spike_bottom": spike_bottom,
        "plateau_spike_top": spike_top,
    }


def _recording_aligned_panel_entries(
    states: Sequence[TraceCuration],
    detection_params: object,
    render_options: RenderOptions,
) -> Tuple[List[PanelEntry], Dict[str, object]]:
    mode = str(render_options.normalization_mode or NORMALIZATION_PERCENT_DFF)
    source_channel = render_options.normalized_aligned_panel_channel()
    entries: List[PanelEntry] = []
    for state in states:
        data = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
        normalized, metadata = _prepared_aligned_panel_channel(data, state, mode, source_channel)
        if normalized is None or data.time.size == 0:
            continue
        n = min(data.time.size, normalized.size)
        if n <= 0:
            continue
        entries.append((state, data, normalized[:n].copy(), metadata))
    if not entries:
        return [], {"y_range": render_options.aligned_panel_y_range()}

    y_low_pct, y_high_pct = render_options.aligned_panel_y_percentiles()
    y_padding = render_options.aligned_panel_y_padding_fraction()
    min_y_span = render_options.aligned_panel_min_y_span_value()
    include_zero = mode == NORMALIZATION_PERCENT_DFF

    unscaled_entries: List[PanelEntry] = []
    for state, data, values, metadata in entries:
        meta = dict(metadata)
        meta.update({"display_unit": meta.get("unit", "")})
        unscaled_entries.append((state, data, values, meta))
    y_range = (
        _nice_panel_y_range(
            [entry[2] for entry in unscaled_entries],
            render_options.aligned_panel_y_range(),
            low_percentile=y_low_pct,
            high_percentile=y_high_pct,
            padding_fraction=y_padding,
            min_span=min_y_span,
            include_zero=include_zero,
        )
        if bool(render_options.aligned_panel_auto_y)
        else render_options.aligned_panel_y_range()
    )
    return unscaled_entries, {"y_range": y_range}


def _add_recording_horizontal_scale_bar(
    plot: object,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_span: float,
    horizontal_native: float,
    horizontal_label: str,
) -> None:
    x_span = max(1.0e-9, float(x_max - x_min))
    x0 = float(x_max) - 0.05 * x_span - float(horizontal_native)
    x1 = x0 + float(horizontal_native)
    y0 = float(y_min) + 0.12 * float(y_span)
    pen = pg.mkPen("#111111", width=2.2)
    plot.plot([x0, x1], [y0, y0], pen=pen)
    label = pg.TextItem(horizontal_label, color="#111111", anchor=(0.5, 0.0))
    label.setFont(QtGui.QFont("Arial", 11))
    plot.addItem(label)
    label.setPos((x0 + x1) * 0.5, y0 - 0.12 * y_span)


def _add_recording_scale_bars(
    plot: object,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_span: float,
    horizontal_native: float,
    horizontal_label: str,
    vertical_value: float,
    vertical_label: str,
) -> None:
    x_span = max(1.0e-9, float(x_max - x_min))
    x0 = float(x_max) - 0.06 * x_span - float(horizontal_native)
    x1 = x0 + float(horizontal_native)
    y0 = float(y_min) + 0.12 * float(y_span)
    y1 = y0 + float(vertical_value)
    pen = pg.mkPen("#111111", width=2.2)
    plot.plot([x0, x1], [y0, y0], pen=pen)
    plot.plot([x0, x0], [y0, y1], pen=pen)

    font = QtGui.QFont("Arial", 11)
    time_label = pg.TextItem(horizontal_label, color="#111111", anchor=(0.5, 0.0))
    time_label.setFont(font)
    plot.addItem(time_label)
    time_label.setPos((x0 + x1) * 0.5, y0 - 0.05 * y_span)

    vertical_text = pg.TextItem(vertical_label, color="#111111", anchor=(1.0, 0.5))
    vertical_text.setFont(font)
    plot.addItem(vertical_text)
    vertical_text.setPos(x0 - 0.015 * x_span, (y0 + y1) * 0.5)


def render_recording_aligned_traces(
    recording_id: str,
    states: Sequence[TraceCuration],
    detection_params: object,
    render_options: RenderOptions,
    *,
    width: int = 1600,
    height: Optional[int] = None,
) -> Tuple[QtGui.QPixmap, Dict[str, object]]:
    _ensure_app()
    mode = str(render_options.normalization_mode or NORMALIZATION_PERCENT_DFF)
    entries: List[Tuple[TraceCuration, PreparedTrace, np.ndarray, Dict[str, object]]] = []
    for state in states:
        data = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
        normalized, metadata = _prepared_percent_dff(data, state, mode)
        if normalized is None or data.time.size == 0:
            continue
        n = min(data.time.size, normalized.size)
        if n <= 0:
            continue
        entries.append((state, data, normalized[:n].copy(), metadata))
    if not entries:
        return QtGui.QPixmap(), {}

    visible_chunks = [entry[2][np.isfinite(entry[2])] for entry in entries if np.any(np.isfinite(entry[2]))]
    all_visible = np.concatenate(visible_chunks) if visible_chunks else np.array([0.0], dtype=float)
    unit_label = PERCENT_DFF_SCALE_LABEL if mode == NORMALIZATION_PERCENT_DFF else " native"
    scale = robust_percent_scale(all_visible, scale_label=unit_label)
    vertical_value = float(scale["value"])
    q5 = float(scale["robust_p05"])
    q95 = float(scale["robust_p95"])
    robust_span = max(q95 - q5, vertical_value)
    gap = max(1.0, vertical_value * 2.4, robust_span * 1.35)
    if _uses_bar_markers(render_options):
        gap *= 1.35

    scale_seconds, scale_native, horizontal_label, x_min, x_max, is_ms = _recording_time_scale(
        [entry[1].time[: min(entry[1].time.size, entry[2].size)] for entry in entries]
    )
    x_span = max(1.0e-9, x_max - x_min)
    if height is None:
        height = int(np.clip(240 + 115 * len(entries), 420, 1400))

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(0)
    plot = pg.PlotWidget(background="#ffffff")
    plot.hideAxis("bottom")
    plot.hideAxis("left")
    plot.showGrid(x=False, y=False)
    plot.setMouseEnabled(x=False, y=False)
    try:
        plot.setMenuEnabled(False)
    except AttributeError:
        plot.getPlotItem().setMenuEnabled(False)
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)

    colors = ["#111111", "#334155", "#0f766e", "#7c2d12", "#1d4ed8", "#86198f", "#365314", "#be123c"]
    shifted_values: List[np.ndarray] = []
    for row, (state, data, y, _metadata) in enumerate(entries):
        n = min(data.time.size, y.size)
        offset = float((len(entries) - row - 1) * gap)
        shifted = y[:n] + offset
        shifted_values.append(shifted)
        plot.plot(data.time[:n], shifted, pen=pg.mkPen(colors[row % len(colors)], width=1.55))
        if _uses_bar_markers(render_options):
            shifted_values.extend(
                _add_bar_marker_annotations(
                    plot,
                    data.time[:n],
                    shifted,
                    data.spikes,
                    data.mask("plateau"),
                    show_spike_markers=render_options.show_spike_markers,
                    show_plateau_bars=_show_bar_plateau_bars(render_options),
                    plateau_masks=_plateau_marker_masks(data),
                )
            )
        _recording, roi_id = split_trace_identity(state.trace_name)
        label = pg.TextItem(str(roi_id), color="#111111", anchor=(0.0, 0.5))
        label.setFont(QtGui.QFont("Arial", 10))
        plot.addItem(label)
        label.setPos(x_min - 0.03 * x_span, offset)

    shifted_chunks = [values[np.isfinite(values)] for values in shifted_values if np.any(np.isfinite(values))]
    finite_shifted = np.concatenate(shifted_chunks) if shifted_chunks else np.array([], dtype=float)
    if finite_shifted.size:
        data_y_min = float(np.min(finite_shifted))
        data_y_max = float(np.max(finite_shifted))
    else:
        data_y_min = 0.0
        data_y_max = float(max(1, len(entries) - 1)) * gap
    bottom_pad = max(vertical_value * 1.9, gap * 0.35, 1.0)
    top_pad = max(vertical_value * 0.8, gap * 0.25, 1.0)
    y_min = data_y_min - bottom_pad
    y_max = data_y_max + top_pad
    y_span = max(1.0e-9, y_max - y_min)
    _add_recording_scale_bars(
        plot,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_span=y_span,
        horizontal_native=scale_native,
        horizontal_label=horizontal_label,
        vertical_value=vertical_value,
        vertical_label=str(scale["label"]),
    )
    plot.setXRange(x_min - 0.04 * x_span, x_max + 0.04 * x_span, padding=0)
    plot.setYRange(y_min, y_max, padding=0)
    layout.addWidget(plot)

    metadata = {
        "recording_id": str(recording_id),
        "trace_count": len(entries),
        "traces": [entry[0].trace_name for entry in entries],
        "normalization_mode": mode,
        "unit": PERCENT_DFF_SCALE_LABEL if mode == NORMALIZATION_PERCENT_DFF else "native",
        "time_start": float(x_min),
        "time_end": float(x_max),
        "visible_duration": float(x_max - x_min),
        "time_axis_is_ms": bool(is_ms),
        "horizontal_scale_seconds": float(scale_seconds),
        "horizontal_scale_native_units": float(scale_native),
        "horizontal_label": str(horizontal_label),
        "vertical_scale_percent": float(vertical_value),
        "vertical_scale_value": float(vertical_value),
        "vertical_label": str(scale["label"]),
        "vertical_polarity": str(scale["polarity"]),
        "robust_p05": float(scale["robust_p05"]),
        "robust_p95": float(scale["robust_p95"]),
        "trace_normalization": [
            {
                "trace_name": entry[0].trace_name,
                "recording_id": split_trace_identity(entry[0].trace_name)[0],
                "roi_id": split_trace_identity(entry[0].trace_name)[1],
                **entry[3],
            }
            for entry in entries
        ],
    }
    return _finalize_widget(container, width, int(height)), metadata


def render_recording_aligned_trace_panels(
    recording_id: str,
    states: Sequence[TraceCuration],
    detection_params: object,
    render_options: RenderOptions,
    *,
    width: int = 1600,
    panel_height: int = 300,
) -> Tuple[QtGui.QPixmap, Dict[str, object]]:
    _ensure_app()
    mode = str(render_options.normalization_mode or NORMALIZATION_PERCENT_DFF)
    source_channel = render_options.normalized_aligned_panel_channel()
    entries, panel_metadata = _recording_aligned_panel_entries(states, detection_params, render_options)
    if not entries:
        return QtGui.QPixmap(), {}

    scale_seconds, scale_native, horizontal_label, x_min, x_max, is_ms = _recording_time_scale(
        [entry[1].time[: min(entry[1].time.size, entry[2].size)] for entry in entries]
    )
    x_span = max(1.0e-9, x_max - x_min)
    dpi = 160
    width_inches = max(8.0, float(width) / float(dpi))
    height_inches = max(2.0, (float(panel_height) / float(dpi)) * len(entries))
    colors = ["#5b9bd5", "#8cc665", "#f0c64a", "#7c3aed", "#0f766e", "#dc2626", "#334155", "#9333ea"]
    all_visible: List[np.ndarray] = []
    y_low_pct, y_high_pct = render_options.aligned_panel_y_percentiles()
    y_padding = render_options.aligned_panel_y_padding_fraction()
    min_y_span = render_options.aligned_panel_min_y_span_value()
    include_zero = mode == NORMALIZATION_PERCENT_DFF
    share_y = bool(render_options.aligned_panel_share_y)
    invert_y = bool(render_options.aligned_panel_negative_indicator)

    plotted_entries: List[Tuple[TraceCuration, PreparedTrace, np.ndarray, np.ndarray, Dict[str, object]]] = []
    for state, data, y, metadata in entries:
        plotted_y = -np.asarray(y, dtype=float) if invert_y else np.asarray(y, dtype=float)
        plotted_entries.append((state, data, y, plotted_y, metadata))

    if bool(render_options.aligned_panel_auto_y):
        shared_y_range = _nice_panel_y_range(
            [entry[3] for entry in plotted_entries],
            render_options.aligned_panel_y_range(),
            low_percentile=y_low_pct,
            high_percentile=y_high_pct,
            padding_fraction=y_padding,
            min_span=min_y_span,
            include_zero=include_zero,
        )
    else:
        shared_y_range = render_options.aligned_panel_y_range()

    row_y_ranges: List[Tuple[float, float]] = []
    for _state, _data, _values, plotted_y, _metadata in plotted_entries:
        if share_y or not bool(render_options.aligned_panel_auto_y):
            row_range = shared_y_range
        else:
            row_range = _nice_panel_y_range(
                [plotted_y],
                render_options.aligned_panel_y_range(),
                low_percentile=y_low_pct,
                high_percentile=y_high_pct,
                padding_fraction=y_padding,
                min_span=min_y_span,
                include_zero=include_zero,
            )
        row_y_ranges.append((float(row_range[0]), float(row_range[1])))
    marker_base_ranges = list(row_y_ranges)
    show_bar_plateaus = _show_bar_plateau_bars(render_options)
    if _uses_bar_markers(render_options) and (render_options.show_spike_markers or show_bar_plateaus):
        row_y_ranges = [
            (low, high + _aligned_panel_marker_lane(max(1.0e-9, high - low)))
            for low, high in row_y_ranges
        ]
    overall_y_min = min((limits[0] for limits in row_y_ranges), default=float(shared_y_range[0]))
    overall_y_max = max((limits[1] for limits in row_y_ranges), default=float(shared_y_range[1]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(entries),
        1,
        figsize=(width_inches, height_inches),
        dpi=dpi,
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    total_peak_markers = 0
    if mode == NORMALIZATION_PERCENT_DFF:
        shared_y_label = "-\u0394F/F\u2080 (%)" if invert_y else "\u0394F/F\u2080 (%)"
    elif source_channel == "raw":
        shared_y_label = "Raw signal"
    else:
        shared_y_label = "Signal"
    trace_y_ranges: Dict[str, Tuple[float, float]] = {}

    for row, (ax, (state, data, y, plotted_y, metadata)) in enumerate(zip(axes_flat, plotted_entries)):
        n = min(data.time.size, y.size)
        row_y_min, row_y_max = row_y_ranges[row]
        marker_base_min, marker_base_max = marker_base_ranges[row]
        row_y_span = max(1.0e-9, row_y_max - row_y_min)
        ax.plot(data.time[:n], plotted_y[:n], color=colors[row % len(colors)], linewidth=1.15)
        if _uses_bar_markers(render_options):
            marker_levels = _aligned_panel_marker_levels(marker_base_max, max(1.0e-9, marker_base_max - marker_base_min))
            plateau_y = marker_levels["plateau_y"]
            spike_bottom = marker_levels["spike_bottom"]
            spike_top = marker_levels["spike_top"]
            plateau_mask = data.mask("plateau")
            if show_bar_plateaus and plateau_mask is not None:
                for name, source_mask in _plateau_marker_masks(data):
                    mask = np.asarray(source_mask, dtype=bool)[:n]
                    if not np.any(mask):
                        continue
                    for start, end in gui_main._contiguous_true_regions(mask):
                        if start >= n:
                            continue
                        s = max(0, min(int(start), n - 1))
                        e = max(s, min(int(end) - 1, n - 1))
                        ax.plot(
                            [float(data.time[s]), float(data.time[e])],
                            [plateau_y, plateau_y],
                            color=PLATEAU_MARKER_COLORS.get(str(name), PLATEAU_MARKER_COLORS["plateau"]),
                            linewidth=ALIGNED_PANEL_PLATEAU_BAR_LINE_WIDTH,
                            solid_capstyle="butt",
                            zorder=5,
                        )
            if render_options.show_spike_markers:
                for spike in filtered_spikes(data, render_options):
                    idx = int(spike.spike_index)
                    if 0 <= idx < n:
                        ax.scatter(
                            [float(data.time[idx])],
                            [0.5 * (float(spike_bottom) + float(spike_top))],
                            s=ALIGNED_PANEL_SPIKE_HOLLOW_MARKER_SIZE,
                            facecolors="none",
                            edgecolors="#111111",
                            linewidths=1.2,
                            zorder=6,
                        )
                        total_peak_markers += 1
        elif render_options.show_spike_markers:
            peak_x: List[float] = []
            peak_y: List[float] = []
            for spike in filtered_spikes(data, render_options):
                idx = int(spike.spike_index)
                if 0 <= idx < n:
                    peak_x.append(float(data.time[idx]))
                    peak_y.append(float(plotted_y[idx]))
            if peak_x:
                ax.scatter(
                    peak_x,
                    peak_y,
                    s=18,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=0.9,
                    zorder=4,
                )
                total_peak_markers += len(peak_x)
        ax.set_xlim(x_min - 0.01 * x_span, x_max + 0.01 * x_span)
        ax.set_ylim(row_y_min, row_y_max)
        ax.set_yticks(_panel_ticks(row_y_min, row_y_max))
        ax.tick_params(axis="y", labelsize=9, length=4, width=1.0)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        finite = plotted_y[:n][np.isfinite(plotted_y[:n])]
        if finite.size:
            all_visible.append(finite)

        _recording, roi_id = split_trace_identity(state.trace_name)
        trace_y_ranges[state.trace_name] = (row_y_min, row_y_max)
        source_label = str(metadata.get("source_label", CHANNEL_LABELS.get(source_channel, source_channel)))
        ax.text(
            0.0,
            1.035,
            f"{roi_id} | {source_label}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            color="#111111",
            clip_on=False,
        )

        if row == len(entries) - 1:
            x0 = float(x_max) - 0.05 * x_span - float(scale_native)
            x1 = x0 + float(scale_native)
            y0 = float(row_y_min) + 0.12 * row_y_span
            ax.plot([x0, x1], [y0, y0], color="#111111", linewidth=1.8, solid_capstyle="butt")
            ax.text(
                (x0 + x1) * 0.5,
                y0 - 0.10 * row_y_span,
                horizontal_label,
                ha="center",
                va="top",
                fontsize=12,
                color="#111111",
            )

    fig.patch.set_facecolor("white")
    fig.text(0.035, 0.5, shared_y_label, rotation="vertical", va="center", ha="center", fontsize=12, color="#111111")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.04, hspace=0.62)

    joined = np.concatenate(all_visible) if all_visible else np.array([], dtype=float)
    finite = joined[np.isfinite(joined)]
    robust_p05 = float(np.percentile(finite, 5.0)) if finite.size else float("nan")
    robust_p95 = float(np.percentile(finite, 95.0)) if finite.size else float("nan")
    actual_channels = sorted({str(entry[3].get("source_channel", source_channel)) for entry in entries})
    metadata = {
        "recording_id": str(recording_id),
        "trace_count": len(entries),
        "traces": [entry[0].trace_name for entry in entries],
        "normalization_mode": mode,
        "unit": "dF/F0 (%)" if mode == NORMALIZATION_PERCENT_DFF else "native",
        "source_channel": actual_channels[0] if len(actual_channels) == 1 else ";".join(actual_channels),
        "time_start": float(x_min),
        "time_end": float(x_max),
        "visible_duration": float(x_max - x_min),
        "time_axis_is_ms": bool(is_ms),
        "horizontal_scale_seconds": float(scale_seconds),
        "horizontal_scale_native_units": float(scale_native),
        "horizontal_label": str(horizontal_label),
        "share_y": bool(share_y),
        "plot_negative_indicator": bool(invert_y),
        "plot_y_label": shared_y_label,
        "y_limit_low_percentile": float(y_low_pct),
        "y_limit_high_percentile": float(y_high_pct),
        "y_limit_padding": float(y_padding),
        "y_limit_min_span": float(min_y_span),
        "y_min": float(overall_y_min),
        "y_max": float(overall_y_max),
        "panel_y_ranges": [
            {
                "trace_name": state.trace_name,
                "recording_id": split_trace_identity(state.trace_name)[0],
                "roi_id": split_trace_identity(state.trace_name)[1],
                "y_min": float(trace_y_ranges.get(state.trace_name, (float("nan"), float("nan")))[0]),
                "y_max": float(trace_y_ranges.get(state.trace_name, (float("nan"), float("nan")))[1]),
            }
            for state, _data, _values, _plotted_y, _row_metadata in plotted_entries
        ],
        "robust_p05": robust_p05,
        "robust_p95": robust_p95,
        "show_peak_markers": bool(render_options.show_spike_markers),
        "peak_marker_count": int(total_peak_markers),
        "trace_normalization": [
            {
                "trace_name": entry[0].trace_name,
                "recording_id": split_trace_identity(entry[0].trace_name)[0],
                "roi_id": split_trace_identity(entry[0].trace_name)[1],
                "panel_y_min": float(trace_y_ranges.get(entry[0].trace_name, (float("nan"), float("nan")))[0]),
                "panel_y_max": float(trace_y_ranges.get(entry[0].trace_name, (float("nan"), float("nan")))[1]),
                **entry[3],
            }
            for entry in entries
        ],
    }
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap, metadata


def render_recording_event_raster(
    recording_id: str,
    states: Sequence[TraceCuration],
    detection_params: object,
    render_options: RenderOptions,
    *,
    width: int = 1600,
    row_height: int = 118,
) -> Tuple[QtGui.QPixmap, Dict[str, object]]:
    _ensure_app()
    entries: List[Tuple[TraceCuration, PreparedTrace]] = []
    for state in states:
        data = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
        if data.time.size:
            entries.append((state, data))
    if not entries:
        return QtGui.QPixmap(), {}

    scale_seconds, scale_native, horizontal_label, x_min, x_max, is_ms = _recording_time_scale(
        [entry[1].time for entry in entries]
    )
    x_span = max(1.0e-9, x_max - x_min)
    x_pad = 0.015 * x_span
    row_count = len(entries)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dpi = 160
    width_inches = max(8.0, float(width) / float(dpi))
    height_inches = max(2.8, (float(row_height) / float(dpi)) * row_count + 0.9)
    fig, ax = plt.subplots(1, 1, figsize=(width_inches, height_inches), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    row_spacing = 1.45
    y_positions: List[float] = []
    y_labels: List[str] = []
    spike_count = 0
    plateau_interval_count = 0
    plateau_sources: set[str] = set()
    spike_tick_bottom_offset = 0.02
    spike_tick_top_offset = 0.30
    plateau_y_offset = -0.20

    for row, (state, data) in enumerate(entries):
        y = float(row_count - row) * row_spacing
        y_positions.append(y)
        _recording, roi_id = split_trace_identity(state.trace_name)
        y_labels.append(str(roi_id))

        n = int(data.time.size)
        for name, source_mask in _plateau_marker_masks(data):
            mask = np.asarray(source_mask, dtype=bool)[:n]
            if not np.any(mask):
                continue
            plateau_sources.add(str(name))
            for start, end in gui_main._contiguous_true_regions(mask):
                if start >= n:
                    continue
                s = max(0, min(int(start), n - 1))
                e = max(s, min(int(end) - 1, n - 1))
                ax.plot(
                    [float(data.time[s]), float(data.time[e])],
                    [y + plateau_y_offset, y + plateau_y_offset],
                    color=PLATEAU_MARKER_COLORS.get(str(name), PLATEAU_MARKER_COLORS["plateau"]),
                    linewidth=5.0,
                    solid_capstyle="butt",
                    zorder=2,
                )
                plateau_interval_count += 1

        if render_options.show_spike_markers:
            for spike in filtered_spikes(data, render_options):
                idx = int(spike.spike_index)
                if 0 <= idx < n:
                    x = float(data.time[idx])
                else:
                    x = float(spike.spike_time)
                if not np.isfinite(x) or x < x_min or x > x_max:
                    continue
                ax.vlines(
                    x,
                    y + spike_tick_bottom_offset,
                    y + spike_tick_top_offset,
                    colors="#111111",
                    linewidth=1.15,
                    zorder=3,
                )
                spike_count += 1

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0.30 * row_spacing, (float(row_count) + 0.70) * row_spacing)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.tick_params(axis="x", labelsize=10, length=4, width=1.0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("Time (ms)" if is_ms else "Time (s)", fontsize=11)
    ax.set_title(f"{recording_id} | event raster", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(False)

    legend_items: List[Line2D] = []
    if render_options.show_spike_markers and spike_count:
        legend_items.append(Line2D([0], [0], color="#111111", linewidth=1.4, label="Spike"))
    plateau_legend_labels = {
        "plateau": "Plateau",
        "long_plateau": "Long plateau",
        "short_plateau": "Short plateau",
        "burst_plateau": "Burst plateau",
    }
    for source in ("long_plateau", "short_plateau", "burst_plateau", "plateau"):
        if source in plateau_sources:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color=PLATEAU_MARKER_COLORS.get(source, PLATEAU_MARKER_COLORS["plateau"]),
                    linewidth=5.0,
                    label=plateau_legend_labels.get(source, source),
                )
            )
    if legend_items:
        ax.legend(
            handles=legend_items,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=9,
            handlelength=1.8,
            borderaxespad=0.2,
        )

    fig.subplots_adjust(left=0.12, right=0.82 if legend_items else 0.985, top=0.84, bottom=0.18)
    metadata = {
        "recording_id": str(recording_id),
        "trace_count": row_count,
        "traces": [entry[0].trace_name for entry in entries],
        "time_start": float(x_min),
        "time_end": float(x_max),
        "visible_duration": float(x_max - x_min),
        "time_axis_is_ms": bool(is_ms),
        "horizontal_scale_seconds": float(scale_seconds),
        "horizontal_scale_native_units": float(scale_native),
        "horizontal_label": str(horizontal_label),
        "spike_count": int(spike_count),
        "plateau_interval_count": int(plateau_interval_count),
        "plateau_sources": sorted(plateau_sources),
    }
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap, metadata


def _add_shaded_regions(plot: object, time: np.ndarray, mask: Optional[np.ndarray], color: str = "#e74c3c") -> None:
    if mask is None or time.size == 0:
        return
    m = np.asarray(mask, dtype=bool)
    n = min(time.size, m.size)
    if n <= 0:
        return
    for start, end in gui_main._contiguous_true_regions(m[:n]):
        s = max(0, min(int(start), n - 1))
        e = max(s, min(int(end) - 1, n - 1))
        region = pg.LinearRegionItem(
            values=(float(time[s]), float(time[e])),
            brush=pg.mkBrush(color + "38"),
            pen=pg.mkPen(color, width=0),
            movable=False,
        )
        region.setZValue(-8)
        plot.addItem(region)


def _add_plateau_labels(plot: object, time: np.ndarray, mask: Optional[np.ndarray], y_values: np.ndarray) -> None:
    if mask is None or time.size == 0:
        return
    finite = np.asarray(y_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    y = float(np.max(finite)) if finite.size else 0.0
    for idx, (start, end) in enumerate(gui_main._contiguous_true_regions(np.asarray(mask, dtype=bool)), 1):
        if start >= time.size:
            continue
        s = max(0, min(int(start), time.size - 1))
        e = max(s, min(int(end) - 1, time.size - 1))
        label = pg.TextItem(f"P{idx:04d}", color="#7f1d1d", anchor=(0.5, 1.0))
        plot.addItem(label)
        label.setPos(float(time[s] + time[e]) * 0.5, y)


def _uses_bar_markers(render_options: RenderOptions) -> bool:
    return render_options.normalized_marker_style() == MARKER_STYLE_BAR


def _uses_plateau_shading(render_options: RenderOptions) -> bool:
    return bool(render_options.shade_plateaus) and not _uses_bar_markers(render_options)


def _show_bar_plateau_bars(render_options: RenderOptions) -> bool:
    return _uses_bar_markers(render_options)


def _bar_marker_level_span(signal_span: float) -> float:
    span = max(1.0e-9, float(signal_span))
    max_span = BAR_MARKER_MAX_SIGNAL_HEIGHT / _BAR_MARKER_SPIKE_HEIGHT_FRACTION
    return min(span, max_span)


def _bar_marker_levels(signal: np.ndarray) -> Optional[Dict[str, float]]:
    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    span = max(1.0e-9, y_max - y_min, abs(y_max) * 0.08, 1.0)
    marker_span = _bar_marker_level_span(span)
    spike_bottom = y_max + 0.28 * marker_span
    spike_top = y_max + 0.40 * marker_span
    return {
        "plateau_y": y_max + 0.16 * marker_span,
        "spike_bottom": spike_bottom,
        "spike_top": spike_top,
        "plateau_spike_bottom": spike_bottom,
        "plateau_spike_top": spike_top,
    }


def _mask_contains_index(mask: Optional[np.ndarray], index: int) -> bool:
    if mask is None:
        return False
    arr = np.asarray(mask, dtype=bool)
    return 0 <= int(index) < arr.size and bool(arr[int(index)])


def _plateau_marker_masks(data: PreparedTrace) -> List[Tuple[str, np.ndarray]]:
    masks: List[Tuple[str, np.ndarray]] = []
    for name in ("long_plateau", "short_plateau", "burst_plateau"):
        mask = data.mask(name)
        if mask is not None and np.any(np.asarray(mask, dtype=bool)):
            masks.append((name, np.asarray(mask, dtype=bool)))
    if masks:
        return masks
    plateau_mask = data.mask("plateau")
    if plateau_mask is not None and np.any(np.asarray(plateau_mask, dtype=bool)):
        return [("plateau", np.asarray(plateau_mask, dtype=bool))]
    return []


def _sliced_plateau_marker_masks(data: PreparedTrace, lo: int, hi: int) -> List[Tuple[str, np.ndarray]]:
    return [
        (name, np.asarray(mask, dtype=bool)[lo:hi])
        for name, mask in _plateau_marker_masks(data)
    ]


def _add_bar_marker_annotations(
    plot: object,
    time: np.ndarray,
    signal: np.ndarray,
    spikes: Sequence[SpikeRecord],
    plateau_mask: Optional[np.ndarray],
    *,
    show_spike_markers: bool,
    show_plateau_bars: bool,
    plateau_masks: Optional[Sequence[Tuple[str, np.ndarray]]] = None,
) -> List[np.ndarray]:
    n = min(np.asarray(time).size, np.asarray(signal).size)
    if n <= 0 or (not show_spike_markers and not show_plateau_bars):
        return []
    x_values = np.asarray(time, dtype=float)[:n]
    y_values = np.asarray(signal, dtype=float)[:n]
    levels = _bar_marker_levels(y_values)
    if levels is None:
        return []

    marker_y_values: List[float] = []
    spike_x: List[float] = []
    spike_y: List[float] = []

    if show_plateau_bars and plateau_mask is not None:
        source_masks = list(plateau_masks or [("plateau", np.asarray(plateau_mask, dtype=bool))])
        for name, source_mask in source_masks:
            mask = np.asarray(source_mask, dtype=bool)[:n]
            if not np.any(mask):
                continue
            plateau_pen = pg.mkPen(
                PLATEAU_MARKER_COLORS.get(str(name), PLATEAU_MARKER_COLORS["plateau"]),
                width=PLATEAU_BAR_LINE_WIDTH,
            )
            for start, end in gui_main._contiguous_true_regions(mask):
                if start >= n:
                    continue
                s = max(0, min(int(start), n - 1))
                e = max(s, min(int(end) - 1, n - 1))
                x0 = float(x_values[s])
                x1 = float(x_values[e])
                plot.plot([x0, x1], [levels["plateau_y"], levels["plateau_y"]], pen=plateau_pen)
                marker_y_values.append(float(levels["plateau_y"]))

    if show_spike_markers:
        for spike in spikes:
            spike_idx = int(spike.spike_index)
            if not (0 <= spike_idx < n):
                continue
            x = float(x_values[spike_idx])
            if not np.isfinite(x):
                continue
            spike_x.append(x)
            spike_y.append(0.5 * (float(levels["spike_bottom"]) + float(levels["spike_top"])))
            marker_y_values.extend([float(levels["spike_bottom"]), float(levels["spike_top"])])
        if spike_x:
            plot.addItem(
                pg.ScatterPlotItem(
                    x=spike_x,
                    y=spike_y,
                    size=SPIKE_HOLLOW_MARKER_SIZE,
                    brush=pg.mkBrush(255, 255, 255, 0),
                    pen=pg.mkPen("#111111", width=1.3),
                )
            )

    return [np.asarray(marker_y_values, dtype=float)] if marker_y_values else []


def _add_spike_annotations(
    plot: object,
    time: np.ndarray,
    signal: np.ndarray,
    spikes: Sequence[SpikeRecord],
    *,
    show_spike_markers: bool,
    label_spikes: bool,
    index_offset: int = 0,
) -> None:
    if not show_spike_markers and not label_spikes:
        return
    points: List[Tuple[float, float, int]] = []
    for idx, spike in enumerate(spikes, 1):
        spike_idx = int(spike.spike_index)
        if 0 <= spike_idx < len(time) and spike_idx < len(signal):
            if np.isfinite(time[spike_idx]) and np.isfinite(signal[spike_idx]):
                points.append((float(time[spike_idx]), float(signal[spike_idx]), idx + index_offset))
    if not points:
        return
    if show_spike_markers:
        plot.addItem(
            pg.ScatterPlotItem(
                x=[point[0] for point in points],
                y=[point[1] for point in points],
                size=8,
                brush=pg.mkBrush("#f1c40f"),
                pen=pg.mkPen("#7a5c00", width=0.8),
            )
        )
    if label_spikes:
        for x, y, idx in points:
            label = pg.TextItem(f"S{idx:04d}", color="#7a5c00", anchor=(0.5, 1.1))
            plot.addItem(label)
            label.setPos(x, y)


def render_trace_channels(
    data: PreparedTrace,
    render_options: RenderOptions,
    *,
    title_suffix: str = "selected trace",
    channels: Optional[Sequence[str]] = None,
    width: int = 1800,
    height: int = 560,
) -> QtGui.QPixmap:
    _ensure_app()
    if data.time.size == 0:
        return QtGui.QPixmap()
    selected = list(channels if channels is not None else render_options.normalized_channels())
    plotted: List[Tuple[str, np.ndarray]] = []
    for channel in selected:
        arr = data.array(channel)
        if arr is not None:
            plotted.append((channel, arr))
    if not plotted:
        corrected = data.array("corrected")
        if corrected is None:
            return QtGui.QPixmap()
        plotted.append(("corrected", corrected))

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.addLegend(offset=(10, 8))
    plot.setLabel("bottom", gui_main._time_axis_label(data.time))
    plot.setLabel("left", "Signal")
    plot.setTitle(f"{data.trace_name} | {title_suffix}")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)

    if _uses_plateau_shading(render_options):
        long_mask = data.mask("long_plateau")
        _add_shaded_regions(plot, data.time, long_mask if long_mask is not None else data.mask("plateau"), "#e74c3c")
        _add_shaded_regions(plot, data.time, data.mask("short_plateau"), "#1565c0")
        _add_shaded_regions(plot, data.time, data.mask("burst_plateau"), "#7c3aed")

    for channel, values in plotted:
        n = min(data.time.size, values.size)
        plot.plot(
            data.time[:n],
            values[:n],
            pen=pg.mkPen(CHANNEL_COLORS.get(channel, "#2979ff"), width=1.5),
            name=CHANNEL_LABELS.get(channel, channel),
        )

    marker_signal = plotted[-1][1]
    marker_range_values: List[np.ndarray] = []
    if _uses_bar_markers(render_options):
        marker_range_values = _add_bar_marker_annotations(
            plot,
            data.time,
            marker_signal,
            data.spikes,
            data.mask("plateau"),
            show_spike_markers=render_options.show_spike_markers,
            show_plateau_bars=_show_bar_plateau_bars(render_options),
            plateau_masks=_plateau_marker_masks(data),
        )
    else:
        _add_spike_annotations(
            plot,
            data.time,
            marker_signal,
            data.spikes,
            show_spike_markers=render_options.show_spike_markers,
            label_spikes=render_options.label_spikes,
        )
        if render_options.label_plateaus:
            _add_plateau_labels(plot, data.time, data.mask("plateau"), plotted[-1][1])

    _set_y_range(plot, [values for _channel, values in plotted] + marker_range_values)
    if data.time.size >= 2:
        plot.setXRange(float(data.time[0]), float(data.time[-1]), padding=0)
    layout.addWidget(plot)
    return _finalize_widget(container, width, height)


def render_raw_processed_comparison(
    data: PreparedTrace,
    processed_channel: str,
    *,
    width: int = 1800,
    height: int = 760,
) -> QtGui.QPixmap:
    _ensure_app()
    raw = data.array("raw")
    processed = data.array(processed_channel)
    if raw is None or processed is None or data.time.size == 0:
        return QtGui.QPixmap()
    n = min(data.time.size, raw.size, processed.size)
    if n <= 0:
        return QtGui.QPixmap()
    time = data.time[:n]

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    panel_h = max(1, height // 2)

    for title, values, color in [
        ("Raw", raw[:n], CHANNEL_COLORS["raw"]),
        (CHANNEL_LABELS.get(processed_channel, processed_channel), processed[:n], CHANNEL_COLORS.get(processed_channel, "#2979ff")),
    ]:
        plot = pg.PlotWidget(background="#f7f9fc")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("left", title)
        if title != "Raw":
            plot.setLabel("bottom", gui_main._time_axis_label(time))
        else:
            plot.setTitle(f"{data.trace_name} | raw vs {CHANNEL_LABELS.get(processed_channel, processed_channel).lower()}")
        plot.setMinimumHeight(panel_h)
        plot.setMaximumHeight(panel_h)
        plot.plot(time, values, pen=pg.mkPen(color, width=1.4))
        _set_y_range(plot, [values])
        if n >= 2:
            plot.setXRange(float(time[0]), float(time[-1]), padding=0)
        layout.addWidget(plot)
    return _finalize_widget(container, width, height)


def render_spike_event(
    data: PreparedTrace,
    spike: SpikeRecord,
    render_options: RenderOptions,
    *,
    spike_label: str,
    default_half_window_ms: float,
    width: int = 1200,
    height: int = 430,
) -> QtGui.QPixmap:
    _ensure_app()
    time = data.time
    corrected = data.array("corrected")
    if corrected is None or time.size == 0:
        return QtGui.QPixmap()
    denoised = data.array("denoised")
    baseline = data.array("baseline")
    threshold = data.array("debug_threshold")
    half_window_ms = render_options.effective_spike_half_window_ms(default_half_window_ms)
    half_window = gui_main._half_window_in_time_units(time, half_window_ms)
    spike_time = float(spike.spike_time)
    lo = max(0, int(np.searchsorted(time, spike_time - half_window, side="left") - 1))
    hi = min(time.size, int(np.searchsorted(time, spike_time + half_window, side="right") + 1))
    if hi <= lo:
        return QtGui.QPixmap()
    x = time[lo:hi] - spike_time

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    panels = [("Corrected", corrected, CHANNEL_COLORS["corrected"])]
    if denoised is not None:
        panels.append(("Denoised", denoised, CHANNEL_COLORS["denoised"]))
    panel_h = max(1, height // len(panels))

    all_values = [corrected[lo:hi]]
    if denoised is not None:
        all_values.append(denoised[lo:hi])
    y_range_values = np.concatenate([np.asarray(values, dtype=float) for values in all_values])

    for idx, (title, values, color) in enumerate(panels):
        plot = pg.PlotWidget(background="#f7f9fc")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.addLegend(offset=(10, 8))
        plot.setLabel("left", title)
        if idx == len(panels) - 1:
            plot.setLabel("bottom", gui_main._time_axis_label(time))
        plot.setMinimumHeight(panel_h)
        plot.setMaximumHeight(panel_h)
        y = values[lo:hi]
        plot.plot(x, y, pen=pg.mkPen(color, width=2.0), name=title)
        if baseline is not None and title == "Corrected":
            plot.plot(x, baseline[lo:hi], pen=pg.mkPen(CHANNEL_COLORS["baseline"], width=1.2), name="Baseline")
        if threshold is not None and title == "Corrected":
            plot.plot(x, threshold[lo:hi], pen=pg.mkPen("#8e24aa", width=1.1, style=QtCore.Qt.PenStyle.DashLine), name="Threshold")
        marker_range_values: List[np.ndarray] = []
        spike_idx = int(spike.spike_index)
        spike_y = float(values[spike_idx]) if 0 <= spike_idx < values.size else float("nan")
        if _uses_bar_markers(render_options):
            local_spike = replace(
                spike,
                spike_index=spike_idx - lo,
                candidate_index=max(0, int(spike.candidate_index) - lo),
            )
            plateau_mask = data.mask("plateau")
            local_plateau = None if plateau_mask is None else np.asarray(plateau_mask, dtype=bool)[lo:hi]
            marker_range_values = _add_bar_marker_annotations(
                plot,
                x,
                y,
                [local_spike],
                local_plateau,
                show_spike_markers=render_options.show_spike_markers,
                show_plateau_bars=_show_bar_plateau_bars(render_options),
                plateau_masks=_sliced_plateau_marker_masks(data, lo, hi),
            )
        elif np.isfinite(spike_y):
            if render_options.show_spike_markers:
                plot.addItem(
                    pg.ScatterPlotItem(
                        x=[0.0],
                        y=[spike_y],
                        size=10,
                        brush=pg.mkBrush("#f1c40f"),
                        pen=pg.mkPen("#ffffff", width=1.0),
                    )
                )
            if render_options.label_spikes:
                label = pg.TextItem(spike_label, color="#7a5c00", anchor=(0.5, 1.1))
                plot.addItem(label)
                label.setPos(0.0, spike_y)
        plot.plot([0.0, 0.0], [float(np.nanmin(y_range_values)), float(np.nanmax(y_range_values))], pen=pg.mkPen("#ff0000", width=1.2, style=QtCore.Qt.PenStyle.DashLine))
        plot.setXRange(-half_window, half_window, padding=0)
        _set_y_range(plot, all_values + marker_range_values)
        layout.addWidget(plot)

    return _finalize_widget(container, width, height)


def render_superimposed_spikes(
    data: PreparedTrace,
    spikes: Sequence[SpikeRecord],
    render_options: RenderOptions,
    *,
    default_half_window_ms: float,
    width: int = 1200,
    height: int = 430,
) -> QtGui.QPixmap:
    _ensure_app()
    corrected = data.array("corrected")
    if corrected is None or data.time.size == 0 or not spikes:
        return QtGui.QPixmap()
    half_window_ms = render_options.effective_spike_half_window_ms(default_half_window_ms)
    half_window = gui_main._half_window_in_time_units(data.time, half_window_ms)

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", gui_main._time_axis_label(data.time))
    plot.setLabel("left", "Signal")
    plot.setTitle(f"{data.trace_name} | spike superimposed")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    colors = ["#2979ff", "#f1c40f", "#e74c3c", "#27ae60", "#9b59b6", "#e67e22", "#1abc9c", "#34495e"]
    values: List[np.ndarray] = []
    for idx, spike in enumerate(spikes, 1):
        spike_time = float(spike.spike_time)
        lo = max(0, int(np.searchsorted(data.time, spike_time - half_window, side="left") - 1))
        hi = min(data.time.size, int(np.searchsorted(data.time, spike_time + half_window, side="right") + 1))
        if hi <= lo:
            continue
        x = data.time[lo:hi] - spike_time
        y = corrected[lo:hi]
        values.append(y)
        plot.plot(x, y, pen=pg.mkPen(colors[(idx - 1) % len(colors)] + "b3", width=1.4))
        if render_options.label_spikes and int(spike.spike_index) < corrected.size:
            label = pg.TextItem(f"S{idx:04d}", color="#7a5c00", anchor=(0.5, 1.1))
            plot.addItem(label)
            label.setPos(0.0, float(corrected[int(spike.spike_index)]))
    plot.setXRange(-half_window, half_window, padding=0)
    _set_y_range(plot, values)
    layout.addWidget(plot)
    return _finalize_widget(container, width, height)


def _time_units_from_ms(time: np.ndarray, value_ms: float) -> float:
    return gui_main._half_window_in_time_units(time, float(value_ms))


def render_full_trace_plateaus(
    data: PreparedTrace,
    render_options: RenderOptions,
    *,
    width: int = 1800,
    height: int = 560,
) -> QtGui.QPixmap:
    return render_trace_channels(
        data,
        render_options,
        title_suffix="full trace plateaus",
        channels=("corrected",),
        width=width,
        height=height,
    )


def render_plateau_zoom(
    data: PreparedTrace,
    region: Tuple[int, int],
    render_options: RenderOptions,
    *,
    plateau_label: str,
    plateau_source: str = "long",
    signal_type: str = "unknown",
    width: int = 1400,
    height: int = 430,
) -> QtGui.QPixmap:
    _ensure_app()
    corrected = data.array("corrected")
    if corrected is None or data.time.size == 0:
        return QtGui.QPixmap()
    start, end_exclusive = int(region[0]), int(region[1])
    n = min(corrected.size, data.time.size)
    if n <= 0:
        return QtGui.QPixmap()
    s = max(0, min(start, n - 1))
    e = max(s, min(end_exclusive - 1, n - 1))
    onset_t = float(data.time[s])
    pre = _time_units_from_ms(data.time, render_options.plateau_pre_ms)
    post = _time_units_from_ms(data.time, render_options.plateau_post_ms)
    lo = max(0, int(np.searchsorted(data.time, onset_t - pre, side="left") - 1))
    hi = min(n, int(np.searchsorted(data.time, onset_t + post, side="right") + 1))
    if hi <= lo:
        return QtGui.QPixmap()
    x = data.time[lo:hi] - onset_t
    y = corrected[lo:hi]
    end_rel = float(data.time[e] - onset_t)

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "Time relative to plateau onset")
    plot.setLabel("left", "Signal")
    title_kind = "failed candidate" if str(plateau_source).lower() in {"failed", "rejected"} else "plateau"
    signal_suffix = "" if title_kind != "plateau" else f" | {str(plateau_source)} / {str(signal_type)}"
    plot.setTitle(f"{data.trace_name} | {title_kind} zoom{signal_suffix}")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    plot.plot(x, y, pen=pg.mkPen(CHANNEL_COLORS["corrected"], width=1.8))
    if _uses_plateau_shading(render_options):
        color = "#64748b" if str(plateau_source).lower() in {"failed", "rejected"} else "#e74c3c"
        if str(plateau_source).lower() == "short":
            color = "#1565c0"
        if str(plateau_source).lower() == "burst":
            color = "#7c3aed"
        region_item = pg.LinearRegionItem(values=(0.0, end_rel), brush=pg.mkBrush(color + "38"), pen=pg.mkPen(color, width=1.1), movable=False)
        region_item.setZValue(-8)
        plot.addItem(region_item)
    marker_range_values: List[np.ndarray] = []
    if _uses_bar_markers(render_options):
        plateau_mask = data.mask("plateau")
        local_plateau = None if plateau_mask is None else np.asarray(plateau_mask, dtype=bool)[lo:hi]
        local_spikes: List[SpikeRecord] = []
        for spike in data.spikes:
            spike_idx = int(spike.spike_index)
            if lo <= spike_idx < hi:
                local_spikes.append(
                    replace(
                        spike,
                        spike_index=spike_idx - lo,
                        candidate_index=max(0, int(spike.candidate_index) - lo),
                    )
                )
        marker_range_values = _add_bar_marker_annotations(
            plot,
            x,
            y,
            local_spikes,
            local_plateau,
            show_spike_markers=render_options.show_spike_markers,
            show_plateau_bars=_show_bar_plateau_bars(render_options),
            plateau_masks=_sliced_plateau_marker_masks(data, lo, hi),
        )
    elif render_options.label_plateaus:
        finite = y[np.isfinite(y)]
        label_y = float(np.max(finite)) if finite.size else 0.0
        label = pg.TextItem(plateau_label, color="#7f1d1d", anchor=(0.5, 1.0))
        plot.addItem(label)
        label.setPos(max(0.0, end_rel * 0.5), label_y)
    _set_y_range(plot, [y] + marker_range_values)
    plot.setXRange(-pre, post, padding=0)
    layout.addWidget(plot)
    return _finalize_widget(container, width, height)


def render_superimposed_plateaus(
    data: PreparedTrace,
    regions: Sequence[Tuple[int, int]],
    render_options: RenderOptions,
    *,
    width: int = 1400,
    height: int = 460,
) -> QtGui.QPixmap:
    _ensure_app()
    corrected = data.array("corrected")
    if corrected is None or data.time.size == 0 or not regions:
        return QtGui.QPixmap()
    pre = _time_units_from_ms(data.time, render_options.plateau_pre_ms)
    post = _time_units_from_ms(data.time, render_options.plateau_post_ms)
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "Time relative to plateau onset")
    plot.setLabel("left", "Signal")
    plot.setTitle(f"{data.trace_name} | plateau superimposed")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    values: List[np.ndarray] = []
    marker_values: List[np.ndarray] = []
    for idx, (start, _end) in enumerate(regions, 1):
        s = max(0, min(int(start), data.time.size - 1))
        e = max(s, min(int(_end) - 1, data.time.size - 1))
        onset_t = float(data.time[s])
        lo = max(0, int(np.searchsorted(data.time, onset_t - pre, side="left") - 1))
        hi = min(data.time.size, int(np.searchsorted(data.time, onset_t + post, side="right") + 1))
        if hi <= lo:
            continue
        x = data.time[lo:hi] - onset_t
        y = corrected[lo:hi]
        values.append(y)
        plot.plot(x, y, pen=pg.mkPen("#2979ff66", width=1.2))
        if _uses_bar_markers(render_options):
            plateau_end = float(data.time[e] - onset_t)
            local_plateau = (x >= 0.0) & (x <= plateau_end)
            source = gui_main._plateau_source_for_region(
                int(start),
                int(_end),
                data.mask("long_plateau"),
                data.mask("short_plateau"),
                data.mask("burst_plateau"),
            )
            marker_name = {
                "long": "long_plateau",
                "short": "short_plateau",
                "burst": "burst_plateau",
            }.get(str(source), "plateau")
            marker_values.extend(
                _add_bar_marker_annotations(
                    plot,
                    x,
                    y,
                    [],
                    local_plateau,
                    show_spike_markers=False,
                    show_plateau_bars=_show_bar_plateau_bars(render_options),
                    plateau_masks=[(marker_name, local_plateau)],
                )
            )
        elif render_options.label_plateaus and y.size:
            finite = y[np.isfinite(y)]
            if finite.size:
                label = pg.TextItem(f"P{idx:04d}", color="#7f1d1d", anchor=(0.5, 1.0))
                plot.addItem(label)
                label.setPos(0.0, float(np.max(finite)))
    plot.setXRange(-pre, post, padding=0)
    _set_y_range(plot, values + marker_values)
    layout.addWidget(plot)
    return _finalize_widget(container, width, height)


def render_preview_pixmap(
    state: TraceCuration,
    detection_params: object,
    render_options: RenderOptions,
    preview_type: str,
    event_index: int = 0,
) -> QtGui.QPixmap:
    default_half = float(detection_params.spike_zoom_half_window_ms)

    if preview_type == "trace_view":
        cropped = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
        return render_trace_channels(cropped, render_options, title_suffix="preview")
    if preview_type == "full_trace_plateau":
        cropped = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
        return render_full_trace_plateaus(cropped, render_options)
    if preview_type == "debug_preview":
        cropped = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
        channels = [channel for channel in ("corrected", "denoised", "baseline") if cropped.array(channel) is not None]
        return render_trace_channels(cropped, render_options, title_suffix="debug preview", channels=channels or ("corrected",))

    full = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=False)
    if preview_type == "spike_event":
        spikes = filtered_spikes(full, render_options)
        if not spikes:
            return QtGui.QPixmap()
        idx = max(0, min(int(event_index), len(spikes) - 1))
        return render_spike_event(full, spikes[idx], render_options, spike_label=f"S{idx + 1:04d}", default_half_window_ms=default_half)
    if preview_type == "spike_superimposed":
        return render_superimposed_spikes(full, filtered_spikes(full, render_options), render_options, default_half_window_ms=default_half)
    if preview_type == "plateau_zoom":
        regions = plateau_regions(full, render_options)
        if not regions:
            return QtGui.QPixmap()
        idx = max(0, min(int(event_index), len(regions) - 1))
        long_mask = full.mask("long_plateau")
        short_mask = full.mask("short_plateau")
        burst_mask = full.mask("burst_plateau")
        source = gui_main._plateau_source_for_region(regions[idx][0], regions[idx][1], long_mask, short_mask, burst_mask)
        signal_type = str(
            gui_main._signal_metrics_for_region(
                state.plateau_signal_classification,
                int(regions[idx][0] + full.startup_idx + full.window_offset),
                int(regions[idx][1] + full.startup_idx + full.window_offset),
            ).get("signal_type", "unknown")
        )
        return render_plateau_zoom(
            full,
            regions[idx],
            render_options,
            plateau_label=f"P{idx + 1:04d}",
            plateau_source=source,
            signal_type=signal_type,
        )
    if preview_type == "plateau_superimposed":
        return render_superimposed_plateaus(full, plateau_regions(full, render_options), render_options)
    cropped = crop_prepared_trace(full, render_options)
    return render_trace_channels(cropped, render_options, title_suffix="preview")
