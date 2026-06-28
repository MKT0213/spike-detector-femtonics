from __future__ import annotations

from dataclasses import dataclass, replace
import json
import datetime
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
)
import pyqtgraph as pg
import PySide6.QtWidgets as QtWidgets
import PySide6.QtGui as QtGui

from .detection import DETECTOR_BUILD_TAG, _butterworth_highpass, _butterworth_lowpass, detect_spikes
from .gonzalez_denoising import (
    denoise_gonzalez_adaptive_wavelet,
    denoise_trace_gonzalez_full_trace,
)
from .io import estimate_sampling_rate_hz, is_ignored_trace_column, load_excel_traces
from .models import AppSession, ArtifactParams, DetectionParams, SpikeRecord, TraceCuration
from .plateau_signal_classification import SIGNAL_TYPE_COLUMNS, classify_plateau_signals
from .pipeline import (
    ADDABLE_PIPELINE_STEP_IDS,
    PIPELINE_ORDER_POLICY_LABELS,
    PIPELINE_ORDER_SUMMARY,
    PIPELINE_PRESETS,
    PIPELINE_STEP_DEFINITIONS,
    PipelineConfig,
)
from .plot_widget import TracePlotWidget
from .preprocessing import apply_artifact_interpolation
from .render_normalization import (
    NORMALIZATION_NATIVE,
    NORMALIZATION_PERCENT_DFF,
    f0_values_for_trace,
    normalize_trace_values,
    normalize_values_with_f0,
)
from .session_io import apply_session_to_traces, load_session, save_session
from .shortcuts import bind_shortcuts
from .trace_identity import recording_groups, split_trace_identity, trace_identity_columns


PLATEAU_EXPORT_X_RANGE = (-100.0, 500.0)
DISPLAY_RENDER_SOURCE_DENOISED = "denoised"
DISPLAY_RENDER_SOURCE_CORRECTED = "corrected"
DISPLAY_RENDER_SOURCE_RAW = "raw"
DISPLAY_RENDER_SOURCE_NORMALIZED_DENOISED = "normalized_denoised"
DISPLAY_RENDER_SOURCE_HIGHPASS = "highpass"
DISPLAY_RENDER_SOURCE_LOWPASS = "lowpass"
DISPLAY_RENDER_SOURCE_ITEMS = [
    ("Denoised trace", DISPLAY_RENDER_SOURCE_DENOISED),
    ("Baseline corrected", DISPLAY_RENDER_SOURCE_CORRECTED),
    ("Raw baseline-subtracted", DISPLAY_RENDER_SOURCE_RAW),
    ("Denoised dF/F0", DISPLAY_RENDER_SOURCE_NORMALIZED_DENOISED),
    ("High-pass", DISPLAY_RENDER_SOURCE_HIGHPASS),
    ("Low-pass", DISPLAY_RENDER_SOURCE_LOWPASS),
]


def _denoising_input_normalization_label(method: str) -> str:
    if str(method) == "gonzalez_s7_dff":
        return "dF/F0 using detector baseline + S7 low-frequency branch"
    if str(method) == "gonzalez_full_trace_dff":
        return "dF/F0 using detector baseline"
    if str(method) == "gonzalez_full_trace_guarded":
        return "Corrected trace + positive-event/downward-artifact guard"
    if str(method) == "none":
        return "Off"
    return "Corrected trace"


def _render_source_label(source: str) -> str:
    for label, value in DISPLAY_RENDER_SOURCE_ITEMS:
        if str(value) == str(source):
            return str(label)
    return "Denoised trace"


def _time_axis_is_ms(time: np.ndarray) -> bool:
    """Infer whether a time axis is in milliseconds instead of seconds."""
    if len(time) < 2:
        return False
    return float(time[-1] - time[0]) > 1000.0


def _time_axis_label(time: np.ndarray) -> str:
    return "Time (ms)" if _time_axis_is_ms(time) else "Time (s)"


def _half_window_in_time_units(time: np.ndarray, half_window_ms: float) -> float:
    """Convert a half-window in ms to the native units of the provided time axis."""
    return float(half_window_ms) if _time_axis_is_ms(time) else float(half_window_ms) / 1000.0


def _sampling_rate_hz(time: np.ndarray) -> float:
    return estimate_sampling_rate_hz(time)


def _trace_sampling_rate_hz(state: TraceCuration) -> float:
    fs = state.sampling_rate_hz
    if fs is not None and np.isfinite(fs) and float(fs) > 0.0:
        return float(fs)
    return _sampling_rate_hz(state.time)


def _detection_mask(detection: object, key: str, fallback_key: str = "baseline_mask") -> np.ndarray:
    qc = getattr(detection, "qc_plot_data", {})
    baseline = np.asarray(getattr(detection, "baseline", np.array([])), dtype=float)
    fallback = np.zeros_like(baseline, dtype=bool)
    if isinstance(qc, dict):
        values = qc.get(key, qc.get(fallback_key, fallback))
    else:
        values = fallback
    arr = np.asarray(values, dtype=bool)
    if arr.shape != fallback.shape:
        out = fallback.copy()
        n = min(out.size, arr.size)
        if n > 0:
            out[:n] = arr[:n]
        return out
    return arr


def _detection_short_events(detection: object) -> np.ndarray:
    qc = getattr(detection, "qc_plot_data", {})
    if isinstance(qc, dict):
        events = np.asarray(qc.get("short_plateau_event_table", np.zeros((0, 6), dtype=float)), dtype=float)
        if events.ndim == 2 and events.shape[1] == 6:
            return events
    return np.zeros((0, 6), dtype=float)


def _detection_burst_events(detection: object) -> np.ndarray:
    qc = getattr(detection, "qc_plot_data", {})
    if isinstance(qc, dict):
        events = np.asarray(qc.get("burst_plateau_event_table", np.zeros((0, 8), dtype=float)), dtype=float)
        if events.ndim == 2 and events.shape[1] == 8:
            return events
    return np.zeros((0, 8), dtype=float)


def _adjust_event_table_for_offset(events: Optional[np.ndarray], offset: int, n: int, width: int) -> np.ndarray:
    table = np.asarray(events if events is not None else np.zeros((0, width), dtype=float), dtype=float)
    if table.ndim != 2 or table.shape[1] != width or table.shape[0] == 0:
        return np.zeros((0, width), dtype=float)
    adjusted = table.copy()
    adjusted[:, 0] -= float(offset)
    adjusted[:, 1] -= float(offset)
    keep = (adjusted[:, 1] > 0.0) & (adjusted[:, 0] < float(n))
    adjusted = adjusted[keep]
    if adjusted.size == 0:
        return np.zeros((0, width), dtype=float)
    adjusted[:, 0] = np.maximum(adjusted[:, 0], 0.0)
    adjusted[:, 1] = np.minimum(adjusted[:, 1], float(n))
    return adjusted


def _adjust_plateau_signal_rows_for_offset(
    rows: Optional[List[Dict[str, Any]]],
    offset: int,
    n: int,
) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    adjusted: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            start = int(row.get("start_index", 0))
            end = int(row.get("end_index_exclusive", start))
        except (TypeError, ValueError):
            continue
        if end <= offset or start >= offset + n:
            continue
        display_start = max(0, start - offset)
        display_end = min(int(n), max(display_start + 1, end - offset))
        out = dict(row)
        out["start_index"] = int(display_start)
        out["end_index_exclusive"] = int(display_end)
        adjusted.append(out)
    return adjusted


def _detection_burst_debug(detection: object) -> Dict[str, Any]:
    qc = getattr(detection, "qc_plot_data", {})
    if not isinstance(qc, dict):
        return {"burst_plateau_candidate_rows": []}
    return {
        "burst_plateau_candidate_rows": list(qc.get("burst_plateau_candidate_rows", [])),
    }


def _detection_highpass_trace(detection: object, enabled: bool) -> Optional[np.ndarray]:
    if not enabled:
        return None
    qc = getattr(detection, "qc_plot_data", {})
    if not isinstance(qc, dict):
        return None
    trace = qc.get("highpass_signal")
    if trace is None:
        return None
    arr = np.asarray(trace, dtype=float)
    return arr if arr.size > 0 else None


def _detection_lowpass_trace(detection: object, enabled: bool) -> Optional[np.ndarray]:
    if not enabled:
        return None
    qc = getattr(detection, "qc_plot_data", {})
    if not isinstance(qc, dict):
        return None
    trace = qc.get("lowpass_signal")
    if trace is None:
        return None
    arr = np.asarray(trace, dtype=float)
    return arr if arr.size > 0 else None


def _finite_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


def _median_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.median(np.abs(finite))) if finite.size else float("inf")


def _as_corrected_units(
    values: Optional[np.ndarray],
    corrected_reference: np.ndarray,
    correction_baseline: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return a trace in baseline-corrected units for the main plot."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    ref = np.asarray(corrected_reference, dtype=float)
    if arr.shape != ref.shape:
        return arr
    if correction_baseline is None:
        return arr
    baseline = np.asarray(correction_baseline, dtype=float)
    if baseline.shape != arr.shape:
        return arr

    current_offset = abs(_finite_median(arr) - _finite_median(ref))
    candidate = arr - baseline
    candidate_offset = abs(_finite_median(candidate) - _finite_median(ref))
    if candidate_offset < current_offset and _median_abs(candidate) < _median_abs(arr):
        return candidate
    return arr


def _raw_baseline_for_display(raw: np.ndarray, correction_baseline: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return raw-unit baseline only when it is compatible with the raw trace."""
    if correction_baseline is None:
        return None
    raw_arr = np.asarray(raw, dtype=float)
    baseline = np.asarray(correction_baseline, dtype=float)
    if baseline.shape != raw_arr.shape:
        return None
    raw_med = _finite_median(raw_arr)
    base_med = _finite_median(baseline)
    raw_scale = max(1.0, _median_abs(raw_arr - raw_med))
    if not np.isfinite(raw_med) or not np.isfinite(base_med):
        return None
    # A zero corrected-baseline should not be drawn as "Raw Baseline" when raw
    # data is hundreds of units above zero.
    if abs(raw_med - base_med) > max(50.0, 10.0 * raw_scale):
        return None
    return baseline


def _zero_baseline_for_corrected(corrected: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(corrected, dtype=float), dtype=float)


def _spike_export_dimensions(half_window_ms: float) -> Tuple[int, int, int, int]:
    """Return (single_width, single_height, row_height, combined_width) for spike PNG exports.

    Dimensions scale with the configured time window so x-axis readability is preserved
    when users increase the spike zoom window.
    """
    half = max(0.5, float(half_window_ms))
    scale = half / 5.0

    # Reference width: at +/-5 ms (10 ms total), export around half of prior width.
    single_width = int(np.clip(round(600.0 * scale), 500, 2400))
    single_height = int(np.clip(round(400.0 + 40.0 * np.log10(max(1.0, scale))), 360, 520))
    row_height = int(np.clip(round(180.0 * (scale ** 0.35)), 160, 280))
    combined_width = int(np.clip(round(700.0 * scale), 650, 2800))
    return single_width, single_height, row_height, combined_width


def _startup_exclusion_index(time: np.ndarray, startup_exclusion_ms: float) -> int:
    time_arr = np.asarray(time, dtype=float)
    if time_arr.size <= 1 or not np.isfinite(startup_exclusion_ms) or startup_exclusion_ms <= 0.0:
        return 0
    fs = _sampling_rate_hz(time_arr)
    if not np.isfinite(fs) or fs <= 0.0:
        return 0
    idx = int(round((float(startup_exclusion_ms) / 1000.0) * fs))
    return idx if 0 < idx < time_arr.size else 0


def _trim_1d_for_startup(values: Optional[np.ndarray], startup_idx: int, n: int, dtype: object = float) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        return arr
    out = np.zeros(max(0, n - startup_idx), dtype=arr.dtype)
    source = arr[:n]
    trimmed = source[startup_idx:]
    if trimmed.size:
        out[: trimmed.size] = trimmed
    return out


def _adjust_spikes_for_startup(spikes: List[SpikeRecord], startup_idx: int, n: int) -> List[SpikeRecord]:
    if startup_idx <= 0:
        return list(spikes)
    adjusted: List[SpikeRecord] = []
    for spike in spikes:
        spike_index = int(spike.spike_index)
        if spike_index < startup_idx or spike_index >= n:
            continue
        candidate_index = int(spike.candidate_index)
        adjusted.append(
            replace(
                spike,
                spike_index=spike_index - startup_idx,
                candidate_index=max(0, candidate_index - startup_idx),
            )
        )
    return adjusted


def _startup_trimmed_trace_export_data(state: TraceCuration, startup_exclusion_ms: float) -> Dict[str, Any]:
    time = np.asarray(state.time, dtype=float)
    corrected = np.asarray(state.corrected, dtype=float)
    n = min(time.size, corrected.size)
    time = time[:n]
    corrected = corrected[:n]
    startup_idx = _startup_exclusion_index(time, startup_exclusion_ms)

    return {
        "startup_idx": startup_idx,
        "time": time[startup_idx:],
        "corrected": corrected[startup_idx:],
        "raw": _trim_1d_for_startup(state.raw, startup_idx, n, dtype=float),
        "hybrid_denoised": _trim_1d_for_startup(state.hybrid_denoised, startup_idx, n, dtype=float),
        "baseline": _trim_1d_for_startup(state.baseline, startup_idx, n, dtype=float),
        "debug_threshold": _trim_1d_for_startup(state.debug_threshold, startup_idx, n, dtype=float),
        "plateau_mask": _trim_1d_for_startup(state.plateau_mask, startup_idx, n, dtype=bool),
        "long_plateau_mask": _trim_1d_for_startup(state.long_plateau_mask, startup_idx, n, dtype=bool),
        "short_plateau_mask": _trim_1d_for_startup(state.short_plateau_mask, startup_idx, n, dtype=bool),
        "burst_plateau_mask": _trim_1d_for_startup(state.burst_plateau_mask, startup_idx, n, dtype=bool),
        "burst_plateau_events": _adjust_event_table_for_offset(state.burst_plateau_events, startup_idx, n - startup_idx, 8),
        "plateau_signal_classification": _adjust_plateau_signal_rows_for_offset(
            state.plateau_signal_classification,
            startup_idx,
            n - startup_idx,
        ),
        "spikes": _adjust_spikes_for_startup(state.spikes, startup_idx, n),
    }


def _rejected_long_plateau_rows_for_export(
    debug: Optional[Dict[str, Any]],
    startup_idx: int,
    n: int,
) -> List[Dict[str, Any]]:
    if not isinstance(debug, dict):
        return []
    rows = debug.get("long_plateau_region_debug_rows", [])
    if not isinstance(rows, list):
        return []

    rejected: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or bool(row.get("accepted", False)):
            continue
        original_start = int(row.get("start_index", 0))
        original_end = int(row.get("end_index_exclusive", original_start + 1))
        if original_end <= startup_idx:
            continue
        display_start = max(0, original_start - startup_idx)
        display_end = max(display_start + 1, original_end - startup_idx)
        display_end = min(display_end, int(n))
        if display_start >= display_end or display_start >= int(n):
            continue
        display_row = dict(row)
        display_row["display_start_index"] = int(display_start)
        display_row["display_end_index_exclusive"] = int(display_end)
        rejected.append(display_row)
    return sorted(rejected, key=lambda item: float(item.get("near_miss_score", 0.0)), reverse=True)


def _render_spike_png(
    time: np.ndarray,
    corrected: np.ndarray,
    hybrid_denoised: Optional[np.ndarray],
    baseline: Optional[np.ndarray],
    threshold_line: Optional[np.ndarray],
    spike: SpikeRecord,
    half_window_ms: float = 5.0,
    width: int = 1200,
    height: int = 400,
    show_spike_markers: bool = True,
) -> "QtGui.QPixmap":
    """Render a single spike event as a PNG pixmap centered on spike."""
    spike_time = spike.spike_time

    half_window = _half_window_in_time_units(time, half_window_ms)
    x_label = _time_axis_label(time)
    
    t_lo = spike_time - half_window
    t_hi = spike_time + half_window

    idx_lo = int(np.searchsorted(time, t_lo))
    idx_hi = int(np.searchsorted(time, t_hi))
    idx_lo = max(0, idx_lo - 1)
    idx_hi = min(len(time), idx_hi + 1)

    t_win = time[idx_lo:idx_hi]
    corr_win = corrected[idx_lo:idx_hi]
    denoised_win = hybrid_denoised[idx_lo:idx_hi] if hybrid_denoised is not None else None
    base_win = baseline[idx_lo:idx_hi] if baseline is not None else None
    thresh_win = threshold_line[idx_lo:idx_hi] if threshold_line is not None else None

    # Shift time so spike is at t=0
    t_win_rescaled = t_win - spike_time

    spike_y = float(corrected[spike.spike_index]) if spike.spike_index < len(corrected) else 0.0

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    plot_height = height // 2 if denoised_win is not None else height
    main_plot = pg.PlotWidget(background="#f7f9fc")
    main_plot.showGrid(x=True, y=True, alpha=0.2)
    main_plot.addLegend(offset=(10, 8))
    if denoised_win is None:
        main_plot.setLabel("bottom", x_label)
    main_plot.setLabel("left", "Not denoised")
    main_plot.setMinimumHeight(plot_height)
    main_plot.setMaximumHeight(plot_height)
    main_plot.plot(t_win_rescaled, corr_win, pen=pg.mkPen("#2979ff", width=2.0), name="Not denoised")
    if base_win is not None:
        main_plot.plot(t_win_rescaled, base_win, pen=pg.mkPen("#ff6d00", width=1.5), name="Baseline")
    if thresh_win is not None:
        main_plot.plot(
            t_win_rescaled,
            thresh_win,
            pen=pg.mkPen("#8e24aa", width=1.2, style=Qt.PenStyle.DashLine),
            name="Detection threshold",
        )
    if show_spike_markers:
        main_plot.addItem(
            pg.ScatterPlotItem(
                x=[0.0],
                y=[spike_y],
                size=10,
                brush=pg.mkBrush("#f1c40f"),
                pen=pg.mkPen("#ffffff", width=1.0),
            )
        )

    # Add vertical line at spike (t=0)
    y_range = main_plot.viewRange()[1]
    main_plot.plot([0, 0], y_range, pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine))

    # Set x range to configured half-window.
    main_plot.setXRange(-half_window, half_window, padding=0)

    # Use one y-axis range for both panels so the denoising effect is directly comparable.
    all_values = [v for v in np.asarray(corr_win, dtype=float).flat if np.isfinite(v)]
    if denoised_win is not None:
        all_values.extend([v for v in np.asarray(denoised_win, dtype=float).flat if np.isfinite(v)])
    if spike_y and np.isfinite(spike_y):
        all_values.append(spike_y)
    finite = np.array(all_values, dtype=float)
    if finite.size > 0:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            pad = 1e-3
        else:
            pad = max(1e-3, (hi - lo) * 0.10)
        main_plot.setYRange(lo - pad, hi + pad, padding=0)

    layout.addWidget(main_plot)
    if denoised_win is not None:
        denoised_plot = pg.PlotWidget(background="#f7f9fc")
        denoised_plot.showGrid(x=True, y=True, alpha=0.2)
        denoised_plot.addLegend(offset=(10, 8))
        denoised_plot.setLabel("bottom", x_label)
        denoised_plot.setLabel("left", "State-guided denoised")
        denoised_plot.setMinimumHeight(plot_height)
        denoised_plot.setMaximumHeight(plot_height)
        denoised_plot.plot(t_win_rescaled, denoised_win, pen=pg.mkPen("#8e24aa", width=2.0), name="State-guided denoised")
        denoised_y = float(hybrid_denoised[spike.spike_index]) if spike.spike_index < len(hybrid_denoised) else 0.0
        if show_spike_markers:
            denoised_plot.addItem(
                pg.ScatterPlotItem(
                    x=[0.0],
                    y=[denoised_y],
                    size=10,
                    brush=pg.mkBrush("#f1c40f"),
                    pen=pg.mkPen("#ffffff", width=1.0),
                )
            )
        denoised_plot.plot([0, 0], denoised_plot.viewRange()[1], pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine))
        denoised_plot.setXRange(-half_window, half_window, padding=0)
        if finite.size > 0:
            denoised_plot.setYRange(lo - pad, hi + pad, padding=0)
        layout.addWidget(denoised_plot)
    container.resize(width, height)

    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _render_aligned_spikes_png(
    time: np.ndarray,
    corrected: np.ndarray,
    baseline: Optional[np.ndarray],
    threshold_line: Optional[np.ndarray],
    spikes: List[SpikeRecord],
    half_window_ms: float = 5.0,
    width: int = 1200,
    height: int = 400,
    show_spike_markers: bool = True,
) -> "QtGui.QPixmap":
    """Render all spikes overlaid on a single plot for easy comparison."""
    if not spikes:
        return None

    half_window = _half_window_in_time_units(time, half_window_ms)
    x_label = _time_axis_label(time)
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    main_plot = pg.PlotWidget(background="#f7f9fc")
    main_plot.showGrid(x=True, y=True, alpha=0.2)
    main_plot.setLabel("bottom", x_label)
    main_plot.setLabel("left", "Signal")
    main_plot.setMinimumHeight(height)
    main_plot.setMaximumHeight(height)

    # Extract and plot all spike waveforms
    spike_xs_all = []
    spike_ys_all = []
    
    for spike in spikes:
        spike_time = spike.spike_time
        t_lo = spike_time - half_window
        t_hi = spike_time + half_window

        idx_lo = int(np.searchsorted(time, t_lo))
        idx_hi = int(np.searchsorted(time, t_hi))
        idx_lo = max(0, idx_lo - 1)
        idx_hi = min(len(time), idx_hi + 1)

        if idx_lo >= idx_hi:
            continue

        t_win = time[idx_lo:idx_hi]
        corr_win = corrected[idx_lo:idx_hi]
        
        # Shift time so spike is at t=0
        t_win_rescaled = t_win - spike_time
        
        # Plot each spike waveform with semi-transparent blue (#66 = 40% alpha)
        main_plot.plot(t_win_rescaled, corr_win, pen=pg.mkPen("#2979ff66", width=1.5))
        
        spike_y = float(corrected[spike.spike_index]) if spike.spike_index < len(corrected) else 0.0
        spike_xs_all.append(0.0)
        spike_ys_all.append(spike_y)

    # Plot average baseline if available
    if baseline is not None:
        baseline_ys_avg = []
        for spike in spikes:
            spike_time = spike.spike_time
            t_lo = spike_time - half_window
            t_hi = spike_time + half_window

            idx_lo = int(np.searchsorted(time, t_lo))
            idx_hi = int(np.searchsorted(time, t_hi))
            idx_lo = max(0, idx_lo - 1)
            idx_hi = min(len(time), idx_hi + 1)

            if idx_lo >= idx_hi:
                continue

            t_win = time[idx_lo:idx_hi]
            base_win = baseline[idx_lo:idx_hi]
            t_win_rescaled = t_win - spike_time
            
            main_plot.plot(t_win_rescaled, base_win, pen=pg.mkPen("#ff6d004d", width=1.0))  # 4d = 30% alpha

    if show_spike_markers and spike_xs_all:
        scatter = pg.ScatterPlotItem(
            x=spike_xs_all,
            y=spike_ys_all,
            size=8,
            brush=pg.mkBrush("#f1c40f99"),
            pen=pg.mkPen("#ffffff", width=0.5),
        )
        main_plot.addItem(scatter)

    # Add vertical line at spike (t=0)
    y_range = main_plot.viewRange()[1]
    main_plot.plot([0, 0], y_range, pen=pg.mkPen("#ff0000", width=2.0, style=Qt.PenStyle.DashLine))

    # Set x range to configured half-window.
    main_plot.setXRange(-half_window, half_window, padding=0)

    # Compute Y range from all waveforms
    all_values = []
    for spike in spikes:
        spike_time = spike.spike_time
        t_lo = spike_time - half_window
        t_hi = spike_time + half_window

        idx_lo = int(np.searchsorted(time, t_lo))
        idx_hi = int(np.searchsorted(time, t_hi))
        idx_lo = max(0, idx_lo - 1)
        idx_hi = min(len(time), idx_hi + 1)

        if idx_lo < idx_hi:
            corr_win = corrected[idx_lo:idx_hi]
            all_values.extend([v for v in np.asarray(corr_win, dtype=float).flat if np.isfinite(v)])

    all_values.extend(spike_ys_all)
    finite = np.array(all_values, dtype=float)
    if finite.size > 0:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            pad = 1e-3
        else:
            pad = max(1e-3, (hi - lo) * 0.10)
        main_plot.setYRange(lo - pad, hi + pad, padding=0)

    layout.addWidget(main_plot)
    container.resize(width, height)

    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _render_superimposed_spikes_png(
    time: np.ndarray,
    corrected: np.ndarray,
    spikes: List[SpikeRecord],
    half_window_ms: float = 5.0,
    width: int = 1200,
    height: int = 400,
) -> "QtGui.QPixmap":
    """Render all spike waveforms superimposed (overlaid) on a single plot for variability inspection."""
    if not spikes:
        return QtGui.QPixmap()

    half_window = _half_window_in_time_units(time, half_window_ms)
    x_label = _time_axis_label(time)
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    main_plot = pg.PlotWidget(background="#f7f9fc")
    main_plot.showGrid(x=True, y=True, alpha=0.2)
    main_plot.setLabel("bottom", x_label)
    main_plot.setLabel("left", "Signal (overlaid)")
    main_plot.setMinimumHeight(height)
    main_plot.setMaximumHeight(height)

    # Colors for each spike (cycle through palette)
    colors = [
        "#2979ff",  # blue
        "#f1c40f",  # yellow
        "#e74c3c",  # red
        "#27ae60",  # green
        "#9b59b6",  # purple
        "#e67e22",  # orange
        "#1abc9c",  # teal
        "#34495e",  # dark
    ]

    all_values = []
    for spike_idx, spike in enumerate(spikes):
        spike_time = spike.spike_time
        t_lo = spike_time - half_window
        t_hi = spike_time + half_window

        idx_lo = int(np.searchsorted(time, t_lo))
        idx_hi = int(np.searchsorted(time, t_hi))
        idx_lo = max(0, idx_lo - 1)
        idx_hi = min(len(time), idx_hi + 1)

        if idx_lo >= idx_hi:
            continue

        t_win = time[idx_lo:idx_hi]
        corr_win = corrected[idx_lo:idx_hi]
        t_win_rescaled = t_win - spike_time

        color = colors[spike_idx % len(colors)]
        pen = pg.mkPen(color + "b3", width=1.5)  # b3 = ~70% alpha in hex
        main_plot.plot(t_win_rescaled, corr_win, pen=pen)

        all_values.extend([v for v in np.asarray(corr_win, dtype=float).flat if np.isfinite(v)])

    # Add vertical line at spike (t=0)
    if all_values:
        lo = float(np.min(all_values))
        hi = float(np.max(all_values))
        main_plot.plot([0, 0], [lo, hi], pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine))

    # Set ranges
    main_plot.setXRange(-half_window, half_window, padding=0)
    if all_values:
        finite = np.array(all_values, dtype=float)
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            pad = 1e-3
        else:
            pad = max(1e-3, (hi - lo) * 0.10)
        main_plot.setYRange(lo - pad, hi + pad, padding=0)

    layout.addWidget(main_plot)
    container.resize(width, height)

    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _render_superimposed_all_traces_png(
    trace_data: List[Tuple[str, np.ndarray, np.ndarray, List[SpikeRecord]]],
    half_window_ms: float = 5.0,
    width: int = 1400,
    height: int = 500,
) -> "QtGui.QPixmap":
    """Render spike waveforms from ALL traces superimposed on a single plot.

    Each trace gets a distinct color. All waveforms are aligned to their spike peak (t=0).

    Args:
        trace_data: List of (trace_name, time, corrected, spikes).
    """
    if not trace_data:
        return QtGui.QPixmap()

    # Detect time units from first trace that has data.
    half_window = float(half_window_ms)
    x_label = "Time (ms)"
    for _, time, _, _ in trace_data:
        if len(time) >= 2:
            half_window = _half_window_in_time_units(time, half_window_ms)
            x_label = _time_axis_label(time)
            break

    # Build a color palette: one distinct color per trace.
    PALETTE = [
        "#2979ff", "#e74c3c", "#27ae60", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
        "#c0392b", "#16a085", "#8e44ad", "#2c3e50",
    ]

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    main_plot = pg.PlotWidget(background="#f7f9fc")
    main_plot.showGrid(x=True, y=True, alpha=0.2)
    main_plot.setLabel("bottom", x_label)
    main_plot.setLabel("left", "Signal (a.u., trace-normalized)")
    main_plot.setMinimumHeight(height)
    main_plot.setMaximumHeight(height)
    main_plot.addLegend(offset=(10, 8))

    all_values: List[float] = []
    for trace_idx, (trace_name, time, corrected, spikes) in enumerate(trace_data):
        if not spikes:
            continue
        color = PALETTE[trace_idx % len(PALETTE)]
        pen = pg.mkPen(color, width=1.2)

        windows: List[Tuple[np.ndarray, np.ndarray]] = []
        trace_values: List[float] = []
        for spike in spikes:
            spike_time = spike.spike_time
            t_lo = spike_time - half_window
            t_hi = spike_time + half_window
            idx_lo = max(0, int(np.searchsorted(time, t_lo)) - 1)
            idx_hi = min(len(time), int(np.searchsorted(time, t_hi)) + 1)
            if idx_lo >= idx_hi:
                continue
            t_win = time[idx_lo:idx_hi] - spike_time
            corr_win = corrected[idx_lo:idx_hi]
            windows.append((t_win, corr_win))
            trace_values.extend([v for v in np.asarray(corr_win, dtype=float).flat if np.isfinite(v)])

        if not windows or not trace_values:
            continue

        trace_arr = np.asarray(trace_values, dtype=float)
        trace_center = float(np.median(trace_arr))
        # Robust per-trace scale for a.u. normalization across traces.
        trace_scale = float(np.percentile(np.abs(trace_arr - trace_center), 95.0))
        if not np.isfinite(trace_scale) or trace_scale <= 1e-12:
            trace_scale = float(np.max(np.abs(trace_arr - trace_center)))
        if not np.isfinite(trace_scale) or trace_scale <= 1e-12:
            trace_scale = 1.0

        first = True
        for t_win, corr_win in windows:
            corr_win_au = (corr_win - trace_center) / trace_scale
            # Only add legend entry for the first waveform of each trace.
            if first:
                main_plot.plot(t_win, corr_win_au, pen=pen, name=trace_name)
                first = False
            else:
                main_plot.plot(t_win, corr_win_au, pen=pen)
            all_values.extend([v for v in np.asarray(corr_win_au, dtype=float).flat if np.isfinite(v)])

    # Vertical line at t=0 (peak alignment).
    if all_values:
        lo = float(np.min(all_values))
        hi = float(np.max(all_values))
        main_plot.plot([0, 0], [lo, hi], pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine))
        pad = max(1e-3, (hi - lo) * 0.10)
        main_plot.setYRange(lo - pad, hi + pad, padding=0)

    main_plot.setXRange(-half_window, half_window, padding=0)

    layout.addWidget(main_plot)
    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _compose_stacked_spike_png(
    spike_pixmaps: List["QtGui.QPixmap"],
    title: str,
    row_height: int = 180,
    margin: int = 16,
    gap: int = 8,
) -> "QtGui.QPixmap":
    """Compose one vertical image by stacking spike PNGs for quick waveform comparison."""
    if not spike_pixmaps:
        return QtGui.QPixmap()

    scaled_rows = [
        pix.scaledToHeight(row_height, Qt.TransformationMode.SmoothTransformation)
        for pix in spike_pixmaps
        if not pix.isNull()
    ]
    if not scaled_rows:
        return QtGui.QPixmap()

    header_h = 44
    content_w = max(pix.width() for pix in scaled_rows)
    canvas_w = content_w + (margin * 2)
    content_h = sum(pix.height() for pix in scaled_rows) + max(0, len(scaled_rows) - 1) * gap
    canvas_h = header_h + margin + content_h + margin

    canvas = QtGui.QPixmap(canvas_w, canvas_h)
    canvas.fill(QtGui.QColor("#f7f9fc"))

    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    font = painter.font()
    font.setPointSize(10)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QtGui.QColor("#222222"))
    painter.drawText(margin, 28, title)

    y = header_h
    for pix in scaled_rows:
        x = (canvas_w - pix.width()) // 2
        painter.drawPixmap(x, y, pix)
        y += pix.height() + gap

    painter.end()
    return canvas


def _compose_horizontal_traces_png(
    trace_pixmaps: List["QtGui.QPixmap"],
    title: str,
    column_height: int = 320,
    margin: int = 16,
    gap: int = 14,
) -> "QtGui.QPixmap":
    """Compose one horizontal comparison image by placing each trace panel side-by-side."""
    if not trace_pixmaps:
        return QtGui.QPixmap()

    scaled_cols = [
        pix.scaledToHeight(column_height, Qt.TransformationMode.SmoothTransformation)
        for pix in trace_pixmaps
        if not pix.isNull()
    ]
    if not scaled_cols:
        return QtGui.QPixmap()

    header_h = 44
    content_w = sum(pix.width() for pix in scaled_cols) + max(0, len(scaled_cols) - 1) * gap
    canvas_w = content_w + (margin * 2)
    canvas_h = header_h + margin + column_height + margin

    canvas = QtGui.QPixmap(canvas_w, canvas_h)
    canvas.fill(QtGui.QColor("#f7f9fc"))

    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    font = painter.font()
    font.setPointSize(10)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QtGui.QColor("#222222"))
    painter.drawText(margin, 28, title)

    x = margin
    y = header_h
    for pix in scaled_cols:
        painter.drawPixmap(x, y, pix)
        x += pix.width() + gap

    painter.end()
    return canvas


def _contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, split_points)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _plateau_source_for_region(
    start: int,
    end: int,
    long_plateau_mask: Optional[np.ndarray],
    short_plateau_mask: Optional[np.ndarray],
    burst_plateau_mask: Optional[np.ndarray] = None,
) -> str:
    long_hit = False
    short_hit = False
    burst_hit = False
    if long_plateau_mask is not None:
        lm = np.asarray(long_plateau_mask, dtype=bool)
        if lm.size > start:
            long_hit = bool(np.any(lm[start : min(end, lm.size)]))
    if short_plateau_mask is not None:
        sm = np.asarray(short_plateau_mask, dtype=bool)
        if sm.size > start:
            short_hit = bool(np.any(sm[start : min(end, sm.size)]))
    if burst_plateau_mask is not None:
        bm = np.asarray(burst_plateau_mask, dtype=bool)
        if bm.size > start:
            burst_hit = bool(np.any(bm[start : min(end, bm.size)]))
    if long_hit:
        return "long"
    if short_hit:
        return "short"
    if burst_hit:
        return "burst"
    return "unknown"


def _event_metrics_for_region(events: Optional[np.ndarray], start: int, end: int) -> Dict[str, float]:
    if events is None:
        return {}
    table = np.asarray(events, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] < 8:
        return {}
    best: Optional[np.ndarray] = None
    best_overlap = 0
    for row in table:
        row_start = int(row[0])
        row_end = int(row[1])
        overlap = max(0, min(end, row_end) - max(start, row_start))
        if overlap > best_overlap:
            best = row
            best_overlap = overlap
    if best is None or best_overlap <= 0:
        return {}
    return {
        "burst_floor_support_fraction": float(best[4]),
        "burst_peak_dominance_noise": float(best[5]),
        "burst_iqr_over_noise": float(best[6]),
        "burst_confidence": float(best[7]),
    }


def _unknown_signal_metrics() -> Dict[str, object]:
    return {
        "signal_type": "unknown",
        "signal_confidence": float("nan"),
        "raw_highfreq_rms": float("nan"),
        "denoised_highfreq_rms": float("nan"),
        "residual_rms": float("nan"),
        "residual_fraction": float("nan"),
        "noise_reduction_fraction": float("nan"),
        "raw_peak_density_hz": float("nan"),
        "denoised_peak_density_hz": float("nan"),
        "preserved_peak_fraction": float("nan"),
    }


def _signal_metrics_for_region(
    rows: Optional[List[Dict[str, Any]]],
    start: int,
    end: int,
) -> Dict[str, object]:
    if not isinstance(rows, list):
        return _unknown_signal_metrics()
    best: Optional[Dict[str, Any]] = None
    best_overlap = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            row_start = int(row.get("start_index", 0))
            row_end = int(row.get("end_index_exclusive", row_start))
        except (TypeError, ValueError):
            continue
        overlap = max(0, min(end, row_end) - max(start, row_start))
        if overlap > best_overlap:
            best = row
            best_overlap = overlap
    if best is None or best_overlap <= 0:
        return _unknown_signal_metrics()
    metrics = _unknown_signal_metrics()
    for key in SIGNAL_TYPE_COLUMNS:
        if key in best:
            metrics[key] = best[key]
    return metrics


def _plateau_rows(
    trace_name: str,
    time: np.ndarray,
    plateau_mask: np.ndarray,
    long_plateau_mask: Optional[np.ndarray] = None,
    short_plateau_mask: Optional[np.ndarray] = None,
    burst_plateau_mask: Optional[np.ndarray] = None,
    burst_plateau_events: Optional[np.ndarray] = None,
    plateau_signal_classification: Optional[List[Dict[str, Any]]] = None,
    sampling_rate_hz: Optional[float] = None,
) -> List[Dict[str, object]]:
    regions = _contiguous_true_regions(plateau_mask)
    if not regions:
        return []

    rows: List[Dict[str, object]] = []
    n_time = int(len(time))
    for idx, (start, end) in enumerate(regions, 1):
        if n_time <= 0:
            continue
        s = max(0, min(int(start), n_time - 1))
        e_exclusive = max(0, min(int(end), n_time))
        e = max(s, min(e_exclusive - 1, n_time - 1))
        source = _plateau_source_for_region(start, end, long_plateau_mask, short_plateau_mask, burst_plateau_mask)
        row = {
            "trace_name": trace_name,
            "plateau_index": int(idx),
            "detector_source": source,
            "start_index": int(start),
            "end_index_exclusive": int(end),
            "onset_time_ms": _absolute_time_ms(time, s, sampling_rate_hz),
            "start_time": float(time[s]),
            "end_time": float(time[e]),
            "sample_count": int(max(0, end - start)),
            "duration_time_units": float(time[e] - time[s]),
        }
        if source == "burst":
            row.update(_event_metrics_for_region(burst_plateau_events, start, end))
        row.update(_signal_metrics_for_region(plateau_signal_classification, start, end))
        rows.append(row)
    return rows


def _render_full_trace_plateau_png(
    trace_name: str,
    time: np.ndarray,
    corrected: np.ndarray,
    plateau_mask: np.ndarray,
    long_plateau_mask: Optional[np.ndarray] = None,
    short_plateau_mask: Optional[np.ndarray] = None,
    burst_plateau_mask: Optional[np.ndarray] = None,
    width: int = 1800,
    height: int = 500,
) -> "QtGui.QPixmap":
    """Render full trace with plateau regions highlighted and labeled by code."""
    if len(time) == 0 or len(corrected) == 0:
        return QtGui.QPixmap()

    n = int(min(len(time), len(corrected), len(plateau_mask)))
    if n <= 0:
        return QtGui.QPixmap()

    t = np.asarray(time[:n], dtype=float)
    y = np.asarray(corrected[:n], dtype=float)
    mask = np.asarray(plateau_mask[:n], dtype=bool)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "Time")
    plot.setLabel("left", "Signal")
    plot.setTitle(f"{trace_name} | Full trace with plateaus")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    plot.plot(t, y, pen=pg.mkPen("#2979ff", width=1.5))

    finite = y[np.isfinite(y)]
    if finite.size > 0:
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if y_max <= y_min:
            y_pad = 1e-3
        else:
            y_pad = max(1e-3, (y_max - y_min) * 0.10)
        plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0)

    def _add_shaded_regions(source_mask: Optional[np.ndarray], brush: object, pen: object) -> None:
        if source_mask is None:
            return
        source = np.asarray(source_mask[:n], dtype=bool)
        for start, end in _contiguous_true_regions(source):
            s = max(0, min(start, n - 1))
            e = max(s, min(end - 1, n - 1))
            x0 = float(t[s])
            x1 = float(t[e])
            region = pg.LinearRegionItem(
                values=(x0, x1),
                brush=brush,
                pen=pen,
                movable=False,
            )
            region.setZValue(-8)
            plot.addItem(region)

    red_mask = long_plateau_mask if long_plateau_mask is not None else mask
    _add_shaded_regions(red_mask, pg.mkBrush(231, 76, 60, 55), pg.mkPen(192, 57, 43, 0))
    _add_shaded_regions(short_plateau_mask, pg.mkBrush(21, 101, 192, 50), pg.mkPen(21, 101, 192, 0))
    _add_shaded_regions(burst_plateau_mask, pg.mkBrush(124, 58, 237, 50), pg.mkPen(124, 58, 237, 0))

    if n >= 2:
        plot.setXRange(float(t[0]), float(t[-1]), padding=0)

    layout.addWidget(plot)
    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _render_plateau_zoom_png(
    trace_name: str,
    time: np.ndarray,
    corrected: np.ndarray,
    start_idx: int,
    end_idx_exclusive: int,
    plateau_source: str = "long",
    signal_type: str = "unknown",
    shade_region: bool = True,
    width: int = 1400,
    height: int = 420,
) -> "QtGui.QPixmap":
    """Render a zoomed plateau window where plateau onset is re-labeled to x=0."""
    n = int(min(len(time), len(corrected)))
    if n <= 0:
        return QtGui.QPixmap()

    s = max(0, min(int(start_idx), n - 1))
    e_exclusive = max(s + 1, min(int(end_idx_exclusive), n))
    e = max(s, e_exclusive - 1)

    t = np.asarray(time[:n], dtype=float)
    y = np.asarray(corrected[:n], dtype=float)
    onset_t = float(t[s])
    end_t = float(t[e])

    win_lo_t = onset_t + PLATEAU_EXPORT_X_RANGE[0]
    win_hi_t = onset_t + PLATEAU_EXPORT_X_RANGE[1]
    lo = max(0, int(np.searchsorted(t, win_lo_t, side="left") - 1))
    hi = min(n, int(np.searchsorted(t, win_hi_t, side="right") + 1))
    if hi <= lo:
        return QtGui.QPixmap()

    t_win = t[lo:hi]
    y_win = y[lo:hi]
    x_rel = t_win - onset_t
    plateau_end_rel = end_t - onset_t

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

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
    plot.setTitle(f"{trace_name} | {title_kind} zoom{signal_suffix}")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    plot.plot(x_rel, y_win, pen=pg.mkPen("#2979ff", width=1.8))

    source = str(plateau_source).lower()
    if source == "short":
        region_brush = pg.mkBrush(21, 101, 192, 50)
        region_pen = pg.mkPen("#1565c0", width=1.1)
    elif source == "burst":
        region_brush = pg.mkBrush(124, 58, 237, 50)
        region_pen = pg.mkPen("#7c3aed", width=1.1)
    elif source in {"failed", "rejected"}:
        region_brush = pg.mkBrush(100, 116, 139, 45)
        region_pen = pg.mkPen("#64748b", width=1.1)
    else:
        region_brush = pg.mkBrush(231, 76, 60, 55)
        region_pen = pg.mkPen("#c0392b", width=1.1)

    if shade_region:
        plateau_region = pg.LinearRegionItem(
            values=(0.0, float(plateau_end_rel)),
            brush=region_brush,
            pen=region_pen,
            movable=False,
        )
        plateau_region.setZValue(-8)
        plot.addItem(plateau_region)

    finite = y_win[np.isfinite(y_win)]
    if finite.size > 0:
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        y_pad = max(1e-3, (y_max - y_min) * 0.10) if y_max > y_min else 1e-3
        plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0)

    plot.setXRange(PLATEAU_EXPORT_X_RANGE[0], PLATEAU_EXPORT_X_RANGE[1], padding=0)

    layout.addWidget(plot)
    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _set_plot_y_range(plot: object, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    y_pad = max(1e-3, (y_max - y_min) * 0.10) if y_max > y_min else 1e-3
    plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0)


def _add_trace_plateau_shading(plot: object, time: np.ndarray, plateau_mask: Optional[np.ndarray]) -> None:
    if plateau_mask is None or time.size == 0:
        return
    mask = np.asarray(plateau_mask, dtype=bool)
    n = min(int(time.size), int(mask.size))
    if n <= 0:
        return
    t = np.asarray(time[:n], dtype=float)
    for start, end in _contiguous_true_regions(mask[:n]):
        s = max(0, min(int(start), n - 1))
        e = max(s, min(int(end) - 1, n - 1))
        region = pg.LinearRegionItem(
            values=(float(t[s]), float(t[e])),
            brush=pg.mkBrush(231, 76, 60, 55),
            pen=pg.mkPen(192, 57, 43, 0),
            movable=False,
        )
        region.setZValue(-8)
        plot.addItem(region)


def _render_whole_trace_png(
    trace_name: str,
    time: np.ndarray,
    signal: np.ndarray,
    title_suffix: str,
    line_color: str = "#2979ff",
    plateau_mask: Optional[np.ndarray] = None,
    spikes: Optional[List[SpikeRecord]] = None,
    width: int = 1800,
    height: int = 500,
    show_spike_markers: bool = True,
) -> "QtGui.QPixmap":
    n = int(min(len(time), len(signal)))
    if n <= 0:
        return QtGui.QPixmap()

    t = np.asarray(time[:n], dtype=float)
    y = np.asarray(signal[:n], dtype=float)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "Time")
    plot.setLabel("left", "Signal")
    plot.setTitle(f"{trace_name} | {title_suffix}")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    plot.plot(t, y, pen=pg.mkPen(line_color, width=1.5))
    _add_trace_plateau_shading(plot, t, plateau_mask)

    if show_spike_markers and spikes:
        points: List[tuple[float, float]] = []
        for spike in sorted(spikes, key=lambda item: int(item.spike_index)):
            idx = int(spike.spike_index)
            if 0 <= idx < n and np.isfinite(t[idx]) and np.isfinite(y[idx]):
                points.append((float(t[idx]), float(y[idx])))
        if points:
            plot.addItem(
                pg.ScatterPlotItem(
                    x=[point[0] for point in points],
                    y=[point[1] for point in points],
                    size=7,
                    brush=pg.mkBrush("#f1c40f"),
                    pen=pg.mkPen("#7a5c00", width=0.8),
                )
            )

    _set_plot_y_range(plot, y)
    if n >= 2:
        plot.setXRange(float(t[0]), float(t[-1]), padding=0)

    layout.addWidget(plot)
    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _render_raw_processed_comparison_png(
    trace_name: str,
    time: np.ndarray,
    raw: np.ndarray,
    processed: np.ndarray,
    processed_label: str,
    processed_color: str,
    width: int = 1800,
    height: int = 700,
) -> "QtGui.QPixmap":
    n = int(min(len(time), len(raw), len(processed)))
    if n <= 0:
        return QtGui.QPixmap()

    t = np.asarray(time[:n], dtype=float)
    raw_y = np.asarray(raw[:n], dtype=float)
    processed_y = np.asarray(processed[:n], dtype=float)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    panel_h = max(1, height // 2)
    raw_plot = pg.PlotWidget(background="#f7f9fc")
    raw_plot.showGrid(x=True, y=True, alpha=0.2)
    raw_plot.setLabel("left", "Raw")
    raw_plot.setTitle(f"{trace_name} | raw vs {processed_label}")
    raw_plot.setMinimumHeight(panel_h)
    raw_plot.setMaximumHeight(panel_h)
    raw_plot.plot(t, raw_y, pen=pg.mkPen("#455a64", width=1.4))
    _set_plot_y_range(raw_plot, raw_y)
    if n >= 2:
        raw_plot.setXRange(float(t[0]), float(t[-1]), padding=0)
    layout.addWidget(raw_plot)

    processed_plot = pg.PlotWidget(background="#f7f9fc")
    processed_plot.showGrid(x=True, y=True, alpha=0.2)
    processed_plot.setLabel("bottom", "Time")
    processed_plot.setLabel("left", processed_label)
    processed_plot.setMinimumHeight(panel_h)
    processed_plot.setMaximumHeight(panel_h)
    processed_plot.plot(t, processed_y, pen=pg.mkPen(processed_color, width=1.4))
    _set_plot_y_range(processed_plot, processed_y)
    if n >= 2:
        processed_plot.setXRange(float(t[0]), float(t[-1]), padding=0)
    layout.addWidget(processed_plot)

    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _collect_plateau_windows_aligned_to_onset(
    time: np.ndarray,
    corrected: np.ndarray,
    regions: List[Tuple[int, int]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    n = int(min(len(time), len(corrected)))
    if n <= 0 or not regions:
        return windows

    t = np.asarray(time[:n], dtype=float)
    y = np.asarray(corrected[:n], dtype=float)

    for start, end_exclusive in regions:
        s = max(0, min(int(start), n - 1))

        onset_t = float(t[s])
        lo_t = onset_t + PLATEAU_EXPORT_X_RANGE[0]
        hi_t = onset_t + PLATEAU_EXPORT_X_RANGE[1]
        lo = max(0, int(np.searchsorted(t, lo_t, side="left") - 1))
        hi = min(n, int(np.searchsorted(t, hi_t, side="right") + 1))
        if hi <= lo:
            continue

        t_win = t[lo:hi] - onset_t
        y_win = y[lo:hi]
        if t_win.size == 0 or y_win.size == 0:
            continue
        windows.append((t_win, y_win))

    return windows


def _render_superimposed_plateaus_png(
    trace_name: str,
    time: np.ndarray,
    corrected: np.ndarray,
    plateau_mask: np.ndarray,
    width: int = 1400,
    height: int = 450,
) -> "QtGui.QPixmap":
    """Render all plateau windows from one trace, overlaid and aligned to plateau onset (x=0)."""
    regions = _contiguous_true_regions(np.asarray(plateau_mask, dtype=bool))
    windows = _collect_plateau_windows_aligned_to_onset(time, corrected, regions)
    if not windows:
        return QtGui.QPixmap()

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "Time relative to plateau onset")
    plot.setLabel("left", "Signal")
    plot.setTitle(f"{trace_name} | plateau superimposed")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)

    all_values: List[float] = []
    for t_win, y_win in windows:
        plot.plot(t_win, y_win, pen=pg.mkPen("#2979ff66", width=1.2))
        arr = np.asarray(y_win, dtype=float)
        all_values.extend([v for v in arr.flat if np.isfinite(v)])

    if all_values:
        y_vals = np.asarray(all_values, dtype=float)
        y_lo = float(np.min(y_vals))
        y_hi = float(np.max(y_vals))
        y_pad = max(1e-3, (y_hi - y_lo) * 0.10) if y_hi > y_lo else 1e-3
        plot.setYRange(y_lo - y_pad, y_hi + y_pad, padding=0)

    plot.setXRange(PLATEAU_EXPORT_X_RANGE[0], PLATEAU_EXPORT_X_RANGE[1], padding=0)

    layout.addWidget(plot)
    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _render_superimposed_all_traces_plateaus_png(
    trace_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    width: int = 1500,
    height: int = 500,
) -> "QtGui.QPixmap":
    """Render all plateau windows from all traces, overlaid and onset-aligned in a.u."""
    if not trace_data:
        return QtGui.QPixmap()

    palette = [
        "#2979ff", "#e74c3c", "#27ae60", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
        "#c0392b", "#16a085", "#8e44ad", "#2c3e50",
    ]

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    plot = pg.PlotWidget(background="#f7f9fc")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "Time relative to plateau onset")
    plot.setLabel("left", "Signal (a.u., trace-normalized)")
    plot.setTitle("All traces | plateau superimposed")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    plot.addLegend(offset=(10, 8))

    all_values: List[float] = []
    for idx, (trace_name, time, corrected, plateau_mask) in enumerate(trace_data):
        regions = _contiguous_true_regions(np.asarray(plateau_mask, dtype=bool))
        windows = _collect_plateau_windows_aligned_to_onset(time, corrected, regions)
        if not windows:
            continue

        trace_values: List[float] = []
        for _, y_win in windows:
            arr = np.asarray(y_win, dtype=float)
            trace_values.extend([v for v in arr.flat if np.isfinite(v)])
        if not trace_values:
            continue

        trace_arr = np.asarray(trace_values, dtype=float)
        trace_center = float(np.median(trace_arr))
        trace_scale = float(np.percentile(np.abs(trace_arr - trace_center), 95.0))
        if (not np.isfinite(trace_scale)) or trace_scale <= 1e-12:
            trace_scale = float(np.max(np.abs(trace_arr - trace_center)))
        if (not np.isfinite(trace_scale)) or trace_scale <= 1e-12:
            trace_scale = 1.0

        color = palette[idx % len(palette)]
        pen = pg.mkPen(color + "99", width=1.1)
        first = True
        for t_win, y_win in windows:
            y_win_au = (y_win - trace_center) / trace_scale
            if first:
                plot.plot(t_win, y_win_au, pen=pen, name=trace_name)
                first = False
            else:
                plot.plot(t_win, y_win_au, pen=pen)

            arr = np.asarray(y_win_au, dtype=float)
            all_values.extend([v for v in arr.flat if np.isfinite(v)])

    if all_values:
        y_vals = np.asarray(all_values, dtype=float)
        y_lo = float(np.min(y_vals))
        y_hi = float(np.max(y_vals))
        y_pad = max(1e-3, (y_hi - y_lo) * 0.10) if y_hi > y_lo else 1e-3
        plot.setYRange(y_lo - y_pad, y_hi + y_pad, padding=0)

    plot.setXRange(PLATEAU_EXPORT_X_RANGE[0], PLATEAU_EXPORT_X_RANGE[1], padding=0)

    layout.addWidget(plot)
    container.resize(width, height)
    container.hide()
    QtWidgets.QApplication.processEvents()
    pix = QtWidgets.QWidget.grab(container)
    container.close()
    return pix


def _nan_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


TRACE_SUMMARY_CONFIG = {
    # Output filenames under Result/export_YYYYMMDD_HHMMSS/
    "csv_name": "trace_summary_stats.csv",
    "events_csv_name": "plateau_event_stats.csv",
    "qc_csv_name": "trace_qc_summary.csv",
    "spike_events_csv_name": "spike_event_stats.csv",
    "spike_summary_csv_name": "trace_spike_summary_stats.csv",
    "png_name": "summary_qc_report.png",
    "spike_png_name": "spike_statistics_summary.png",
    "pdf_name": "summary_qc_report.pdf",
    "save_pdf": True,
    "png_dpi": 400,
    # CSV schema
    "id_col": "trace_id",
    "duration_col": "duration_ms",
    "fwhm_col": "fwhm_ms",
    "plateau_sd_col": "plateau_sd",
    "mean_amp_col": "mean_amp_above_baseline",
    # Display labels/units
    "units": {
        "duration_ms": "ms",
        "fwhm_ms": "ms",
        "plateau_sd": "a.u.",
        "mean_amp_above_baseline": "a.u.",
    },
}

EXPORT_TABLES_DIR = "tables"
EXPORT_REPORTS_DIR = "reports"
EXPORT_RELOAD_DIR = "reload"
EXPORT_MANIFEST_NAME = "export_manifest.json"

PLATEAU_EVENT_COLUMNS = [
    "trace_id",
    "recording_id",
    "roi_id",
    "plateau_id",
    "detector_source",
    *SIGNAL_TYPE_COLUMNS,
    "start_index",
    "end_index_exclusive",
    "onset_time_ms",
    "start_time",
    "end_time",
    "duration_ms",
    "fwhm_ms",
    "rise_time_ms",
    "decay_time_ms",
    "plateau_sd",
    "mean_amp_above_baseline",
    "peak_amp_above_baseline",
    "sample_count",
]

TRACE_SUMMARY_COLUMNS = [
    "trace_id",
    "recording_id",
    "roi_id",
    "plateau_count",
    "first_onset_ms",
    "mean_duration_ms",
    "median_duration_ms",
    "mean_fwhm_ms",
    "median_fwhm_ms",
    "mean_rise_time_ms",
    "median_rise_time_ms",
    "mean_peak_amp_above_baseline",
    "median_peak_amp_above_baseline",
    "mean_plateau_sd",
    "median_plateau_sd",
    "total_plateau_time_ms",
    "plateau_time_fraction",
    "mean_inter_plateau_interval_ms",
    "median_inter_plateau_interval_ms",
]

SPIKE_EVENT_COLUMNS = [
    "trace_id",
    "recording_id",
    "roi_id",
    "spike_id",
    "peak_index",
    "peak_time",
    "amplitude",
    "delta_f_over_f0",
    "snr",
    "rise_time_ms",
    "fwhm_ms",
    "return_to_baseline_time_ms",
    "post_spike_level",
    "isi_ms",
    "instantaneous_firing_rate_hz",
    "source",
]


QC_TRACE_METRICS = {
    "plateau_count": "Plateau count",
    "median_plateau_sd": "Median plateau SD",
    "median_duration_ms": "Median duration",
    "median_fwhm_ms": "Median FWHM",
    "plateau_time_fraction": "Plateau time fraction",
}

QC_EVENT_METRICS = {
    "duration_ms": "duration_qc",
    "fwhm_ms": "fwhm_qc",
    "peak_amp_above_baseline": "amplitude_qc",
    "plateau_sd": "noise_qc",
}

QC_OUTLIER_THRESHOLD = 4.5


def _time_to_ms_scale(time: np.ndarray, sampling_rate_hz: Optional[float] = None) -> float:
    t = np.asarray(time, dtype=float)
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    dt_med = float(np.median(dt))
    sample_period_ms = _time_sample_period_ms(t, sampling_rate_hz)
    if not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
        return 1.0
    return float(sample_period_ms / dt_med)


def _absolute_time_ms(time: np.ndarray, index: int, sampling_rate_hz: Optional[float] = None) -> float:
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return float("nan")
    idx = max(0, min(int(index), t.size - 1))
    value = float(t[idx])
    if not np.isfinite(value):
        return float("nan")
    return float(value * _time_to_ms_scale(t, sampling_rate_hz))


def _time_sample_period_ms(time: np.ndarray, sampling_rate_hz: Optional[float] = None) -> float:
    fs = float(sampling_rate_hz) if sampling_rate_hz is not None else estimate_sampling_rate_hz(time)
    if not np.isfinite(fs) or fs <= 0.0:
        return float("nan")
    return float(1000.0 / fs)


def _trace_total_time_ms(time: np.ndarray, sampling_rate_hz: Optional[float] = None) -> float:
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float("nan")
    sample_period_ms = _time_sample_period_ms(t, sampling_rate_hz)
    if not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
        return float("nan")
    return float(t.size * sample_period_ms)


def _finite_numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([], dtype=float)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _robust_high_outlier_mask(values: np.ndarray, threshold: float = QC_OUTLIER_THRESHOLD) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.zeros(arr.shape, dtype=bool)
    finite = np.isfinite(arr)
    finite_vals = arr[finite]
    if finite_vals.size < 4:
        return mask

    median = float(np.median(finite_vals))
    mad = float(np.median(np.abs(finite_vals - median)))
    if mad > 0.0 and np.isfinite(mad):
        robust_z = 0.6745 * (arr - median) / mad
        return finite & (robust_z > threshold)

    q1, q3 = np.percentile(finite_vals, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr > 0.0 and np.isfinite(iqr):
        return finite & (arr > (float(q3) + 3.0 * iqr))

    max_val = float(np.max(finite_vals))
    if max_val > median:
        return finite & (arr == max_val)

    return mask


def _label_outliers(df: pd.DataFrame, value_col: str, flag_col: str) -> None:
    if value_col not in df.columns:
        df[flag_col] = ""
        return
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    outliers = _robust_high_outlier_mask(vals)
    labels = np.full(vals.shape, "", dtype=object)
    labels[outliers] = "high_outlier"
    df[flag_col] = labels


def _add_event_qc_flags(events_df: pd.DataFrame) -> pd.DataFrame:
    out = events_df.copy()
    for value_col, flag_col in QC_EVENT_METRICS.items():
        _label_outliers(out, value_col, flag_col)
    return out


def _trace_duration_ms(time: np.ndarray, plateau_mask: Optional[np.ndarray]) -> float:
    if plateau_mask is None:
        return 0.0
    mask = np.asarray(plateau_mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0

    t = np.asarray(time, dtype=float)
    n = min(mask.size, t.size)
    if n <= 1:
        return 0.0
    mask = mask[:n]
    t = t[:n]

    sample_period_ms = _time_sample_period_ms(t)
    if not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
        return 0.0
    return float(np.sum(mask) * sample_period_ms)


def _trace_plateau_sd(corrected: np.ndarray, plateau_mask: Optional[np.ndarray]) -> float:
    if plateau_mask is None:
        return float("nan")
    y = np.asarray(corrected, dtype=float)
    m = np.asarray(plateau_mask, dtype=bool)
    n = min(y.size, m.size)
    if n <= 1:
        return float("nan")
    vals = y[:n][m[:n]]
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    return float(np.std(vals, ddof=0))


def _event_fwhm_samples(amplitude: np.ndarray) -> float:
    y = np.asarray(amplitude, dtype=float).copy()
    if y.size < 2:
        return float("nan")

    finite = np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return float("nan")
    if not np.all(finite):
        idx = np.arange(y.size, dtype=float)
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])

    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        return float("nan")

    half = 0.5 * peak_val

    li = peak_idx
    while li > 0 and y[li] >= half:
        li -= 1
    if li < peak_idx and y[li] < half and y[li + 1] != y[li]:
        left = float(li) + (half - float(y[li])) / float(y[li + 1] - y[li])
    else:
        left = float(li)

    ri = peak_idx
    while ri < (y.size - 1) and y[ri] >= half:
        ri += 1
    if ri > peak_idx and y[ri] < half and y[ri] != y[ri - 1]:
        right = float(ri - 1) + (half - float(y[ri - 1])) / float(y[ri] - y[ri - 1])
    else:
        right = float(ri)

    width = float(max(0.0, right - left))
    if width <= 0.0:
        width = float(max(1, y.size - 1))
    return width


def _rise_decay_time_samples(amplitude: np.ndarray) -> tuple[float, float]:
    """Return (rise_time_samples, decay_time_samples) using 10-90% of peak amplitude.

    Rise  = first crossing of 10% to first crossing of 90%, before the peak.
    Decay = last crossing of 90% to first crossing of 10%, after the peak.
    """
    y = np.asarray(amplitude, dtype=float).copy()
    if y.size < 3:
        return float("nan"), float("nan")
    finite = np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return float("nan"), float("nan")
    if not np.all(finite):
        idx = np.arange(y.size, dtype=float)
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])

    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        return float("nan"), float("nan")

    lo = 0.10 * peak_val
    hi = 0.90 * peak_val

    # Rise: scan pre-peak for first 10% crossing to first 90% crossing.
    rise_start: float = 0.0
    rise_end: float = float(peak_idx)
    for i in range(peak_idx):
        if y[i] >= lo:
            rise_start = float(i)
            break
    for i in range(peak_idx):
        if y[i] >= hi:
            rise_end = float(i)
            break
    rise_samples = float(max(0.0, rise_end - rise_start))

    # Decay: scan post-peak for last 90% crossing to first 10% crossing.
    decay_start: float = float(peak_idx)
    decay_end: float = float(y.size - 1)
    for i in range(y.size - 1, peak_idx - 1, -1):
        if y[i] >= hi:
            decay_start = float(i)
            break
    for i in range(peak_idx, y.size):
        if y[i] <= lo:
            decay_end = float(i)
            break
    decay_samples = float(max(0.0, decay_end - decay_start))

    return rise_samples, decay_samples


def _trace_plateau_fwhm_ms(state: TraceCuration) -> float:
    if state.plateau_mask is None:
        return float("nan")

    y = np.asarray(state.corrected, dtype=float)
    b = np.asarray(state.baseline, dtype=float) if state.baseline is not None else np.zeros_like(y)
    t = np.asarray(state.time, dtype=float)
    m = np.asarray(state.plateau_mask, dtype=bool)

    n = min(y.size, b.size, t.size, m.size)
    if n <= 2:
        return float("nan")

    y = y[:n]
    b = b[:n]
    t = t[:n]
    m = m[:n]
    if not np.any(m):
        return float("nan")

    sample_period_ms = _time_sample_period_ms(t, _trace_sampling_rate_hz(state))
    if not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
        return float("nan")

    widths_ms: List[float] = []
    for start, end in _contiguous_true_regions(m):
        if end <= start or (end - start) < 3:
            continue
        amp = y[start:end] - b[start:end]
        width_samples = _event_fwhm_samples(amp)
        if np.isfinite(width_samples):
            widths_ms.append(float(width_samples * sample_period_ms))

    if not widths_ms:
        return float("nan")
    return float(np.mean(np.asarray(widths_ms, dtype=float)))


def _trace_mean_amp_above_baseline(state: TraceCuration) -> float:
    if state.plateau_mask is None:
        return float("nan")

    y = np.asarray(state.corrected, dtype=float)
    b = np.asarray(state.baseline, dtype=float) if state.baseline is not None else np.zeros_like(y)
    m = np.asarray(state.plateau_mask, dtype=bool)
    n = min(y.size, b.size, m.size)
    if n <= 0:
        return float("nan")

    amp = y[:n] - b[:n]
    vals = amp[m[:n]]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _interp_crossing_sample(y0: float, y1: float, level: float, i0: int, i1: int) -> float:
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
        return float("nan")
    return float(i0 + ((float(level) - float(y0)) / (float(y1) - float(y0))) * float(i1 - i0))


def _pre_peak_crossing(y: np.ndarray, level: float, start: int, peak: int) -> float:
    for idx in range(int(peak) - 1, int(start) - 1, -1):
        y0 = float(y[idx])
        y1 = float(y[idx + 1])
        if y0 <= level <= y1:
            return _interp_crossing_sample(y0, y1, level, idx, idx + 1)
    return float("nan")


def _post_peak_crossing(y: np.ndarray, level: float, peak: int, stop: int) -> float:
    for idx in range(int(peak), int(stop) - 1):
        y0 = float(y[idx])
        y1 = float(y[idx + 1])
        if y0 >= level >= y1:
            return _interp_crossing_sample(y0, y1, level, idx, idx + 1)
    return float("nan")


def _finite_spike_metric(spike: SpikeRecord, name: str) -> float:
    try:
        value = float(getattr(spike, name, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _measure_spike_waveform_stats(state: TraceCuration, spike: SpikeRecord, half_window_ms: float = 5.0) -> Dict[str, float]:
    cached = {
        "rise_time_ms": _finite_spike_metric(spike, "rise_time_ms"),
        "fwhm_ms": _finite_spike_metric(spike, "fwhm_ms"),
        "return_to_baseline_time_ms": _finite_spike_metric(spike, "return_to_baseline_time_ms"),
        "post_spike_level": _finite_spike_metric(spike, "post_spike_level"),
    }

    def _with_cached(computed: Dict[str, float]) -> Dict[str, float]:
        out = dict(computed)
        if cached["rise_time_ms"] >= 0.0:
            out["rise_time_ms"] = cached["rise_time_ms"]
        if cached["fwhm_ms"] > 0.0:
            out["fwhm_ms"] = cached["fwhm_ms"]
        if cached["return_to_baseline_time_ms"] >= 0.0:
            out["return_to_baseline_time_ms"] = cached["return_to_baseline_time_ms"]
        if np.isfinite(cached["post_spike_level"]):
            out["post_spike_level"] = cached["post_spike_level"]
        return out

    nan_result = {
        "rise_time_ms": float("nan"),
        "fwhm_ms": float("nan"),
        "return_to_baseline_time_ms": float("nan"),
        "post_spike_level": float("nan"),
    }

    y = np.asarray(state.corrected, dtype=float)
    baseline = np.asarray(state.baseline, dtype=float) if state.baseline is not None else np.zeros_like(y)
    n = min(y.size, baseline.size)
    if n <= 2:
        return _with_cached(nan_result)

    peak = int(spike.spike_index)
    if peak <= 0 or peak >= n:
        return _with_cached(nan_result)

    fs = _trace_sampling_rate_hz(state)
    window = max(1, int(round((float(half_window_ms) / 1000.0) * fs)))
    start = max(0, peak - window)
    stop = min(n, peak + window + 1)
    amp = y[:n] - baseline[:n]
    if stop - start < 3 or not np.all(np.isfinite(amp[start:stop])):
        return _with_cached(nan_result)

    peak_amp = float(amp[peak])
    if not np.isfinite(peak_amp) or peak_amp <= 0.0:
        return _with_cached(nan_result)

    left = amp[start : peak + 1]
    right = amp[peak:stop]
    shoulder = max(float(np.min(left)), float(np.min(right)))
    local_peak_amp = peak_amp - shoulder
    if not np.isfinite(local_peak_amp) or local_peak_amp <= 0.0:
        return _with_cached(nan_result)

    lo = shoulder + (0.10 * local_peak_amp)
    hi = shoulder + (0.90 * local_peak_amp)
    half = shoulder + (0.50 * local_peak_amp)
    sample_to_ms = 1000.0 / max(float(fs), 1e-9)

    rise_10 = _pre_peak_crossing(amp, lo, start, peak)
    rise_90 = _pre_peak_crossing(amp, hi, start, peak)
    rise_time_ms = float((rise_90 - rise_10) * sample_to_ms) if np.isfinite(rise_10) and np.isfinite(rise_90) and rise_90 >= rise_10 else float("nan")

    half_left = _pre_peak_crossing(amp, half, start, peak)
    half_right = _post_peak_crossing(amp, half, peak, stop)
    fwhm_ms = float((half_right - half_left) * sample_to_ms) if np.isfinite(half_left) and np.isfinite(half_right) and half_right >= half_left else float("nan")

    return_cross = _post_peak_crossing(amp, lo, peak, stop)
    return_to_baseline_time_ms = float((return_cross - float(peak)) * sample_to_ms) if np.isfinite(return_cross) and return_cross >= float(peak) else float("nan")

    if np.isfinite(return_cross):
        post_start = max(peak + 1, min(stop, int(np.ceil(return_cross))))
    else:
        post_start = min(stop, peak + max(1, int(round(0.001 * fs))))
    post_vals = amp[post_start:stop]
    post_vals = post_vals[np.isfinite(post_vals)]
    post_spike_level = float(np.median(post_vals)) if post_vals.size > 0 else float("nan")

    return _with_cached(
        {
            "rise_time_ms": rise_time_ms,
            "fwhm_ms": fwhm_ms,
            "return_to_baseline_time_ms": return_to_baseline_time_ms,
            "post_spike_level": post_spike_level,
        }
    )


def _spike_delta_f_over_f0(spike: SpikeRecord) -> float:
    cached = _finite_spike_metric(spike, "delta_f_over_f0")
    if np.isfinite(cached):
        return cached
    f0 = float(spike.baseline_at_spike)
    if not np.isfinite(f0) or f0 == 0.0:
        return float("nan")
    return float(spike.amplitude_above_baseline / f0)


def _build_spike_event_rows(trace_names: List[str], traces: Dict[str, TraceCuration]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trace_name in sorted(trace_names):
        state = traces.get(trace_name)
        if state is None:
            continue
        spikes = sorted(state.spikes, key=lambda spike: spike.spike_index)
        prev_index: Optional[int] = None
        sample_period_ms = _time_sample_period_ms(np.asarray(state.time, dtype=float), _trace_sampling_rate_hz(state))
        for idx, spike in enumerate(spikes, 1):
            peak_time = float(spike.spike_time)
            peak_index = int(spike.spike_index)
            if prev_index is None or not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
                isi_ms = float("nan")
                rate_hz = float("nan")
            else:
                isi_ms = float((peak_index - prev_index) * sample_period_ms)
                rate_hz = float(1000.0 / isi_ms) if np.isfinite(isi_ms) and isi_ms > 0.0 else float("nan")
            prev_index = peak_index

            waveform = _measure_spike_waveform_stats(state, spike, half_window_ms=5.0)
            rows.append(
                {
                    "trace_id": str(trace_name),
                    **trace_identity_columns(trace_name),
                    "spike_id": f"{idx:04d}",
                    "peak_index": peak_index,
                    "peak_time": peak_time,
                    "amplitude": float(spike.amplitude_above_baseline),
                    "delta_f_over_f0": _spike_delta_f_over_f0(spike),
                    "snr": float(spike.snr),
                    "rise_time_ms": waveform["rise_time_ms"],
                    "fwhm_ms": waveform["fwhm_ms"],
                    "return_to_baseline_time_ms": waveform["return_to_baseline_time_ms"],
                    "post_spike_level": waveform["post_spike_level"],
                    "isi_ms": isi_ms,
                    "instantaneous_firing_rate_hz": rate_hz,
                    "source": spike.source,
                }
            )
    return rows


def _build_trace_spike_summary_rows(trace_names: List[str], spike_event_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    events_df = pd.DataFrame(spike_event_rows)
    metric_cols = [
        "amplitude",
        "delta_f_over_f0",
        "snr",
        "rise_time_ms",
        "fwhm_ms",
        "return_to_baseline_time_ms",
        "post_spike_level",
        "isi_ms",
        "instantaneous_firing_rate_hz",
    ]

    rows: List[Dict[str, object]] = []
    for trace_name in sorted(trace_names):
        if not events_df.empty and "trace_id" in events_df.columns:
            per = events_df[events_df["trace_id"] == str(trace_name)]
        else:
            per = pd.DataFrame()
        row: Dict[str, object] = {
            "trace_id": str(trace_name),
            **trace_identity_columns(trace_name),
            "spike_count": int(per.shape[0]),
        }
        for col in metric_cols:
            vals = _finite_numeric_series(per, col)
            row[f"mean_{col}"] = float(np.mean(vals)) if vals.size > 0 else float("nan")
            row[f"median_{col}"] = float(np.median(vals)) if vals.size > 0 else float("nan")
        rows.append(row)
    return rows


def _build_plateau_event_rows(trace_names: List[str], traces: Dict[str, TraceCuration]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for trace_name in sorted(trace_names):
        state = traces[trace_name]
        if state.plateau_mask is None:
            continue

        y = np.asarray(state.corrected, dtype=float)
        b = np.asarray(state.baseline, dtype=float) if state.baseline is not None else np.zeros_like(y)
        t = np.asarray(state.time, dtype=float)
        m = np.asarray(state.plateau_mask, dtype=bool)
        long_mask = np.asarray(state.long_plateau_mask, dtype=bool) if state.long_plateau_mask is not None else None
        short_mask = np.asarray(state.short_plateau_mask, dtype=bool) if state.short_plateau_mask is not None else None
        burst_mask = np.asarray(state.burst_plateau_mask, dtype=bool) if state.burst_plateau_mask is not None else None
        burst_events = np.asarray(state.burst_plateau_events, dtype=float) if state.burst_plateau_events is not None else None
        signal_rows = state.plateau_signal_classification

        n = min(y.size, b.size, t.size, m.size)
        if n <= 2:
            continue

        y = y[:n]
        b = b[:n]
        t = t[:n]
        m = m[:n]
        if not np.any(m):
            continue

        sampling_rate_hz = _trace_sampling_rate_hz(state)
        sample_period_ms = _time_sample_period_ms(t, sampling_rate_hz)
        if not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
            continue

        regions = _contiguous_true_regions(m)
        for idx, (start, end) in enumerate(regions, 1):
            if end <= start:
                continue

            amp = y[start:end] - b[start:end]
            amp_f = amp[np.isfinite(amp)]

            onset_time_ms = _absolute_time_ms(t, start, sampling_rate_hz)
            duration_ms = float((end - start) * sample_period_ms)

            fwhm_samples = _event_fwhm_samples(amp)
            fwhm_ms = float(fwhm_samples * sample_period_ms) if np.isfinite(fwhm_samples) else float("nan")

            rise_s, decay_s = _rise_decay_time_samples(amp)
            rise_time_ms = float(rise_s * sample_period_ms) if np.isfinite(rise_s) else float("nan")
            decay_time_ms = float(decay_s * sample_period_ms) if np.isfinite(decay_s) else float("nan")

            plateau_sd = float(np.std(amp_f, ddof=0)) if amp_f.size >= 2 else float("nan")
            mean_amp = float(np.mean(amp_f)) if amp_f.size > 0 else float("nan")
            peak_amp = float(np.max(amp_f)) if amp_f.size > 0 else float("nan")

            source = _plateau_source_for_region(start, end, long_mask, short_mask, burst_mask)
            row = {
                    "trace_id": str(trace_name),
                    **trace_identity_columns(trace_name),
                    "plateau_id": f"{idx:03d}",
                    "detector_source": source,
                    "start_index": int(start),
                    "end_index_exclusive": int(end),
                    "onset_time_ms": onset_time_ms,
                    "start_time": float(t[start]),
                    "end_time": float(t[end - 1]),
                    "duration_ms": duration_ms,
                    "fwhm_ms": fwhm_ms,
                    "rise_time_ms": rise_time_ms,
                    "decay_time_ms": decay_time_ms,
                    "plateau_sd": plateau_sd,
                    "mean_amp_above_baseline": mean_amp,
                    "peak_amp_above_baseline": peak_amp,
                    "sample_count": int(end - start),
            }
            if source == "burst":
                row.update(_event_metrics_for_region(burst_events, start, end))
            row.update(_signal_metrics_for_region(signal_rows, start, end))
            rows.append(row)

    return rows


def _trace_spike_means(state: TraceCuration) -> tuple[float, float]:
    spikes = list(state.spikes)
    if not spikes:
        return float("nan"), float("nan")

    fwhm_vals = np.asarray([float(s.fwhm_ms) for s in spikes], dtype=float)
    amp_vals = np.asarray([float(s.amplitude_above_baseline) for s in spikes], dtype=float)
    fwhm_vals = fwhm_vals[np.isfinite(fwhm_vals)]
    amp_vals = amp_vals[np.isfinite(amp_vals)]

    fwhm = float(np.mean(fwhm_vals)) if fwhm_vals.size > 0 else float("nan")
    amp = float(np.mean(amp_vals)) if amp_vals.size > 0 else float("nan")
    return fwhm, amp


def _build_trace_summary_rows(
    trace_names: List[str],
    traces: Dict[str, TraceCuration],
    event_rows: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    if event_rows is None:
        event_rows = _build_plateau_event_rows(trace_names, traces)

    events_df = pd.DataFrame(event_rows)

    def _col_vals(df: pd.DataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    def _stats(arr: np.ndarray) -> tuple[float, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.median(arr))

    rows: List[Dict[str, object]] = []
    for trace_name in sorted(trace_names):
        if not events_df.empty and "trace_id" in events_df.columns:
            per = events_df[events_df["trace_id"] == str(trace_name)]
        else:
            per = pd.DataFrame()

        if per.empty:
            state = traces.get(trace_name)
            total_time_ms = _trace_total_time_ms(state.time, _trace_sampling_rate_hz(state)) if state is not None else float("nan")
            rows.append(
                {
                    "trace_id": str(trace_name),
                    **trace_identity_columns(trace_name),
                    "plateau_count": 0,
                    "first_onset_ms": float("nan"),
                    "mean_duration_ms": float("nan"),
                    "median_duration_ms": float("nan"),
                    "mean_fwhm_ms": float("nan"),
                    "median_fwhm_ms": float("nan"),
                    "mean_rise_time_ms": float("nan"),
                    "median_rise_time_ms": float("nan"),
                    "mean_peak_amp_above_baseline": float("nan"),
                    "median_peak_amp_above_baseline": float("nan"),
                    "mean_plateau_sd": float("nan"),
                    "median_plateau_sd": float("nan"),
                    "total_plateau_time_ms": 0.0,
                    "plateau_time_fraction": 0.0 if np.isfinite(total_time_ms) and total_time_ms > 0.0 else float("nan"),
                    "mean_inter_plateau_interval_ms": float("nan"),
                    "median_inter_plateau_interval_ms": float("nan"),
                }
            )
            continue

        onset_vals = _col_vals(per, "onset_time_ms")
        duration_vals = _col_vals(per, "duration_ms")
        fwhm_vals = _col_vals(per, "fwhm_ms")
        rise_vals = _col_vals(per, "rise_time_ms")
        peak_vals = _col_vals(per, "peak_amp_above_baseline")
        sd_vals = _col_vals(per, "plateau_sd")

        mean_dur, med_dur = _stats(duration_vals)
        mean_fwhm, med_fwhm = _stats(fwhm_vals)
        mean_rise, med_rise = _stats(rise_vals)
        mean_peak, med_peak = _stats(peak_vals)
        mean_sd, med_sd = _stats(sd_vals)

        finite_onsets = onset_vals[np.isfinite(onset_vals)]
        first_onset = float(np.min(finite_onsets)) if finite_onsets.size > 0 else float("nan")

        sorted_onsets = np.sort(finite_onsets)
        if sorted_onsets.size > 1:
            gaps = np.diff(sorted_onsets)
            gaps = gaps[gaps > 0]
            mean_ipi = float(np.mean(gaps)) if gaps.size > 0 else float("nan")
            median_ipi = float(np.median(gaps)) if gaps.size > 0 else float("nan")
        else:
            mean_ipi = float("nan")
            median_ipi = float("nan")

        total_plateau_time = float(np.nansum(duration_vals[np.isfinite(duration_vals)]))
        state = traces.get(trace_name)
        total_time_ms = _trace_total_time_ms(state.time, _trace_sampling_rate_hz(state)) if state is not None else float("nan")
        if np.isfinite(total_time_ms) and total_time_ms > 0.0:
            plateau_fraction = float(total_plateau_time / total_time_ms)
        else:
            plateau_fraction = float("nan")

        rows.append(
            {
                "trace_id": str(trace_name),
                **trace_identity_columns(trace_name),
                "plateau_count": int(per.shape[0]),
                "first_onset_ms": first_onset,
                "mean_duration_ms": mean_dur,
                "median_duration_ms": med_dur,
                "mean_fwhm_ms": mean_fwhm,
                "median_fwhm_ms": med_fwhm,
                "mean_rise_time_ms": mean_rise,
                "median_rise_time_ms": med_rise,
                "mean_peak_amp_above_baseline": mean_peak,
                "median_peak_amp_above_baseline": med_peak,
                "mean_plateau_sd": mean_sd,
                "median_plateau_sd": med_sd,
                "total_plateau_time_ms": total_plateau_time,
                "plateau_time_fraction": plateau_fraction,
                "mean_inter_plateau_interval_ms": mean_ipi,
                "median_inter_plateau_interval_ms": median_ipi,
            }
        )
    return rows


def _add_trace_qc_flags(trace_df: pd.DataFrame) -> pd.DataFrame:
    out = trace_df.copy()
    notes: List[List[str]] = [[] for _ in range(len(out))]

    if "plateau_count" in out.columns:
        counts = pd.to_numeric(out["plateau_count"], errors="coerce").to_numpy(dtype=float)
        for i, value in enumerate(counts):
            if np.isfinite(value) and value <= 0.0:
                notes[i].append("no_plateaus")

    for col, label in QC_TRACE_METRICS.items():
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        for i, flagged in enumerate(_robust_high_outlier_mask(vals)):
            if flagged:
                notes[i].append(f"high_{label.lower().replace(' ', '_')}")

    out["qc_status"] = ["review" if row_notes else "ok" for row_notes in notes]
    out["qc_notes"] = ["; ".join(row_notes) for row_notes in notes]
    return out


def _build_all_trace_summary_row(events_df: pd.DataFrame) -> Dict[str, object]:
    def _col_vals_all(col: str) -> np.ndarray:
        if col not in events_df.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(events_df[col], errors="coerce").to_numpy(dtype=float)

    def _agg(arr: np.ndarray) -> tuple[float, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.median(arr))

    pooled_gaps: List[float] = []
    if "trace_id" in events_df.columns and "onset_time_ms" in events_df.columns:
        for _trace_id, per_events in events_df.groupby("trace_id", sort=False):
            per_onsets = pd.to_numeric(per_events["onset_time_ms"], errors="coerce").to_numpy(dtype=float)
            per_onsets = np.sort(per_onsets[np.isfinite(per_onsets)])
            if per_onsets.size > 1:
                gaps = np.diff(per_onsets)
                pooled_gaps.extend([float(gap) for gap in gaps if np.isfinite(gap) and gap > 0.0])
    if pooled_gaps:
        pooled_gap_arr = np.asarray(pooled_gaps, dtype=float)
        mean_ipi_all = float(np.mean(pooled_gap_arr))
        median_ipi_all = float(np.median(pooled_gap_arr))
    else:
        mean_ipi_all = float("nan")
        median_ipi_all = float("nan")

    m_dur, med_dur = _agg(_col_vals_all("duration_ms"))
    m_fwhm, med_fwhm = _agg(_col_vals_all("fwhm_ms"))
    m_rise, med_rise = _agg(_col_vals_all("rise_time_ms"))
    m_peak, med_peak = _agg(_col_vals_all("peak_amp_above_baseline"))
    m_sd, med_sd = _agg(_col_vals_all("plateau_sd"))
    total_plateau = _col_vals_all("duration_ms")
    total_plateau_all = float(np.nansum(total_plateau[np.isfinite(total_plateau)]))

    return {
        "trace_id": "ALL",
        "recording_id": "ALL",
        "roi_id": "",
        "plateau_count": int(events_df.shape[0]),
        "first_onset_ms": float("nan"),
        "mean_duration_ms": m_dur,
        "median_duration_ms": med_dur,
        "mean_fwhm_ms": m_fwhm,
        "median_fwhm_ms": med_fwhm,
        "mean_rise_time_ms": m_rise,
        "median_rise_time_ms": med_rise,
        "mean_peak_amp_above_baseline": m_peak,
        "median_peak_amp_above_baseline": med_peak,
        "mean_plateau_sd": m_sd,
        "median_plateau_sd": med_sd,
        "total_plateau_time_ms": total_plateau_all,
        "plateau_time_fraction": float("nan"),
        "mean_inter_plateau_interval_ms": mean_ipi_all,
        "median_inter_plateau_interval_ms": median_ipi_all,
    }


def _ordered_dataframe(rows: List[Dict[str, object]], preferred_columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=preferred_columns)
    ordered = [col for col in preferred_columns if col in df.columns]
    tail = [col for col in df.columns if col not in ordered]
    return df[ordered + tail]


def _pad_numeric(values: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == size:
        return arr
    out = np.full(size, np.nan, dtype=float)
    n = min(size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _build_corrected_traces_dataframe(trace_names: List[str], traces: Dict[str, TraceCuration]) -> pd.DataFrame:
    available = [(str(name), traces[name]) for name in trace_names if name in traces]
    if not available:
        return pd.DataFrame(columns=["time"])

    max_len = max(
        max(np.asarray(state.time, dtype=float).size, np.asarray(state.corrected, dtype=float).size)
        for _name, state in available
    )
    first_time = np.asarray(available[0][1].time, dtype=float)
    data: Dict[str, np.ndarray] = {"time": _pad_numeric(first_time, max_len)}
    for trace_name, state in available:
        data[trace_name] = _pad_numeric(np.asarray(state.corrected, dtype=float), max_len)
    return pd.DataFrame(data)


def _build_export_dataframes(trace_names: List[str], traces: Dict[str, TraceCuration]) -> Dict[str, pd.DataFrame]:
    event_rows = _build_plateau_event_rows(trace_names, traces)
    events_df = _ordered_dataframe(event_rows, PLATEAU_EVENT_COLUMNS)
    events_df = _add_event_qc_flags(events_df)
    if {"trace_id", "plateau_id"}.issubset(events_df.columns):
        events_df = events_df.sort_values(["trace_id", "plateau_id"], kind="mergesort").reset_index(drop=True)

    trace_rows = _build_trace_summary_rows(trace_names, traces, event_rows=event_rows)
    trace_df = _ordered_dataframe(trace_rows, TRACE_SUMMARY_COLUMNS)
    if not events_df.empty:
        trace_df = pd.concat([trace_df, pd.DataFrame([_build_all_trace_summary_row(events_df)])], ignore_index=True)
    trace_df = _ordered_dataframe(trace_df.to_dict("records"), TRACE_SUMMARY_COLUMNS)

    qc_trace_df = _add_trace_qc_flags(trace_df)
    qc_columns = TRACE_SUMMARY_COLUMNS + ["qc_status", "qc_notes"]
    qc_trace_df = _ordered_dataframe(qc_trace_df.to_dict("records"), qc_columns)

    spike_event_rows = _build_spike_event_rows(trace_names, traces)
    spike_event_df = _ordered_dataframe(spike_event_rows, SPIKE_EVENT_COLUMNS)

    spike_summary_rows = _build_trace_spike_summary_rows(trace_names, spike_event_rows)
    spike_summary_df = pd.DataFrame(spike_summary_rows)

    return {
        "plateau_events": events_df,
        "trace_summary": trace_df,
        "trace_qc": qc_trace_df,
        "spike_events": spike_event_df,
        "spike_summary": spike_summary_df,
        "corrected_traces": _build_corrected_traces_dataframe(trace_names, traces),
    }


def _prepare_export_layout(out_dir: Path) -> Dict[str, Path]:
    layout = {
        "root": out_dir,
        "tables": out_dir / EXPORT_TABLES_DIR,
        "reports": out_dir / EXPORT_REPORTS_DIR,
        "reload": out_dir / EXPORT_RELOAD_DIR,
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _find_export_file(folder_path: Path, filename: str) -> Optional[Path]:
    folder = Path(folder_path)
    candidate_roots = [folder]
    if folder.name in {EXPORT_TABLES_DIR, EXPORT_REPORTS_DIR, EXPORT_RELOAD_DIR}:
        candidate_roots.append(folder.parent)

    seen: set[Path] = set()
    for root in candidate_roots:
        for candidate in (
            root / filename,
            root / EXPORT_RELOAD_DIR / filename,
            root / EXPORT_TABLES_DIR / filename,
            root / EXPORT_REPORTS_DIR / filename,
        ):
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
    return None


def _validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _short_trace_label(name: object, max_len: int = 26) -> str:
    text = str(name)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}..."


def _format_report_value(value: object, digits: int = 2) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    if abs(val) >= 1000.0:
        return f"{val:.0f}"
    if abs(val) >= 100.0:
        return f"{val:.1f}"
    return f"{val:.{digits}f}"


def _save_trace_summary_figure(csv_path: Path, png_path: Path, pdf_path: Optional[Path] = None) -> None:
    """Backward-compatible wrapper for older callers."""
    trace_df = pd.read_csv(csv_path)
    _save_qc_summary_report(trace_df=trace_df, events_df=pd.DataFrame(), png_path=png_path, pdf_path=pdf_path)


def _save_qc_summary_report(
    trace_df: pd.DataFrame,
    events_df: pd.DataFrame,
    png_path: Path,
    pdf_path: Optional[Path] = None,
) -> None:
    cfg = TRACE_SUMMARY_CONFIG
    df = trace_df.copy()
    if "trace_id" in df.columns:
        df = df[df["trace_id"] != "ALL"].copy()
        df = df.sort_values("trace_id", kind="mergesort").reset_index(drop=True)
    events = events_df.copy()

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.style.use("default")
    fig = plt.figure(figsize=(17, 14), constrained_layout=True)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(4, 2, height_ratios=[1.35, 1.45, 1.35, 1.25], width_ratios=[1.15, 1.0])
    ax_table = fig.add_subplot(grid[0, :])
    ax_heat = fig.add_subplot(grid[1, :])
    ax_dist = fig.add_subplot(grid[2, :])
    ax_count = fig.add_subplot(grid[3, 0])
    ax_fraction = fig.add_subplot(grid[3, 1])
    fig.suptitle("Plateau Detection QC Summary", fontsize=18, fontweight="bold")

    ax_table.axis("off")
    table_cols = [
        ("trace_id", "Trace"),
        ("plateau_count", "N"),
        ("first_onset_ms", "First onset ms"),
        ("median_duration_ms", "Median duration ms"),
        ("median_fwhm_ms", "Median FWHM ms"),
        ("median_rise_time_ms", "Median rise ms"),
        ("median_peak_amp_above_baseline", "Median peak a.u."),
        ("median_plateau_sd", "Median plateau SD a.u."),
        ("qc_status", "QC"),
        ("qc_notes", "Notes"),
    ]
    display = df.head(18).copy()
    table_data: List[List[str]] = []
    for _, row in display.iterrows():
        table_row: List[str] = []
        for col, _label in table_cols:
            if col == "trace_id":
                table_row.append(_short_trace_label(row.get(col, ""), max_len=22))
            elif col in {"qc_status", "qc_notes"}:
                max_len = 36 if col == "qc_notes" else 8
                table_row.append(_short_trace_label(row.get(col, ""), max_len=max_len))
            else:
                table_row.append(_format_report_value(row.get(col, "")))
        table_data.append(table_row)
    if table_data:
        col_widths = [0.17, 0.045, 0.09, 0.10, 0.09, 0.085, 0.105, 0.12, 0.07, 0.125]
        table = ax_table.table(
            cellText=table_data,
            colLabels=[label for _col, label in table_cols],
            loc="center",
            cellLoc="center",
            colLoc="center",
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.4)
        table.scale(1.0, 1.35)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#d1d5db")
            if row_idx == 0:
                cell.set_facecolor("#e5e7eb")
                cell.set_text_props(weight="bold", color="#111827")
            elif col_idx == len(table_cols) - 2 and cell.get_text().get_text() == "review":
                cell.set_facecolor("#fee2e2")
                cell.set_text_props(weight="bold", color="#991b1b")
    else:
        ax_table.text(0.5, 0.5, "No traces available", ha="center", va="center", fontsize=12, color="#6b7280")
    if len(df) > len(display):
        ax_table.text(0.99, 0.02, f"Showing first {len(display)} of {len(df)} traces", ha="right", va="bottom", transform=ax_table.transAxes, fontsize=8, color="#6b7280")

    heat_cols = [
        ("plateau_count", "Count"),
        ("first_onset_ms", "First onset"),
        ("median_duration_ms", "Duration"),
        ("median_fwhm_ms", "FWHM"),
        ("median_rise_time_ms", "Rise"),
        ("median_peak_amp_above_baseline", "Peak amp"),
        ("median_plateau_sd", "Plateau SD"),
        ("plateau_time_fraction", "Time fraction"),
    ]
    heat = np.full((len(df), len(heat_cols)), np.nan, dtype=float)
    for col_idx, (col, _label) in enumerate(heat_cols):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 2:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if mad > 0.0 and np.isfinite(mad):
            heat[:, col_idx] = np.clip(0.6745 * (vals - median) / mad, -4.0, 4.0)
        else:
            span = float(np.nanmax(finite) - np.nanmin(finite))
            if span > 0.0:
                heat[:, col_idx] = ((vals - float(np.nanmin(finite))) / span) * 2.0 - 1.0
            else:
                heat[:, col_idx] = 0.0
    cmap = LinearSegmentedColormap.from_list("qc_heat", ["#2563eb", "#f8fafc", "#dc2626"])
    shown_heat = heat[:30, :]
    image = ax_heat.imshow(shown_heat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-4.0, vmax=4.0)
    ax_heat.set_title("Trace metric heatmap (robust z-score; red = unusually high)", fontsize=11, fontweight="bold")
    ax_heat.set_xticks(range(len(heat_cols)))
    ax_heat.set_xticklabels([label for _col, label in heat_cols], rotation=30, ha="right", fontsize=9)
    labels = [_short_trace_label(name, max_len=34) for name in df.get("trace_id", pd.Series(dtype=object)).to_numpy()[:30]]
    ax_heat.set_yticks(range(len(labels)))
    ax_heat.set_yticklabels(labels, fontsize=8)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    if shown_heat.size > 0:
        fig.colorbar(image, ax=ax_heat, shrink=0.75, label="robust z")

    dist_metrics = [
        ("duration_ms", "Duration (ms)"),
        ("fwhm_ms", "FWHM (ms)"),
        ("peak_amp_above_baseline", "Peak amplitude (a.u.)"),
        ("plateau_sd", "Plateau SD (a.u.)"),
    ]
    ax_dist.set_title("Event distributions (each point is one plateau)", fontsize=11, fontweight="bold")
    plotted = False
    rng = np.random.default_rng(12345)
    for pos, (col, label) in enumerate(dist_metrics, 1):
        vals = _finite_numeric_series(events, col)
        if vals.size == 0:
            continue
        plotted = True
        jitter = rng.uniform(-0.10, 0.10, size=vals.size)
        ax_dist.scatter(np.full(vals.size, pos) + jitter, vals, s=18, alpha=0.65, color="#334155", edgecolors="none")
        ax_dist.boxplot(vals, positions=[pos], widths=0.35, showfliers=False, patch_artist=True, boxprops={"facecolor": "#dbeafe", "edgecolor": "#1d4ed8"}, medianprops={"color": "#111827", "linewidth": 1.4})
    ax_dist.set_xticks(range(1, len(dist_metrics) + 1))
    ax_dist.set_xticklabels([label for _col, label in dist_metrics], rotation=15, ha="right", fontsize=9)
    ax_dist.grid(axis="y", color="#e5e7eb")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    if not plotted:
        ax_dist.text(0.5, 0.5, "No plateau events available", ha="center", va="center", transform=ax_dist.transAxes, color="#6b7280")

    trace_labels = [_short_trace_label(name, max_len=22) for name in df.get("trace_id", pd.Series(dtype=object)).to_numpy()]
    y_pos = np.arange(len(df))
    count_vals = pd.to_numeric(df.get("plateau_count", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    frac_vals = pd.to_numeric(df.get("plateau_time_fraction", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    status = df.get("qc_status", pd.Series(["ok"] * len(df))).astype(str).to_numpy()
    colors = np.where(status == "review", "#dc2626", "#2563eb")

    ax_count.barh(y_pos, np.nan_to_num(count_vals, nan=0.0), color=colors, alpha=0.85)
    ax_count.set_title("Plateau count by trace", fontsize=11, fontweight="bold")
    ax_count.set_yticks(y_pos)
    ax_count.set_yticklabels(trace_labels, fontsize=8)
    ax_count.invert_yaxis()
    ax_count.grid(axis="x", color="#e5e7eb")
    ax_count.spines["top"].set_visible(False)
    ax_count.spines["right"].set_visible(False)

    ax_fraction.barh(y_pos, np.nan_to_num(frac_vals, nan=0.0), color=colors, alpha=0.85)
    ax_fraction.set_title("Total plateau time fraction", fontsize=11, fontweight="bold")
    ax_fraction.set_yticks(y_pos)
    ax_fraction.set_yticklabels([])
    ax_fraction.invert_yaxis()
    ax_fraction.grid(axis="x", color="#e5e7eb")
    ax_fraction.spines["top"].set_visible(False)
    ax_fraction.spines["right"].set_visible(False)
    ax_fraction.set_xlabel("fraction of trace")

    fig.text(0.01, 0.01, "Plateau SD is the standard deviation of baseline-subtracted amplitude samples within detected plateau regions.", fontsize=8, color="#6b7280")
    fig.savefig(png_path, dpi=int(cfg["png_dpi"]), facecolor="white", bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_distribution_panel(ax: object, df: pd.DataFrame, metrics: List[Tuple[str, str]], title: str) -> None:
    import matplotlib.pyplot as plt

    ax.set_title(title, fontsize=11, fontweight="bold")
    plotted = False
    rng = np.random.default_rng(24680)
    for pos, (col, label) in enumerate(metrics, 1):
        vals = _finite_numeric_series(df, col)
        if vals.size == 0:
            continue
        plotted = True
        jitter = rng.uniform(-0.10, 0.10, size=vals.size)
        ax.scatter(np.full(vals.size, pos) + jitter, vals, s=18, alpha=0.65, color="#475569", edgecolors="none")
        ax.boxplot(
            vals,
            positions=[pos],
            widths=0.35,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#dcfce7", "edgecolor": "#15803d"},
            medianprops={"color": "#111827", "linewidth": 1.4},
        )
    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels([label for _col, label in metrics], rotation=18, ha="right", fontsize=9)
    ax.grid(axis="y", color="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not plotted:
        ax.text(0.5, 0.5, "No spike values available", ha="center", va="center", transform=ax.transAxes, color="#6b7280")


def _save_spike_statistics_summary(
    spike_summary_df: pd.DataFrame,
    spike_events_df: pd.DataFrame,
    png_path: Path,
) -> None:
    cfg = TRACE_SUMMARY_CONFIG
    summary = spike_summary_df.copy()
    events = spike_events_df.copy()
    if "trace_id" in summary.columns:
        summary = summary.sort_values("trace_id", kind="mergesort").reset_index(drop=True)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.style.use("default")
    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.75, 1.35, 1.35], width_ratios=[1.05, 1.0])
    ax_cards = fig.add_subplot(grid[0, :])
    ax_count = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])
    ax_dist = fig.add_subplot(grid[2, 0])
    ax_timing = fig.add_subplot(grid[2, 1])
    fig.suptitle("Spike Statistics Summary", fontsize=18, fontweight="bold")

    trace_count = int(summary.shape[0]) if "trace_id" in summary.columns else int(events.get("trace_id", pd.Series(dtype=object)).nunique())
    total_spikes = int(events.shape[0])
    if "spike_count" in summary.columns:
        spike_counts = pd.to_numeric(summary["spike_count"], errors="coerce").to_numpy(dtype=float)
        traces_with_spikes = int(np.count_nonzero(np.nan_to_num(spike_counts, nan=0.0) > 0.0))
    else:
        traces_with_spikes = int(events.get("trace_id", pd.Series(dtype=object)).nunique())

    def _median_label(col: str, suffix: str = "") -> str:
        vals = _finite_numeric_series(events, col)
        if vals.size == 0:
            return "n/a"
        return f"{_format_report_value(float(np.median(vals)))}{suffix}"

    cards = [
        ("Traces", str(trace_count)),
        ("Total spikes", str(total_spikes)),
        ("Traces with spikes", str(traces_with_spikes)),
        ("Median amplitude", _median_label("amplitude")),
        ("Median FWHM", _median_label("fwhm_ms", " ms")),
        ("Median ISI", _median_label("isi_ms", " ms")),
        ("Median rate", _median_label("instantaneous_firing_rate_hz", " Hz")),
    ]
    ax_cards.axis("off")
    for i, (label, value) in enumerate(cards):
        x = (i + 0.5) / len(cards)
        ax_cards.text(x, 0.62, value, ha="center", va="center", fontsize=16, fontweight="bold", color="#111827", transform=ax_cards.transAxes)
        ax_cards.text(x, 0.30, label, ha="center", va="center", fontsize=9, color="#475569", transform=ax_cards.transAxes)

    if "trace_id" in summary.columns and "spike_count" in summary.columns and not summary.empty:
        count_df = summary[["trace_id", "spike_count"]].copy()
        count_df["spike_count"] = pd.to_numeric(count_df["spike_count"], errors="coerce").fillna(0.0)
    elif "trace_id" in events.columns and not events.empty:
        count_df = events.groupby("trace_id").size().reset_index(name="spike_count")
    else:
        count_df = pd.DataFrame(columns=["trace_id", "spike_count"])
    count_df = count_df.sort_values(["spike_count", "trace_id"], ascending=[True, True], kind="mergesort").tail(24)
    if not count_df.empty:
        labels = [_short_trace_label(name, max_len=24) for name in count_df["trace_id"].to_numpy()]
        y_pos = np.arange(len(count_df))
        ax_count.barh(y_pos, count_df["spike_count"].to_numpy(dtype=float), color="#2563eb", alpha=0.85)
        ax_count.set_yticks(y_pos)
        ax_count.set_yticklabels(labels, fontsize=8)
        ax_count.set_xlabel("spikes")
    else:
        ax_count.text(0.5, 0.5, "No spikes available", ha="center", va="center", transform=ax_count.transAxes, color="#6b7280")
    ax_count.set_title("Spike count by trace", fontsize=11, fontweight="bold")
    ax_count.grid(axis="x", color="#e5e7eb")
    ax_count.spines["top"].set_visible(False)
    ax_count.spines["right"].set_visible(False)

    heat_cols = [
        ("spike_count", "Count"),
        ("median_amplitude", "Amplitude"),
        ("median_delta_f_over_f0", "Delta F/F0"),
        ("median_snr", "SNR"),
        ("median_fwhm_ms", "FWHM"),
        ("median_isi_ms", "ISI"),
        ("median_instantaneous_firing_rate_hz", "Rate"),
    ]
    heat_df = summary.head(30).copy()
    heat = np.full((len(heat_df), len(heat_cols)), np.nan, dtype=float)
    for col_idx, (col, _label) in enumerate(heat_cols):
        if col not in heat_df.columns:
            continue
        vals = pd.to_numeric(heat_df[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        if finite.size < 2:
            heat[np.isfinite(vals), col_idx] = 0.0
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if mad > 0.0 and np.isfinite(mad):
            heat[:, col_idx] = np.clip(0.6745 * (vals - median) / mad, -4.0, 4.0)
        else:
            span = float(np.nanmax(finite) - np.nanmin(finite))
            heat[:, col_idx] = ((vals - float(np.nanmin(finite))) / span) * 2.0 - 1.0 if span > 0.0 else 0.0
    if heat.size > 0 and heat.shape[0] > 0:
        cmap = LinearSegmentedColormap.from_list("spike_heat", ["#2563eb", "#f8fafc", "#dc2626"])
        image = ax_heat.imshow(heat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-4.0, vmax=4.0)
        ax_heat.set_xticks(range(len(heat_cols)))
        ax_heat.set_xticklabels([label for _col, label in heat_cols], rotation=30, ha="right", fontsize=9)
        labels = [_short_trace_label(name, max_len=28) for name in heat_df.get("trace_id", pd.Series(dtype=object)).to_numpy()]
        ax_heat.set_yticks(range(len(labels)))
        ax_heat.set_yticklabels(labels, fontsize=8)
        fig.colorbar(image, ax=ax_heat, shrink=0.74, label="robust z")
    else:
        ax_heat.text(0.5, 0.5, "No trace summary available", ha="center", va="center", transform=ax_heat.transAxes, color="#6b7280")
    ax_heat.set_title("Per-trace spike metric heatmap", fontsize=11, fontweight="bold")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    _plot_distribution_panel(
        ax_dist,
        events,
        [
            ("amplitude", "Amplitude"),
            ("delta_f_over_f0", "Delta F/F0"),
            ("snr", "SNR"),
            ("fwhm_ms", "FWHM ms"),
        ],
        "Spike metric distributions",
    )
    _plot_distribution_panel(
        ax_timing,
        events,
        [
            ("isi_ms", "ISI ms"),
            ("instantaneous_firing_rate_hz", "Rate Hz"),
            ("rise_time_ms", "Rise ms"),
            ("return_to_baseline_time_ms", "Return ms"),
        ],
        "ISI, rate, and timing distributions",
    )

    fig.text(0.01, 0.01, "ISI and firing rate are computed from spike-index distance and trace sampling rate.", fontsize=8, color="#6b7280")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(cfg["png_dpi"]), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _debug_array(values: object, dtype: object = float) -> Optional[np.ndarray]:
    if values is None or isinstance(values, dict):
        return None
    try:
        return np.asarray(values, dtype=dtype)
    except (TypeError, ValueError):
        return None


def _downsample_1d_for_debug(x: np.ndarray, y: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = min(x_arr.size, y_arr.size)
    if n <= 0:
        return x_arr[:0], y_arr[:0]
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    if n <= int(max_points):
        return x_arr, y_arr
    idx = np.linspace(0, n - 1, int(max_points)).astype(int)
    return x_arr[idx], y_arr[idx]


def _downsample_heatmap_for_debug(matrix: np.ndarray, max_cols: int = 2000) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[1] <= int(max_cols):
        return arr
    idx = np.linspace(0, arr.shape[1] - 1, int(max_cols)).astype(int)
    return arr[:, idx]


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item())
        summary: Dict[str, object] = {
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
        }
        if value.size and np.issubdtype(value.dtype, np.number):
            if np.iscomplexobj(value):
                magnitude = np.abs(value)
                summary["magnitude_min"] = _json_safe(np.nanmin(magnitude))
                summary["magnitude_max"] = _json_safe(np.nanmax(magnitude))
            else:
                summary["min"] = _json_safe(np.nanmin(value))
                summary["max"] = _json_safe(np.nanmax(value))
        return summary
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _save_debug_line_plot(
    path: Path,
    title: str,
    time: np.ndarray,
    series: List[tuple],
    ylabel: str = "Signal",
) -> bool:
    import matplotlib.pyplot as plt

    if not series:
        return False
    fig, ax = plt.subplots(figsize=(14, 4.5))
    plotted = False
    for item in series:
        label, values, color = item[:3]
        linestyle = item[3] if len(item) > 3 else "-"
        linewidth = float(item[4]) if len(item) > 4 else 1.1
        zorder = int(item[5]) if len(item) > 5 else 2
        x_plot, y_plot = _downsample_1d_for_debug(time, values)
        if x_plot.size and y_plot.size:
            ax.plot(x_plot, y_plot, label=label, linewidth=linewidth, color=color, linestyle=linestyle, zorder=zorder)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel(_time_axis_label(time))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _save_debug_heatmap(
    path: Path,
    title: str,
    time: np.ndarray,
    frequencies: np.ndarray,
    matrix: np.ndarray,
    colorbar_label: str,
) -> bool:
    import matplotlib.pyplot as plt

    data = np.asarray(matrix)
    freqs = np.asarray(frequencies, dtype=float)
    if data.ndim != 2 or data.size == 0 or freqs.size != data.shape[0] or time.size == 0:
        return False
    render_data = np.asarray(_downsample_heatmap_for_debug(data), dtype=float)
    finite = render_data[np.isfinite(render_data)]
    if finite.size == 0:
        return False
    vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    extent = [float(time[0]), float(time[-1]), float(freqs[0]), float(freqs[-1])]
    im = ax.imshow(
        render_data,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel(_time_axis_label(time))
    ax.set_ylabel("Frequency (Hz)")
    if np.all(freqs > 0) and float(np.max(freqs)) / max(float(np.min(freqs)), 1e-12) > 10.0:
        ax.set_yscale("log")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _save_frequency_clusters_png(path: Path, frequencies: np.ndarray, labels: np.ndarray) -> bool:
    import matplotlib.pyplot as plt

    freqs = np.asarray(frequencies, dtype=float)
    cluster_labels = np.asarray(labels, dtype=int)
    if freqs.size == 0 or freqs.shape != cluster_labels.shape:
        return False
    unique = sorted(int(v) for v in np.unique(cluster_labels))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cluster_id in enumerate(unique):
        mask = cluster_labels == cluster_id
        ax.scatter(
            np.full(int(np.count_nonzero(mask)), int(cluster_id)),
            freqs[mask],
            s=26,
            color=cmap(i % 20),
            label=f"cluster {cluster_id}: {float(np.min(freqs[mask])):.3g}-{float(np.max(freqs[mask])):.3g} Hz",
        )
    ax.set_title("Ward Frequency Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Frequency (Hz)")
    if np.all(freqs > 0) and float(np.max(freqs)) / max(float(np.min(freqs)), 1e-12) > 10.0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _save_cluster_thresholds_png(path: Path, time: np.ndarray, threshold_debug: Dict[object, object]) -> bool:
    import matplotlib.pyplot as plt

    cluster_items = []
    for cluster_id, info in threshold_debug.items():
        if not isinstance(info, dict):
            continue
        activity = _debug_array(info.get("activity"))
        threshold = _debug_array(info.get("threshold"))
        moving = _debug_array(info.get("moving_average"))
        if activity is None or threshold is None or moving is None:
            continue
        cluster_items.append((int(cluster_id), activity, moving, threshold, info))
    if not cluster_items:
        return False
    cluster_items.sort(key=lambda item: item[0])
    rows = len(cluster_items)
    fig, axes = plt.subplots(rows, 1, figsize=(14, max(3.0, 2.2 * rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (cluster_id, activity, moving, threshold, info) in zip(axes, cluster_items):
        x_a, y_a = _downsample_1d_for_debug(time, activity)
        x_m, y_m = _downsample_1d_for_debug(time, moving)
        x_t, y_t = _downsample_1d_for_debug(time, threshold)
        ax.plot(x_a, y_a, label="activity", color="#1f77b4", linewidth=1.0)
        ax.plot(x_m, y_m, label="moving average", color="#ff7f0e", linewidth=1.0)
        ax.plot(x_t, y_t, label="threshold", color="#d62728", linewidth=1.0)
        max_freq = info.get("max_frequency", float("nan"))
        ax.set_title(f"Cluster {cluster_id} activity and threshold | max frequency {float(max_freq):.3g} Hz")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(_time_axis_label(time))
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _save_noise_classification_png(path: Path, decisions: List[dict], retained: List[int]) -> bool:
    import matplotlib.pyplot as plt

    if not decisions and not retained:
        return False
    rows = []
    for decision in decisions:
        rows.append(
            {
                "cluster_id": int(decision.get("cluster_id", 0)),
                "score": float(decision.get("score", 0.0)),
                "rejected": bool(decision.get("rejected", False)),
                "reason": str(decision.get("reason", "")),
            }
        )
    known = {row["cluster_id"] for row in rows}
    for cluster_id in retained:
        if int(cluster_id) not in known:
            rows.append({"cluster_id": int(cluster_id), "score": 0.0, "rejected": False, "reason": "retained"})
    rows.sort(key=lambda row: row["cluster_id"])
    colors = ["#c62828" if row["rejected"] else "#2e7d32" for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    ax.bar(x, [row["score"] for row in rows], color=colors)
    ax.set_xticks(x, [f"C{row['cluster_id']}" for row in rows])
    ax.set_ylabel("Noise-candidate score")
    ax.set_title("Frequency Cluster Noise Classification")
    for xpos, row in zip(x, rows):
        label = "rejected" if row["rejected"] else row["reason"]
        ax.text(xpos, row["score"], label, rotation=45, ha="left", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _save_attenuation_masks_png(path: Path, time: np.ndarray, masks: Dict[object, object]) -> bool:
    import matplotlib.pyplot as plt

    rows = []
    for cluster_id, mask in masks.items():
        arr = _debug_array(mask)
        if arr is not None and arr.ndim == 1:
            rows.append((int(cluster_id), arr))
    if not rows:
        return False
    rows.sort(key=lambda item: item[0])
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for cluster_id, values in rows:
        x_plot, y_plot = _downsample_1d_for_debug(time, values)
        ax.plot(x_plot, y_plot, label=f"cluster {cluster_id}", linewidth=1.0)
    ax.set_title("Coefficient Gain / Attenuation Masks")
    ax.set_xlabel(_time_axis_label(time))
    ax.set_ylabel("Gain (1 = retained, lower = suppressed)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _save_reconstructed_cluster_traces_png(path: Path, time: np.ndarray, traces: Dict[object, object]) -> bool:
    import matplotlib.pyplot as plt

    rows = []
    for cluster_id, trace in traces.items():
        arr = _debug_array(trace)
        if arr is not None and arr.ndim == 1:
            rows.append((int(cluster_id), arr))
    if not rows:
        return False
    rows.sort(key=lambda item: item[0])
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for cluster_id, values in rows:
        x_plot, y_plot = _downsample_1d_for_debug(time, values)
        ax.plot(x_plot, y_plot, label=f"cluster {cluster_id}", linewidth=1.0)
    ax.set_title("Reconstructed Cluster Contributions")
    ax.set_xlabel(_time_axis_label(time))
    ax.set_ylabel("Reconstructed signal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def _write_gonzalez_cwt_debug_export(
    trace_name: str,
    time: np.ndarray,
    corrected: np.ndarray,
    debug: Dict[str, Any],
    out_dir: Path,
    startup_exclusion_ms: float = 0.0,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    skipped: Dict[str, str] = {}
    time_arr = np.asarray(time, dtype=float)
    corrected_arr = np.asarray(corrected, dtype=float)
    n = min(time_arr.size, corrected_arr.size)
    time_arr = time_arr[:n]
    corrected_arr = corrected_arr[:n]
    startup_idx = _startup_exclusion_index(time_arr, startup_exclusion_ms)
    full_n = n
    if startup_idx > 0:
        time_arr = time_arr[startup_idx:]
        corrected_arr = corrected_arr[startup_idx:]
        n = time_arr.size

    def _trim_debug_time_axis(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values)
        if startup_idx <= 0:
            return arr
        if arr.ndim == 1 and arr.size >= full_n:
            return arr[startup_idx:full_n]
        if arr.ndim == 2 and arr.shape[1] >= full_n:
            return arr[:, startup_idx:full_n]
        return arr

    def _trim_nested_time_values(values: object) -> object:
        if startup_idx <= 0:
            return values
        if isinstance(values, dict):
            return {key: _trim_nested_time_values(value) for key, value in values.items()}
        arr = _debug_array(values)
        if arr is not None:
            trimmed = _trim_debug_time_axis(arr)
            return trimmed
        return values

    frequencies = _debug_array(debug.get("frequency_vector"))
    raw_coeffs = _trim_debug_time_axis(_debug_array(debug.get("original_complex_cwt_coefficients"), dtype=complex))
    cleaned_coeffs = _trim_debug_time_axis(_debug_array(debug.get("cleaned_complex_cwt_coefficients"), dtype=complex))
    normalized = _trim_debug_time_axis(_debug_array(debug.get("normalized_coefficient_magnitudes")))
    labels = _debug_array(debug.get("ward_cluster_labels"), dtype=int)
    output = _trim_debug_time_axis(_debug_array(debug.get("output")))
    wrapper_output = _trim_debug_time_axis(_debug_array(debug.get("wrapper_output")))
    wrapper_correction = _trim_debug_time_axis(_debug_array(debug.get("wrapper_correction_component")))

    def _record(name: str, ok: bool, reason: str = "source data unavailable") -> None:
        if ok:
            written.append(out_dir / name)
        else:
            skipped[name] = reason

    _record(
        "01_input_trace.png",
        _save_debug_line_plot(
            out_dir / "01_input_trace.png",
            f"{trace_name} | Gonzalez input trace",
            time_arr,
            [("corrected input", corrected_arr, "#1f77b4")],
        ),
    )
    _record(
        "02_morlet_cwt_scalogram_raw.png",
        bool(
            frequencies is not None
            and raw_coeffs is not None
            and _save_debug_heatmap(
                out_dir / "02_morlet_cwt_scalogram_raw.png",
                f"{trace_name} | Morlet CWT coefficient magnitude",
                time_arr,
                frequencies,
                np.abs(raw_coeffs),
                "|CWT coefficient|",
            )
        ),
    )
    _record(
        "03_morlet_cwt_scalogram_normalized.png",
        bool(
            frequencies is not None
            and normalized is not None
            and _save_debug_heatmap(
                out_dir / "03_morlet_cwt_scalogram_normalized.png",
                f"{trace_name} | Normalized CWT coefficient magnitude",
                time_arr,
                frequencies,
                normalized,
                "normalized magnitude",
            )
        ),
    )
    _record(
        "04_frequency_clusters.png",
        bool(
            frequencies is not None
            and labels is not None
            and _save_frequency_clusters_png(out_dir / "04_frequency_clusters.png", frequencies, labels)
        ),
    )
    threshold_debug = debug.get("threshold_per_cluster", {})
    if isinstance(threshold_debug, dict):
        threshold_debug = _trim_nested_time_values(threshold_debug)
    _record(
        "05_cluster_activity_thresholds.png",
        isinstance(threshold_debug, dict)
        and _save_cluster_thresholds_png(out_dir / "05_cluster_activity_thresholds.png", time_arr, threshold_debug),
    )
    decisions = debug.get("rejected_frequency_clusters", [])
    retained = debug.get("retained_frequency_clusters", [])
    _record(
        "06_noise_classification.png",
        _save_noise_classification_png(
            out_dir / "06_noise_classification.png",
            decisions if isinstance(decisions, list) else [],
            retained if isinstance(retained, list) else [],
        ),
    )
    masks = debug.get("attenuation_mask_per_cluster", {})
    if isinstance(masks, dict):
        masks = _trim_nested_time_values(masks)
    _record(
        "07_attenuation_masks.png",
        isinstance(masks, dict) and _save_attenuation_masks_png(out_dir / "07_attenuation_masks.png", time_arr, masks),
    )
    before_after_path = out_dir / "08_coefficients_before_after.png"
    if frequencies is not None and raw_coeffs is not None and cleaned_coeffs is not None and time_arr.size:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        for ax, title, matrix in [
            (axes[0], "Before attenuation", np.abs(raw_coeffs)),
            (axes[1], "After attenuation", np.abs(cleaned_coeffs)),
        ]:
            render = np.asarray(_downsample_heatmap_for_debug(matrix), dtype=float)
            finite = render[np.isfinite(render)]
            vmin, vmax = np.nanpercentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
            extent = [float(time_arr[0]), float(time_arr[-1]), float(frequencies[0]), float(frequencies[-1])]
            im = ax.imshow(
                render,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            ax.set_title(title)
            ax.set_ylabel("Frequency (Hz)")
            if np.all(frequencies > 0) and float(np.max(frequencies)) / max(float(np.min(frequencies)), 1e-12) > 10.0:
                ax.set_yscale("log")
            fig.colorbar(im, ax=ax, label="|CWT coefficient|")
        axes[-1].set_xlabel(_time_axis_label(time_arr))
        fig.suptitle(f"{trace_name} | CWT coefficients before vs after suppression")
        fig.savefig(before_after_path, dpi=160, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        _record("08_coefficients_before_after.png", True)
    else:
        _record("08_coefficients_before_after.png", False, "coefficient matrix stored as summary-only or unavailable")
    reconstructed = debug.get("reconstructed_cluster_traces", {})
    if isinstance(reconstructed, dict):
        reconstructed = _trim_nested_time_values(reconstructed)
    _record(
        "09_reconstructed_cluster_traces.png",
        isinstance(reconstructed, dict)
        and _save_reconstructed_cluster_traces_png(out_dir / "09_reconstructed_cluster_traces.png", time_arr, reconstructed),
    )
    output_for_plot = output[:n] if output is not None and output.size >= n else None
    wrapper_for_plot = wrapper_output[:n] if wrapper_output is not None and wrapper_output.size >= n else None
    final_for_plot = wrapper_for_plot if wrapper_for_plot is not None else output_for_plot
    overlay_series = [("input", corrected_arr, "#1f77b4")]
    if output_for_plot is not None:
        overlay_series.append(("raw Gonzalez", output_for_plot, "#ff7f0e"))
    if wrapper_for_plot is not None:
        overlay_series.append(("Gonzalez full-trace wrapper output", wrapper_for_plot, "#d62728"))
    _record(
        "10_final_overlay.png",
        len(overlay_series) > 1
        and _save_debug_line_plot(
            out_dir / "10_final_overlay.png",
            f"{trace_name} | Gonzalez input, raw CWT output, and wrapper output",
            time_arr,
            overlay_series,
        ),
    )
    _record(
        "11_removed_component.png",
        final_for_plot is not None
        and _save_debug_line_plot(
            out_dir / "11_removed_component.png",
            f"{trace_name} | Suppressed component (input - final output)",
            time_arr,
            [("removed component", corrected_arr - final_for_plot, "#6a3d9a")],
            ylabel="Input - final output",
        ),
    )
    correction_for_plot = (
        wrapper_correction[:n]
        if wrapper_correction is not None and wrapper_correction.size >= n
        else (
            wrapper_for_plot - output_for_plot
            if wrapper_for_plot is not None and output_for_plot is not None
            else None
        )
    )
    _record(
        "12_wrapper_correction_component.png",
        correction_for_plot is not None
        and _save_debug_line_plot(
            out_dir / "12_wrapper_correction_component.png",
            f"{trace_name} | Wrapper correction component",
            time_arr,
            [("wrapper output - raw Gonzalez", correction_for_plot, "#0f766e")],
            ylabel="Wrapper correction",
        ),
    )

    metrics = {
        "trace_name": trace_name,
        "startup_exclusion_ms": float(startup_exclusion_ms),
        "startup_exclusion_samples": int(startup_idx),
        "denoiser_used": debug.get("denoiser_used"),
        "input_mode": debug.get("input_mode"),
        "wavelet": debug.get("wavelet", "cmor1.5-1.0"),
        "frequency_min_hz": float(np.nanmin(frequencies)) if frequencies is not None and frequencies.size else float("nan"),
        "frequency_max_hz": float(np.nanmax(frequencies)) if frequencies is not None and frequencies.size else float("nan"),
        "frequency_count": int(frequencies.size) if frequencies is not None else 0,
        "chosen_cluster_count": debug.get("chosen_cluster_count"),
        "selected_pc_count": debug.get("selected_pc_count"),
        "retained_frequency_clusters": debug.get("retained_frequency_clusters"),
        "rejected_frequency_clusters": debug.get("rejected_frequency_clusters"),
        "threshold_per_cluster": threshold_debug,
        "enable_noise_cluster_rejection": debug.get("enable_noise_cluster_rejection"),
        "enable_event_template_rejection": debug.get("enable_event_template_rejection"),
        "safety_metrics": debug.get("safety_metrics"),
        "warnings": debug.get("warnings"),
        "amplitude_preservation_diagnostics": debug.get("amplitude_preservation_diagnostics"),
        "wrapper_output_mode": debug.get("wrapper_output_mode"),
        "has_wrapper_output": wrapper_output is not None,
        "has_wrapper_correction_component": correction_for_plot is not None,
        "skipped_images": skipped,
    }
    metrics_path = out_dir / "gonzalez_cwt_debug_metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    written.append(metrics_path)
    return written


def _debug_mask(values: object, n: int) -> np.ndarray:
    arr = _debug_array(values, dtype=bool)
    if arr is None:
        return np.zeros(int(max(0, n)), dtype=bool)
    out = np.zeros(int(max(0, n)), dtype=bool)
    m = min(out.size, arr.size)
    if m > 0:
        out[:m] = np.asarray(arr[:m], dtype=bool)
    return out


def _plot_mask_bands(ax: object, time: np.ndarray, masks: List[tuple[str, np.ndarray, str]]) -> None:
    import matplotlib.patches as patches

    ax.set_ylim(-0.5, len(masks) - 0.5)
    ax.set_yticks(np.arange(len(masks)), [label for label, _mask, _color in masks])
    ax.set_xlabel(_time_axis_label(time))
    ax.grid(axis="x", alpha=0.2)
    if time.size == 0:
        return
    for row_idx, (_label, mask, color) in enumerate(masks):
        for start, end in _contiguous_true_regions(np.asarray(mask, dtype=bool)):
            s = max(0, min(int(start), time.size - 1))
            e = max(s, min(int(end) - 1, time.size - 1))
            ax.add_patch(
                patches.Rectangle(
                    (float(time[s]), row_idx - 0.35),
                    max(float(time[e] - time[s]), 1e-12),
                    0.7,
                    color=color,
                    alpha=0.75,
                )
            )


def _shade_mask_regions(ax: object, time: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.18) -> None:
    if time.size == 0:
        return
    for start, end in _contiguous_true_regions(np.asarray(mask, dtype=bool)):
        s = max(0, min(int(start), time.size - 1))
        e = max(s, min(int(end) - 1, time.size - 1))
        ax.axvspan(float(time[s]), float(time[e]), color=color, alpha=alpha, linewidth=0)


def _write_long_plateau_debug_export(
    trace_name: str,
    time: np.ndarray,
    corrected: np.ndarray,
    debug: Dict[str, Any],
    out_dir: Path,
    max_rejected_zooms: int = 50,
    startup_exclusion_ms: float = 0.0,
) -> List[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    skipped: Dict[str, str] = {}
    time_arr = np.asarray(time, dtype=float)
    corrected_arr = np.asarray(corrected, dtype=float)
    n = min(time_arr.size, corrected_arr.size)
    time_arr = time_arr[:n]
    corrected_arr = corrected_arr[:n]
    full_n = n
    startup_idx = _startup_exclusion_index(time_arr, startup_exclusion_ms)
    if startup_idx > 0:
        time_arr = time_arr[startup_idx:]
        corrected_arr = corrected_arr[startup_idx:]
        n = time_arr.size
    if n <= 0:
        return written

    def _trim_line(values: object, fallback: np.ndarray) -> np.ndarray:
        arr = _debug_array(values)
        if arr is None:
            return np.asarray(fallback, dtype=float)
        arr = np.asarray(arr, dtype=float)
        out = np.zeros(n, dtype=float)
        if startup_idx > 0 and arr.size >= full_n:
            source = arr[startup_idx:full_n]
        else:
            source = arr[:n]
        m = min(out.size, source.size)
        if m > 0:
            out[:m] = source[:m]
        return out

    def _trim_mask(values: object) -> np.ndarray:
        arr = _debug_array(values, dtype=bool)
        out = np.zeros(n, dtype=bool)
        if arr is None:
            return out
        if startup_idx > 0 and arr.size >= full_n:
            source = np.asarray(arr[startup_idx:full_n], dtype=bool)
        else:
            source = np.asarray(arr[:n], dtype=bool)
        m = min(out.size, source.size)
        if m > 0:
            out[:m] = source[:m]
        return out

    input_signal = _trim_line(debug.get("long_plateau_input_signal"), corrected_arr)
    proxy = _trim_line(debug.get("long_plateau_plateau_proxy"), input_signal)
    drift = _trim_line(debug.get("long_plateau_slow_drift"), np.zeros(n, dtype=float))
    detrended = _trim_line(debug.get("long_plateau_detrended_proxy"), proxy - drift)
    high = _trim_line(debug.get("long_plateau_high_threshold_line"), np.zeros(n, dtype=float))
    low = _trim_line(debug.get("long_plateau_low_threshold_line"), np.zeros(n, dtype=float))
    otsu = _trim_line(debug.get("long_plateau_otsu_threshold_line"), np.zeros(n, dtype=float))

    valid = _trim_mask(debug.get("long_plateau_valid_mask"))
    excluded = _trim_mask(debug.get("long_plateau_excluded_mask"))
    above = _trim_mask(debug.get("long_plateau_above_entry_mask"))
    below = _trim_mask(debug.get("long_plateau_below_exit_mask"))
    hysteresis = _trim_mask(debug.get("long_plateau_hysteresis_mask"))
    otsu_candidate = _trim_mask(debug.get("long_plateau_otsu_candidate_mask"))
    after_min = _trim_mask(debug.get("long_plateau_after_min_duration_mask"))
    after_refine = _trim_mask(debug.get("long_plateau_after_refinement_mask"))
    after_impulse = _trim_mask(debug.get("long_plateau_after_impulse_suppression_mask"))
    after_merge = _trim_mask(debug.get("long_plateau_after_merge_mask"))
    after_rescue = _trim_mask(debug.get("long_plateau_after_overmask_rescue_mask"))
    final_mask = _trim_mask(debug.get("long_plateau_final_mask"))

    def _record(name: str, ok: bool, reason: str = "source data unavailable") -> None:
        if ok:
            written.append(out_dir / name)
        else:
            skipped[name] = reason

    path = out_dir / "01_input_and_proxy.png"
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _shade_mask_regions(ax, time_arr, final_mask, "#dc2626", alpha=0.16)
    ax.plot(*_downsample_1d_for_debug(time_arr, input_signal), label="detector input", color="#2563eb", linewidth=1.0)
    ax.plot(*_downsample_1d_for_debug(time_arr, proxy), label="median plateau proxy", color="#f97316", linewidth=1.0)
    ax.set_title(f"{trace_name} | Long plateau input and median proxy")
    ax.set_xlabel(_time_axis_label(time_arr))
    ax.set_ylabel("Signal")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _record(path.name, True)

    path = out_dir / "02_drift_correction.png"
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(*_downsample_1d_for_debug(time_arr, proxy), label="plateau proxy", color="#f97316", linewidth=1.0)
    axes[0].plot(*_downsample_1d_for_debug(time_arr, drift), label="slow drift", color="#475569", linewidth=1.0)
    axes[0].set_title("Proxy and slow drift")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.22)
    axes[1].plot(*_downsample_1d_for_debug(time_arr, detrended), label="detrended proxy", color="#16a34a", linewidth=1.0)
    axes[1].set_title("Detrended proxy")
    axes[1].set_xlabel(_time_axis_label(time_arr))
    axes[1].grid(True, alpha=0.22)
    fig.suptitle(f"{trace_name} | Long plateau drift correction")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _record(path.name, True)

    def _constant_line_value(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        return float(np.median(finite)) if finite.size else float("nan")

    otsu_label = "Otsu threshold"
    otsu_value = _constant_line_value(otsu)
    high_value = _constant_line_value(high)
    low_value = _constant_line_value(low)
    overlap_tol = max(1e-9, 1e-6 * max(1.0, abs(otsu_value), abs(high_value), abs(low_value)))
    if np.isfinite(otsu_value) and np.isfinite(high_value) and abs(otsu_value - high_value) <= overlap_tol:
        otsu_label = "Otsu threshold (overlaps entry/high)"
    elif np.isfinite(otsu_value) and np.isfinite(low_value) and abs(otsu_value - low_value) <= overlap_tol:
        otsu_label = "Otsu threshold (overlaps exit/low)"

    path = out_dir / "03_thresholds_overlay.png"
    _record(
        path.name,
        _save_debug_line_plot(
            path,
            f"{trace_name} | Long plateau thresholds",
            time_arr,
            [
                ("detrended proxy", detrended, "#16a34a", "-", 1.1, 2),
                ("entry/high threshold", high, "#dc2626", "-", 1.5, 3),
                ("exit/low threshold", low, "#0891b2", "-", 1.5, 3),
                (otsu_label, otsu, "#7c3aed", "--", 2.3, 5),
            ],
            ylabel="Detrended signal",
        ),
    )

    path = out_dir / "04_hysteresis_logic.png"
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _plot_mask_bands(
        ax,
        time_arr,
        [
            ("valid", valid, "#94a3b8"),
            ("excluded", excluded, "#111827"),
            ("above entry", above, "#dc2626"),
            ("below exit", below, "#0891b2"),
            ("hysteresis", hysteresis, "#f97316"),
        ],
    )
    ax.set_title(f"{trace_name} | Long plateau hysteresis logic")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _record(path.name, True)

    path = out_dir / "05_candidate_filtering.png"
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _plot_mask_bands(
        ax,
        time_arr,
        [
            ("hysteresis", hysteresis, "#f97316"),
            ("region-filter candidate", otsu_candidate, "#7c3aed"),
            ("min duration", after_min, "#16a34a"),
            ("final long plateau", final_mask, "#dc2626"),
        ],
    )
    ax.set_title(f"{trace_name} | Candidate filtering")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _record(path.name, True)

    path = out_dir / "06_pipeline_masks.png"
    fig, ax = plt.subplots(figsize=(14, 6.5))
    _plot_mask_bands(
        ax,
        time_arr,
        [
            ("above entry", above, "#dc2626"),
            ("hysteresis", hysteresis, "#f97316"),
            ("candidate", otsu_candidate, "#7c3aed"),
            ("min duration", after_min, "#16a34a"),
            ("refined", after_refine, "#0d9488"),
            ("impulse filtered", after_impulse, "#0891b2"),
            ("merged", after_merge, "#2563eb"),
            ("overmask rescue", after_rescue, "#4f46e5"),
            ("final", final_mask, "#991b1b"),
        ],
    )
    ax.set_title(f"{trace_name} | Long plateau pipeline masks")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _record(path.name, True)

    rows = debug.get("long_plateau_region_debug_rows", [])
    if not isinstance(rows, list):
        rows = []
    display_rows: List[Dict[str, Any]] = []
    omitted_startup_region_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        original_start = int(row.get("start_index", 0))
        original_end = int(row.get("end_index_exclusive", original_start + 1))
        if original_end <= startup_idx:
            omitted_startup_region_count += 1
            continue
        display_row = dict(row)
        display_start = max(0, original_start - startup_idx)
        display_end = max(display_start + 1, original_end - startup_idx)
        display_end = min(display_end, n)
        display_row["display_start_index"] = int(display_start)
        display_row["display_end_index_exclusive"] = int(display_end)
        display_row["startup_exclusion_ms"] = float(startup_exclusion_ms)
        display_rows.append(display_row)
    rows = display_rows
    accepted_rows = [row for row in rows if bool(row.get("accepted", False))]
    rejected_rows = [row for row in rows if not bool(row.get("accepted", False))]
    path = out_dir / "07_final_overlap.png"
    fig, ax = plt.subplots(figsize=(14, 5.2))
    _shade_mask_regions(ax, time_arr, final_mask, "#dc2626", alpha=0.16)
    ax.plot(*_downsample_1d_for_debug(time_arr, input_signal), label="detector input", color="#2563eb", linewidth=1.0)
    y_text = float(np.nanmax(input_signal)) if np.any(np.isfinite(input_signal)) else 0.0
    for row in accepted_rows:
        s = int(row.get("display_start_index", row.get("start_index", 0)))
        e = int(row.get("display_end_index_exclusive", s + 1)) - 1
        if not (0 <= s < n and 0 <= e < n):
            continue
    for row in rejected_rows[:50]:
        s = int(row.get("display_start_index", row.get("start_index", 0)))
        e = int(row.get("display_end_index_exclusive", s + 1)) - 1
        if 0 <= s < n and 0 <= e < n:
            ax.axvspan(float(time_arr[s]), float(time_arr[e]), color="#64748b", alpha=0.10, linewidth=0)
    ax.set_title(f"{trace_name} | Accepted long plateaus and rejected candidates")
    ax.set_xlabel(_time_axis_label(time_arr))
    ax.set_ylabel("Signal")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best")
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _record(path.name, True)

    regions_path = out_dir / "long_plateau_debug_regions.csv"
    pd.DataFrame(rows).to_csv(regions_path, index=False)
    written.append(regions_path)

    rejected_sorted = sorted(rejected_rows, key=lambda row: float(row.get("near_miss_score", 0.0)), reverse=True)
    for row in rejected_sorted[: int(max_rejected_zooms)]:
        region_id = int(row.get("region_id", 0))
        start = int(row.get("display_start_index", row.get("start_index", 0)))
        end = int(row.get("display_end_index_exclusive", start + 1))
        pad = max(10, int(round(0.05 * n)))
        lo = max(0, start - pad)
        hi = min(n, end + pad)
        if hi <= lo:
            continue
        path = out_dir / f"rejected_candidate_{region_id:03d}_zoom.png"
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(time_arr[lo:hi], input_signal[lo:hi], color="#2563eb", linewidth=1.0, label="input")
        axes[0].axvspan(float(time_arr[start]), float(time_arr[max(start, end - 1)]), color="#64748b", alpha=0.18)
        axes[0].set_title(f"Rejected candidate: {row.get('rejection_reason', '')}")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.22)
        axes[1].plot(time_arr[lo:hi], detrended[lo:hi], color="#16a34a", linewidth=1.0, label="detrended proxy")
        axes[1].plot(time_arr[lo:hi], high[lo:hi], color="#dc2626", linewidth=1.0, label="entry")
        axes[1].plot(time_arr[lo:hi], low[lo:hi], color="#0891b2", linewidth=1.0, label="exit")
        axes[1].plot(time_arr[lo:hi], otsu[lo:hi], color="#7c3aed", linewidth=1.0, label="Otsu")
        axes[1].axvspan(float(time_arr[start]), float(time_arr[max(start, end - 1)]), color="#64748b", alpha=0.18)
        axes[1].set_xlabel(_time_axis_label(time_arr))
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.22)
        fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    metrics = dict(debug.get("long_plateau_debug_metrics", {}) if isinstance(debug.get("long_plateau_debug_metrics"), dict) else {})
    metrics.update(
        {
            "trace_name": trace_name,
            "startup_exclusion_ms": float(startup_exclusion_ms),
            "startup_exclusion_samples": int(startup_idx),
            "startup_excluded_region_count": int(omitted_startup_region_count),
            "accepted_region_count": len(accepted_rows),
            "rejected_region_count": len(rejected_rows),
            "final_plateau_count": len(_contiguous_true_regions(final_mask)),
            "rejected_zoom_count": min(len(rejected_sorted), int(max_rejected_zooms)),
            "omitted_rejected_zoom_count": max(0, len(rejected_sorted) - int(max_rejected_zooms)),
            "skipped_images": skipped,
        }
    )
    metrics_path = out_dir / "long_plateau_debug_metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    written.append(metrics_path)
    return written




def _safe_excel_sheet_name(name: str, used_names: set[str]) -> str:
    forbidden = set('[]:*?/\\')
    cleaned = "".join("_" if ch in forbidden else ch for ch in str(name)).strip()
    if not cleaned:
        cleaned = "trace"
    base = cleaned[:31]
    candidate = base
    idx = 1
    while candidate in used_names:
        suffix = f"_{idx}"
        candidate = f"{base[: max(1, 31 - len(suffix))]}{suffix}"
        idx += 1
    used_names.add(candidate)
    return candidate


def _spike_record_from_event_stats_row(row: pd.Series, trace_name: str) -> SpikeRecord:
    def _num(key: str, default: float = float("nan")) -> float:
        return float(pd.to_numeric(row.get(key, default), errors="coerce"))

    peak_index = int(_num("peak_index", 0.0))
    amplitude = _num("amplitude", 0.0)
    snr = _num("snr", 0.0)
    dff = _num("delta_f_over_f0")
    baseline = float(amplitude / dff) if np.isfinite(amplitude) and np.isfinite(dff) and dff != 0.0 else 0.0
    peak_value = float(baseline + amplitude) if np.isfinite(baseline) else float(amplitude)
    return SpikeRecord(
        trace_name=trace_name,
        candidate_index=peak_index,
        spike_index=peak_index,
        spike_time=_num("peak_time", 0.0),
        spike_amplitude_raw_or_corrected=peak_value,
        baseline_at_spike=baseline,
        amplitude_above_baseline=amplitude,
        prominence=0.0,
        width=1.0,
        local_noise_estimate=float(amplitude / snr) if np.isfinite(amplitude) and np.isfinite(snr) and snr != 0.0 else 0.0,
        snr=snr,
        detection_threshold_used=0.0,
        fwhm_ms=_num("fwhm_ms"),
        delta_f_over_f0=dff,
        rise_time_ms=_num("rise_time_ms"),
        return_to_baseline_time_ms=_num("return_to_baseline_time_ms"),
        post_spike_level=_num("post_spike_level"),
        status="pending",
        source=str(row.get("source", "auto")),
        notes="loaded-spike-event-stats",
    )


@dataclass(frozen=True)
class TraceAnalysisResult:
    time: np.ndarray
    raw: np.ndarray
    corrected_source: np.ndarray
    corrected: np.ndarray
    hybrid_denoised: np.ndarray
    gonzalez_cwt_debug: Dict[str, Any]
    long_plateau_debug: Dict[str, Any]
    plateau_signal_classification: List[Dict[str, Any]]
    correction_baseline: np.ndarray
    detection: object
    artifact_mask: np.ndarray
    sampling_rate_hz: float
    highpass_trace: Optional[np.ndarray]
    lowpass_trace: Optional[np.ndarray]


@dataclass(frozen=True)
class TraceAnalysisRequest:
    trace_name: str
    time: np.ndarray
    raw: np.ndarray
    corrected_source: np.ndarray
    sampling_rate_hz: float
    detection_params: DetectionParams
    artifact_params: ArtifactParams
    pipeline_config: Dict[str, Any]
    generation: int


def _clone_detection_params(params: DetectionParams) -> DetectionParams:
    return DetectionParams.from_dict(params.to_dict())


def _clone_artifact_params(params: ArtifactParams) -> ArtifactParams:
    return ArtifactParams.from_dict(params.to_dict())


def _user_template_indices_from_trace(state: TraceCuration) -> List[int]:
    indices: List[int] = []
    for spike in list(state.spikes):
        status = str(getattr(spike, "status", "")).strip().lower()
        source = str(getattr(spike, "source", "")).strip().lower()
        if status in {"accepted", "manual"} or source == "manual":
            indices.append(int(spike.spike_index))
    return sorted(set(index for index in indices if index >= 0))


_DOWNSAMPLE_STEP_ID = "rolling_average_downsample"
_FILTER_STEP_IDS = ("highpass_filter", "lowpass_filter")
_PIPELINE_TRANSFORM_STEP_IDS = (*_FILTER_STEP_IDS, _DOWNSAMPLE_STEP_ID)


def _pipeline_config_from_payload(data: Dict[str, Any]) -> PipelineConfig:
    if isinstance(data, dict):
        return PipelineConfig.from_dict(data)
    return PipelineConfig.from_preset("current")


def _pipeline_has_step(config: PipelineConfig, step_id: str) -> bool:
    return str(step_id) in [str(item) for item in config.ordered_step_ids]


def _pipeline_step_enabled(config: PipelineConfig, step_id: str) -> bool:
    return _pipeline_has_step(config, step_id) and bool(config.enabled_steps.get(step_id, False))


def _pipeline_enabled_order(config: PipelineConfig) -> List[str]:
    ordered: List[str] = []
    for step_id in config.ordered_step_ids:
        step = str(step_id)
        if step in PIPELINE_STEP_DEFINITIONS and step not in ordered and bool(config.enabled_steps.get(step, True)):
            ordered.append(step)
    return ordered


def _pipeline_stage_index(ordered_steps: List[str], step_id: str) -> int:
    try:
        return ordered_steps.index(step_id)
    except ValueError:
        return len(ordered_steps)


def _pipeline_transform_steps_between(
    config: PipelineConfig,
    start_step_id: Optional[str],
    end_step_id: str,
) -> List[str]:
    ordered = _pipeline_enabled_order(config)
    start = -1 if start_step_id is None else _pipeline_stage_index(ordered, start_step_id)
    end = _pipeline_stage_index(ordered, end_step_id)
    if end < start:
        end = len(ordered)
    return [
        step_id
        for index, step_id in enumerate(ordered)
        if start < index < end and step_id in _PIPELINE_TRANSFORM_STEP_IDS
    ]


def _pipeline_filter_steps_between(
    config: PipelineConfig,
    start_step_id: Optional[str],
    end_step_id: str,
) -> List[str]:
    return [
        step_id
        for step_id in _pipeline_transform_steps_between(config, start_step_id, end_step_id)
        if step_id in _FILTER_STEP_IDS
    ]


def _legacy_filter_steps(params: DetectionParams) -> List[str]:
    steps: List[str] = []
    if bool(getattr(params, "use_highpass_filter", False)):
        steps.append("highpass_filter")
    if bool(getattr(params, "use_lowpass_filter", False)):
        steps.append("lowpass_filter")
    return steps


def _pipeline_uses_transform_steps(config: PipelineConfig) -> bool:
    return any(step_id in _PIPELINE_TRANSFORM_STEP_IDS for step_id in config.ordered_step_ids)


def _rolling_downsample_factor(params: DetectionParams) -> int:
    try:
        value = int(getattr(params, "rolling_downsample_factor", 2))
    except (TypeError, ValueError):
        value = 2
    return max(2, min(100, value))


def _rolling_block_mean_1d(values: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    factor = max(2, int(factor))
    usable = (arr.size // factor) * factor
    if usable <= 0:
        raise ValueError(
            f"Rolling-average downsampling factor {factor} is larger than the available sample count {arr.size}."
        )
    blocks = arr[:usable].reshape(-1, factor)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=1)
    sums = np.where(finite, blocks, 0.0).sum(axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    )


def _rolling_block_any_1d(values: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(values, dtype=bool)
    factor = max(2, int(factor))
    usable = (arr.size // factor) * factor
    if usable <= 0:
        raise ValueError(
            f"Rolling-average downsampling factor {factor} is larger than the available mask length {arr.size}."
        )
    return np.any(arr[:usable].reshape(-1, factor), axis=1)


def _downsample_optional_numeric(values: Optional[np.ndarray], factor: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    return _rolling_block_mean_1d(np.asarray(values, dtype=float), factor)


def _downsample_optional_bool(values: Optional[np.ndarray], factor: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    return _rolling_block_any_1d(np.asarray(values, dtype=bool), factor)


def _downsample_filter_traces(filter_traces: Dict[str, np.ndarray], factor: int) -> None:
    for key, values in list(filter_traces.items()):
        filter_traces[key] = _rolling_block_mean_1d(np.asarray(values, dtype=float), factor)


def _downsample_pipeline_state(
    state: Dict[str, Any],
    factor: int,
    filter_traces: Dict[str, np.ndarray],
) -> None:
    state["time"] = _rolling_block_mean_1d(np.asarray(state["time"], dtype=float), factor)
    state["raw"] = _rolling_block_mean_1d(np.asarray(state["raw"], dtype=float), factor)
    state["corrected_source"] = _rolling_block_mean_1d(
        np.asarray(state["corrected_source"], dtype=float),
        factor,
    )
    state["signal"] = _rolling_block_mean_1d(np.asarray(state["signal"], dtype=float), factor)
    state["sampling_rate_hz"] = float(state["sampling_rate_hz"]) / float(factor)
    state["artifact_mask"] = _downsample_optional_bool(state.get("artifact_mask"), factor)
    state["baseline"] = _downsample_optional_numeric(state.get("baseline"), factor)
    state["plateau_mask"] = _downsample_optional_bool(state.get("plateau_mask"), factor)
    side_numeric = state.get("side_numeric")
    if isinstance(side_numeric, dict):
        for key, values in list(side_numeric.items()):
            side_numeric[key] = _rolling_block_mean_1d(np.asarray(values, dtype=float), factor)
    side_bool = state.get("side_bool")
    if isinstance(side_bool, dict):
        for key, values in list(side_bool.items()):
            side_bool[key] = _rolling_block_any_1d(np.asarray(values, dtype=bool), factor)
    _downsample_filter_traces(filter_traces, factor)


def _apply_pipeline_transform_steps(
    state: Dict[str, Any],
    steps: List[str],
    params: DetectionParams,
    filter_traces: Dict[str, np.ndarray],
) -> None:
    current = np.asarray(state["signal"], dtype=float).copy()
    sampling_rate_hz = float(state["sampling_rate_hz"])
    for step_id in steps:
        if step_id == "highpass_filter":
            current = _butterworth_highpass(
                current,
                fs=sampling_rate_hz,
                cutoff_hz=float(getattr(params, "highpass_cutoff_hz", 1.0)),
                order=int(getattr(params, "highpass_order", 3)),
            )
            filter_traces["highpass_filter"] = current.copy()
        elif step_id == "lowpass_filter":
            current = _butterworth_lowpass(
                current,
                fs=sampling_rate_hz,
                cutoff_hz=float(getattr(params, "lowpass_cutoff_hz", 50.0)),
                order=int(getattr(params, "lowpass_order", 5)),
            )
            filter_traces["lowpass_filter"] = current.copy()
        elif step_id == _DOWNSAMPLE_STEP_ID:
            state["signal"] = current
            _downsample_pipeline_state(state, _rolling_downsample_factor(params), filter_traces)
            current = np.asarray(state["signal"], dtype=float).copy()
            sampling_rate_hz = float(state["sampling_rate_hz"])
    state["signal"] = current


def _apply_pipeline_filter_steps(
    signal: np.ndarray,
    steps: List[str],
    params: DetectionParams,
    sampling_rate_hz: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    current = np.asarray(signal, dtype=float).copy()
    traces: Dict[str, np.ndarray] = {}
    for step_id in steps:
        if step_id == "highpass_filter":
            current = _butterworth_highpass(
                current,
                fs=sampling_rate_hz,
                cutoff_hz=float(getattr(params, "highpass_cutoff_hz", 1.0)),
                order=int(getattr(params, "highpass_order", 3)),
            )
            traces["highpass_filter"] = current.copy()
        elif step_id == "lowpass_filter":
            current = _butterworth_lowpass(
                current,
                fs=sampling_rate_hz,
                cutoff_hz=float(getattr(params, "lowpass_cutoff_hz", 50.0)),
                order=int(getattr(params, "lowpass_order", 5)),
            )
            traces["lowpass_filter"] = current.copy()
    return current, traces


def _analysis_gonzalez_denoise_kwargs(params: DetectionParams) -> Dict[str, object]:
    return {
        "threshold_sd": float(params.gonzalez_threshold_sd),
        "max_clusters": int(params.gonzalez_max_clusters),
        "attenuation_min": float(params.gonzalez_attenuation_min),
        "attenuation_max": float(params.gonzalez_attenuation_max),
        "enable_noise_cluster_rejection": bool(params.gonzalez_enable_noise_cluster_rejection),
        "enable_event_template_rejection": bool(params.gonzalez_enable_event_template_rejection),
        "enable_denoise_safety_gate": bool(params.enable_denoise_safety_gate),
        "min_event_peak_preservation": float(params.min_event_peak_preservation),
        "min_local_max_ratio": float(params.min_local_max_ratio),
    }


def analyze_trace_data(
    request: TraceAnalysisRequest,
) -> TraceAnalysisResult:
    """Run optional artifact interpolation, plateau detection, denoising, and spike detection for one trace."""
    time = np.asarray(request.time, dtype=float)
    raw = np.asarray(request.raw, dtype=float)
    corrected_source = np.asarray(request.corrected_source, dtype=float)
    n = min(time.size, raw.size, corrected_source.size)
    time = time[:n]
    raw = raw[:n]
    corrected_source = corrected_source[:n]
    pipeline_config = _pipeline_config_from_payload(request.pipeline_config)
    transform_pipeline_mode = _pipeline_uses_transform_steps(pipeline_config)
    analysis_params = _clone_detection_params(request.detection_params)
    # Ordered pipeline filters are applied explicitly here. Disable the legacy
    # detector-side preprocessing flags so a filter is not applied twice.
    analysis_params.use_highpass_filter = False
    analysis_params.use_lowpass_filter = False
    sampling_rate_hz = (
        float(request.sampling_rate_hz)
        if np.isfinite(request.sampling_rate_hz) and float(request.sampling_rate_hz) > 0.0
        else _sampling_rate_hz(time)
    )
    filter_traces: Dict[str, np.ndarray] = {}
    working_state: Dict[str, Any] = {
        "time": time.copy(),
        "raw": raw.copy(),
        "corrected_source": corrected_source.copy(),
        "signal": corrected_source.copy(),
        "sampling_rate_hz": sampling_rate_hz,
        "artifact_mask": None,
        "baseline": None,
        "plateau_mask": None,
        "side_numeric": {},
        "side_bool": {},
    }
    downsampled_before_artifact = False
    if _pipeline_step_enabled(pipeline_config, "artifact_cleanup"):
        pre_artifact_downsample = [
            step_id
            for step_id in _pipeline_transform_steps_between(pipeline_config, None, "artifact_cleanup")
            if step_id == _DOWNSAMPLE_STEP_ID
        ]
        if pre_artifact_downsample:
            _apply_pipeline_transform_steps(
                working_state,
                pre_artifact_downsample,
                request.detection_params,
                filter_traces,
            )
            downsampled_before_artifact = True
    if _pipeline_step_enabled(pipeline_config, "artifact_cleanup"):
        corrected, artifact_mask = apply_artifact_interpolation(
            np.asarray(working_state["time"], dtype=float),
            np.asarray(working_state["signal"], dtype=float),
            params=request.artifact_params.to_dict(),
        )
    else:
        corrected = np.asarray(working_state["signal"], dtype=float).copy()
        artifact_mask = np.zeros(corrected.size, dtype=bool)
    working_state["signal"] = corrected
    working_state["artifact_mask"] = np.asarray(artifact_mask, dtype=bool)
    pre_baseline_steps = (
        _pipeline_transform_steps_between(pipeline_config, None, "baseline_correction")
        if transform_pipeline_mode
        else _legacy_filter_steps(request.detection_params)
    )
    if downsampled_before_artifact:
        pre_baseline_steps = [step_id for step_id in pre_baseline_steps if step_id != _DOWNSAMPLE_STEP_ID]
    _apply_pipeline_transform_steps(
        working_state,
        pre_baseline_steps,
        request.detection_params,
        filter_traces,
    )

    baseline_result = detect_spikes(
        request.trace_name,
        np.asarray(working_state["time"], dtype=float),
        np.asarray(working_state["signal"], dtype=float),
        analysis_params,
        artifact_mask=np.asarray(working_state["artifact_mask"], dtype=bool),
        sampling_rate_hz=float(working_state["sampling_rate_hz"]),
    )
    baseline = np.asarray(baseline_result.baseline, dtype=float)
    computation_source = np.asarray(baseline_result.processed_signal, dtype=float)
    plateau_mask = _detection_mask(baseline_result, "long_plateau_mask")
    denoising_method = str(analysis_params.denoising_method or "gonzalez_full_trace")
    corrected_baseline_corrected = computation_source - baseline
    working_state["signal"] = corrected_baseline_corrected
    working_state["baseline"] = baseline
    working_state["plateau_mask"] = plateau_mask
    working_state["side_numeric"] = {"corrected_for_result": corrected_baseline_corrected.copy()}
    working_state["side_bool"] = {}
    denoising_enabled = (
        _pipeline_step_enabled(pipeline_config, "denoising")
        if transform_pipeline_mode
        else denoising_method != "none"
    )
    if not denoising_enabled:
        denoising_method = "none"
    if denoising_enabled:
        pre_denoise_steps = (
            _pipeline_transform_steps_between(pipeline_config, "baseline_correction", "denoising")
            if transform_pipeline_mode
            else []
        )
        post_denoise_steps = (
            _pipeline_transform_steps_between(pipeline_config, "denoising", "spike_detection")
            if transform_pipeline_mode
            else []
        )
    else:
        pre_denoise_steps = []
        post_denoise_steps = (
            _pipeline_transform_steps_between(pipeline_config, "baseline_correction", "spike_detection")
            if transform_pipeline_mode
            else []
        )
    _apply_pipeline_transform_steps(
        working_state,
        pre_denoise_steps,
        request.detection_params,
        filter_traces,
    )
    denoise_input = np.asarray(working_state["signal"], dtype=float)
    corrected_for_result = np.asarray(
        working_state.get("side_numeric", {}).get("corrected_for_result", denoise_input),
        dtype=float,
    )
    baseline = np.asarray(working_state["baseline"], dtype=float)
    plateau_mask = np.asarray(working_state["plateau_mask"], dtype=bool)
    sampling_rate_hz = float(working_state["sampling_rate_hz"])
    # Denoising is used as a detection aid. The returned corrected trace below
    # remains the preferred source for quantitative amplitude, width, area,
    # plateau level, and export summary measurements.
    if denoising_method == "none":
        gonzalez_cwt_debug = dict(
            denoise_trace_gonzalez_full_trace(
                denoise_input,
                fs=sampling_rate_hz,
                denoising_method="none",
                return_debug=True,
            )
        )
        denoised_baseline_corrected = np.asarray(gonzalez_cwt_debug.get("output", denoise_input), dtype=float)
    else:
        denoise_debug_payload = denoise_trace_gonzalez_full_trace(
            denoise_input,
            fs=sampling_rate_hz,
            denoising_method=denoising_method,
            normalization_baseline=baseline,
            return_debug=True,
            **_analysis_gonzalez_denoise_kwargs(analysis_params),
        )
        if isinstance(denoise_debug_payload, dict):
            denoised_baseline_corrected = np.asarray(
                denoise_debug_payload.get("output", denoise_input),
                dtype=float,
            )
            core_debug = denoise_debug_payload.get("gonzalez_debug")
            gonzalez_cwt_debug = dict(core_debug) if isinstance(core_debug, dict) else dict(denoise_debug_payload)
        else:
            denoised_baseline_corrected = np.asarray(denoise_debug_payload, dtype=float)
            gonzalez_cwt_debug = {
                "denoiser_used": denoising_method,
                "output": denoised_baseline_corrected.copy(),
            }
        if "denoiser_used" not in gonzalez_cwt_debug:
            gonzalez_cwt_debug["denoiser_used"] = denoising_method
        if "output" not in gonzalez_cwt_debug:
            gonzalez_cwt_debug["output"] = denoised_baseline_corrected.copy()
    gonzalez_cwt_debug["wrapper_output"] = denoised_baseline_corrected.copy()
    raw_gonzalez_output = np.asarray(gonzalez_cwt_debug.get("output", denoised_baseline_corrected), dtype=float)
    if raw_gonzalez_output.shape == denoised_baseline_corrected.shape:
        gonzalez_cwt_debug["wrapper_correction_component"] = denoised_baseline_corrected - raw_gonzalez_output
    gonzalez_cwt_debug["wrapper_output_mode"] = denoising_method
    gonzalez_cwt_debug["wrapper_input_normalization"] = _denoising_input_normalization_label(denoising_method)
    side_numeric: Dict[str, np.ndarray] = {
        "corrected_for_result": corrected_for_result,
        "denoise_input": denoise_input,
        "hybrid_denoised": denoised_baseline_corrected,
    }
    if raw_gonzalez_output.shape == denoised_baseline_corrected.shape:
        side_numeric["gonzalez_output"] = raw_gonzalez_output
    working_state["signal"] = denoised_baseline_corrected
    working_state["side_numeric"] = side_numeric
    working_state["side_bool"] = {}
    _apply_pipeline_transform_steps(
        working_state,
        post_denoise_steps,
        request.detection_params,
        filter_traces,
    )
    detection_source = np.asarray(working_state["signal"], dtype=float)
    side_numeric = working_state.get("side_numeric", {})
    denoise_input = np.asarray(side_numeric.get("denoise_input", denoise_input), dtype=float)
    denoised_baseline_corrected = np.asarray(
        side_numeric.get("hybrid_denoised", denoised_baseline_corrected),
        dtype=float,
    )
    corrected_for_result = np.asarray(
        side_numeric.get("corrected_for_result", corrected_for_result),
        dtype=float,
    )
    if "gonzalez_output" in side_numeric:
        gonzalez_output = np.asarray(side_numeric["gonzalez_output"], dtype=float)
        gonzalez_cwt_debug["output"] = gonzalez_output.copy()
        if gonzalez_output.shape == denoised_baseline_corrected.shape:
            gonzalez_cwt_debug["wrapper_correction_component"] = denoised_baseline_corrected - gonzalez_output
    gonzalez_cwt_debug["wrapper_output"] = denoised_baseline_corrected.copy()
    sampling_rate_hz = float(working_state["sampling_rate_hz"])
    artifact_mask = np.asarray(working_state["artifact_mask"], dtype=bool)
    plateau_mask = np.asarray(working_state["plateau_mask"], dtype=bool)
    detection = detect_spikes(
        request.trace_name,
        np.asarray(working_state["time"], dtype=float),
        detection_source,
        analysis_params,
        artifact_mask=artifact_mask,
        baseline_override=np.zeros_like(detection_source),
        plateau_mask_override=plateau_mask,
        sampling_rate_hz=sampling_rate_hz,
        short_plateau_trace_override=denoise_input,
    )
    highpass_trace = filter_traces.get("highpass_filter")
    lowpass_trace = filter_traces.get("lowpass_filter")
    detection.qc_plot_data["highpass_signal"] = (
        np.asarray(highpass_trace, dtype=float) if highpass_trace is not None else np.zeros_like(detection_source)
    )
    detection.qc_plot_data["lowpass_signal"] = (
        np.asarray(lowpass_trace, dtype=float) if lowpass_trace is not None else np.zeros_like(detection_source)
    )
    final_plateau_mask = _detection_mask(detection, "baseline_mask")
    long_plateau_mask = _detection_mask(detection, "long_plateau_mask")
    short_plateau_mask = _detection_mask(detection, "short_plateau_mask", fallback_key="")
    burst_plateau_mask = _detection_mask(detection, "burst_plateau_mask", fallback_key="")
    local_noise = np.asarray(
        detection.qc_plot_data.get("local_noise_estimate", np.zeros_like(denoise_input)),
        dtype=float,
    )
    plateau_signal_classification = classify_plateau_signals(
        denoise_input,
        denoised_baseline_corrected,
        final_plateau_mask,
        sampling_rate_hz=sampling_rate_hz,
        long_plateau_mask=long_plateau_mask,
        short_plateau_mask=short_plateau_mask,
        burst_plateau_mask=burst_plateau_mask,
        local_noise=local_noise,
        denoising_enabled=denoising_method != "none" and denoising_enabled,
    )
    detection.qc_plot_data["plateau_signal_classification_rows"] = list(plateau_signal_classification)
    return TraceAnalysisResult(
        time=np.asarray(working_state["time"], dtype=float),
        raw=np.asarray(working_state["raw"], dtype=float),
        corrected_source=np.asarray(working_state["corrected_source"], dtype=float),
        corrected=corrected_for_result,
        hybrid_denoised=denoised_baseline_corrected,
        gonzalez_cwt_debug=gonzalez_cwt_debug,
        long_plateau_debug={
            str(key): value
            for key, value in baseline_result.qc_plot_data.items()
            if str(key).startswith("long_plateau_")
        },
        plateau_signal_classification=plateau_signal_classification,
        correction_baseline=np.asarray(working_state["baseline"], dtype=float),
        detection=detection,
        artifact_mask=np.asarray(artifact_mask, dtype=bool),
        sampling_rate_hz=sampling_rate_hz,
        highpass_trace=None if highpass_trace is None else np.asarray(highpass_trace, dtype=float),
        lowpass_trace=None if lowpass_trace is None else np.asarray(lowpass_trace, dtype=float),
    )


def build_pending_trace_states(
    time_map: Dict[str, np.ndarray],
    raw_map: Dict[str, np.ndarray],
    corrected_map: Dict[str, np.ndarray],
    sampling_rate_map: Dict[str, float],
    generation: int = 0,
) -> Dict[str, TraceCuration]:
    states: Dict[str, TraceCuration] = {}
    for trace_name, raw in raw_map.items():
        time = np.asarray(time_map[trace_name], dtype=float)
        raw_array = np.asarray(raw, dtype=float)
        corrected_source = np.asarray(corrected_map[trace_name], dtype=float)
        n = min(time.size, raw_array.size, corrected_source.size)
        time = time[:n]
        raw_array = raw_array[:n]
        corrected_source = corrected_source[:n]
        states[trace_name] = TraceCuration(
            trace_name=trace_name,
            raw=raw_array,
            corrected_source=corrected_source,
            corrected=corrected_source.copy(),
            time=time,
            sampling_rate_hz=float(sampling_rate_map.get(trace_name, _sampling_rate_hz(time))),
            original_time=time.copy(),
            original_raw=raw_array.copy(),
            original_corrected_source=corrected_source.copy(),
            original_sampling_rate_hz=float(sampling_rate_map.get(trace_name, _sampling_rate_hz(time))),
            baseline=np.zeros_like(corrected_source, dtype=float),
            correction_baseline=None,
            artifact_mask=None,
            spikes=[],
            smoothed=None,
            debug_threshold=None,
            candidate_windows=[],
            analysis_status="pending",
            analysis_error="",
            analysis_generation=int(generation),
        )
    return states


class _ExcelImportWorkerSignals(QObject):
    finished = Signal(str, object, object, object, object, object)


class _ExcelImportTask(QRunnable):
    def __init__(self, file_path: str, generation: int) -> None:
        super().__init__()
        self.file_path = str(file_path)
        self.generation = int(generation)
        self.signals = _ExcelImportWorkerSignals()

    def run(self) -> None:
        try:
            time_map, raw_map, corrected_map, sampling_rate_map = load_excel_traces(self.file_path)
            self.signals.finished.emit(self.file_path, self.generation, time_map, raw_map, corrected_map, sampling_rate_map)
        except Exception as exc:
            self.signals.finished.emit(self.file_path, self.generation, None, None, None, str(exc))


class _TraceAnalysisWorkerSignals(QObject):
    finished = Signal(str, int, object, object)


class _TraceAnalysisTask(QRunnable):
    def __init__(
        self,
        request: TraceAnalysisRequest,
    ) -> None:
        super().__init__()
        self.request = request
        self.signals = _TraceAnalysisWorkerSignals()

    def run(self) -> None:
        try:
            result = analyze_trace_data(self.request)
            self.signals.finished.emit(self.request.trace_name, self.request.generation, result, None)
        except Exception as exc:
            self.signals.finished.emit(self.request.trace_name, self.request.generation, None, str(exc))


class _DenoiseWorkerSignals(QObject):
    finished = Signal(str, object, object, object)


class _GonzalezDenoiseTask(QRunnable):
    def __init__(
        self,
        trace_name: str,
        cache_key: Tuple[object, ...],
        corrected: np.ndarray,
        fs: float,
        baseline: Optional[np.ndarray],
        denoising_method: str,
        denoise_kwargs: Dict[str, object],
        normalization_baseline: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.trace_name = trace_name
        self.cache_key = cache_key
        self.corrected = np.asarray(corrected, dtype=float).copy()
        self.fs = float(fs)
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=float).copy()
        self.denoising_method = str(denoising_method)
        self.denoise_kwargs = dict(denoise_kwargs)
        self.normalization_baseline = (
            None if normalization_baseline is None else np.asarray(normalization_baseline, dtype=float).copy()
        )
        self.signals = _DenoiseWorkerSignals()

    def run(self) -> None:
        try:
            denoise_input = self.corrected
            if self.baseline is not None and self.baseline.shape == self.corrected.shape:
                denoise_input = self.corrected - self.baseline
            result = np.asarray(
                denoise_trace_gonzalez_full_trace(
                    denoise_input,
                    fs=self.fs,
                    denoising_method=self.denoising_method,
                    normalization_baseline=self.normalization_baseline,
                    **self.denoise_kwargs,
                ),
                dtype=float,
            )
            self.signals.finished.emit(self.trace_name, self.cache_key, result, None)
        except Exception as exc:
            self.signals.finished.emit(self.trace_name, self.cache_key, None, str(exc))


class SpikeCurationMainWindow(QMainWindow):
    def __init__(
        self,
        denoise_fn: Optional[Callable[..., np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"Spike Detector [{DETECTOR_BUILD_TAG}]")
        self.resize(1500, 920)

        self.excel_path: str = ""
        self.detection_params = DetectionParams()
        self.artifact_params = ArtifactParams()
        self._load_detection_params_config()
        self.traces: Dict[str, TraceCuration] = {}
        self.trace_names: List[str] = []
        self.current_trace_name: Optional[str] = None
        self.current_recording_id: Optional[str] = None
        self._table_spikes: List[SpikeRecord] = []
        self._table_signature: Tuple[Tuple[str, str, str, str, str], ...] = ()
        self._denoise_fn = denoise_fn
        self._denoise_thread_pool = QThreadPool(self)
        self._denoise_tasks: List[_GonzalezDenoiseTask] = []
        self._import_thread_pool = QThreadPool(self)
        self._import_thread_pool.setMaxThreadCount(1)
        self._analysis_thread_pool = QThreadPool(self)
        self._analysis_thread_pool.setMaxThreadCount(2)
        self._import_task: Optional[_ExcelImportTask] = None
        self._analysis_tasks: List[_TraceAnalysisTask] = []
        self._analysis_generation = 0
        self._loading_excel = False
        self._pending_session: Optional[AppSession] = None
        self._pending_session_generation: Optional[int] = None
        self.pipeline_config = PipelineConfig.from_preset("current")
        self._seed_pipeline_from_filter_params()
        self._last_packaged_trace_names: set[str] = set()

        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)

        top_bar = QHBoxLayout()
        root_layout.addLayout(top_bar)

        self.btn_load_excel = QPushButton("Load Excel")
        self.btn_load_exported = QPushButton("Load Exported Result")
        self.btn_load_session = QPushButton("Load Session")
        self.btn_save_session = QPushButton("Save Session")
        self.btn_save_analysis_package = QPushButton("Save Analysis Package")
        self.btn_export = QPushButton("Legacy Export All")
        self.btn_export_png_current = QPushButton("Legacy Export Current")
        self.btn_reload_params_cfg = QPushButton("Reload Params Config")
        self.btn_rerun_current = QPushButton("Re-detect Current")
        self.btn_rerun_all = QPushButton("Re-detect All")

        for button in [
            self.btn_load_excel,
            self.btn_load_exported,
            self.btn_load_session,
            self.btn_save_session,
            self.btn_save_analysis_package,
            self.btn_export,
            self.btn_export_png_current,
            self.btn_reload_params_cfg,
            self.btn_rerun_current,
            self.btn_rerun_all,
        ]:
            top_bar.addWidget(button)

        self.status_label = QLabel("No file loaded")
        top_bar.addWidget(self.status_label, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.recording_selector = QComboBox()
        left_layout.addWidget(QLabel("Recording"))
        left_layout.addWidget(self.recording_selector)

        self.trace_list = QListWidget()
        self.trace_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.trace_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.trace_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.trace_list.setAlternatingRowColors(True)
        left_layout.addWidget(QLabel("ROIs / Trace Queue"))
        left_layout.addWidget(self.trace_list, stretch=1)

        nav_layout = QHBoxLayout()
        self.btn_prev_trace = QPushButton("Prev Trace")
        self.btn_next_trace = QPushButton("Next Trace")
        nav_layout.addWidget(self.btn_prev_trace)
        nav_layout.addWidget(self.btn_next_trace)
        left_layout.addLayout(nav_layout)

        run_layout = QHBoxLayout()
        self.btn_run_current = QPushButton("Run Current")
        self.btn_run_selected = QPushButton("Run Selected")
        self.btn_run_all = QPushButton("Run All")
        run_layout.addWidget(self.btn_run_current)
        run_layout.addWidget(self.btn_run_selected)
        run_layout.addWidget(self.btn_run_all)
        left_layout.addLayout(run_layout)

        self.trace_queue_status = QLabel("0 traces")
        left_layout.addWidget(self.trace_queue_status)
        left_layout.addStretch(1)

        self.param_detection_mode = QComboBox()
        self.param_detection_mode.addItem("STD threshold", "std")
        self.param_detection_mode.addItem("MAD threshold", "mad")
        self.param_detection_mode.addItem("SNR threshold", "snr")
        self.param_detection_mode.addItem("Template matching", "template_matching")
        self.param_detection_mode.setToolTip(
            "Choose the spike detector. STD threshold is the default; Template matching uses user-selected templates."
        )
        self.param_baseline_mode = QComboBox()
        self.param_baseline_mode.addItem("Percentile", "percentile")
        self.param_baseline_mode.addItem("Savitzky-Golay", "savgol")
        self.param_startup_exclusion_ms = self._make_double_spin(0.0, 1000.0, self.detection_params.startup_exclusion_ms, 1.0)
        self.param_baseline_rolling_window_ms = self._make_double_spin(5.0, 5000.0, self.detection_params.baseline_rolling_window_ms, 5.0)
        self.param_baseline_rolling_percentile = self._make_double_spin(1.0, 50.0, self.detection_params.baseline_rolling_percentile, 1.0)
        self.param_baseline_savgol_window_ms = self._make_double_spin(5.0, 5000.0, self.detection_params.baseline_savgol_window_ms, 5.0)
        self.param_baseline_savgol_polyorder = self._make_int_spin(1, 7, self.detection_params.baseline_savgol_polyorder)
        self.param_highpass_cutoff_hz = self._make_double_spin(0.001, 1000.0, self.detection_params.highpass_cutoff_hz, 0.1)
        self.param_highpass_order = self._make_int_spin(1, 10, self.detection_params.highpass_order)
        self.param_lowpass_cutoff_hz = self._make_double_spin(0.001, 1000.0, self.detection_params.lowpass_cutoff_hz, 0.1)
        self.param_lowpass_order = self._make_int_spin(1, 10, self.detection_params.lowpass_order)
        self.param_rolling_downsample_factor = self._make_int_spin(
            2,
            100,
            int(self.detection_params.rolling_downsample_factor),
        )
        self.param_plateau_median_window_ms = self._make_double_spin(5.0, 500.0, self.detection_params.plateau_median_window_ms, 1.0)
        self.param_long_plateau_drift_window_ms = self._make_double_spin(50.0, 60000.0, self.detection_params.long_plateau_drift_window_ms, 50.0)
        self.param_long_plateau_drift_percentile = self._make_double_spin(1.0, 50.0, self.detection_params.long_plateau_drift_percentile, 1.0)
        self.param_long_plateau_entry_noise_k = self._make_double_spin(0.0, 50.0, self.detection_params.long_plateau_entry_noise_k, 0.1)
        self.param_long_plateau_exit_noise_k = self._make_double_spin(0.0, 50.0, self.detection_params.long_plateau_exit_noise_k, 0.1)
        self.param_long_plateau_exit_fraction = self._make_double_spin(0.0, 1.0, self.detection_params.long_plateau_exit_fraction, 0.05)
        self.param_long_plateau_enter_sustain_ms = self._make_double_spin(1.0, 5000.0, self.detection_params.long_plateau_enter_sustain_ms, 1.0)
        self.param_long_plateau_exit_sustain_ms = self._make_double_spin(1.0, 5000.0, self.detection_params.long_plateau_exit_sustain_ms, 1.0)
        self.param_short_plateau_step_noise_k = self._make_double_spin(0.0, 20.0, self.detection_params.short_plateau_step_noise_k, 0.1)
        self.param_short_plateau_elevated_noise_k = self._make_double_spin(0.0, 20.0, self.detection_params.short_plateau_elevated_noise_k, 0.1)
        self.param_short_plateau_survival_noise_k = self._make_double_spin(0.0, 20.0, self.detection_params.short_plateau_survival_noise_k, 0.1)
        self.param_short_plateau_loss_noise_k = self._make_double_spin(0.0, 20.0, self.detection_params.short_plateau_loss_noise_k, 0.1)
        self.param_short_plateau_median_noise_k = self._make_double_spin(0.0, 20.0, self.detection_params.short_plateau_median_noise_k, 0.1)
        self.param_short_plateau_pre_quiet_noise_k = self._make_double_spin(0.0, 20.0, self.detection_params.short_plateau_pre_quiet_noise_k, 0.1)
        self.param_short_plateau_min_support_fraction = self._make_double_spin(0.0, 1.0, self.detection_params.short_plateau_min_support_fraction, 0.05)
        self.param_short_plateau_min_confidence = self._make_double_spin(0.0, 1.0, self.detection_params.short_plateau_min_confidence, 0.05)
        self.param_short_plateau_early_dwell_ms = self._make_double_spin(1.0, 500.0, self.detection_params.short_plateau_early_dwell_ms, 1.0)
        self.param_short_plateau_min_support_ms = self._make_double_spin(1.0, 500.0, self.detection_params.short_plateau_min_support_ms, 1.0)
        self.param_short_plateau_min_duration_ms = self._make_double_spin(1.0, 1000.0, self.detection_params.short_plateau_min_duration_ms, 1.0)
        self.param_spike_noise_k = self._make_double_spin(0.1, 100.0, self.detection_params.spike_noise_k, 0.1)
        self.param_spike_noise_k.setToolTip(
            "Noise multiplier for MAD/STD/SNR threshold detection. Default is 7x for conservative STD detection."
        )
        self.param_spike_prominence_k = self._make_double_spin(0.0, 100.0, self.detection_params.spike_prominence_k, 0.1)
        self.param_spike_prominence_k.setToolTip(
            "Prominence multiplier used with threshold detectors to reject small local bumps."
        )
        self.param_template_matching_enable_autopick = QCheckBox("Enable safe auto-pick")
        self.param_template_matching_enable_autopick.setChecked(
            bool(self.detection_params.template_matching_enable_autopick)
        )
        self.param_template_matching_enable_autopick.setToolTip(
            "Allow template matching to seed templates from STD detection at 12x noise. "
            "When off, only user-provided or accepted/manual templates are used."
        )
        self.param_template_matching_template_indices = QLineEdit()
        self.param_template_matching_template_indices.setPlaceholderText("e.g. 253, 523, 843")
        self.param_template_matching_template_indices.setToolTip(
            "Comma-separated peak sample indices chosen by the user. "
            "If blank, accepted/manual spikes already saved on this trace are used when re-running."
        )
        self.param_template_matching_pre_ms = self._make_double_spin(0.1, 100.0, self.detection_params.template_matching_pre_ms, 0.5)
        self.param_template_matching_post_ms = self._make_double_spin(0.1, 100.0, self.detection_params.template_matching_post_ms, 0.5)
        self.param_template_matching_filter_cutoff_hz = self._make_double_spin(0.001, 1000.0, self.detection_params.template_matching_filter_cutoff_hz, 0.5)
        self.param_template_matching_false_positive_rate = self._make_double_spin(0.000001, 0.5, self.detection_params.template_matching_false_positive_rate, 0.005)
        self.param_template_matching_pre_ms.setToolTip("Milliseconds before each selected peak included in the template window.")
        self.param_template_matching_post_ms.setToolTip("Milliseconds after each selected peak included in the template window.")
        self.param_template_matching_filter_cutoff_hz.setToolTip(
            "Second-order high-pass cutoff used before Evans-style template matching. Default is 20 Hz."
        )
        self.param_template_matching_false_positive_rate.setToolTip(
            "Total desired false-positive rate; internally divided across template-length pieces."
        )
        self.param_min_separation_ms = self._make_double_spin(1.0, 1000.0, self.detection_params.min_separation_ms, 1.0)
        self.param_spike_zoom_half_window_ms = self._make_double_spin(0.5, 500.0, self.detection_params.spike_zoom_half_window_ms, 0.5)
        self.param_min_separation_ms.setToolTip("Minimum separation between accepted spike candidates.")
        self.param_spike_zoom_half_window_ms.setToolTip("Half-width used for spike zoom plots and FWHM measurement.")

        self.param_artifact_abs_low = self._make_double_spin(0.0, 100000.0, self.artifact_params.absolute_low, 1.0)
        self.param_artifact_min_depth = self._make_double_spin(0.0, 100000.0, self.artifact_params.min_depth, 0.5)
        self.param_artifact_drop_start = self._make_double_spin(0.0, 100000.0, self.artifact_params.drop_start_abs, 0.5)

        self.chk_show_raw = QCheckBox("Show raw")
        self.chk_show_corrected = QCheckBox("Show corrected")
        self.chk_show_baseline = QCheckBox("Show baseline")
        self.chk_show_highpass = QCheckBox("Show high-pass")
        self.chk_show_lowpass = QCheckBox("Show low-pass")
        self.chk_show_hybrid_denoised = QCheckBox("Show Gonzalez denoised")
        self.chk_show_normalized_denoised = QCheckBox("Show denoised % dF/F0")
        self.chk_show_denoise_residual = QCheckBox("Show denoise residual")
        self.display_units_combo = QComboBox()
        self.display_units_combo.addItem("Native", NORMALIZATION_NATIVE)
        self.display_render_source_combo = QComboBox()
        for label, value in DISPLAY_RENDER_SOURCE_ITEMS:
            self.display_render_source_combo.addItem(label, value)
        self.display_units_combo.addItem("% ΔF/F0", NORMALIZATION_PERCENT_DFF)
        self.display_units_combo.setToolTip("Choose whether the trace plot uses native units or percent dF/F0.")
        self.display_render_source_combo.setToolTip("Choose which signal is drawn by the selected overlay line.")
        self.chk_show_hybrid_denoised.setToolTip("Show or hide the signal chosen in the selected overlay menu.")
        self.chk_show_normalized_denoised.setToolTip("Show the denoised trace as a percent dF/F0 overlay.")
        self.chk_show_denoise_residual.setToolTip("Show corrected minus denoised signal as a residual overlay.")

        self.chk_show_raw.setChecked(True)
        self.chk_show_corrected.setChecked(True)
        self.chk_show_baseline.setChecked(True)
        self.chk_show_highpass.setChecked(False)
        self.chk_show_lowpass.setChecked(False)
        self.chk_show_hybrid_denoised.setChecked(True)
        self.chk_show_normalized_denoised.setChecked(True)
        self.chk_show_denoise_residual.setChecked(False)

        self.overlay_controls_group = QGroupBox("Signal Overlays")
        overlay_layout = QFormLayout(self.overlay_controls_group)
        overlay_layout.addRow("Units", self.display_units_combo)
        overlay_layout.addRow("Overlay", self.display_render_source_combo)
        overlay_layout.addRow(self.chk_show_raw)
        overlay_layout.addRow(self.chk_show_corrected)
        overlay_layout.addRow(self.chk_show_baseline)
        overlay_layout.addRow(self.chk_show_highpass)
        overlay_layout.addRow(self.chk_show_lowpass)
        overlay_layout.addRow(self.chk_show_hybrid_denoised)
        overlay_layout.addRow(self.chk_show_normalized_denoised)
        overlay_layout.addRow(self.chk_show_denoise_residual)
        left_layout.addWidget(self.overlay_controls_group)

        self.pipeline_display_units_combo = QComboBox()
        self.pipeline_display_units_combo.addItem("Native", NORMALIZATION_NATIVE)
        self.pipeline_display_units_combo.addItem("% ΔF/F0", NORMALIZATION_PERCENT_DFF)
        self.pipeline_display_render_source_combo = QComboBox()
        for label, value in DISPLAY_RENDER_SOURCE_ITEMS:
            self.pipeline_display_render_source_combo.addItem(label, value)
        self.pipeline_chk_show_raw = QCheckBox("Show raw")
        self.pipeline_chk_show_corrected = QCheckBox("Show corrected")
        self.pipeline_chk_show_baseline = QCheckBox("Show baseline")
        self.pipeline_chk_show_highpass = QCheckBox("Show high-pass")
        self.pipeline_chk_show_lowpass = QCheckBox("Show low-pass")
        self.pipeline_chk_show_hybrid_denoised = QCheckBox("Show selected overlay")
        self.pipeline_chk_show_normalized_denoised = QCheckBox("Show denoised % dF/F0")
        self.pipeline_chk_show_denoise_residual = QCheckBox("Show denoise residual")
        self.pipeline_display_units_combo.setToolTip(self.display_units_combo.toolTip())
        self.pipeline_display_render_source_combo.setToolTip(self.display_render_source_combo.toolTip())
        self.pipeline_chk_show_hybrid_denoised.setToolTip(self.chk_show_hybrid_denoised.toolTip())
        self.pipeline_chk_show_normalized_denoised.setToolTip(self.chk_show_normalized_denoised.toolTip())
        self.pipeline_chk_show_denoise_residual.setToolTip(self.chk_show_denoise_residual.toolTip())
        self._sync_pipeline_overlay_controls_from_main()

        self.param_denoising_method = QComboBox()
        self.param_denoising_method.addItem("On (corrected trace)", "gonzalez_full_trace")
        self.param_denoising_method.addItem("Guarded (corrected trace)", "gonzalez_full_trace_guarded")
        self.param_denoising_method.addItem("On (dF/F0 input)", "gonzalez_full_trace_dff")
        self.param_denoising_method.addItem("S7 reproduction (dF/F0)", "gonzalez_s7_dff")
        self.param_denoising_method.addItem("Off", "none")
        self.param_denoising_method.setToolTip(
            "Enable denoising and choose the input units for the single Gonzalez denoiser."
        )
        denoising_method_index = self.param_denoising_method.findData(str(self.detection_params.denoising_method))
        self.param_denoising_method.setCurrentIndex(max(0, denoising_method_index))
        self.param_denoising_input_normalization = QLabel("")
        self.param_denoising_input_normalization.setWordWrap(True)
        self.param_denoising_input_normalization.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.param_denoising_input_normalization.setToolTip("Shows the input used by the Gonzalez denoiser.")
        self.param_gonzalez_threshold_sd = self._make_double_spin(
            0.0,
            10.0,
            self.detection_params.gonzalez_threshold_sd,
            0.1,
        )
        self.param_gonzalez_threshold_sd.setToolTip(
            "Higher values denoise more aggressively by requiring stronger wavelet activity to remain fully preserved."
        )
        self.param_gonzalez_max_clusters = self._make_int_spin(
            1,
            100,
            int(self.detection_params.gonzalez_max_clusters),
        )
        self.param_gonzalez_max_clusters.setToolTip(
            "Number of wavelet-frequency groups used before thresholding; higher values separate frequency bands more finely."
        )
        self.param_gonzalez_attenuation_min = self._make_double_spin(
            0.0,
            1.0,
            self.detection_params.gonzalez_attenuation_min,
            0.05,
        )
        self.param_gonzalez_attenuation_min.setToolTip(
            "Minimum suppression applied to samples below the wavelet threshold."
        )
        self.param_gonzalez_attenuation_max = self._make_double_spin(
            0.0,
            1.0,
            self.detection_params.gonzalez_attenuation_max,
            0.05,
        )
        self.param_gonzalez_attenuation_max.setToolTip(
            "Maximum suppression applied to below-threshold wavelet coefficients; lower values preserve more signal."
        )
        self.param_gonzalez_noise_cluster_rejection = QCheckBox("Noise-cluster rejection")
        self.param_gonzalez_noise_cluster_rejection.setChecked(
            bool(self.detection_params.gonzalez_enable_noise_cluster_rejection)
        )
        self.param_gonzalez_noise_cluster_rejection.setToolTip(
            "Optionally discard one high-frequency, low-event cluster. Conservative but can remove real high-frequency spike content."
        )
        self.param_gonzalez_event_template_rejection = QCheckBox("Event-template cleanup")
        self.param_gonzalez_event_template_rejection.setChecked(
            bool(self.detection_params.gonzalez_enable_event_template_rejection)
        )
        self.param_gonzalez_event_template_rejection.setToolTip(
            "Optionally attenuate events whose wavelet-cluster shape differs from the cluster template. Experimental and spike-risky."
        )
        artifact_group = QGroupBox("Artifact Cleanup")
        artifact_layout = QFormLayout(artifact_group)
        artifact_layout.addRow("absolute_low", self.param_artifact_abs_low)
        artifact_layout.addRow("min_depth", self.param_artifact_min_depth)
        artifact_layout.addRow("drop_start_abs", self.param_artifact_drop_start)

        baseline_group = QGroupBox("Baseline Correction")
        baseline_layout = QFormLayout(baseline_group)
        baseline_layout.addRow("baseline_mode", self.param_baseline_mode)
        baseline_layout.addRow("baseline_rolling_window_ms", self.param_baseline_rolling_window_ms)
        baseline_layout.addRow("baseline_rolling_percentile", self.param_baseline_rolling_percentile)
        baseline_layout.addRow("baseline_savgol_window_ms", self.param_baseline_savgol_window_ms)
        baseline_layout.addRow("baseline_savgol_polyorder", self.param_baseline_savgol_polyorder)

        highpass_group = QGroupBox("High-Pass Filter")
        highpass_layout = QFormLayout(highpass_group)
        highpass_layout.addRow("highpass_cutoff_hz", self.param_highpass_cutoff_hz)
        highpass_layout.addRow("highpass_order", self.param_highpass_order)

        lowpass_group = QGroupBox("Low-Pass Filter")
        lowpass_layout = QFormLayout(lowpass_group)
        lowpass_layout.addRow("lowpass_cutoff_hz", self.param_lowpass_cutoff_hz)
        lowpass_layout.addRow("lowpass_order", self.param_lowpass_order)

        rolling_downsample_group = QGroupBox("Rolling-Average Downsampling")
        rolling_downsample_layout = QFormLayout(rolling_downsample_group)
        rolling_downsample_layout.addRow("rolling_downsample_factor", self.param_rolling_downsample_factor)

        plateau_group = QGroupBox("Plateau Detection")
        plateau_layout = QVBoxLayout(plateau_group)
        plateau_shared_group = QGroupBox("Shared Setup")
        plateau_shared_group.setFlat(True)
        plateau_shared_layout = QFormLayout(plateau_shared_group)
        plateau_shared_layout.addRow("startup_exclusion_ms", self.param_startup_exclusion_ms)
        plateau_shared_layout.addRow("plateau_median_window_ms", self.param_plateau_median_window_ms)

        plateau_long_group = QGroupBox("Long Baseline-Protection Plateaus")
        plateau_long_group.setFlat(True)
        plateau_long_layout = QFormLayout(plateau_long_group)
        plateau_long_layout.addRow("drift_window_ms", self.param_long_plateau_drift_window_ms)
        plateau_long_layout.addRow("drift_percentile", self.param_long_plateau_drift_percentile)
        plateau_long_layout.addRow("entry_noise_k", self.param_long_plateau_entry_noise_k)
        plateau_long_layout.addRow("exit_noise_k", self.param_long_plateau_exit_noise_k)
        plateau_long_layout.addRow("exit_fraction", self.param_long_plateau_exit_fraction)
        plateau_long_layout.addRow("enter_sustain_ms", self.param_long_plateau_enter_sustain_ms)
        plateau_long_layout.addRow("exit_sustain_ms", self.param_long_plateau_exit_sustain_ms)

        plateau_short_burst_group = QGroupBox("Short / Burst Raised-Floor Plateaus")
        plateau_short_burst_group.setFlat(True)
        plateau_short_burst_layout = QFormLayout(plateau_short_burst_group)
        plateau_short_burst_layout.addRow("step_noise_k", self.param_short_plateau_step_noise_k)
        plateau_short_burst_layout.addRow("elevated_noise_k", self.param_short_plateau_elevated_noise_k)
        plateau_short_burst_layout.addRow("survival_noise_k", self.param_short_plateau_survival_noise_k)
        plateau_short_burst_layout.addRow("loss_noise_k", self.param_short_plateau_loss_noise_k)
        plateau_short_burst_layout.addRow("median_noise_k", self.param_short_plateau_median_noise_k)
        plateau_short_burst_layout.addRow("pre_quiet_noise_k", self.param_short_plateau_pre_quiet_noise_k)
        plateau_short_burst_layout.addRow("min_support_fraction", self.param_short_plateau_min_support_fraction)
        plateau_short_burst_layout.addRow("min_confidence", self.param_short_plateau_min_confidence)
        plateau_short_burst_layout.addRow("early_dwell_ms", self.param_short_plateau_early_dwell_ms)
        plateau_short_burst_layout.addRow("min_support_ms", self.param_short_plateau_min_support_ms)
        plateau_short_burst_layout.addRow("min_duration_ms", self.param_short_plateau_min_duration_ms)
        plateau_layout.addWidget(plateau_shared_group)
        plateau_layout.addWidget(plateau_long_group)
        plateau_layout.addWidget(plateau_short_burst_group)

        denoise_group = QGroupBox("Denoising")
        denoise_layout = QFormLayout(denoise_group)
        denoise_layout.addRow("Denoising", self.param_denoising_method)
        denoise_layout.addRow("Denoiser input", self.param_denoising_input_normalization)
        denoise_layout.addRow("Noise threshold SD", self.param_gonzalez_threshold_sd)
        denoise_layout.addRow("Cluster count", self.param_gonzalez_max_clusters)
        denoise_layout.addRow("Min suppression", self.param_gonzalez_attenuation_min)
        denoise_layout.addRow("Max suppression", self.param_gonzalez_attenuation_max)
        denoise_layout.addRow(self.param_gonzalez_noise_cluster_rejection)
        denoise_layout.addRow(self.param_gonzalez_event_template_rejection)

        spike_group = QGroupBox("Spike Detection")
        spike_layout = QVBoxLayout(spike_group)

        spike_shared_group = QGroupBox("Shared Detection Setup")
        spike_shared_group.setFlat(True)
        spike_shared_layout = QFormLayout(spike_shared_group)
        spike_shared_layout.addRow("detection_mode", self.param_detection_mode)
        spike_shared_layout.addRow("min_separation_ms", self.param_min_separation_ms)
        spike_shared_layout.addRow("spike_zoom_half_window_ms", self.param_spike_zoom_half_window_ms)

        spike_threshold_group = QGroupBox("Threshold Methods")
        spike_threshold_group.setFlat(True)
        spike_threshold_layout = QFormLayout(spike_threshold_group)
        spike_threshold_layout.addRow("spike_noise_k", self.param_spike_noise_k)
        spike_threshold_layout.addRow("spike_prominence_k", self.param_spike_prominence_k)

        spike_template_group = QGroupBox("Template Matching")
        spike_template_group.setFlat(True)
        spike_template_layout = QFormLayout(spike_template_group)
        spike_template_layout.addRow("template_matching_template_indices", self.param_template_matching_template_indices)
        spike_template_layout.addRow("template_matching_enable_autopick", self.param_template_matching_enable_autopick)
        spike_template_layout.addRow("template_matching_pre_ms", self.param_template_matching_pre_ms)
        spike_template_layout.addRow("template_matching_post_ms", self.param_template_matching_post_ms)
        spike_template_layout.addRow("template_matching_filter_cutoff_hz", self.param_template_matching_filter_cutoff_hz)
        spike_template_layout.addRow("template_matching_false_positive_rate", self.param_template_matching_false_positive_rate)

        spike_layout.addWidget(spike_shared_group)
        spike_layout.addWidget(spike_threshold_group)
        spike_layout.addWidget(spike_template_group)

        display_group = QGroupBox("Signal Display / Overlay")
        display_layout = QFormLayout(display_group)
        display_layout.addRow("Units", self.pipeline_display_units_combo)
        display_layout.addRow("Selected overlay", self.pipeline_display_render_source_combo)
        display_layout.addRow(self.pipeline_chk_show_raw)
        display_layout.addRow(self.pipeline_chk_show_corrected)
        display_layout.addRow(self.pipeline_chk_show_baseline)
        display_layout.addRow(self.pipeline_chk_show_highpass)
        display_layout.addRow(self.pipeline_chk_show_lowpass)
        display_layout.addRow(self.pipeline_chk_show_hybrid_denoised)
        display_layout.addRow(self.pipeline_chk_show_normalized_denoised)
        display_layout.addRow(self.pipeline_chk_show_denoise_residual)

        review_group = QGroupBox("Review / Save Package")
        review_layout = QVBoxLayout(review_group)
        self.btn_review_save_package = QPushButton("Save Analysis Package")
        self.btn_review_rerun_current = QPushButton("Re-detect Current")
        self.pipeline_review_status = QLabel("Current Algorithm")
        review_layout.addWidget(self.btn_review_save_package)
        review_layout.addWidget(self.btn_review_rerun_current)
        review_layout.addWidget(self.pipeline_review_status)
        review_layout.addStretch(1)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        self.plot = TracePlotWidget()
        center_layout.addWidget(self.plot, stretch=1)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["idx", "time", "amp", "snr", "prom"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { gridline-color: #666666; }"
            "QTableWidget::item { color: #111111; }"
            "QHeaderView::section { background-color: #30343a; color: #f5f7fa; padding: 4px; border: 1px solid #4b4f56; }"
        )
        self.table.hide()

        workflow_panel = QWidget()
        workflow_layout = QVBoxLayout(workflow_panel)

        preset_group = QGroupBox("Pipeline Builder")
        preset_layout = QVBoxLayout(preset_group)
        self.pipeline_preset_combo = QComboBox()
        for preset_id, preset in PIPELINE_PRESETS.items():
            self.pipeline_preset_combo.addItem(str(preset["label"]), preset_id)
        preset_layout.addWidget(self.pipeline_preset_combo)
        self.pipeline_warning_label = QLabel(self.pipeline_config.warning())
        self.pipeline_warning_label.setWordWrap(True)
        preset_layout.addWidget(self.pipeline_warning_label)
        self.pipeline_order_summary_label = QLabel(PIPELINE_ORDER_SUMMARY)
        self.pipeline_order_summary_label.setWordWrap(True)
        self.pipeline_order_summary_label.setStyleSheet("color: #4f5965;")
        preset_layout.addWidget(self.pipeline_order_summary_label)
        self.pipeline_steps = QListWidget()
        self.pipeline_steps.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.pipeline_steps.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.pipeline_steps.setAlternatingRowColors(True)
        preset_layout.addWidget(self.pipeline_steps, stretch=1)
        add_step_layout = QHBoxLayout()
        self.pipeline_add_step_combo = QComboBox()
        self.btn_pipeline_add_step = QPushButton("Add Step")
        add_step_layout.addWidget(self.pipeline_add_step_combo, stretch=1)
        add_step_layout.addWidget(self.btn_pipeline_add_step)
        preset_layout.addLayout(add_step_layout)
        step_button_layout = QHBoxLayout()
        self.btn_pipeline_step_up = QPushButton("Move Up")
        self.btn_pipeline_step_down = QPushButton("Move Down")
        self.btn_pipeline_remove_step = QPushButton("Remove")
        step_button_layout.addWidget(self.btn_pipeline_step_up)
        step_button_layout.addWidget(self.btn_pipeline_step_down)
        step_button_layout.addWidget(self.btn_pipeline_remove_step)
        preset_layout.addLayout(step_button_layout)
        workflow_layout.addWidget(preset_group, stretch=1)

        self.pipeline_params_stack = QtWidgets.QStackedWidget()
        self._pipeline_step_page_indexes: Dict[str, int] = {}
        for step_id, group in [
            ("artifact_cleanup", artifact_group),
            ("baseline_correction", baseline_group),
            ("highpass_filter", highpass_group),
            ("lowpass_filter", lowpass_group),
            ("rolling_average_downsample", rolling_downsample_group),
            ("plateau_detection", plateau_group),
            ("denoising", denoise_group),
            ("spike_detection", spike_group),
            ("review_save", review_group),
            ("signal_display", display_group),
        ]:
            self._pipeline_step_page_indexes[step_id] = self.pipeline_params_stack.addWidget(group)
        workflow_layout.addWidget(self.pipeline_params_stack, stretch=2)
        self._refresh_pipeline_steps()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)

        workflow_scroll = QScrollArea()
        workflow_scroll.setWidgetResizable(True)
        workflow_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        workflow_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        workflow_scroll.setWidget(workflow_panel)

        splitter.addWidget(left_scroll)
        splitter.addWidget(center_panel)
        splitter.addWidget(workflow_scroll)
        splitter.setSizes([260, 920, 360])

        self._connect_signals()
        self._init_shortcuts()

    def _make_double_spin(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(step)
        widget.setDecimals(3)
        return widget

    def _make_int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _parse_index_list(text: str) -> List[int]:
        indices: List[int] = []
        for piece in str(text or "").replace(";", ",").split(","):
            item = piece.strip()
            if not item:
                continue
            try:
                indices.append(int(round(float(item))))
            except ValueError:
                continue
        return indices

    @staticmethod
    def _format_index_list(indices: List[int]) -> str:
        return ", ".join(str(int(index)) for index in indices)

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _insert_pipeline_step_once(self, step_id: str, before_step_id: str = "spike_detection") -> int:
        step_id = str(step_id)
        if step_id not in PIPELINE_STEP_DEFINITIONS:
            return -1
        if step_id in self.pipeline_config.ordered_step_ids:
            self.pipeline_config.enabled_steps[step_id] = True
            return self.pipeline_config.ordered_step_ids.index(step_id)
        try:
            insert_at = self.pipeline_config.ordered_step_ids.index(before_step_id)
        except ValueError:
            insert_at = len(self.pipeline_config.ordered_step_ids)
        self.pipeline_config.ordered_step_ids.insert(insert_at, step_id)
        self.pipeline_config.enabled_steps[step_id] = True
        return insert_at

    def _seed_pipeline_from_filter_params(self) -> None:
        filter_steps: List[str] = []
        if bool(getattr(self.detection_params, "use_highpass_filter", False)):
            filter_steps.append("highpass_filter")
        if bool(getattr(self.detection_params, "use_lowpass_filter", False)):
            filter_steps.append("lowpass_filter")
        if not filter_steps:
            return
        self.pipeline_config = PipelineConfig.from_preset("custom")
        for step_id in filter_steps:
            self._insert_pipeline_step_once(step_id)

    def _refresh_pipeline_steps(self) -> None:
        step_ids = []
        for step_id in self.pipeline_config.ordered_step_ids:
            if step_id in PIPELINE_STEP_DEFINITIONS and step_id not in step_ids:
                step_ids.append(step_id)
        self.pipeline_config.ordered_step_ids = step_ids

        locked = self.pipeline_config.is_locked()
        self.pipeline_steps.blockSignals(True)
        self.pipeline_steps.clear()
        for row, step_id in enumerate(step_ids, start=1):
            definition = PIPELINE_STEP_DEFINITIONS[step_id]
            order_policy = str(definition.get("order_policy", ""))
            order_label = PIPELINE_ORDER_POLICY_LABELS.get(order_policy, order_policy)
            suffix = f" [{order_label}]" if order_label else ""
            item = QListWidgetItem(f"{row}. {definition['label']}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, step_id)
            tooltip_parts = [
                str(definition.get("description", "")),
                str(definition.get("order_note", "")),
            ]
            item.setToolTip("\n".join(part for part in tooltip_parts if part))
            item.setCheckState(
                Qt.CheckState.Checked
                if self.pipeline_config.enabled_steps.get(step_id, True)
                else Qt.CheckState.Unchecked
            )
            flags = item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            if locked:
                flags &= ~Qt.ItemFlag.ItemIsUserCheckable
                flags &= ~Qt.ItemFlag.ItemIsDragEnabled
                flags &= ~Qt.ItemFlag.ItemIsDropEnabled
            else:
                flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
            item.setFlags(flags)
            self.pipeline_steps.addItem(item)
        self.pipeline_steps.blockSignals(False)

        self.pipeline_steps.setDragDropMode(
            QAbstractItemView.DragDropMode.NoDragDrop
            if locked
            else QAbstractItemView.DragDropMode.InternalMove
        )
        self.btn_pipeline_step_up.setEnabled(not locked)
        self.btn_pipeline_step_down.setEnabled(not locked)
        self.btn_pipeline_remove_step.setEnabled(
            not locked and self.pipeline_steps.currentRow() >= 0 and self.pipeline_steps.count() > 0
        )
        self._refresh_pipeline_add_controls()
        self.pipeline_warning_label.setText(self.pipeline_config.warning())
        self.pipeline_review_status.setText(
            str(PIPELINE_PRESETS.get(self.pipeline_config.active_preset, {}).get("label", "Custom Pipeline"))
        )
        if self.pipeline_steps.currentRow() < 0 and self.pipeline_steps.count() > 0:
            self.pipeline_steps.setCurrentRow(0)
        else:
            self._on_pipeline_step_selected(self.pipeline_steps.currentRow())

    def _sync_pipeline_config_from_widgets(self) -> None:
        if not hasattr(self, "pipeline_steps"):
            return
        ordered_step_ids: List[str] = []
        enabled_steps: Dict[str, bool] = {}
        for row in range(self.pipeline_steps.count()):
            item = self.pipeline_steps.item(row)
            if item is None:
                continue
            step_id = str(item.data(Qt.ItemDataRole.UserRole))
            ordered_step_ids.append(step_id)
            enabled_steps[step_id] = item.checkState() == Qt.CheckState.Checked
        self.pipeline_config.ordered_step_ids = ordered_step_ids
        self.pipeline_config.enabled_steps.update(enabled_steps)
        for step_id in ADDABLE_PIPELINE_STEP_IDS:
            if step_id not in ordered_step_ids:
                self.pipeline_config.enabled_steps[step_id] = False
        self.pipeline_config.trace_order = list(self.trace_names)

    def _refresh_pipeline_add_controls(self) -> None:
        if not hasattr(self, "pipeline_add_step_combo"):
            return
        existing = set(self.pipeline_config.ordered_step_ids)
        self.pipeline_add_step_combo.blockSignals(True)
        self.pipeline_add_step_combo.clear()
        for step_id in ADDABLE_PIPELINE_STEP_IDS:
            if step_id in existing:
                continue
            definition = PIPELINE_STEP_DEFINITIONS.get(step_id, {})
            self.pipeline_add_step_combo.addItem(str(definition.get("label", step_id)), step_id)
        self.pipeline_add_step_combo.blockSignals(False)
        can_add = self.pipeline_add_step_combo.count() > 0
        self.pipeline_add_step_combo.setEnabled(can_add)
        self.btn_pipeline_add_step.setEnabled(can_add)

    def _pipeline_config_dict(self) -> Dict[str, Any]:
        self._sync_pipeline_config_from_widgets()
        return self.pipeline_config.to_dict()

    def _set_pipeline_combo_to_preset(self, preset_id: str) -> None:
        index = self.pipeline_preset_combo.findData(str(preset_id))
        if index < 0:
            return
        self.pipeline_preset_combo.blockSignals(True)
        self.pipeline_preset_combo.setCurrentIndex(index)
        self.pipeline_preset_combo.blockSignals(False)

    def _mark_pipeline_custom(self) -> None:
        if self.pipeline_config.active_preset == "custom":
            return
        self.pipeline_config.active_preset = "custom"
        self._set_pipeline_combo_to_preset("custom")
        self.pipeline_warning_label.setText(self.pipeline_config.warning())
        self.pipeline_review_status.setText("Custom Editable Pipeline")

    def _remember_current_denoiser_for_restore(self) -> None:
        current = str(self.param_denoising_method.currentData() or "gonzalez_full_trace")
        if current != "none":
            self._last_non_none_denoising_method = current

    def _apply_no_denoising_if_needed(self) -> None:
        denoising_enabled = self.pipeline_config.enabled_steps.get("denoising", True)
        if denoising_enabled:
            current = str(self.param_denoising_method.currentData() or "")
            if current == "none":
                restore = str(getattr(self, "_last_non_none_denoising_method", "gonzalez_full_trace"))
                self._set_combo_data(self.param_denoising_method, restore)
            return
        self._remember_current_denoiser_for_restore()
        self._set_combo_data(self.param_denoising_method, "none")

    def _on_pipeline_preset_changed(self, index: int) -> None:
        preset_id = str(self.pipeline_preset_combo.itemData(index) or "current")
        self.pipeline_config = PipelineConfig.from_preset(preset_id)
        self.pipeline_config.trace_order = list(self.trace_names)
        self._apply_no_denoising_if_needed()
        self._refresh_pipeline_steps()

    def _on_pipeline_step_selected(self, row: int) -> None:
        if row < 0 or row >= self.pipeline_steps.count():
            self.btn_pipeline_remove_step.setEnabled(False)
            return
        item = self.pipeline_steps.item(row)
        if item is None:
            self.btn_pipeline_remove_step.setEnabled(False)
            return
        step_id = str(item.data(Qt.ItemDataRole.UserRole))
        self.btn_pipeline_remove_step.setEnabled(
            not self.pipeline_config.is_locked() and step_id in ADDABLE_PIPELINE_STEP_IDS
        )
        index = self._pipeline_step_page_indexes.get(step_id)
        if index is not None:
            self.pipeline_params_stack.setCurrentIndex(index)

    def _on_pipeline_item_changed(self, item: QListWidgetItem) -> None:
        if self.pipeline_config.is_locked():
            return
        step_id = str(item.data(Qt.ItemDataRole.UserRole))
        self._mark_pipeline_custom()
        self._sync_pipeline_config_from_widgets()
        if step_id == "denoising":
            self._apply_no_denoising_if_needed()

    def _on_pipeline_steps_reordered(self, *_args: object) -> None:
        if self.pipeline_config.is_locked():
            return
        self._mark_pipeline_custom()
        self._sync_pipeline_config_from_widgets()
        self._refresh_pipeline_steps()

    def _move_pipeline_step(self, offset: int) -> None:
        if self.pipeline_config.is_locked():
            return
        row = self.pipeline_steps.currentRow()
        new_row = row + int(offset)
        if row < 0 or new_row < 0 or new_row >= self.pipeline_steps.count():
            return
        item = self.pipeline_steps.takeItem(row)
        if item is None:
            return
        self.pipeline_steps.insertItem(new_row, item)
        self.pipeline_steps.setCurrentRow(new_row)
        self._on_pipeline_steps_reordered()

    def _add_pipeline_step(self) -> None:
        step_id = str(self.pipeline_add_step_combo.currentData() or "")
        if step_id not in ADDABLE_PIPELINE_STEP_IDS:
            return
        if self.pipeline_config.is_locked():
            self._mark_pipeline_custom()
        row = self.pipeline_steps.currentRow()
        insert_before = "spike_detection"
        if row >= 0 and row < self.pipeline_steps.count():
            current_item = self.pipeline_steps.item(row)
            if current_item is not None:
                current_step = str(current_item.data(Qt.ItemDataRole.UserRole))
                current_index = self.pipeline_config.ordered_step_ids.index(current_step)
                self.pipeline_config.ordered_step_ids.insert(current_index + 1, step_id)
                self.pipeline_config.enabled_steps[step_id] = True
                self._mark_pipeline_custom()
                self._refresh_pipeline_steps()
                self.pipeline_steps.setCurrentRow(current_index + 1)
                return
        new_row = self._insert_pipeline_step_once(step_id, before_step_id=insert_before)
        self._mark_pipeline_custom()
        self._refresh_pipeline_steps()
        if new_row >= 0:
            self.pipeline_steps.setCurrentRow(new_row)

    def _remove_pipeline_step(self) -> None:
        if self.pipeline_config.is_locked():
            return
        row = self.pipeline_steps.currentRow()
        if row < 0 or row >= self.pipeline_steps.count():
            return
        item = self.pipeline_steps.item(row)
        if item is None:
            return
        step_id = str(item.data(Qt.ItemDataRole.UserRole))
        if step_id not in ADDABLE_PIPELINE_STEP_IDS:
            return
        self.pipeline_config.ordered_step_ids = [
            existing for existing in self.pipeline_config.ordered_step_ids if existing != step_id
        ]
        self.pipeline_config.enabled_steps[step_id] = False
        self._mark_pipeline_custom()
        self._refresh_pipeline_steps()
        if self.pipeline_steps.count() > 0:
            self.pipeline_steps.setCurrentRow(min(row, self.pipeline_steps.count() - 1))

    def _trace_names_for_recording(self, recording_id: Optional[str]) -> List[str]:
        if recording_id is None:
            return []
        return [
            name
            for name in self.trace_names
            if split_trace_identity(name)[0] == recording_id and name in self.traces
        ]

    def _visible_trace_names(self) -> List[str]:
        if self.current_recording_id is None:
            groups = recording_groups(self.trace_names)
            self.current_recording_id = next(iter(groups), None)
        return self._trace_names_for_recording(self.current_recording_id)

    def _sync_recording_selector(self, preferred_trace_name: Optional[str] = None) -> None:
        groups = recording_groups(self.trace_names)
        if preferred_trace_name in self.trace_names:
            preferred_recording = split_trace_identity(str(preferred_trace_name))[0]
        elif self.current_recording_id in groups:
            preferred_recording = self.current_recording_id
        else:
            preferred_recording = next(iter(groups), None)

        self.current_recording_id = preferred_recording
        self.recording_selector.blockSignals(True)
        self.recording_selector.clear()
        for recording_id, names in groups.items():
            roi_count = len(names)
            label = f"{recording_id} ({roi_count} ROI{'s' if roi_count != 1 else ''})"
            self.recording_selector.addItem(label, recording_id)
        if preferred_recording is not None:
            index = self.recording_selector.findData(preferred_recording)
            if index >= 0:
                self.recording_selector.setCurrentIndex(index)
        self.recording_selector.blockSignals(False)

    def _trace_list_label(self, trace_name: str, state: Optional[TraceCuration]) -> str:
        _recording_id, roi_id = split_trace_identity(trace_name)
        return f"{roi_id}{self._trace_status_suffix(trace_name, state)}"

    def _select_current_visible_trace(self, *, start_analysis: bool = True, render: bool = True) -> None:
        visible_names = self._visible_trace_names()
        if not visible_names:
            self.current_trace_name = None
            return
        if self.current_trace_name not in visible_names:
            self.current_trace_name = visible_names[0]
        row = visible_names.index(str(self.current_trace_name))
        self.trace_list.blockSignals(True)
        self.trace_list.setCurrentRow(row)
        self.trace_list.blockSignals(False)
        if start_analysis:
            self._start_analysis_for_trace(str(self.current_trace_name), priority=10)
        if render:
            self._render_current_trace(reset_view=True)

    def _selected_trace_names(self) -> List[str]:
        indexes = sorted(self.trace_list.selectedIndexes(), key=lambda idx: idx.row())
        names: List[str] = []
        for index in indexes:
            item = self.trace_list.item(index.row())
            if item is None:
                continue
            trace_name = str(item.data(Qt.ItemDataRole.UserRole))
            if trace_name in self.traces and trace_name not in names:
                names.append(trace_name)
        if not names and self.current_trace_name in self.traces:
            names.append(str(self.current_trace_name))
        return names

    def _run_trace_names(self, trace_names: List[str]) -> None:
        active_names = [name for name in trace_names if name in self.traces]
        if not active_names:
            return
        self._pull_params_from_widgets()
        self._analysis_generation += 1
        self._pending_session = None
        self._pending_session_generation = None
        for name in active_names:
            state = self.traces[name]
            state.analysis_generation = int(self._analysis_generation)
            state.analysis_status = "pending"
            state.analysis_error = ""
            self._last_packaged_trace_names.discard(name)
        current = self.current_trace_name if self.current_trace_name in active_names else active_names[0]
        self._start_analysis_for_trace(str(current), priority=10, force=True)
        for name in active_names:
            if name != current:
                self._start_analysis_for_trace(name, priority=0, force=True)
        if current in self.trace_names:
            self.current_trace_name = str(current)
            self.current_recording_id = split_trace_identity(current)[0]
        self._refresh_trace_list()
        self._render_current_trace()
        self.status_label.setText(f"Running {len(active_names)} trace(s) with current pipeline settings...")

    def run_current_trace(self) -> None:
        if self.current_trace_name:
            self._run_trace_names([self.current_trace_name])

    def run_selected_traces(self) -> None:
        self._run_trace_names(self._selected_trace_names())

    def run_all_traces(self) -> None:
        self._run_trace_names(list(self.trace_names))

    def _on_trace_queue_reordered(self, *_args: object) -> None:
        ordered: List[str] = []
        for row in range(self.trace_list.count()):
            item = self.trace_list.item(row)
            if item is None:
                continue
            trace_name = str(item.data(Qt.ItemDataRole.UserRole))
            if trace_name in self.traces and trace_name not in ordered:
                ordered.append(trace_name)
        recording_names = self._trace_names_for_recording(self.current_recording_id)
        if set(ordered) != set(recording_names):
            return
        ordered_set = set(ordered)
        updated: List[str] = []
        inserted = False
        for name in self.trace_names:
            if name in ordered_set:
                if not inserted:
                    updated.extend(ordered)
                    inserted = True
                continue
            updated.append(name)
        if not inserted or set(updated) != set(self.trace_names):
            return
        self.trace_names = updated
        self.pipeline_config.trace_order = list(self.trace_names)
        self._update_trace_queue_status()

    def _update_trace_queue_status(self) -> None:
        if not hasattr(self, "trace_queue_status"):
            return
        counts = {"pending": 0, "running": 0, "done": 0, "error": 0}
        for name in self.trace_names:
            state = self.traces.get(name)
            if state is not None:
                counts[state.analysis_status] = counts.get(state.analysis_status, 0) + 1
        self.trace_queue_status.setText(
            f"{len(self.trace_names)} traces | pending {counts.get('pending', 0)} | "
            f"running {counts.get('running', 0)} | reviewed {counts.get('done', 0)} | "
            f"failed {counts.get('error', 0)}"
        )

    def _connect_signals(self) -> None:
        self.btn_load_excel.clicked.connect(self.load_excel_dialog)
        self.btn_load_exported.clicked.connect(self.load_exported_dialog)
        self.btn_load_session.clicked.connect(self.load_session_dialog)
        self.btn_save_session.clicked.connect(self.save_session_dialog)
        self.btn_save_analysis_package.clicked.connect(self.save_analysis_package_dialog)
        self.btn_export.clicked.connect(self.export_all_dialog)
        self.btn_export_png_current.clicked.connect(self.export_current_dialog)
        self.btn_reload_params_cfg.clicked.connect(self.reload_detection_params_config)
        self.btn_rerun_current.clicked.connect(self.rerun_current)
        self.btn_rerun_all.clicked.connect(self.rerun_all)

        self.btn_prev_trace.clicked.connect(self.prev_trace)
        self.btn_next_trace.clicked.connect(self.next_trace)
        self.btn_run_current.clicked.connect(self.run_current_trace)
        self.btn_run_selected.clicked.connect(self.run_selected_traces)
        self.btn_run_all.clicked.connect(self.run_all_traces)
        self.btn_review_save_package.clicked.connect(self.save_analysis_package_dialog)
        self.btn_review_rerun_current.clicked.connect(self.rerun_current)

        self.recording_selector.currentIndexChanged.connect(self._on_recording_selected)
        self.trace_list.currentRowChanged.connect(self.on_trace_selected)
        self.trace_list.model().rowsMoved.connect(self._on_trace_queue_reordered)
        self.plot.clicked.connect(self.on_plot_clicked)
        self.table.cellClicked.connect(self.on_table_clicked)
        self.chk_show_raw.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_corrected.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_baseline.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_highpass.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_lowpass.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_hybrid_denoised.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_normalized_denoised.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_denoise_residual.stateChanged.connect(self._on_overlay_changed)
        self.display_units_combo.currentIndexChanged.connect(self._on_display_units_changed)
        self.display_render_source_combo.currentIndexChanged.connect(self._on_overlay_changed)
        for checkbox in [
            self.chk_show_raw,
            self.chk_show_corrected,
            self.chk_show_baseline,
            self.chk_show_highpass,
            self.chk_show_lowpass,
            self.chk_show_hybrid_denoised,
            self.chk_show_normalized_denoised,
            self.chk_show_denoise_residual,
        ]:
            checkbox.stateChanged.connect(self._sync_pipeline_overlay_controls_from_main)
        self.display_units_combo.currentIndexChanged.connect(self._sync_pipeline_overlay_controls_from_main)
        self.display_render_source_combo.currentIndexChanged.connect(self._sync_pipeline_overlay_controls_from_main)
        for checkbox in [
            self.pipeline_chk_show_raw,
            self.pipeline_chk_show_corrected,
            self.pipeline_chk_show_baseline,
            self.pipeline_chk_show_highpass,
            self.pipeline_chk_show_lowpass,
            self.pipeline_chk_show_hybrid_denoised,
            self.pipeline_chk_show_normalized_denoised,
            self.pipeline_chk_show_denoise_residual,
        ]:
            checkbox.stateChanged.connect(self._on_pipeline_overlay_controls_changed)
        self.pipeline_display_units_combo.currentIndexChanged.connect(self._on_pipeline_overlay_controls_changed)
        self.pipeline_display_render_source_combo.currentIndexChanged.connect(self._on_pipeline_overlay_controls_changed)
        self.param_denoising_method.currentIndexChanged.connect(self._update_denoise_control_state)
        self.pipeline_preset_combo.currentIndexChanged.connect(self._on_pipeline_preset_changed)
        self.pipeline_steps.currentRowChanged.connect(self._on_pipeline_step_selected)
        self.pipeline_steps.itemChanged.connect(self._on_pipeline_item_changed)
        self.pipeline_steps.model().rowsMoved.connect(self._on_pipeline_steps_reordered)
        self.btn_pipeline_add_step.clicked.connect(self._add_pipeline_step)
        self.btn_pipeline_step_up.clicked.connect(lambda: self._move_pipeline_step(-1))
        self.btn_pipeline_step_down.clicked.connect(lambda: self._move_pipeline_step(1))
        self.btn_pipeline_remove_step.clicked.connect(self._remove_pipeline_step)
        self._update_denoise_control_state()

    def _init_shortcuts(self) -> None:
        self.shortcuts = bind_shortcuts(
            self,
            {
                "Right": self.next_trace,
                "Left": self.prev_trace,
                "N": self.jump_next_spike,
                "B": self.jump_prev_spike,
                "Ctrl+S": self.save_session_dialog,
                "Ctrl+E": self.save_analysis_package_dialog,
                "Ctrl+Shift+E": self.export_current_dialog,
                "Q": lambda: self._toggle_checkbox(self.chk_show_raw),
                "W": lambda: self._toggle_checkbox(self.chk_show_corrected),
            },
        )

    def _toggle_checkbox(self, checkbox: QCheckBox) -> None:
        checkbox.setChecked(not checkbox.isChecked())

    @staticmethod
    def _set_combo_to_matching_data(target: QComboBox, source: QComboBox) -> None:
        index = target.findData(source.currentData())
        if index >= 0:
            target.setCurrentIndex(index)

    def _overlay_checkbox_pairs(self) -> List[Tuple[QCheckBox, QCheckBox]]:
        if not hasattr(self, "pipeline_chk_show_raw"):
            return []
        return [
            (self.chk_show_raw, self.pipeline_chk_show_raw),
            (self.chk_show_corrected, self.pipeline_chk_show_corrected),
            (self.chk_show_baseline, self.pipeline_chk_show_baseline),
            (self.chk_show_highpass, self.pipeline_chk_show_highpass),
            (self.chk_show_lowpass, self.pipeline_chk_show_lowpass),
            (self.chk_show_hybrid_denoised, self.pipeline_chk_show_hybrid_denoised),
            (self.chk_show_normalized_denoised, self.pipeline_chk_show_normalized_denoised),
            (self.chk_show_denoise_residual, self.pipeline_chk_show_denoise_residual),
        ]

    def _sync_pipeline_overlay_controls_from_main(self, *_args: object) -> None:
        if not hasattr(self, "pipeline_display_units_combo"):
            return
        for source, target in self._overlay_checkbox_pairs():
            target.blockSignals(True)
            target.setChecked(source.isChecked())
            target.blockSignals(False)
        for source, target in [
            (self.display_units_combo, self.pipeline_display_units_combo),
            (self.display_render_source_combo, self.pipeline_display_render_source_combo),
        ]:
            target.blockSignals(True)
            self._set_combo_to_matching_data(target, source)
            target.blockSignals(False)

    def _on_pipeline_overlay_controls_changed(self, *_args: object) -> None:
        for target, source in self._overlay_checkbox_pairs():
            target.blockSignals(True)
            target.setChecked(source.isChecked())
            target.blockSignals(False)
        for target, source in [
            (self.display_units_combo, self.pipeline_display_units_combo),
            (self.display_render_source_combo, self.pipeline_display_render_source_combo),
        ]:
            target.blockSignals(True)
            self._set_combo_to_matching_data(target, source)
            target.blockSignals(False)
        self._render_current_trace(refit_y=True)

    def _display_normalization_mode(self) -> str:
        if not hasattr(self, "display_units_combo"):
            return NORMALIZATION_NATIVE
        return str(self.display_units_combo.currentData() or NORMALIZATION_NATIVE)

    def _selected_render_source(self) -> str:
        if not hasattr(self, "display_render_source_combo"):
            return DISPLAY_RENDER_SOURCE_DENOISED
        return str(self.display_render_source_combo.currentData() or DISPLAY_RENDER_SOURCE_DENOISED)

    @staticmethod
    def _raw_overlay_signal(raw: np.ndarray, raw_baseline: Optional[np.ndarray]) -> Optional[np.ndarray]:
        raw_arr = np.asarray(raw, dtype=float)
        baseline_arr = None if raw_baseline is None else np.asarray(raw_baseline, dtype=float)
        if baseline_arr is not None and baseline_arr.shape == raw_arr.shape:
            return raw_arr - baseline_arr
        return None

    def _selected_pipeline_signal_for_display(
        self,
        source: str,
        raw: np.ndarray,
        raw_baseline: Optional[np.ndarray],
        corrected: np.ndarray,
        denoised: Optional[np.ndarray],
        normalized_denoised: Optional[np.ndarray],
        highpass: Optional[np.ndarray],
        lowpass: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if source == DISPLAY_RENDER_SOURCE_CORRECTED:
            return np.asarray(corrected, dtype=float)
        if source == DISPLAY_RENDER_SOURCE_RAW:
            return self._raw_overlay_signal(raw, raw_baseline)
        if source == DISPLAY_RENDER_SOURCE_NORMALIZED_DENOISED:
            return None if normalized_denoised is None else np.asarray(normalized_denoised, dtype=float)
        if source == DISPLAY_RENDER_SOURCE_HIGHPASS:
            return None if highpass is None else np.asarray(highpass, dtype=float)
        if source == DISPLAY_RENDER_SOURCE_LOWPASS:
            return None if lowpass is None else np.asarray(lowpass, dtype=float)
        return None if denoised is None else np.asarray(denoised, dtype=float)

    def _on_display_units_changed(self, *_args: object) -> None:
        self._render_current_trace(refit_y=True)

    def _pull_params_from_widgets(self) -> None:
        self.detection_params.detection_mode = str(self.param_detection_mode.currentData() or "std")
        self.detection_params.baseline_mode = str(self.param_baseline_mode.currentData() or "percentile")
        self.detection_params.startup_exclusion_ms = float(self.param_startup_exclusion_ms.value())
        self.detection_params.baseline_rolling_window_ms = float(self.param_baseline_rolling_window_ms.value())
        self.detection_params.baseline_rolling_percentile = float(self.param_baseline_rolling_percentile.value())
        self.detection_params.baseline_savgol_window_ms = float(self.param_baseline_savgol_window_ms.value())
        self.detection_params.baseline_savgol_polyorder = int(self.param_baseline_savgol_polyorder.value())
        self.detection_params.highpass_cutoff_hz = float(self.param_highpass_cutoff_hz.value())
        self.detection_params.highpass_order = int(self.param_highpass_order.value())
        self.detection_params.lowpass_cutoff_hz = float(self.param_lowpass_cutoff_hz.value())
        self.detection_params.lowpass_order = int(self.param_lowpass_order.value())
        self.detection_params.rolling_downsample_factor = int(self.param_rolling_downsample_factor.value())
        self.detection_params.plateau_median_window_ms = float(self.param_plateau_median_window_ms.value())
        self.detection_params.long_plateau_drift_window_ms = float(self.param_long_plateau_drift_window_ms.value())
        self.detection_params.long_plateau_drift_percentile = float(self.param_long_plateau_drift_percentile.value())
        self.detection_params.long_plateau_entry_noise_k = float(self.param_long_plateau_entry_noise_k.value())
        self.detection_params.long_plateau_exit_noise_k = float(self.param_long_plateau_exit_noise_k.value())
        self.detection_params.long_plateau_exit_fraction = float(self.param_long_plateau_exit_fraction.value())
        self.detection_params.long_plateau_enter_sustain_ms = float(self.param_long_plateau_enter_sustain_ms.value())
        self.detection_params.long_plateau_exit_sustain_ms = float(self.param_long_plateau_exit_sustain_ms.value())
        self.detection_params.short_plateau_step_noise_k = float(self.param_short_plateau_step_noise_k.value())
        self.detection_params.short_plateau_elevated_noise_k = float(self.param_short_plateau_elevated_noise_k.value())
        self.detection_params.short_plateau_survival_noise_k = float(self.param_short_plateau_survival_noise_k.value())
        self.detection_params.short_plateau_loss_noise_k = float(self.param_short_plateau_loss_noise_k.value())
        self.detection_params.short_plateau_median_noise_k = float(self.param_short_plateau_median_noise_k.value())
        self.detection_params.short_plateau_pre_quiet_noise_k = float(self.param_short_plateau_pre_quiet_noise_k.value())
        self.detection_params.short_plateau_min_support_fraction = float(self.param_short_plateau_min_support_fraction.value())
        self.detection_params.short_plateau_min_confidence = float(self.param_short_plateau_min_confidence.value())
        self.detection_params.short_plateau_early_dwell_ms = float(self.param_short_plateau_early_dwell_ms.value())
        self.detection_params.short_plateau_min_support_ms = float(self.param_short_plateau_min_support_ms.value())
        self.detection_params.short_plateau_min_duration_ms = float(self.param_short_plateau_min_duration_ms.value())
        self.detection_params.spike_noise_k = float(self.param_spike_noise_k.value())
        self.detection_params.spike_prominence_k = float(self.param_spike_prominence_k.value())
        self.detection_params.template_matching_enable_autopick = bool(
            self.param_template_matching_enable_autopick.isChecked()
        )
        self.detection_params.template_matching_template_indices = self._parse_index_list(
            self.param_template_matching_template_indices.text()
        )
        self.detection_params.template_matching_pre_ms = float(self.param_template_matching_pre_ms.value())
        self.detection_params.template_matching_post_ms = float(self.param_template_matching_post_ms.value())
        self.detection_params.template_matching_filter_cutoff_hz = float(self.param_template_matching_filter_cutoff_hz.value())
        self.detection_params.template_matching_false_positive_rate = float(self.param_template_matching_false_positive_rate.value())
        self.detection_params.min_separation_ms = float(self.param_min_separation_ms.value())
        self.detection_params.spike_zoom_half_window_ms = float(self.param_spike_zoom_half_window_ms.value())
        self.detection_params.denoising_method = str(self.param_denoising_method.currentData() or "gonzalez_full_trace")
        self.detection_params.gonzalez_threshold_sd = float(self.param_gonzalez_threshold_sd.value())
        self.detection_params.gonzalez_max_clusters = int(self.param_gonzalez_max_clusters.value())
        self.detection_params.gonzalez_attenuation_min = float(self.param_gonzalez_attenuation_min.value())
        self.detection_params.gonzalez_attenuation_max = float(self.param_gonzalez_attenuation_max.value())
        self.detection_params.gonzalez_enable_noise_cluster_rejection = bool(
            self.param_gonzalez_noise_cluster_rejection.isChecked()
        )
        self.detection_params.gonzalez_enable_event_template_rejection = bool(
            self.param_gonzalez_event_template_rejection.isChecked()
        )
        self.artifact_params.absolute_low = float(self.param_artifact_abs_low.value())
        self.artifact_params.min_depth = float(self.param_artifact_min_depth.value())
        self.artifact_params.drop_start_abs = float(self.param_artifact_drop_start.value())
        self._sync_pipeline_config_from_widgets()
        self.detection_params.use_highpass_filter = _pipeline_step_enabled(self.pipeline_config, "highpass_filter")
        self.detection_params.use_lowpass_filter = _pipeline_step_enabled(self.pipeline_config, "lowpass_filter")
        if not self.pipeline_config.enabled_steps.get("denoising", True):
            self.detection_params.denoising_method = "none"

    def _push_params_to_widgets(self) -> None:
        mode_index = self.param_detection_mode.findData(str(self.detection_params.detection_mode))
        self.param_detection_mode.setCurrentIndex(max(0, mode_index))
        baseline_mode_index = self.param_baseline_mode.findData(str(self.detection_params.baseline_mode))
        self.param_baseline_mode.setCurrentIndex(max(0, baseline_mode_index))
        self.param_startup_exclusion_ms.setValue(self.detection_params.startup_exclusion_ms)
        self.param_baseline_rolling_window_ms.setValue(self.detection_params.baseline_rolling_window_ms)
        self.param_baseline_rolling_percentile.setValue(self.detection_params.baseline_rolling_percentile)
        self.param_baseline_savgol_window_ms.setValue(self.detection_params.baseline_savgol_window_ms)
        self.param_baseline_savgol_polyorder.setValue(int(self.detection_params.baseline_savgol_polyorder))
        self.param_highpass_cutoff_hz.setValue(float(self.detection_params.highpass_cutoff_hz))
        self.param_highpass_order.setValue(int(self.detection_params.highpass_order))
        self.param_lowpass_cutoff_hz.setValue(float(self.detection_params.lowpass_cutoff_hz))
        self.param_lowpass_order.setValue(int(self.detection_params.lowpass_order))
        self.param_rolling_downsample_factor.setValue(int(self.detection_params.rolling_downsample_factor))
        self.param_plateau_median_window_ms.setValue(self.detection_params.plateau_median_window_ms)
        self.param_long_plateau_drift_window_ms.setValue(float(self.detection_params.long_plateau_drift_window_ms))
        self.param_long_plateau_drift_percentile.setValue(float(self.detection_params.long_plateau_drift_percentile))
        self.param_long_plateau_entry_noise_k.setValue(float(self.detection_params.long_plateau_entry_noise_k))
        self.param_long_plateau_exit_noise_k.setValue(float(self.detection_params.long_plateau_exit_noise_k))
        self.param_long_plateau_exit_fraction.setValue(float(self.detection_params.long_plateau_exit_fraction))
        self.param_long_plateau_enter_sustain_ms.setValue(float(self.detection_params.long_plateau_enter_sustain_ms))
        self.param_long_plateau_exit_sustain_ms.setValue(float(self.detection_params.long_plateau_exit_sustain_ms))
        self.param_short_plateau_step_noise_k.setValue(float(self.detection_params.short_plateau_step_noise_k))
        self.param_short_plateau_elevated_noise_k.setValue(float(self.detection_params.short_plateau_elevated_noise_k))
        self.param_short_plateau_survival_noise_k.setValue(float(self.detection_params.short_plateau_survival_noise_k))
        self.param_short_plateau_loss_noise_k.setValue(float(self.detection_params.short_plateau_loss_noise_k))
        self.param_short_plateau_median_noise_k.setValue(float(self.detection_params.short_plateau_median_noise_k))
        self.param_short_plateau_pre_quiet_noise_k.setValue(float(self.detection_params.short_plateau_pre_quiet_noise_k))
        self.param_short_plateau_min_support_fraction.setValue(float(self.detection_params.short_plateau_min_support_fraction))
        self.param_short_plateau_min_confidence.setValue(float(self.detection_params.short_plateau_min_confidence))
        self.param_short_plateau_early_dwell_ms.setValue(float(self.detection_params.short_plateau_early_dwell_ms))
        self.param_short_plateau_min_support_ms.setValue(float(self.detection_params.short_plateau_min_support_ms))
        self.param_short_plateau_min_duration_ms.setValue(float(self.detection_params.short_plateau_min_duration_ms))
        self.param_spike_noise_k.setValue(self.detection_params.spike_noise_k)
        self.param_spike_prominence_k.setValue(self.detection_params.spike_prominence_k)
        self.param_template_matching_enable_autopick.setChecked(
            bool(self.detection_params.template_matching_enable_autopick)
        )
        self.param_template_matching_template_indices.setText(
            self._format_index_list(list(self.detection_params.template_matching_template_indices))
        )
        self.param_template_matching_pre_ms.setValue(float(self.detection_params.template_matching_pre_ms))
        self.param_template_matching_post_ms.setValue(float(self.detection_params.template_matching_post_ms))
        self.param_template_matching_filter_cutoff_hz.setValue(float(self.detection_params.template_matching_filter_cutoff_hz))
        self.param_template_matching_false_positive_rate.setValue(float(self.detection_params.template_matching_false_positive_rate))
        self.param_min_separation_ms.setValue(self.detection_params.min_separation_ms)
        self.param_spike_zoom_half_window_ms.setValue(self.detection_params.spike_zoom_half_window_ms)
        denoising_method_index = self.param_denoising_method.findData(str(self.detection_params.denoising_method))
        self.param_denoising_method.setCurrentIndex(max(0, denoising_method_index))
        self.param_gonzalez_threshold_sd.setValue(float(self.detection_params.gonzalez_threshold_sd))
        self.param_gonzalez_max_clusters.setValue(int(self.detection_params.gonzalez_max_clusters))
        self.param_gonzalez_attenuation_min.setValue(float(self.detection_params.gonzalez_attenuation_min))
        self.param_gonzalez_attenuation_max.setValue(float(self.detection_params.gonzalez_attenuation_max))
        self.param_gonzalez_noise_cluster_rejection.setChecked(
            bool(self.detection_params.gonzalez_enable_noise_cluster_rejection)
        )
        self.param_gonzalez_event_template_rejection.setChecked(
            bool(self.detection_params.gonzalez_enable_event_template_rejection)
        )
        self._update_denoise_control_state()
        self.param_artifact_abs_low.setValue(self.artifact_params.absolute_low)
        self.param_artifact_min_depth.setValue(self.artifact_params.min_depth)
        self.param_artifact_drop_start.setValue(self.artifact_params.drop_start_abs)

    def _update_denoise_control_state(self, *_args: object) -> None:
        denoising_method = str(self.param_denoising_method.currentData() or "gonzalez_full_trace")
        self.param_denoising_input_normalization.setText(_denoising_input_normalization_label(denoising_method))
        denoising_enabled = denoising_method != "none"
        uses_s7 = denoising_method == "gonzalez_s7_dff"
        self.param_gonzalez_threshold_sd.setEnabled(denoising_enabled)
        self.param_gonzalez_max_clusters.setEnabled(denoising_enabled)
        self.param_gonzalez_attenuation_min.setEnabled(denoising_enabled and not uses_s7)
        self.param_gonzalez_attenuation_max.setEnabled(denoising_enabled and not uses_s7)
        if uses_s7:
            self.param_gonzalez_max_clusters.setToolTip(
                "For S7 reproduction, Ward/silhouette clustering searches up to 20 clusters and keeps broad paper-style domains with wavelet mask support."
            )
            self.param_gonzalez_attenuation_min.setToolTip(
                "Fixed at 0.5 in paper S7 reproduction."
            )
            self.param_gonzalez_attenuation_max.setToolTip(
                "Fixed at 0.5 for paper S7 thresholding and PC1 noisy-event cleanup."
            )
        else:
            self.param_gonzalez_max_clusters.setToolTip(
                "Number of wavelet-frequency groups used before thresholding; higher values separate frequency bands more finely."
            )
            self.param_gonzalez_attenuation_min.setToolTip(
                "Minimum suppression applied to samples below the wavelet threshold."
            )
            self.param_gonzalez_attenuation_max.setToolTip(
                "Maximum suppression applied to below-threshold wavelet coefficients; lower values preserve more signal."
            )
        self.chk_show_hybrid_denoised.setText("Show selected overlay")
        if hasattr(self, "pipeline_chk_show_hybrid_denoised"):
            self.pipeline_chk_show_hybrid_denoised.setText("Show selected overlay")

    def _result_root(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent / "Result"
        return Path(__file__).resolve().parents[3] / "Result"

    def _make_export_dir(self, label: str) -> Path:
        """Create and return a timestamped subfolder under the Result directory.

        Pattern: Result/<label>_YYYYMMDD_HHMMSS/
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._result_root() / f"{label}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _detection_params_config_path(self) -> Path:
        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            editable_cfg = exe_dir / "detection_params.json"
            bundled_cfg = exe_dir / "_internal" / "detection_params.json"

            if editable_cfg.exists():
                return editable_cfg

            if bundled_cfg.exists():
                # Copy bundled defaults next to the exe so users can edit config easily.
                try:
                    editable_cfg.write_text(bundled_cfg.read_text(encoding="utf-8"), encoding="utf-8")
                    return editable_cfg
                except Exception:
                    return bundled_cfg

            return editable_cfg
        return Path(__file__).resolve().parents[2] / "detection_params.json"

    def _load_detection_params_config(self) -> None:
        cfg_path = self._detection_params_config_path()
        if not cfg_path.exists():
            cfg_path.write_text(json.dumps(DetectionParams().to_dict(), indent=2), encoding="utf-8")
            return
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.detection_params = DetectionParams.from_dict(data)
        except Exception:
            pass

    def reload_detection_params_config(self) -> None:
        self._load_detection_params_config()
        self.pipeline_config = PipelineConfig.from_preset("current")
        self._seed_pipeline_from_filter_params()
        self._push_params_to_widgets()
        self._refresh_pipeline_steps()
        self.status_label.setText(f"Reloaded detector params: {self._detection_params_config_path().name}")

    def load_excel_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Excel", "", "Excel Files (*.xlsx *.xls)")
        if not file_path:
            return
        self.load_excel(file_path)

    def _set_loading_controls_enabled(self, enabled: bool) -> None:
        for button in [
            self.btn_load_excel,
            self.btn_load_exported,
            self.btn_load_session,
            self.btn_save_analysis_package,
            self.btn_rerun_current,
            self.btn_rerun_all,
        ]:
            button.setEnabled(enabled)

    def _spike_from_event_stats_row(self, row: pd.Series, trace_name: str) -> SpikeRecord:
        return _spike_record_from_event_stats_row(row, trace_name)

    def load_exported_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Export Folder")
        if not folder:
            return
        self.load_exported_result(folder)

    def load_exported_result(self, folder: str) -> None:
        try:
            folder_path = Path(folder)
            corrected_path = _find_export_file(folder_path, "corrected_traces.csv")
            if corrected_path is None:
                raise FileNotFoundError("corrected_traces.csv not found in selected folder or export subfolders")

            corrected_df = pd.read_csv(corrected_path)
            if "time" not in corrected_df.columns:
                raise ValueError("corrected_traces.csv must contain a 'time' column")

            time = pd.to_numeric(corrected_df["time"], errors="coerce").to_numpy(dtype=float)
            trace_cols = [col for col in corrected_df.columns if col != "time" and not is_ignored_trace_column(col)]
            if not trace_cols:
                raise ValueError("No trace columns found in corrected_traces.csv")

            self.traces.clear()
            self._last_packaged_trace_names.clear()
            self.current_trace_name = None
            self.current_recording_id = None
            self.recording_selector.clear()
            self.trace_list.clear()
            for trace_name in trace_cols:
                signal = pd.to_numeric(corrected_df[trace_name], errors="coerce").to_numpy(dtype=float)
                state = TraceCuration(
                    trace_name=trace_name,
                    raw=signal.copy(),
                    corrected_source=signal.copy(),
                    corrected=signal.copy(),
                    time=time,
                    sampling_rate_hz=_sampling_rate_hz(time),
                    original_time=time.copy(),
                    original_raw=signal.copy(),
                    original_corrected_source=signal.copy(),
                    original_sampling_rate_hz=_sampling_rate_hz(time),
                    artifact_mask=None,
                    spikes=[],
                    smoothed=None,
                    baseline=None,
                    debug_threshold=None,
                    candidate_windows=[],
                )
                self.traces[trace_name] = state

            spike_event_path = _find_export_file(folder_path, "spike_event_stats.csv")
            if spike_event_path is not None:
                spike_event_df = pd.read_csv(spike_event_path)
                for _, row in spike_event_df.iterrows():
                    trace_name = str(row.get("trace_id", row.get("trace_name", "")))
                    state = self.traces.get(trace_name)
                    if state is None:
                        continue
                    state.spikes.append(self._spike_from_event_stats_row(row, trace_name))

            for state in self.traces.values():
                state.sort_spikes()

            self.trace_names = sorted(self.traces.keys())
            self.pipeline_config.trace_order = list(self.trace_names)
            self._refresh_trace_list()
            if self.trace_names:
                self._select_current_visible_trace(start_analysis=False, render=True)
            self.status_label.setText(f"Loaded exported result: {folder_path.name} | traces: {len(self.trace_names)}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Export Error", str(exc))

    def load_excel(self, file_path: str, pending_session: Optional[AppSession] = None) -> None:
        self._pull_params_from_widgets()
        self._analysis_generation += 1
        generation = self._analysis_generation
        self.excel_path = file_path
        self.traces.clear()
        self._last_packaged_trace_names.clear()
        self.trace_names = []
        self.current_trace_name = None
        self.current_recording_id = None
        self._table_spikes = []
        self._analysis_tasks = []
        self._pending_session = pending_session
        self._pending_session_generation = generation if pending_session is not None else None
        self.recording_selector.clear()
        self.trace_list.clear()
        self._table_signature = ()
        self.table.setRowCount(0)
        self._loading_excel = True
        self._set_loading_controls_enabled(False)
        self.status_label.setText(f"Loading Excel: {Path(file_path).name}...")

        task = _ExcelImportTask(file_path, generation)
        task.signals.finished.connect(self._on_excel_import_finished)
        self._import_task = task
        self._import_thread_pool.start(task)

    def _on_excel_import_finished(
        self,
        file_path: str,
        generation: int,
        time_map: Optional[Dict[str, np.ndarray]],
        raw_map: Optional[Dict[str, np.ndarray]],
        corrected_map: Optional[Dict[str, np.ndarray]],
        sampling_or_error: object,
    ) -> None:
        if int(generation) != self._analysis_generation:
            return
        self._import_task = None
        self._loading_excel = False
        self._set_loading_controls_enabled(True)
        if time_map is None or raw_map is None or corrected_map is None or isinstance(sampling_or_error, str):
            self._pending_session = None
            self._pending_session_generation = None
            QMessageBox.critical(self, "Load Error", str(sampling_or_error))
            self.status_label.setText("Excel load failed")
            return

        sampling_rate_map = sampling_or_error
        if not isinstance(sampling_rate_map, dict):
            self._pending_session = None
            self._pending_session_generation = None
            QMessageBox.critical(self, "Load Error", "Unexpected Excel import result")
            self.status_label.setText("Excel load failed")
            return

        self.traces = build_pending_trace_states(
            time_map=time_map,
            raw_map=raw_map,
            corrected_map=corrected_map,
            sampling_rate_map=sampling_rate_map,
            generation=int(generation),
        )
        self.trace_names = sorted(self.traces.keys())
        self.pipeline_config.trace_order = list(self.trace_names)
        if self._pending_session is not None:
            apply_session_to_traces(self._pending_session, self.traces)
        if self.trace_names:
            self.current_trace_name = self.trace_names[0]
            self.current_recording_id = split_trace_identity(self.current_trace_name)[0]
        self._refresh_trace_list()
        if self.trace_names:
            self._select_current_visible_trace(start_analysis=False, render=True)
            self._start_analysis_for_trace(self.trace_names[0], priority=10)
            for trace_name in self.trace_names[1:]:
                self._start_analysis_for_trace(trace_name, priority=0)
        self.status_label.setText(
            f"Loaded {Path(file_path).name} | traces: {len(self.trace_names)} | analyzing selected trace..."
        )

    def load_session_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Session", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            app_session = load_session(file_path)
            self.detection_params = app_session.detection_params
            self.artifact_params = app_session.artifact_params
            if app_session.pipeline_config:
                self.pipeline_config = PipelineConfig.from_dict(app_session.pipeline_config)
            else:
                self.pipeline_config = PipelineConfig.from_preset("current")
                self._seed_pipeline_from_filter_params()
            self._push_params_to_widgets()
            self._refresh_pipeline_steps()

            if app_session.excel_path:
                self.load_excel(app_session.excel_path, pending_session=app_session)
                self.status_label.setText(f"Loading session traces: {Path(file_path).name}")
                return
            if self.traces:
                apply_session_to_traces(app_session, self.traces)
                self._render_current_trace(reset_view=True)
                self.status_label.setText(f"Loaded session: {Path(file_path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Session Error", str(exc))

    def save_session_dialog(self) -> None:
        if not self.traces:
            return
        if not self._analysis_ready_for_trace_names(self.trace_names, action="save session"):
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Session", "spike_session.json", "JSON Files (*.json)")
        if not file_path:
            return
        self._pull_params_from_widgets()
        save_session(
            file_path,
            self.excel_path,
            self.detection_params,
            self.artifact_params,
            self.traces,
            pipeline_config=self._pipeline_config_dict(),
        )
        self.status_label.setText(f"Session saved: {Path(file_path).name}")

    def save_analysis_package_dialog(self) -> None:
        if not self.traces:
            return
        if not self._analysis_ready_for_trace_names(self.trace_names, action="save analysis package"):
            return
        parent = QFileDialog.getExistingDirectory(
            self,
            "Choose Analysis Package Parent Folder",
            str(self._result_root()),
        )
        if not parent:
            return
        self._pull_params_from_widgets()
        try:
            from detector.analysis.package import default_analysis_package_dir, save_analysis_package

            out_dir = default_analysis_package_dir(Path(parent), self.excel_path)
            saved_dir = save_analysis_package(
                out_dir,
                source_excel=self.excel_path,
                detection_params=self.detection_params,
                artifact_params=self.artifact_params,
                trace_names=self.trace_names,
                traces=self.traces,
                pipeline_config=self._pipeline_config_dict(),
                render_normalization_mode=self._display_normalization_mode(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Package Error", str(exc))
            return
        self._last_packaged_trace_names = set(self.trace_names)
        self._refresh_trace_list()
        self.status_label.setText(f"Analysis package saved: {saved_dir}")

    def _analysis_ready_for_trace_names(self, trace_names: List[str], action: str) -> bool:
        not_done = [
            name
            for name in trace_names
            if name in self.traces and self.traces[name].analysis_status != "done"
        ]
        if not not_done:
            return True
        current = self.current_trace_name
        if current in not_done:
            self._start_analysis_for_trace(current, priority=10)
        for name in not_done:
            if name != current:
                self._start_analysis_for_trace(name, priority=0)
        self.status_label.setText(
            f"Cannot {action} yet; analysis is still pending/running for {len(not_done)} trace(s)."
        )
        QMessageBox.information(
            self,
            "Analysis In Progress",
            f"Please wait until analysis finishes before you {action}. Pending/running traces: {len(not_done)}.",
        )
        return False

    def _write_export_tables_and_report(self, out_dir: Path, trace_names: List[str]) -> List[Path]:
        active_trace_names = [name for name in trace_names if name in self.traces]
        layout = _prepare_export_layout(out_dir)
        cfg = TRACE_SUMMARY_CONFIG

        export_tables = _build_export_dataframes(active_trace_names, self.traces)

        table_paths = {
            "trace_summary": layout["tables"] / str(cfg["csv_name"]),
            "trace_qc": layout["tables"] / str(cfg["qc_csv_name"]),
            "plateau_events": layout["tables"] / str(cfg["events_csv_name"]),
            "spike_events": layout["tables"] / str(cfg["spike_events_csv_name"]),
            "spike_summary": layout["tables"] / str(cfg["spike_summary_csv_name"]),
        }
        for key, path in table_paths.items():
            export_tables[key].to_csv(path, index=False)

        reload_corrected_path = layout["reload"] / "corrected_traces.csv"
        reload_spike_event_path = layout["reload"] / str(cfg["spike_events_csv_name"])
        export_tables["corrected_traces"].to_csv(reload_corrected_path, index=False)
        export_tables["spike_events"].to_csv(reload_spike_event_path, index=False)

        png_path = layout["reports"] / str(cfg["png_name"])
        pdf_path = layout["reports"] / str(cfg["pdf_name"]) if bool(cfg.get("save_pdf", False)) else None
        _save_qc_summary_report(
            trace_df=export_tables["trace_qc"],
            events_df=export_tables["plateau_events"],
            png_path=png_path,
            pdf_path=pdf_path,
        )
        spike_png_path = layout["reports"] / str(cfg["spike_png_name"])
        _save_spike_statistics_summary(
            spike_summary_df=export_tables["spike_summary"],
            spike_events_df=export_tables["spike_events"],
            png_path=spike_png_path,
        )

        written_files = list(table_paths.values()) + [reload_corrected_path, reload_spike_event_path, png_path, spike_png_path]
        if pdf_path is not None:
            written_files.append(pdf_path)
        return written_files

    def _write_export_manifest(self, out_dir: Path, export_type: str, trace_names: List[str]) -> Path:
        active_trace_names = [name for name in trace_names if name in self.traces]
        grouped_recordings = recording_groups(active_trace_names)
        files = sorted(
            path
            for path in out_dir.rglob("*")
            if path.is_file() and path.name != EXPORT_MANIFEST_NAME
        )
        folders = sorted(path for path in out_dir.rglob("*") if path.is_dir())
        manifest = {
            "export_version": 2,
            "export_type": export_type,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_excel": self.excel_path,
            "trace_count": len(active_trace_names),
            "traces": [str(name) for name in active_trace_names],
            "recordings": [
                {
                    "recording_id": recording_id,
                    "roi_count": len(names),
                    "roi_ids": [split_trace_identity(name)[1] for name in names],
                    "traces": [str(name) for name in names],
                }
                for recording_id, names in grouped_recordings.items()
            ],
            "folders": {
                "tables": EXPORT_TABLES_DIR,
                "reports": EXPORT_REPORTS_DIR,
                "reload": EXPORT_RELOAD_DIR,
            },
            "generated_folders": [str(path.relative_to(out_dir)) for path in folders],
            "files": [str(path.relative_to(out_dir)) for path in files],
        }
        manifest_path = out_dir / EXPORT_MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def export_dialog(self) -> None:
        if not self.traces:
            return
        if not self._analysis_ready_for_trace_names(self.trace_names, action="export"):
            return

        out_dir = Path(self._make_export_dir("export_tables"))
        written_files = self._write_export_tables_and_report(out_dir, self.trace_names)
        self._write_export_manifest(out_dir, "tables", self.trace_names)

        self.status_label.setText(
            f"Exported {len(written_files)} table/report/reload files to {out_dir}"
        )


    def export_all_dialog(self) -> None:
        if not self.traces:
            return
        if not self._analysis_ready_for_trace_names(self.trace_names, action="export all"):
            return
        self._pull_params_from_widgets()

        out_dir = self._make_export_dir("export_all")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_export_tables_and_report(out_dir, self.trace_names)

        images_dir = out_dir / "images"
        traces_dir = images_dir / "traces"
        spikes_dir = images_dir / "spikes"
        plateaus_dir = images_dir / "plateaus"
        recordings_dir = images_dir / "recordings"
        denoising_debug_dir = images_dir / "denoising_debug"
        long_plateau_debug_dir = images_dir / "plateau_debug" / "long"
        summary_dir = images_dir / "summary"
        traces_dir.mkdir(parents=True, exist_ok=True)
        spikes_dir.mkdir(parents=True, exist_ok=True)
        plateaus_dir.mkdir(parents=True, exist_ok=True)
        recordings_dir.mkdir(parents=True, exist_ok=True)
        denoising_debug_dir.mkdir(parents=True, exist_ok=True)
        long_plateau_debug_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        pg.setConfigOptions(antialias=True)
        exported_traces = 0
        denoising_debug_traces = 0
        long_plateau_debug_traces = 0
        all_trace_aligned_pixmaps: List[QtGui.QPixmap] = []
        all_trace_superimposed_pixmaps: List[QtGui.QPixmap] = []
        all_trace_plateau_superimposed_pixmaps: List[QtGui.QPixmap] = []
        all_trace_data_for_combined: List[Tuple[str, np.ndarray, np.ndarray, List[SpikeRecord]]] = []
        all_trace_plateau_data_for_combined: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        startup_exclusion_ms = float(self.detection_params.startup_exclusion_ms)
        for trace_name in self.trace_names:
            state = self.traces[trace_name]
            export_view = _startup_trimmed_trace_export_data(state, startup_exclusion_ms)
            safe_name = "".join(c if c.isalnum() else "_" for c in state.trace_name)
            debug = self._gonzalez_cwt_debug_for_export(state)
            written_debug = _write_gonzalez_cwt_debug_export(
                trace_name=state.trace_name,
                time=state.time,
                corrected=state.corrected,
                debug=debug,
                out_dir=denoising_debug_dir / safe_name,
                startup_exclusion_ms=startup_exclusion_ms,
            )
            if written_debug:
                denoising_debug_traces += 1
            if state.long_plateau_debug is not None:
                written_plateau_debug = _write_long_plateau_debug_export(
                    trace_name=state.trace_name,
                    time=state.time,
                    corrected=state.corrected,
                    debug=dict(state.long_plateau_debug),
                    out_dir=long_plateau_debug_dir / safe_name,
                    startup_exclusion_ms=startup_exclusion_ms,
                )
                if written_plateau_debug:
                    long_plateau_debug_traces += 1
            exported_count, aligned_pix, superimposed_pix = self._export_trace_spike_bundle(
                state,
                out_dir,
                spikes_dir=spikes_dir,
                plateaus_dir=plateaus_dir,
            )
            exported_trace_images = self._export_trace_level_bundle(state, traces_dir)
            plateau_mask_for_export = export_view["plateau_mask"]
            has_plateau_map = bool(
                plateau_mask_for_export is not None and np.any(np.asarray(plateau_mask_for_export, dtype=bool))
            )
            has_failed_plateau_candidates = bool(
                _rejected_long_plateau_rows_for_export(
                    state.long_plateau_debug,
                    int(export_view["startup_idx"]),
                    int(len(export_view["time"])),
                )
            )
            if exported_count <= 0 and exported_trace_images <= 0 and not has_plateau_map and not has_failed_plateau_candidates:
                continue
            exported_traces += 1
            if not aligned_pix.isNull():
                all_trace_aligned_pixmaps.append(aligned_pix)
            if not superimposed_pix.isNull():
                all_trace_superimposed_pixmaps.append(superimposed_pix)
            if has_plateau_map:
                plateau_superimposed_pix = _render_superimposed_plateaus_png(
                    trace_name=state.trace_name,
                    time=export_view["time"],
                    corrected=export_view["corrected"],
                    plateau_mask=np.asarray(plateau_mask_for_export, dtype=bool),
                    width=1400,
                    height=450,
                )
                if not plateau_superimposed_pix.isNull():
                    all_trace_plateau_superimposed_pixmaps.append(plateau_superimposed_pix)
            if export_view["spikes"]:
                all_trace_data_for_combined.append(
                    (trace_name, export_view["time"], export_view["corrected"], export_view["spikes"])
                )
            if has_plateau_map:
                all_trace_plateau_data_for_combined.append(
                    (trace_name, export_view["time"], export_view["corrected"], np.asarray(plateau_mask_for_export, dtype=bool))
                )

        if self.trace_names:
            try:
                from detector.exporter.render_options import RenderOptions
                from detector.exporter.rendering import render_recording_aligned_trace_panels

                panel_rows: List[Dict[str, object]] = []
                panel_options = RenderOptions(
                    normalization_mode=NORMALIZATION_PERCENT_DFF,
                    aligned_panel_channel="denoised",
                    aligned_panel_auto_y=True,
                    aligned_panel_share_y=False,
                )
                for recording_id, names in recording_groups(self.trace_names).items():
                    states = [self.traces[name] for name in names if name in self.traces]
                    pix, metadata = render_recording_aligned_trace_panels(
                        recording_id,
                        states,
                        self.detection_params,
                        panel_options,
                    )
                    if pix.isNull():
                        continue
                    safe_recording = "".join(c if c.isalnum() else "_" for c in str(recording_id))
                    panel_path = recordings_dir / f"{safe_recording}_aligned_trace_panels_percent_dff.png"
                    if pix.save(str(panel_path)):
                        traces_text = metadata.get("traces", [])
                        if isinstance(traces_text, list):
                            traces_text = ";".join(str(value) for value in traces_text)
                        panel_rows.append(
                            {
                                "recording_id": recording_id,
                                "trace_count": metadata.get("trace_count", 0),
                                "traces": traces_text,
                                "image_path": str(panel_path.relative_to(out_dir)),
                                "normalization_mode": metadata.get("normalization_mode", ""),
                                "unit": metadata.get("unit", ""),
                                "source_channel": metadata.get("source_channel", ""),
                                "share_y": metadata.get("share_y", False),
                                "plot_y_label": metadata.get("plot_y_label", ""),
                                "y_limit_low_percentile": metadata.get("y_limit_low_percentile", float("nan")),
                                "y_limit_high_percentile": metadata.get("y_limit_high_percentile", float("nan")),
                                "y_limit_min_span": metadata.get("y_limit_min_span", float("nan")),
                                "time_start": metadata.get("time_start", float("nan")),
                                "time_end": metadata.get("time_end", float("nan")),
                                "y_min": metadata.get("y_min", float("nan")),
                                "y_max": metadata.get("y_max", float("nan")),
                            }
                        )
                if panel_rows:
                    pd.DataFrame(panel_rows).to_csv(
                        out_dir / EXPORT_TABLES_DIR / "recording_aligned_trace_panel_scales.csv",
                        index=False,
                    )
            except Exception as exc:
                print(f"Recording aligned trace panel export failed: {exc}", file=sys.stderr)

            # Grid of per-trace superimposed waveform plots placed side-by-side.
            superimposed_grid = _compose_horizontal_traces_png(
                all_trace_superimposed_pixmaps,
                title=f"All traces | superimposed waveforms (trace count={len(all_trace_superimposed_pixmaps)})",
                column_height=400,
            )
            if not superimposed_grid.isNull():
                superimposed_grid.save(str(summary_dir / "all_traces_superimposed_grid.png"))

            plateau_superimposed_grid = _compose_horizontal_traces_png(
                all_trace_plateau_superimposed_pixmaps,
                title=f"All traces | plateau superimposed per trace (trace count={len(all_trace_plateau_superimposed_pixmaps)})",
                column_height=400,
            )
            if not plateau_superimposed_grid.isNull():
                plateau_superimposed_grid.save(str(summary_dir / "all_traces_plateaus_superimposed_grid.png"))

            # Single plot with all spike waveforms from all traces overlaid together.
            _, _, _, combined_w = _spike_export_dimensions(self.detection_params.spike_zoom_half_window_ms)
            combined_superimposed = _render_superimposed_all_traces_png(
                all_trace_data_for_combined,
                half_window_ms=self.detection_params.spike_zoom_half_window_ms,
                width=combined_w,
                height=500,
            )
            if not combined_superimposed.isNull():
                combined_superimposed.save(str(summary_dir / "all_traces_superimposed_combined.png"))

            combined_aligned = _compose_horizontal_traces_png(
                all_trace_aligned_pixmaps,
                title=f"All traces | aligned stacks (trace count={len(all_trace_aligned_pixmaps)})",
                column_height=300,
            )
            if not combined_aligned.isNull():
                combined_aligned.save(str(summary_dir / "all_traces_aligned.png"))

            combined_plateau_superimposed = _render_superimposed_all_traces_plateaus_png(
                all_trace_plateau_data_for_combined,
                width=1500,
                height=500,
            )
            if not combined_plateau_superimposed.isNull():
                combined_plateau_superimposed.save(str(summary_dir / "all_traces_plateaus_superimposed_combined.png"))

        self._write_export_manifest(out_dir, "all", self.trace_names)
        image_note = f" and images for {exported_traces} traces" if exported_traces > 0 else "; no spike/plateau images were available"
        self.status_label.setText(
            f"Exported all tables, reports, reload files{image_note}; denoising debug for {denoising_debug_traces} traces and long-plateau debug for {long_plateau_debug_traces} traces to {out_dir}"
        )

    def export_current_dialog(self) -> None:
        state = self._current_trace()
        if state is None:
            return
        if not self._analysis_ready_for_trace_names([state.trace_name], action="export current trace"):
            return
        self._pull_params_from_widgets()

        startup_exclusion_ms = float(self.detection_params.startup_exclusion_ms)
        export_view = _startup_trimmed_trace_export_data(state, startup_exclusion_ms)
        has_spikes = bool(export_view["spikes"])
        plateau_mask_for_export = export_view["plateau_mask"]
        has_plateau_map = bool(
            plateau_mask_for_export is not None and np.any(np.asarray(plateau_mask_for_export, dtype=bool))
        )
        has_failed_plateau_candidates = bool(
            _rejected_long_plateau_rows_for_export(
                state.long_plateau_debug,
                int(export_view["startup_idx"]),
                int(len(export_view["time"])),
            )
        )

        safe_name = "".join(c if c.isalnum() else "_" for c in state.trace_name)
        out_dir = self._make_export_dir(f"export_current_{safe_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_export_tables_and_report(out_dir, [state.trace_name])

        images_dir = out_dir / "images"
        traces_dir = images_dir / "traces"
        spikes_dir = images_dir / "spikes"
        plateaus_dir = images_dir / "plateaus"
        denoising_debug_dir = images_dir / "denoising_debug"
        long_plateau_debug_dir = images_dir / "plateau_debug" / "long"
        summary_dir = images_dir / "summary"
        traces_dir.mkdir(parents=True, exist_ok=True)
        spikes_dir.mkdir(parents=True, exist_ok=True)
        plateaus_dir.mkdir(parents=True, exist_ok=True)
        denoising_debug_dir.mkdir(parents=True, exist_ok=True)
        long_plateau_debug_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        pg.setConfigOptions(antialias=True)
        debug = self._gonzalez_cwt_debug_for_export(state)
        _write_gonzalez_cwt_debug_export(
            trace_name=state.trace_name,
            time=state.time,
            corrected=state.corrected,
            debug=debug,
            out_dir=denoising_debug_dir / safe_name,
            startup_exclusion_ms=startup_exclusion_ms,
        )
        if state.long_plateau_debug is not None:
            _write_long_plateau_debug_export(
                trace_name=state.trace_name,
                time=state.time,
                corrected=state.corrected,
                debug=dict(state.long_plateau_debug),
                out_dir=long_plateau_debug_dir / safe_name,
                startup_exclusion_ms=startup_exclusion_ms,
            )
        exported_count = 0
        exported_trace_images = self._export_trace_level_bundle(state, traces_dir)
        if has_spikes or has_plateau_map or has_failed_plateau_candidates:
            exported_count, _aligned_pix, _superimposed_pix = self._export_trace_spike_bundle(
                state,
                out_dir,
                spikes_dir=spikes_dir,
                plateaus_dir=plateaus_dir,
            )
        self._write_export_manifest(out_dir, "current", [state.trace_name])
        image_note = (
            " with images"
            if exported_count > 0 or exported_trace_images > 0 or has_plateau_map or has_failed_plateau_candidates
            else "; no spike/plateau images were available"
        )
        self.status_label.setText(
            f"Exported current trace tables, reports, reload files{image_note}; denoising debug images to {out_dir}"
        )

    def export_png_dialog(self) -> None:
        self.export_all_dialog()

    def export_png_current_dialog(self) -> None:
        self.export_current_dialog()

    def _export_trace_level_bundle(self, state: TraceCuration, traces_dir: Path) -> int:
        safe_name = "".join(c if c.isalnum() else "_" for c in state.trace_name)
        export_view = _startup_trimmed_trace_export_data(state, float(self.detection_params.startup_exclusion_ms))
        time = np.asarray(export_view["time"], dtype=float)
        raw = export_view["raw"]
        corrected = export_view["corrected"]
        denoised = export_view["hybrid_denoised"]
        plateau_mask = export_view["plateau_mask"]
        spikes = export_view["spikes"]
        written = 0

        if raw is not None:
            raw_pix = _render_whole_trace_png(
                trace_name=state.trace_name,
                time=time,
                signal=np.asarray(raw, dtype=float),
                title_suffix="raw trace",
                line_color="#455a64",
                width=1800,
                height=500,
            )
            if not raw_pix.isNull():
                raw_pix.save(str(traces_dir / f"{safe_name}_raw_trace.png"))
                written += 1

        corrected_pix = _render_whole_trace_png(
            trace_name=state.trace_name,
            time=time,
            signal=np.asarray(corrected, dtype=float),
            title_suffix="corrected trace with plateaus",
            line_color="#2979ff",
            plateau_mask=np.asarray(plateau_mask, dtype=bool) if plateau_mask is not None else None,
            width=1800,
            height=500,
        )
        if not corrected_pix.isNull():
            corrected_pix.save(str(traces_dir / f"{safe_name}_corrected_trace_plateaus.png"))
            written += 1

        if denoised is not None:
            denoised_arr = np.asarray(denoised, dtype=float)
            denoised_pix = _render_whole_trace_png(
                trace_name=state.trace_name,
                time=time,
                signal=denoised_arr,
                title_suffix="denoised trace with spikes",
                line_color="#8e24aa",
                spikes=spikes,
                width=1800,
                height=500,
            )
            if not denoised_pix.isNull():
                denoised_pix.save(str(traces_dir / f"{safe_name}_denoised_trace_spikes.png"))
                written += 1
            corrected_arr = np.asarray(corrected, dtype=float)
            if denoised_arr.shape == corrected_arr.shape:
                residual_pix = _render_whole_trace_png(
                    trace_name=state.trace_name,
                    time=time,
                    signal=corrected_arr - denoised_arr,
                    title_suffix="denoise residual",
                    line_color="#d81b60",
                    width=1800,
                    height=500,
                )
                if not residual_pix.isNull():
                    residual_pix.save(str(traces_dir / f"{safe_name}_denoise_residual.png"))
                    written += 1

        if raw is not None:
            raw_arr = np.asarray(raw, dtype=float)
            raw_corrected_pix = _render_raw_processed_comparison_png(
                trace_name=state.trace_name,
                time=time,
                raw=raw_arr,
                processed=np.asarray(corrected, dtype=float),
                processed_label="corrected",
                processed_color="#2979ff",
                width=1800,
                height=700,
            )
            if not raw_corrected_pix.isNull():
                raw_corrected_pix.save(str(traces_dir / f"{safe_name}_raw_vs_corrected.png"))
                written += 1

            if denoised is not None:
                raw_denoised_pix = _render_raw_processed_comparison_png(
                    trace_name=state.trace_name,
                    time=time,
                    raw=raw_arr,
                    processed=np.asarray(denoised, dtype=float),
                    processed_label="denoised",
                    processed_color="#8e24aa",
                    width=1800,
                    height=700,
                )
                if not raw_denoised_pix.isNull():
                    raw_denoised_pix.save(str(traces_dir / f"{safe_name}_raw_vs_denoised.png"))
                    written += 1

        return written

    def _export_trace_spike_bundle(
        self,
        state: TraceCuration,
        out_dir: Path,
        spikes_dir: Optional[Path] = None,
        plateaus_dir: Optional[Path] = None,
    ) -> tuple[int, "QtGui.QPixmap", "QtGui.QPixmap"]:
        """Export per-spike PNGs, superimposed plot, and one stacked aligned PNG for a single trace.

        Returns:
            (spike_count, aligned_pixmap, superimposed_pixmap)
        """
        safe_name = "".join(c if c.isalnum() else "_" for c in state.trace_name)
        spike_out = spikes_dir if spikes_dir is not None else out_dir
        plateau_out = plateaus_dir if plateaus_dir is not None else out_dir
        export_view = _startup_trimmed_trace_export_data(state, float(self.detection_params.startup_exclusion_ms))
        time = export_view["time"]
        corrected = export_view["corrected"]
        hybrid_denoised = export_view["hybrid_denoised"]
        baseline = export_view["baseline"]
        debug_threshold = export_view["debug_threshold"]
        spikes = export_view["spikes"]
        spike_pixmaps: List[QtGui.QPixmap] = []
        half_window_ms = float(self.detection_params.spike_zoom_half_window_ms)
        single_w, single_h, aligned_row_h, _ = _spike_export_dimensions(half_window_ms)

        for spike_idx, spike in enumerate(spikes, 1):
            pix = _render_spike_png(
                time=time,
                corrected=corrected,
                hybrid_denoised=hybrid_denoised,
                baseline=baseline,
                threshold_line=debug_threshold,
                spike=spike,
                half_window_ms=half_window_ms,
                width=single_w,
                height=single_h * 2 if hybrid_denoised is not None else single_h,
            )
            if pix.isNull():
                continue
            spike_pixmaps.append(pix)
            pix.save(str(spike_out / f"{safe_name}_spike_event_{spike_idx:04d}.png"))

        # Render superimposed (overlaid) spikes
        superimposed_pix = _render_superimposed_spikes_png(
            time=time,
            corrected=corrected,
            spikes=spikes,
            half_window_ms=half_window_ms,
            width=single_w,
            height=single_h,
        )
        if not superimposed_pix.isNull():
            superimposed_pix.save(str(spike_out / f"{safe_name}_superimposed.png"))

        aligned_pix = _compose_stacked_spike_png(
            spike_pixmaps,
            title=f"{state.trace_name} | aligned spikes (count={len(spike_pixmaps)})",
            row_height=aligned_row_h,
        )
        if not aligned_pix.isNull():
            aligned_pix.save(str(spike_out / f"{safe_name}_aligned.png"))

        if export_view["plateau_mask"] is not None and np.any(np.asarray(export_view["plateau_mask"], dtype=bool)):
            plateau_mask = np.asarray(export_view["plateau_mask"], dtype=bool)
            long_plateau_mask = (
                np.asarray(export_view["long_plateau_mask"], dtype=bool)
                if export_view["long_plateau_mask"] is not None
                else plateau_mask
            )
            short_plateau_mask = (
                np.asarray(export_view["short_plateau_mask"], dtype=bool)
                if export_view["short_plateau_mask"] is not None
                else np.zeros_like(plateau_mask, dtype=bool)
            )
            burst_plateau_mask = (
                np.asarray(export_view["burst_plateau_mask"], dtype=bool)
                if export_view["burst_plateau_mask"] is not None
                else np.zeros_like(plateau_mask, dtype=bool)
            )
            full_trace_plateau_pix = _render_full_trace_plateau_png(
                trace_name=state.trace_name,
                time=time,
                corrected=corrected,
                plateau_mask=plateau_mask,
                long_plateau_mask=long_plateau_mask,
                short_plateau_mask=short_plateau_mask,
                burst_plateau_mask=burst_plateau_mask,
                width=1800,
                height=500,
            )
            if not full_trace_plateau_pix.isNull():
                full_trace_plateau_pix.save(str(plateau_out / f"{safe_name}_full_trace_plateaus.png"))

            plateau_superimposed_pix = _render_superimposed_plateaus_png(
                trace_name=state.trace_name,
                time=time,
                corrected=corrected,
                plateau_mask=plateau_mask,
                width=1400,
                height=450,
            )
            if not plateau_superimposed_pix.isNull():
                plateau_superimposed_pix.save(str(plateau_out / f"{safe_name}_plateaus_superimposed_onset0.png"))

            plateau_rows = _plateau_rows(
                trace_name=state.trace_name,
                time=time,
                plateau_mask=plateau_mask,
                long_plateau_mask=long_plateau_mask,
                short_plateau_mask=short_plateau_mask,
                burst_plateau_mask=burst_plateau_mask,
                burst_plateau_events=export_view["burst_plateau_events"],
                plateau_signal_classification=export_view["plateau_signal_classification"],
                sampling_rate_hz=_trace_sampling_rate_hz(state),
            )
            if plateau_rows:
                pd.DataFrame(plateau_rows).to_csv(str(plateau_out / f"{safe_name}_plateaus.csv"), index=False)

                regions = _contiguous_true_regions(plateau_mask)
                for idx, (start, end) in enumerate(regions, 1):
                    plateau_source = _plateau_source_for_region(start, end, long_plateau_mask, short_plateau_mask, burst_plateau_mask)
                    signal_type = str(
                        _signal_metrics_for_region(
                            export_view["plateau_signal_classification"],
                            start,
                            end,
                        ).get("signal_type", "unknown")
                    )
                    zoom_pix = _render_plateau_zoom_png(
                        trace_name=state.trace_name,
                        time=time,
                        corrected=corrected,
                        start_idx=int(start),
                        end_idx_exclusive=int(end),
                        plateau_source=plateau_source,
                        signal_type=signal_type,
                        width=1400,
                        height=420,
                    )
                    if not zoom_pix.isNull():
                        zoom_pix.save(str(plateau_out / f"{safe_name}_plateau_zoom_{idx:03d}.png"))
                    zoom_unshaded_pix = _render_plateau_zoom_png(
                        trace_name=state.trace_name,
                        time=time,
                        corrected=corrected,
                        start_idx=int(start),
                        end_idx_exclusive=int(end),
                        plateau_source=plateau_source,
                        signal_type=signal_type,
                        shade_region=False,
                        width=1400,
                        height=420,
                    )
                    if not zoom_unshaded_pix.isNull():
                        zoom_unshaded_pix.save(str(plateau_out / f"{safe_name}_plateau_zoom_{idx:03d}_no_shading.png"))

            short_events = np.asarray(
                state.short_plateau_events if state.short_plateau_events is not None else np.zeros((0, 6), dtype=float),
                dtype=float,
            )
            if short_events.ndim == 2 and short_events.shape[0] > 0 and short_events.shape[1] == 6:
                short_df = pd.DataFrame(
                    short_events,
                    columns=[
                        "onset_index",
                        "offset_index_exclusive",
                        "duration_ms",
                        "median_level",
                        "floor_support_fraction",
                        "confidence",
                    ],
                )
                short_df.insert(0, "trace_name", state.trace_name)
                short_df["onset_index"] = short_df["onset_index"].astype(int)
                short_df["offset_index_exclusive"] = short_df["offset_index_exclusive"].astype(int)
                short_df.to_csv(str(plateau_out / f"{safe_name}_short_plateau_detector_events.csv"), index=False)

            burst_events = np.asarray(
                export_view["burst_plateau_events"],
                dtype=float,
            )
            if burst_events.ndim == 2 and burst_events.shape[0] > 0 and burst_events.shape[1] == 8:
                burst_df = pd.DataFrame(
                    burst_events,
                    columns=[
                        "onset_index",
                        "offset_index_exclusive",
                        "duration_ms",
                        "median_level",
                        "floor_support_fraction",
                        "peak_dominance_noise",
                        "iqr_over_noise",
                        "confidence",
                    ],
                )
                burst_df.insert(0, "trace_name", state.trace_name)
                burst_df["onset_index"] = burst_df["onset_index"].astype(int)
                burst_df["offset_index_exclusive"] = burst_df["offset_index_exclusive"].astype(int)
                burst_df.to_csv(str(plateau_out / f"{safe_name}_burst_plateau_detector_events.csv"), index=False)

            burst_debug_rows = []
            if isinstance(state.burst_plateau_debug, dict):
                raw_rows = state.burst_plateau_debug.get("burst_plateau_candidate_rows", [])
                if isinstance(raw_rows, list):
                    burst_debug_rows = [row for row in raw_rows if isinstance(row, dict)]
            if burst_debug_rows:
                pd.DataFrame(burst_debug_rows).to_csv(
                    str(plateau_out / f"{safe_name}_burst_plateau_candidate_debug.csv"),
                    index=False,
                )

        rejected_rows = _rejected_long_plateau_rows_for_export(
            state.long_plateau_debug,
            int(export_view["startup_idx"]),
            int(len(time)),
        )
        for row in rejected_rows[:50]:
            region_id = int(row.get("region_id", 0))
            start = int(row.get("display_start_index", row.get("start_index", 0)))
            end = int(row.get("display_end_index_exclusive", start + 1))
            candidate_pix = _render_plateau_zoom_png(
                trace_name=state.trace_name,
                time=time,
                corrected=corrected,
                start_idx=start,
                end_idx_exclusive=end,
                plateau_source="rejected",
                shade_region=False,
                width=1400,
                height=420,
            )
            if not candidate_pix.isNull():
                candidate_pix.save(str(plateau_out / f"{safe_name}_failed_candidate_zoom_{region_id:03d}.png"))

        return len(spike_pixmaps), aligned_pix, superimposed_pix


    def _refresh_trace_list(self) -> None:
        selected = self.current_trace_name
        self._sync_recording_selector(selected)
        visible_names = self._visible_trace_names()
        if selected not in visible_names and visible_names:
            selected = visible_names[0]
            self.current_trace_name = selected
        self.trace_list.blockSignals(True)
        self.trace_list.clear()
        for name in visible_names:
            state = self.traces.get(name)
            item = QListWidgetItem(self._trace_list_label(name, state))
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.trace_list.addItem(item)
        if selected in visible_names:
            self.trace_list.setCurrentRow(visible_names.index(str(selected)))
        self.trace_list.blockSignals(False)
        self._update_trace_queue_status()

    def _trace_status_suffix(self, trace_name: str, state: Optional[TraceCuration]) -> str:
        if trace_name in self._last_packaged_trace_names:
            return " [packaged]"
        if state is None:
            return ""
        if state.analysis_status == "pending":
            return " [pending]"
        if state.analysis_status == "running":
            return " [analyzing]"
        if state.analysis_status == "error":
            return " [failed]"
        if state.analysis_status == "done":
            return " [reviewed]"
        return ""

    def _refresh_trace_list_item(self, trace_name: str) -> None:
        visible_names = self._visible_trace_names()
        try:
            row = visible_names.index(trace_name)
        except ValueError:
            self._update_trace_queue_status()
            return
        item = self.trace_list.item(row)
        state = self.traces.get(trace_name)
        if item is None or state is None:
            self._update_trace_queue_status()
            return
        item.setText(self._trace_list_label(trace_name, state))
        item.setData(Qt.ItemDataRole.UserRole, trace_name)
        self._update_trace_queue_status()

    def _source_arrays_for_analysis(self, state: TraceCuration) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        time = (
            np.asarray(state.original_time, dtype=float)
            if state.original_time is not None
            else np.asarray(state.time, dtype=float)
        )
        raw = (
            np.asarray(state.original_raw, dtype=float)
            if state.original_raw is not None
            else np.asarray(state.raw, dtype=float)
        )
        corrected_source = (
            np.asarray(state.original_corrected_source, dtype=float)
            if state.original_corrected_source is not None
            else np.asarray(state.corrected_source, dtype=float)
        )
        sampling_rate_hz = state.original_sampling_rate_hz
        if sampling_rate_hz is None or not np.isfinite(sampling_rate_hz) or float(sampling_rate_hz) <= 0.0:
            sampling_rate_hz = _sampling_rate_hz(time)
        return time, raw, corrected_source, float(sampling_rate_hz)

    def _make_analysis_request(
        self,
        state: TraceCuration,
        template_indices_override: Optional[List[int]] = None,
    ) -> TraceAnalysisRequest:
        source_time, source_raw, source_corrected, source_sampling_rate_hz = self._source_arrays_for_analysis(state)
        detection_params = _clone_detection_params(self.detection_params)
        mode = str(detection_params.detection_mode).strip().lower()
        if mode in {"template_matching", "template-matching", "template matching", "evans", "evans_log_likelihood"}:
            if not list(getattr(detection_params, "template_matching_template_indices", [])):
                detection_params.template_matching_template_indices = list(template_indices_override or [])
        return TraceAnalysisRequest(
            trace_name=state.trace_name,
            time=source_time.copy(),
            raw=source_raw.copy(),
            corrected_source=source_corrected.copy(),
            sampling_rate_hz=source_sampling_rate_hz,
            detection_params=detection_params,
            artifact_params=_clone_artifact_params(self.artifact_params),
            pipeline_config=self._pipeline_config_dict(),
            generation=int(self._analysis_generation),
        )

    def _start_analysis_for_trace(self, trace_name: str, priority: int = 0, force: bool = False) -> None:
        state = self.traces.get(trace_name)
        if state is None:
            return
        if not force and state.analysis_status in {"running", "done"}:
            return
        template_indices = _user_template_indices_from_trace(state)
        if force:
            source_time, source_raw, source_corrected, source_sampling_rate_hz = self._source_arrays_for_analysis(state)
            state.time = source_time.copy()
            state.raw = source_raw.copy()
            state.corrected_source = source_corrected.copy()
            state.corrected = source_corrected.copy()
            state.sampling_rate_hz = source_sampling_rate_hz
            state.hybrid_denoised = None
            state.hybrid_denoised_cache_key = None
            state.hybrid_denoised_pending_key = None
            state.gonzalez_cwt_debug = None
            state.gonzalez_cwt_debug_cache_key = None
            state.baseline = np.zeros_like(state.corrected, dtype=float)
            state.correction_baseline = None
            state.highpass_trace = None
            state.lowpass_trace = None
            state.artifact_mask = None
            state.plateau_mask = None
            state.long_plateau_mask = None
            state.long_plateau_debug = None
            state.short_plateau_mask = None
            state.short_plateau_events = None
            state.burst_plateau_mask = None
            state.burst_plateau_events = None
            state.burst_plateau_debug = None
            state.plateau_signal_classification = None
            state.debug_threshold = None
            state.candidate_windows = []
            state.spikes = []
            state.deleted_spikes = []
            state.bursts = []
            state.deleted_bursts = []
        state.analysis_status = "running"
        state.analysis_error = ""
        state.analysis_generation = int(self._analysis_generation)
        self._refresh_trace_list_item(trace_name)
        request = self._make_analysis_request(state, template_indices_override=template_indices)
        task = _TraceAnalysisTask(request)
        task.signals.finished.connect(self._on_trace_analysis_finished)
        self._analysis_tasks.append(task)
        self._analysis_thread_pool.start(task, priority=int(priority))

    def _apply_saved_session_state_to_trace(self, trace_name: str) -> None:
        if self._pending_session is None or trace_name not in self._pending_session.trace_states:
            return
        if self._pending_session_generation != self._analysis_generation:
            return
        state = self.traces.get(trace_name)
        if state is None:
            return
        trace_session = AppSession(
            excel_path=self._pending_session.excel_path,
            detection_params=self._pending_session.detection_params,
            artifact_params=self._pending_session.artifact_params,
            pipeline_config=dict(self._pending_session.pipeline_config),
            trace_states={trace_name: self._pending_session.trace_states[trace_name]},
        )
        apply_session_to_traces(trace_session, {trace_name: state})

    def _apply_analysis_result(self, state: TraceCuration, result: TraceAnalysisResult) -> None:
        detection = result.detection
        state.spikes = detection.spikes
        state.time = np.asarray(result.time, dtype=float)
        state.raw = np.asarray(result.raw, dtype=float)
        state.corrected_source = np.asarray(result.corrected_source, dtype=float)
        state.corrected = result.corrected
        state.sampling_rate_hz = result.sampling_rate_hz
        state.hybrid_denoised = result.hybrid_denoised
        state.gonzalez_cwt_debug = dict(result.gonzalez_cwt_debug)
        state.gonzalez_cwt_debug_cache_key = self._gonzalez_cwt_debug_cache_key()
        state.hybrid_denoised_cache_key = self._denoise_cache_key()
        state.hybrid_denoised_pending_key = None
        state.smoothed = detection.smoothed
        state.baseline = detection.baseline
        state.correction_baseline = result.correction_baseline
        state.highpass_trace = (
            np.asarray(result.highpass_trace, dtype=float)
            if result.highpass_trace is not None
            else _detection_highpass_trace(detection, self.detection_params.use_highpass_filter)
        )
        state.lowpass_trace = (
            np.asarray(result.lowpass_trace, dtype=float)
            if result.lowpass_trace is not None
            else _detection_lowpass_trace(detection, self.detection_params.use_lowpass_filter)
        )
        state.artifact_mask = np.asarray(result.artifact_mask, dtype=bool)
        state.debug_threshold = detection.threshold
        state.candidate_windows = detection.candidate_windows
        state.plateau_mask = _detection_mask(detection, "baseline_mask")
        state.long_plateau_mask = _detection_mask(detection, "long_plateau_mask")
        state.long_plateau_debug = dict(result.long_plateau_debug)
        state.short_plateau_mask = _detection_mask(detection, "short_plateau_mask", fallback_key="")
        state.short_plateau_events = _detection_short_events(detection)
        state.burst_plateau_mask = _detection_mask(detection, "burst_plateau_mask", fallback_key="")
        state.burst_plateau_events = _detection_burst_events(detection)
        state.burst_plateau_debug = _detection_burst_debug(detection)
        state.plateau_signal_classification = list(result.plateau_signal_classification)
        state.analysis_status = "done"
        state.analysis_error = ""
        self._apply_saved_session_state_to_trace(state.trace_name)
        state.sort_spikes()

    def _on_trace_analysis_finished(
        self,
        trace_name: str,
        generation: int,
        result: Optional[TraceAnalysisResult],
        error: Optional[str],
    ) -> None:
        self._analysis_tasks = [
            task
            for task in self._analysis_tasks
            if not (task.request.trace_name == trace_name and task.request.generation == generation)
        ]
        state = self.traces.get(trace_name)
        if state is None or int(generation) != int(self._analysis_generation):
            return
        if result is None or error is not None:
            state.analysis_status = "error"
            state.analysis_error = str(error or "Unknown analysis error")
            self._refresh_trace_list_item(trace_name)
            if self.current_trace_name == trace_name:
                self._render_current_trace(reset_view=True)
            if self._pending_session is not None and all(
                trace.analysis_status in {"done", "error"} for trace in self.traces.values()
            ):
                self._pending_session = None
                self._pending_session_generation = None
            return

        self._apply_analysis_result(state, result)
        self._refresh_trace_list_item(trace_name)
        if self.current_trace_name == trace_name:
            self._render_current_trace(reset_view=True)
        if self._pending_session is not None and all(
            trace.analysis_status in {"done", "error"} for trace in self.traces.values()
        ):
            self._pending_session = None
            self._pending_session_generation = None

    def _on_recording_selected(self, index: int) -> None:
        if index < 0:
            return
        recording_id = self.recording_selector.itemData(index)
        if recording_id is None:
            return
        self.current_recording_id = str(recording_id)
        visible_names = self._visible_trace_names()
        if visible_names:
            self.current_trace_name = visible_names[0]
        else:
            self.current_trace_name = None
        self._refresh_trace_list()
        if self.current_trace_name:
            self._start_analysis_for_trace(self.current_trace_name, priority=10)
            self._render_current_trace(reset_view=True)

    def on_trace_selected(self, row: int) -> None:
        visible_names = self._visible_trace_names()
        if row < 0 or row >= len(visible_names):
            return
        self.current_trace_name = visible_names[row]
        self._start_analysis_for_trace(self.current_trace_name, priority=10)
        self._render_current_trace(reset_view=True)

    def _current_trace(self) -> Optional[TraceCuration]:
        if self.current_trace_name is None:
            return None
        return self.traces.get(self.current_trace_name)

    def _render_current_trace(self, reset_view: bool = False, refit_y: bool = False) -> None:
        state = self._current_trace()
        if state is None:
            return

        spikes_for_plot = sorted(state.spikes, key=lambda spike: spike.spike_index)
        self.plot.set_overlay_visibility(
            show_raw=self.chk_show_raw.isChecked(),
            show_corrected=self.chk_show_corrected.isChecked(),
            show_baseline=self.chk_show_baseline.isChecked(),
            show_highpass=self.chk_show_highpass.isChecked(),
            show_lowpass=self.chk_show_lowpass.isChecked(),
            show_hybrid_denoised=self.chk_show_hybrid_denoised.isChecked(),
            show_normalized_denoised=self.chk_show_normalized_denoised.isChecked(),
            show_denoise_residual=self.chk_show_denoise_residual.isChecked(),
        )

        raw_for_view = np.asarray(state.raw, dtype=float)
        corrected_for_view = np.asarray(state.corrected, dtype=float)
        raw_baseline_for_view = _raw_baseline_for_display(raw_for_view, state.correction_baseline)
        denoised_for_view = _as_corrected_units(
            state.hybrid_denoised,
            corrected_reference=corrected_for_view,
            correction_baseline=state.correction_baseline,
        )
        if denoised_for_view is not None and state.hybrid_denoised is not None:
            state.hybrid_denoised = np.asarray(denoised_for_view, dtype=float)
        corrected_baseline_for_view = _zero_baseline_for_corrected(corrected_for_view)
        highpass_for_view = state.highpass_trace
        lowpass_for_view = state.lowpass_trace
        normalized_denoised_for_view = (
            normalize_trace_values(denoised_for_view, state, NORMALIZATION_PERCENT_DFF)
            if denoised_for_view is not None
            else None
        )
        display_mode = self._display_normalization_mode()
        if display_mode == NORMALIZATION_PERCENT_DFF:
            raw_f0, _raw_f0_source, _raw_f0_median = f0_values_for_trace(state, raw_for_view.size)
            raw_for_view = normalize_values_with_f0(raw_for_view - raw_f0, raw_f0)
            raw_baseline_for_view = np.zeros_like(raw_for_view, dtype=float)
            corrected_for_view = normalize_trace_values(corrected_for_view, state, display_mode)
            corrected_baseline_for_view = np.zeros_like(corrected_for_view, dtype=float)
            if denoised_for_view is not None:
                denoised_for_view = normalize_trace_values(denoised_for_view, state, display_mode)
            if highpass_for_view is not None:
                highpass_for_view = normalize_trace_values(highpass_for_view, state, display_mode)
            if lowpass_for_view is not None:
                lowpass_for_view = normalize_trace_values(lowpass_for_view, state, display_mode)

        selected_render_source = self._selected_render_source()
        selected_overlay_for_view = self._selected_pipeline_signal_for_display(
            selected_render_source,
            raw=raw_for_view,
            raw_baseline=raw_baseline_for_view,
            corrected=corrected_for_view,
            denoised=denoised_for_view,
            normalized_denoised=normalized_denoised_for_view,
            highpass=highpass_for_view,
            lowpass=lowpass_for_view,
        )
        denoise_residual_for_view = (
            corrected_for_view - denoised_for_view
            if denoised_for_view is not None and np.asarray(denoised_for_view).shape == corrected_for_view.shape
            else None
        )

        self.plot.set_trace(
            time=state.time,
            raw=raw_for_view,
            corrected=corrected_for_view,
            hybrid_denoised=selected_overlay_for_view,
            normalized_denoised=normalized_denoised_for_view,
            denoise_residual=denoise_residual_for_view,
            highpass=highpass_for_view,
            lowpass=lowpass_for_view,
            raw_baseline=raw_baseline_for_view,
            corrected_baseline=corrected_baseline_for_view,
            spikes=spikes_for_plot,
            plateau_mask=state.plateau_mask,
            long_plateau_mask=state.long_plateau_mask,
            short_plateau_mask=state.short_plateau_mask,
            burst_plateau_mask=state.burst_plateau_mask,
            artifact_mask=state.artifact_mask,
            selected_overlay_label=_render_source_label(selected_render_source),
            reset_view=reset_view,
            refit_y=refit_y,
            startup_exclusion_ms=self.detection_params.startup_exclusion_ms,
        )
        self._refresh_table(state)
        if state.analysis_status != "done":
            fs_status = f" | fs: {_trace_sampling_rate_hz(state):.3g} Hz"
            if state.analysis_status == "error":
                self.status_label.setText(
                    f"Trace: {state.trace_name}{fs_status} | analysis error: {state.analysis_error or 'unknown error'}"
                )
            else:
                self.status_label.setText(f"Trace: {state.trace_name}{fs_status} | analysis {state.analysis_status}")
            return
        plateau_count = 0
        if state.plateau_mask is not None:
            plateau_count = len(_contiguous_true_regions(np.asarray(state.plateau_mask, dtype=bool)))
        hybrid_status = ""
        if self.chk_show_hybrid_denoised.isChecked():
            label = _render_source_label(selected_render_source)
            hybrid_status = f" | overlay {label}: shown" if selected_overlay_for_view is not None else f" | overlay {label}: unavailable"
        if self.chk_show_denoise_residual.isChecked():
            hybrid_status += " | residual: shown" if denoise_residual_for_view is not None else ""
        fs_status = f" | fs: {_trace_sampling_rate_hz(state):.3g} Hz"
        self.status_label.setText(
            f"Trace: {state.trace_name}{fs_status} | plateaus: {plateau_count} | spikes: {len(state.spikes)}{hybrid_status}"
        )

    def _gonzalez_denoise_kwargs(self) -> Dict[str, object]:
        return {
            "threshold_sd": float(self.param_gonzalez_threshold_sd.value()),
            "max_clusters": int(self.param_gonzalez_max_clusters.value()),
            "attenuation_min": float(self.param_gonzalez_attenuation_min.value()),
            "attenuation_max": float(self.param_gonzalez_attenuation_max.value()),
            "enable_noise_cluster_rejection": bool(self.param_gonzalez_noise_cluster_rejection.isChecked()),
            "enable_event_template_rejection": bool(self.param_gonzalez_event_template_rejection.isChecked()),
            "enable_denoise_safety_gate": bool(self.detection_params.enable_denoise_safety_gate),
            "min_event_peak_preservation": float(self.detection_params.min_event_peak_preservation),
            "min_local_max_ratio": float(self.detection_params.min_local_max_ratio),
        }

    def _denoise_cache_key(self) -> Tuple[object, ...]:
        denoising_method = str(self.param_denoising_method.currentData() or "gonzalez_full_trace")
        return (
            denoising_method,
            float(self.param_gonzalez_threshold_sd.value()),
            int(self.param_gonzalez_max_clusters.value()),
            float(self.param_gonzalez_attenuation_min.value()),
            float(self.param_gonzalez_attenuation_max.value()),
            bool(self.param_gonzalez_noise_cluster_rejection.isChecked()),
            bool(self.param_gonzalez_event_template_rejection.isChecked()),
            bool(self.detection_params.enable_denoise_safety_gate),
            float(self.detection_params.min_event_peak_preservation),
            float(self.detection_params.min_local_max_ratio),
        )

    def _gonzalez_cwt_debug_cache_key(self) -> Tuple[object, ...]:
        return (
            "gonzalez_full_trace_cwt_debug",
            str(self.param_denoising_method.currentData() or "gonzalez_full_trace"),
            float(self.param_gonzalez_threshold_sd.value()),
            int(self.param_gonzalez_max_clusters.value()),
            float(self.param_gonzalez_attenuation_min.value()),
            float(self.param_gonzalez_attenuation_max.value()),
            bool(self.param_gonzalez_noise_cluster_rejection.isChecked()),
            bool(self.param_gonzalez_event_template_rejection.isChecked()),
            bool(self.detection_params.enable_denoise_safety_gate),
            float(self.detection_params.min_event_peak_preservation),
            float(self.detection_params.min_local_max_ratio),
        )

    def _gonzalez_cwt_debug_for_export(self, state: TraceCuration) -> Dict[str, Any]:
        cache_key = self._gonzalez_cwt_debug_cache_key()
        if (
            state.gonzalez_cwt_debug is not None
            and state.gonzalez_cwt_debug_cache_key == cache_key
        ):
            return dict(state.gonzalez_cwt_debug)
        mode = str(self.param_denoising_method.currentData() or "gonzalez_full_trace")
        if mode == "none":
            debug = dict(
                denoise_trace_gonzalez_full_trace(
                    np.asarray(state.corrected, dtype=float),
                    fs=_trace_sampling_rate_hz(state),
                    denoising_method="none",
                    return_debug=True,
                )
            )
            debug["wrapper_output"] = np.asarray(state.corrected, dtype=float).copy()
            debug["wrapper_correction_component"] = np.zeros_like(np.asarray(state.corrected, dtype=float))
            debug["wrapper_output_mode"] = mode
            debug["wrapper_input_normalization"] = _denoising_input_normalization_label(mode)
            state.gonzalez_cwt_debug = debug
            state.gonzalez_cwt_debug_cache_key = cache_key
            return dict(debug)
        debug = dict(
            denoise_gonzalez_adaptive_wavelet(
                np.asarray(state.corrected, dtype=float),
                fs=_trace_sampling_rate_hz(state),
                input_mode="corrected",
                return_debug=True,
                threshold_sd=float(self.param_gonzalez_threshold_sd.value()),
                max_clusters=int(self.param_gonzalez_max_clusters.value()),
                attenuation_min=float(self.param_gonzalez_attenuation_min.value()),
                attenuation_max=float(self.param_gonzalez_attenuation_max.value()),
                enable_noise_cluster_rejection=bool(self.param_gonzalez_noise_cluster_rejection.isChecked()),
                enable_event_template_rejection=bool(self.param_gonzalez_event_template_rejection.isChecked()),
                enable_denoise_safety_gate=bool(self.detection_params.enable_denoise_safety_gate),
                min_event_peak_preservation=float(self.detection_params.min_event_peak_preservation),
                min_local_max_ratio=float(self.detection_params.min_local_max_ratio),
            )
        )
        if state.hybrid_denoised is not None:
            wrapper_output = np.asarray(state.hybrid_denoised, dtype=float)
            if wrapper_output.shape == np.asarray(state.corrected, dtype=float).shape:
                debug["wrapper_output"] = wrapper_output.copy()
                raw_output = np.asarray(debug.get("output", wrapper_output), dtype=float)
                if raw_output.shape == wrapper_output.shape:
                    debug["wrapper_correction_component"] = wrapper_output - raw_output
                debug["wrapper_output_mode"] = mode
                debug["wrapper_input_normalization"] = _denoising_input_normalization_label(mode)
        state.gonzalez_cwt_debug = debug
        state.gonzalez_cwt_debug_cache_key = cache_key
        return dict(debug)

    def _ensure_denoised(self, state: TraceCuration) -> None:
        cache_key = self._denoise_cache_key()
        if (
            state.hybrid_denoised is not None
            and len(state.hybrid_denoised) == len(state.corrected)
            and state.hybrid_denoised_cache_key == cache_key
        ):
            return
        if state.hybrid_denoised_pending_key == cache_key:
            return
        state.hybrid_denoised_pending_key = cache_key
        denoising_method = str(self.param_denoising_method.currentData() or "gonzalez_full_trace")
        task = _GonzalezDenoiseTask(
            trace_name=state.trace_name,
            cache_key=cache_key,
            corrected=state.corrected,
            fs=_trace_sampling_rate_hz(state),
            baseline=state.baseline,
            denoising_method=denoising_method,
            denoise_kwargs=self._gonzalez_denoise_kwargs(),
            normalization_baseline=state.correction_baseline,
        )
        task.signals.finished.connect(self._on_denoise_finished)
        self._denoise_tasks.append(task)
        self._denoise_thread_pool.start(task)

    def _detect_on_denoised(
        self,
        trace_name: str,
        time: np.ndarray,
        corrected: np.ndarray,
        artifact_mask: np.ndarray,
        sampling_rate_hz: Optional[float] = None,
    ):
        """Detect plateaus on source data, then detect spikes on Gonzalez denoising."""
        params = _clone_detection_params(self.detection_params)
        params.denoising_method = str(self.param_denoising_method.currentData() or params.denoising_method)
        params.gonzalez_threshold_sd = float(self.param_gonzalez_threshold_sd.value())
        params.gonzalez_max_clusters = int(self.param_gonzalez_max_clusters.value())
        params.gonzalez_attenuation_min = float(self.param_gonzalez_attenuation_min.value())
        params.gonzalez_attenuation_max = float(self.param_gonzalez_attenuation_max.value())
        params.gonzalez_enable_noise_cluster_rejection = bool(self.param_gonzalez_noise_cluster_rejection.isChecked())
        params.gonzalez_enable_event_template_rejection = bool(self.param_gonzalez_event_template_rejection.isChecked())
        request = TraceAnalysisRequest(
            trace_name=trace_name,
            time=np.asarray(time, dtype=float),
            raw=np.asarray(corrected, dtype=float),
            corrected_source=np.asarray(corrected, dtype=float),
            sampling_rate_hz=float(sampling_rate_hz) if sampling_rate_hz is not None else _sampling_rate_hz(time),
            detection_params=params,
            artifact_params=_clone_artifact_params(self.artifact_params),
            pipeline_config=self._pipeline_config_dict(),
            generation=int(self._analysis_generation),
        )
        result = analyze_trace_data(request)
        return result.corrected, result.hybrid_denoised, result.correction_baseline, result.detection

    def _on_denoise_finished(
        self,
        trace_name: str,
        cache_key: Tuple[object, ...],
        result: Optional[np.ndarray],
        error: Optional[str],
    ) -> None:
        self._denoise_tasks = [task for task in self._denoise_tasks if task.cache_key != cache_key or task.trace_name != trace_name]
        state = self.traces.get(trace_name)
        if state is None or state.hybrid_denoised_pending_key != cache_key:
            return
        if result is not None:
            result_array = np.asarray(result, dtype=float)
            if result_array.shape != np.asarray(state.corrected).shape or not np.all(np.isfinite(result_array)):
                result = None
                error = "Gonzalez denoising returned invalid data"
        state.hybrid_denoised_pending_key = None
        if error is not None or result is None:
            state.hybrid_denoised = None
            state.hybrid_denoised_cache_key = None
            if self.current_trace_name == trace_name:
                self.chk_show_hybrid_denoised.setChecked(False)
                QMessageBox.warning(self, "Denoising Error", str(error or "Unknown error"))
            return
        result_array = np.asarray(result, dtype=float)
        normalized_result = _as_corrected_units(
            result_array,
            corrected_reference=np.asarray(state.corrected, dtype=float),
            correction_baseline=state.correction_baseline,
        )
        result_array = np.asarray(normalized_result, dtype=float)
        state.hybrid_denoised = result_array
        state.hybrid_denoised_cache_key = cache_key
        if self.current_trace_name == trace_name and self.chk_show_hybrid_denoised.isChecked():
            self._render_current_trace(refit_y=True)

    def _on_overlay_changed(self, _: int) -> None:
        self._render_current_trace(refit_y=True)

    def _refresh_table(self, state: TraceCuration) -> None:
        table_spikes = sorted(state.spikes, key=lambda spike: spike.spike_index)
        self._table_spikes = table_spikes
        signature = tuple(
            (
                str(spike.spike_index),
                f"{spike.spike_time:.2f}",
                f"{spike.amplitude_above_baseline:.2f}",
                f"{spike.snr:.2f}",
                f"{spike.prominence:.2f}",
            )
            for spike in table_spikes
        )
        if signature == self._table_signature:
            return
        self._table_signature = signature
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(table_spikes))
            for row, values in enumerate(signature):
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    self.table.setItem(row, col, item)
        finally:
            self.table.setUpdatesEnabled(True)

    def _nearest_sample_index(self, state: TraceCuration, clicked_time: float) -> int:
        return int(np.argmin(np.abs(state.time - clicked_time)))

    def _nearest_spike_row(self, state: TraceCuration, sample_index: int, tolerance: int = 8) -> Optional[int]:
        if not state.spikes:
            return None
        distances = [abs(spike.spike_index - sample_index) for spike in state.spikes]
        row = int(np.argmin(distances))
        if distances[row] <= tolerance:
            return row
        return None

    def on_plot_clicked(self, clicked_time: float) -> None:
        state = self._current_trace()
        if state is None:
            return
        sample_index = self._nearest_sample_index(state, clicked_time)
        row = self._nearest_spike_row(state, sample_index)
        if row is not None:
            self.table.setCurrentCell(row, 0)

    def on_table_clicked(self, row: int, _: int) -> None:
        state = self._current_trace()
        if state is None or row < 0 or row >= len(self._table_spikes):
            return
        spike = self._table_spikes[row]
        t = spike.spike_time
        half_window_ms = float(self.param_spike_zoom_half_window_ms.value())
        half_window = _half_window_in_time_units(state.time, half_window_ms)
        self.plot.main_plot.setXRange(t - half_window, t + half_window, padding=0)

    def rerun_all(self) -> None:
        if not self.traces:
            return
        self._pull_params_from_widgets()
        self._analysis_generation += 1
        self._pending_session = None
        self._pending_session_generation = None
        current = self.current_trace_name
        for name, state in self.traces.items():
            state.analysis_generation = int(self._analysis_generation)
            state.analysis_status = "pending"
            state.analysis_error = ""
        if current:
            self._start_analysis_for_trace(current, priority=10, force=True)
        for name in self.trace_names:
            if name != current:
                self._start_analysis_for_trace(name, priority=0, force=True)
        self._refresh_trace_list()
        self._render_current_trace()
        self.status_label.setText(f"Re-detecting {len(self.trace_names)} traces in background...")

    def rerun_current(self) -> None:
        state = self._current_trace()
        if state is None:
            return
        self._pull_params_from_widgets()
        self._analysis_generation += 1
        self._pending_session = None
        self._pending_session_generation = None
        state.analysis_generation = int(self._analysis_generation)
        self._start_analysis_for_trace(state.trace_name, priority=10, force=True)
        self._render_current_trace()
        self.status_label.setText(f"Re-detecting {state.trace_name} in background...")

    def next_trace(self) -> None:
        row = self.trace_list.currentRow()
        if row < len(self._visible_trace_names()) - 1:
            self.trace_list.setCurrentRow(row + 1)

    def prev_trace(self) -> None:
        row = self.trace_list.currentRow()
        if row > 0:
            self.trace_list.setCurrentRow(row - 1)

    def jump_next_spike(self) -> None:
        row = self.table.currentRow()
        if row < self.table.rowCount() - 1:
            self.table.setCurrentCell(row + 1, 0)
            self.on_table_clicked(row + 1, 0)

    def jump_prev_spike(self) -> None:
        row = self.table.currentRow()
        if row > 0:
            self.table.setCurrentCell(row - 1, 0)
            self.on_table_clicked(row - 1, 0)

    def _filtered_rows(self) -> List[int]:
        return list(range(len(self._table_spikes)))

    def jump_next_filtered(self) -> None:
        rows = self._filtered_rows()
        if not rows:
            return
        current = self.table.currentRow()
        for row in rows:
            if row > current:
                self.table.setCurrentCell(row, 0)
                self.on_table_clicked(row, 0)
                return
        self.table.setCurrentCell(rows[0], 0)
        self.on_table_clicked(rows[0], 0)

    def jump_prev_filtered(self) -> None:
        rows = self._filtered_rows()
        if not rows:
            return
        current = self.table.currentRow()
        for row in reversed(rows):
            if row < current:
                self.table.setCurrentCell(row, 0)
                self.on_table_clicked(row, 0)
                return
        self.table.setCurrentCell(rows[-1], 0)
        self.on_table_clicked(rows[-1], 0)
