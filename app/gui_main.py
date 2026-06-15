from __future__ import annotations

from dataclasses import dataclass
import json
import datetime
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

from .detection import DETECTOR_BUILD_TAG, detect_spikes
from .hybrid_denoising import denoise_hybrid_fluorescence_trace, denoise_trace_state_guided_low_plateau_tv
from .io import estimate_sampling_rate_hz, load_excel_traces
from .models import AppSession, ArtifactParams, DetectionParams, SpikeRecord, TraceCuration
from .plot_widget import TracePlotWidget
from .preprocessing import apply_artifact_interpolation
from .session_io import apply_session_to_traces, load_session, save_session
from .shortcuts import bind_shortcuts


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
    spike_x = 0.0  # Spike is at t=0

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

    # Mark the spike
    scatter = pg.ScatterPlotItem(x=[spike_x], y=[spike_y], size=10, brush=pg.mkBrush("#f1c40f"), pen=pg.mkPen("#ffffff", width=1.0))
    main_plot.addItem(scatter)

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
        denoised_plot.addItem(pg.ScatterPlotItem(x=[0.0], y=[denoised_y], size=10, brush=pg.mkBrush("#f1c40f"), pen=pg.mkPen("#ffffff", width=1.0)))
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
    spike_ys_all = []
    spike_xs_all = []
    
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
        
        # Track spike points for scatter plot
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

    # Mark all spikes
    if spike_xs_all:
        scatter = pg.ScatterPlotItem(x=spike_xs_all, y=spike_ys_all, size=8, brush=pg.mkBrush("#f1c40f99"), pen=pg.mkPen("#ffffff", width=0.5))
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
) -> str:
    long_hit = False
    short_hit = False
    if long_plateau_mask is not None:
        lm = np.asarray(long_plateau_mask, dtype=bool)
        if lm.size > start:
            long_hit = bool(np.any(lm[start : min(end, lm.size)]))
    if short_plateau_mask is not None:
        sm = np.asarray(short_plateau_mask, dtype=bool)
        if sm.size > start:
            short_hit = bool(np.any(sm[start : min(end, sm.size)]))
    if long_hit:
        return "long"
    if short_hit:
        return "short"
    return "unknown"


def _plateau_code_rows(
    trace_name: str,
    time: np.ndarray,
    plateau_mask: np.ndarray,
    long_plateau_mask: Optional[np.ndarray] = None,
    short_plateau_mask: Optional[np.ndarray] = None,
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
        rows.append(
            {
                "trace_name": trace_name,
                "plateau_code": f"P{idx:03d}",
                "detector_source": _plateau_source_for_region(start, end, long_plateau_mask, short_plateau_mask),
                "start_index": int(start),
                "end_index_exclusive": int(end),
                "start_time": float(time[s]),
                "end_time": float(time[e]),
                "sample_count": int(max(0, end - start)),
                "duration_time_units": float(time[e] - time[s]),
            }
        )
    return rows


def _render_full_trace_plateau_png(
    trace_name: str,
    time: np.ndarray,
    corrected: np.ndarray,
    plateau_mask: np.ndarray,
    long_plateau_mask: Optional[np.ndarray] = None,
    short_plateau_mask: Optional[np.ndarray] = None,
    spikes: Optional[List[SpikeRecord]] = None,
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
    regions = _contiguous_true_regions(mask)

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
    plot.setTitle(f"{trace_name} | Full trace with plateau labels")
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
        label_y = y_max + (0.02 * max(1e-6, (y_max - y_min)))
    else:
        label_y = 0.0

    # Overlay detected spikes so plateau export shows spike morphology context.
    if spikes:
        spike_points: List[tuple[float, float, int]] = []
        for spike in sorted(spikes, key=lambda s: int(s.spike_index)):
            idx = int(spike.spike_index)
            if 0 <= idx < n and np.isfinite(t[idx]) and np.isfinite(y[idx]):
                spike_points.append((float(t[idx]), float(y[idx]), idx))

        if spike_points:
            sx = [p[0] for p in spike_points]
            sy = [p[1] for p in spike_points]
            scatter = pg.ScatterPlotItem(
                x=sx,
                y=sy,
                size=8,
                brush=pg.mkBrush("#f1c40f"),
                pen=pg.mkPen("#7a5c00", width=1.0),
            )
            plot.addItem(scatter)

            y_span = max(1e-6, float(y_max - y_min)) if finite.size > 0 else 1.0
            for i, (sx_i, sy_i, idx_i) in enumerate(spike_points, 1):
                dy = 0.035 * y_span if (i % 2 == 1) else 0.06 * y_span
                label = pg.TextItem(
                    f"S{i:03d}",
                    color="#7a5c00",
                    fill=pg.mkColor("#fff8d6e6"),
                    border=pg.mkColor("#c8a600"),
                    anchor=(0.5, 1.0),
                )
                plot.addItem(label)
                label.setPos(sx_i, sy_i + dy)

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

    for idx, (start, end) in enumerate(regions, 1):
        s = max(0, min(start, n - 1))
        e = max(s, min(end - 1, n - 1))
        x0 = float(t[s])
        x1 = float(t[e])

        label = pg.TextItem(
            f"P{idx:03d}",
            color="#7f1d1d",
            fill=pg.mkColor("#ffffffd9"),
            border=pg.mkColor("#c0392b"),
            anchor=(0.5, 1.0),
        )
        plot.addItem(label)
        label.setPos((x0 + x1) * 0.5, label_y)

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
    plateau_code: str,
    time: np.ndarray,
    corrected: np.ndarray,
    start_idx: int,
    end_idx_exclusive: int,
    plateau_source: str = "long",
    spikes: Optional[List[SpikeRecord]] = None,
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
    plateau_span = max(0.0, end_t - onset_t)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    dt_med = float(np.median(dt)) if dt.size else 1.0
    pad = max(10.0 * dt_med, 0.25 * plateau_span)

    win_lo_t = onset_t - pad
    win_hi_t = end_t + pad
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
    plot.setTitle(f"{trace_name} | {plateau_code} zoom (onset=0)")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    plot.plot(x_rel, y_win, pen=pg.mkPen("#2979ff", width=1.8))

    is_short = str(plateau_source).lower() == "short"
    region_brush = pg.mkBrush(21, 101, 192, 50) if is_short else pg.mkBrush(231, 76, 60, 55)
    region_pen = pg.mkPen("#1565c0", width=1.1) if is_short else pg.mkPen("#c0392b", width=1.1)
    label_color = "#0f3d73" if is_short else "#7f1d1d"
    label_border = pg.mkColor("#1565c0") if is_short else pg.mkColor("#c0392b")

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
        y_line = y_max + 0.02 * max(1e-6, (y_max - y_min))
    else:
        y_line = 0.0

    plot.plot([0.0, 0.0], [float(np.min(finite)) if finite.size else -1.0, float(np.max(finite)) if finite.size else 1.0], pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine))
    onset_label = pg.TextItem(
        "onset=0",
        color=label_color,
        fill=pg.mkColor("#ffffffd9"),
        border=label_border,
        anchor=(0.5, 1.0),
    )
    plot.addItem(onset_label)
    onset_label.setPos(0.0, y_line)

    # Overlay spikes in this zoom window and label those inside the plateau span.
    if spikes:
        spike_points: List[tuple[float, float, int, bool]] = []
        for spike in sorted(spikes, key=lambda s: int(s.spike_index)):
            idx = int(spike.spike_index)
            if lo <= idx < hi and np.isfinite(t[idx]) and np.isfinite(y[idx]):
                x_rel_i = float(t[idx] - onset_t)
                in_plateau = bool(s <= idx < e_exclusive)
                spike_points.append((x_rel_i, float(y[idx]), idx, in_plateau))

        if spike_points:
            sx = [p[0] for p in spike_points]
            sy = [p[1] for p in spike_points]
            scatter = pg.ScatterPlotItem(
                x=sx,
                y=sy,
                size=8,
                brush=pg.mkBrush("#f1c40f"),
                pen=pg.mkPen("#7a5c00", width=1.0),
            )
            plot.addItem(scatter)

            if finite.size > 0:
                y_span = max(1e-6, float(np.max(finite) - np.min(finite)))
            else:
                y_span = 1.0

            label_idx = 1
            for x_i, y_i, _idx_i, in_plateau in spike_points:
                if not in_plateau:
                    continue
                dy = 0.04 * y_span if (label_idx % 2 == 1) else 0.07 * y_span
                label = pg.TextItem(
                    f"S{label_idx:03d}",
                    color="#7a5c00",
                    fill=pg.mkColor("#fff8d6e6"),
                    border=pg.mkColor("#c8a600"),
                    anchor=(0.5, 1.0),
                )
                plot.addItem(label)
                label.setPos(x_i, y_i + dy)
                label_idx += 1

    if x_rel.size >= 2:
        plot.setXRange(float(x_rel[0]), float(x_rel[-1]), padding=0)

    layout.addWidget(plot)
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

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    dt_med = float(np.median(dt)) if dt.size else 1.0

    for start, end_exclusive in regions:
        s = max(0, min(int(start), n - 1))
        e_exclusive = max(s + 1, min(int(end_exclusive), n))
        e = max(s, e_exclusive - 1)

        onset_t = float(t[s])
        end_t = float(t[e])
        span = max(0.0, end_t - onset_t)
        pad = max(10.0 * dt_med, 0.25 * span)

        lo_t = onset_t - pad
        hi_t = end_t + pad
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
    plot.setTitle(f"{trace_name} | plateau superimposed (onset=0)")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)

    all_values: List[float] = []
    x_min = float("inf")
    x_max = float("-inf")
    for t_win, y_win in windows:
        plot.plot(t_win, y_win, pen=pg.mkPen("#2979ff66", width=1.2))
        arr = np.asarray(y_win, dtype=float)
        all_values.extend([v for v in arr.flat if np.isfinite(v)])
        if t_win.size > 0:
            x_min = min(x_min, float(np.min(t_win)))
            x_max = max(x_max, float(np.max(t_win)))

    if all_values:
        y_vals = np.asarray(all_values, dtype=float)
        y_lo = float(np.min(y_vals))
        y_hi = float(np.max(y_vals))
        y_pad = max(1e-3, (y_hi - y_lo) * 0.10) if y_hi > y_lo else 1e-3
        plot.setYRange(y_lo - y_pad, y_hi + y_pad, padding=0)
        plot.plot([0.0, 0.0], [y_lo - y_pad, y_hi + y_pad], pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine))

    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        plot.setXRange(x_min, x_max, padding=0)

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
    plot.setTitle("All traces | plateau superimposed (onset=0, a.u.)")
    plot.setMinimumHeight(height)
    plot.setMaximumHeight(height)
    plot.addLegend(offset=(10, 8))

    all_values: List[float] = []
    x_min = float("inf")
    x_max = float("-inf")
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
            if t_win.size > 0:
                x_min = min(x_min, float(np.min(t_win)))
                x_max = max(x_max, float(np.max(t_win)))

    if all_values:
        y_vals = np.asarray(all_values, dtype=float)
        y_lo = float(np.min(y_vals))
        y_hi = float(np.max(y_vals))
        y_pad = max(1e-3, (y_hi - y_lo) * 0.10) if y_hi > y_lo else 1e-3
        plot.setYRange(y_lo - y_pad, y_hi + y_pad, padding=0)
        plot.plot([0.0, 0.0], [y_lo - y_pad, y_hi + y_pad], pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine))

    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        plot.setXRange(x_min, x_max, padding=0)

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
    "plateau_id",
    "detector_source",
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


def _time_to_ms_scale(time: np.ndarray) -> float:
    t = np.asarray(time, dtype=float)
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    dt_med = float(np.median(dt))
    sample_period_ms = _time_sample_period_ms(t)
    if not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
        return 1.0
    return float(sample_period_ms / dt_med)


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
                    "spike_id": f"S{idx:04d}",
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

        n = min(y.size, b.size, t.size, m.size)
        if n <= 2:
            continue

        y = y[:n]
        b = b[:n]
        t = t[:n]
        m = m[:n]
        if not np.any(m):
            continue

        sample_period_ms = _time_sample_period_ms(t, _trace_sampling_rate_hz(state))
        if not np.isfinite(sample_period_ms) or sample_period_ms <= 0.0:
            continue

        regions = _contiguous_true_regions(m)
        for idx, (start, end) in enumerate(regions, 1):
            if end <= start:
                continue

            amp = y[start:end] - b[start:end]
            amp_f = amp[np.isfinite(amp)]

            onset_time_ms = float(start * sample_period_ms)
            duration_ms = float((end - start) * sample_period_ms)

            fwhm_samples = _event_fwhm_samples(amp)
            fwhm_ms = float(fwhm_samples * sample_period_ms) if np.isfinite(fwhm_samples) else float("nan")

            rise_s, decay_s = _rise_decay_time_samples(amp)
            rise_time_ms = float(rise_s * sample_period_ms) if np.isfinite(rise_s) else float("nan")
            decay_time_ms = float(decay_s * sample_period_ms) if np.isfinite(decay_s) else float("nan")

            plateau_sd = float(np.std(amp_f, ddof=0)) if amp_f.size >= 2 else float("nan")
            mean_amp = float(np.mean(amp_f)) if amp_f.size > 0 else float("nan")
            peak_amp = float(np.max(amp_f)) if amp_f.size > 0 else float("nan")

            rows.append(
                {
                    "trace_id": str(trace_name),
                    "plateau_id": f"P{idx:03d}",
                    "detector_source": _plateau_source_for_region(start, end, long_mask, short_mask),
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
            )

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
    corrected: np.ndarray
    hybrid_denoised: np.ndarray
    correction_baseline: np.ndarray
    detection: object
    artifact_mask: np.ndarray
    sampling_rate_hz: float


@dataclass(frozen=True)
class TraceAnalysisRequest:
    trace_name: str
    time: np.ndarray
    raw: np.ndarray
    corrected_source: np.ndarray
    sampling_rate_hz: float
    detection_params: DetectionParams
    artifact_params: ArtifactParams
    generation: int


def _clone_detection_params(params: DetectionParams) -> DetectionParams:
    return DetectionParams.from_dict(params.to_dict())


def _clone_artifact_params(params: ArtifactParams) -> ArtifactParams:
    return ArtifactParams.from_dict(params.to_dict())


def _analysis_low_state_kwargs(params: DetectionParams) -> Dict[str, object]:
    return {
        "threshold_sd": float(params.gonzalez_threshold_sd),
        "max_clusters": int(params.gonzalez_max_clusters),
        "attenuation_min": float(params.gonzalez_attenuation_min),
        "attenuation_max": float(params.gonzalez_attenuation_max),
        "enable_noise_cluster_rejection": bool(params.gonzalez_enable_noise_cluster_rejection),
        "enable_event_template_rejection": bool(params.gonzalez_enable_event_template_rejection),
        "enable_low_state_safety_gate": bool(params.enable_low_state_safety_gate),
        "min_event_peak_preservation": float(params.min_event_peak_preservation),
        "min_local_max_ratio": float(params.min_local_max_ratio),
        "min_reliable_low_state_fraction": float(params.min_reliable_low_state_fraction),
        "min_reliable_low_state_samples": int(params.min_reliable_low_state_samples),
        "plateau_tv_min_duration_ms": float(params.plateau_tv_min_duration_ms),
        "plateau_tv_max_weight_factor": float(params.plateau_tv_max_weight_factor),
    }


def analyze_trace_data(
    request: TraceAnalysisRequest,
    legacy_denoise_fn: Callable[..., np.ndarray] = denoise_hybrid_fluorescence_trace,
) -> TraceAnalysisResult:
    """Run artifact interpolation, plateau detection, denoising, and spike detection for one trace."""
    time = np.asarray(request.time, dtype=float)
    corrected_source = np.asarray(request.corrected_source, dtype=float)
    sampling_rate_hz = (
        float(request.sampling_rate_hz)
        if np.isfinite(request.sampling_rate_hz) and float(request.sampling_rate_hz) > 0.0
        else _sampling_rate_hz(time)
    )
    corrected, artifact_mask = apply_artifact_interpolation(
        time,
        corrected_source,
        params=request.artifact_params.to_dict(),
    )

    baseline_result = detect_spikes(
        request.trace_name,
        time,
        corrected,
        request.detection_params,
        artifact_mask=artifact_mask,
        sampling_rate_hz=sampling_rate_hz,
    )
    baseline = np.asarray(baseline_result.baseline, dtype=float)
    computation_source = np.asarray(baseline_result.processed_signal, dtype=float)
    plateau_mask = _detection_mask(baseline_result, "long_plateau_mask")
    low_state_denoiser = str(request.detection_params.low_state_denoiser or "gonzalez_adaptive_wavelet")
    corrected_baseline_corrected = computation_source - baseline
    # State-guided denoising is used as a detection aid. The returned
    # corrected trace below remains the preferred source for quantitative
    # amplitude, width, area, plateau level, and export summary measurements.
    denoised_baseline_corrected = np.asarray(
        denoise_trace_state_guided_low_plateau_tv(
            corrected_baseline_corrected,
            fs=sampling_rate_hz,
            plateau_mask=plateau_mask,
            plateau_tv_weight=float(request.detection_params.plateau_tv_weight),
            boundary_protect_ms=float(request.detection_params.low_state_boundary_protect_ms),
            low_state_denoiser=low_state_denoiser,
            low_state_denoise_fn=legacy_denoise_fn if low_state_denoiser == "legacy_hybrid" else None,
            **_analysis_low_state_kwargs(request.detection_params),
        ),
        dtype=float,
    )
    detection = detect_spikes(
        request.trace_name,
        time,
        denoised_baseline_corrected,
        request.detection_params,
        artifact_mask=artifact_mask,
        baseline_override=np.zeros_like(baseline),
        plateau_mask_override=plateau_mask,
        sampling_rate_hz=sampling_rate_hz,
        short_plateau_trace_override=corrected_baseline_corrected,
    )
    detection.qc_plot_data["highpass_signal"] = np.asarray(
        baseline_result.qc_plot_data.get("highpass_signal", np.zeros_like(corrected)),
        dtype=float,
    )
    return TraceAnalysisResult(
        corrected=corrected_baseline_corrected,
        hybrid_denoised=denoised_baseline_corrected,
        correction_baseline=baseline,
        detection=detection,
        artifact_mask=np.asarray(artifact_mask, dtype=bool),
        sampling_rate_hz=sampling_rate_hz,
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
        legacy_denoise_fn: Callable[..., np.ndarray],
    ) -> None:
        super().__init__()
        self.request = request
        self.legacy_denoise_fn = legacy_denoise_fn
        self.signals = _TraceAnalysisWorkerSignals()

    def run(self) -> None:
        try:
            result = analyze_trace_data(self.request, legacy_denoise_fn=self.legacy_denoise_fn)
            self.signals.finished.emit(self.request.trace_name, self.request.generation, result, None)
        except Exception as exc:
            self.signals.finished.emit(self.request.trace_name, self.request.generation, None, str(exc))


class _HybridWorkerSignals(QObject):
    finished = Signal(str, object, object, object)


class _HybridDenoiseTask(QRunnable):
    def __init__(
        self,
        trace_name: str,
        cache_key: Tuple[object, ...],
        corrected: np.ndarray,
        fs: float,
        baseline: Optional[np.ndarray],
        plateau_mask: Optional[np.ndarray],
        low_state_denoiser: str,
        low_state_kwargs: Dict[str, object],
        plateau_tv_weight: float,
        boundary_protect_ms: float,
        legacy_denoise_fn: Callable[..., np.ndarray],
    ) -> None:
        super().__init__()
        self.trace_name = trace_name
        self.cache_key = cache_key
        self.corrected = np.asarray(corrected, dtype=float).copy()
        self.fs = float(fs)
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=float).copy()
        self.plateau_mask = None if plateau_mask is None else np.asarray(plateau_mask, dtype=bool).copy()
        self.low_state_denoiser = str(low_state_denoiser)
        self.low_state_kwargs = dict(low_state_kwargs)
        self.plateau_tv_weight = float(plateau_tv_weight)
        self.boundary_protect_ms = float(boundary_protect_ms)
        self.legacy_denoise_fn = legacy_denoise_fn
        self.signals = _HybridWorkerSignals()

    def run(self) -> None:
        try:
            denoise_input = self.corrected
            if self.baseline is not None and self.baseline.shape == self.corrected.shape:
                denoise_input = self.corrected - self.baseline
            plateau_mask = self.plateau_mask
            if plateau_mask is None or plateau_mask.shape != denoise_input.shape:
                plateau_mask = np.zeros(denoise_input.shape, dtype=bool)
            result = np.asarray(
                denoise_trace_state_guided_low_plateau_tv(
                    denoise_input,
                    fs=self.fs,
                    plateau_mask=plateau_mask,
                    plateau_tv_weight=self.plateau_tv_weight,
                    boundary_protect_ms=self.boundary_protect_ms,
                    low_state_denoiser=self.low_state_denoiser,
                    low_state_denoise_fn=self.legacy_denoise_fn if self.low_state_denoiser == "legacy_hybrid" else None,
                    **self.low_state_kwargs,
                ),
                dtype=float,
            )
            self.signals.finished.emit(self.trace_name, self.cache_key, result, None)
        except Exception as exc:
            self.signals.finished.emit(self.trace_name, self.cache_key, None, str(exc))


class SpikeCurationMainWindow(QMainWindow):
    def __init__(
        self,
        hybrid_denoise_fn: Callable[..., np.ndarray] = denoise_hybrid_fluorescence_trace,
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
        self._table_spikes: List[SpikeRecord] = []
        self._hybrid_denoise_fn = hybrid_denoise_fn if denoise_fn is None else denoise_fn
        self._hybrid_thread_pool = QThreadPool(self)
        self._hybrid_tasks: List[_HybridDenoiseTask] = []
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

        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)

        top_bar = QHBoxLayout()
        root_layout.addLayout(top_bar)

        self.btn_load_excel = QPushButton("Load Excel")
        self.btn_load_exported = QPushButton("Load Exported Result")
        self.btn_load_session = QPushButton("Load Session")
        self.btn_save_session = QPushButton("Save Session")
        self.btn_export = QPushButton("Export All")
        self.btn_export_png_current = QPushButton("Export Current")
        self.btn_reload_params_cfg = QPushButton("Reload Params Config")
        self.btn_rerun_current = QPushButton("Re-detect Current")
        self.btn_rerun_all = QPushButton("Re-detect All")

        for button in [
            self.btn_load_excel,
            self.btn_load_exported,
            self.btn_load_session,
            self.btn_save_session,
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

        self.trace_list = QListWidget()
        left_layout.addWidget(QLabel("Traces"))
        left_layout.addWidget(self.trace_list, stretch=1)

        nav_layout = QHBoxLayout()
        self.btn_prev_trace = QPushButton("Prev Trace")
        self.btn_next_trace = QPushButton("Next Trace")
        nav_layout.addWidget(self.btn_prev_trace)
        nav_layout.addWidget(self.btn_next_trace)
        left_layout.addLayout(nav_layout)

        params_group = QGroupBox("Detection Params")
        params_layout = QFormLayout(params_group)

        self.param_detection_mode = QComboBox()
        self.param_detection_mode.addItem("MAD threshold", "mad")
        self.param_detection_mode.addItem("STD threshold", "std")
        self.param_detection_mode.addItem("SNR threshold", "snr")
        self.param_baseline_mode = QComboBox()
        self.param_baseline_mode.addItem("Percentile", "percentile")
        self.param_baseline_mode.addItem("Savitzky-Golay", "savgol")
        self.param_startup_exclusion_ms = self._make_double_spin(0.0, 1000.0, self.detection_params.startup_exclusion_ms, 1.0)
        self.param_baseline_rolling_window_ms = self._make_double_spin(5.0, 5000.0, self.detection_params.baseline_rolling_window_ms, 5.0)
        self.param_baseline_rolling_percentile = self._make_double_spin(1.0, 50.0, self.detection_params.baseline_rolling_percentile, 1.0)
        self.param_baseline_savgol_window_ms = self._make_double_spin(5.0, 5000.0, self.detection_params.baseline_savgol_window_ms, 5.0)
        self.param_baseline_savgol_polyorder = self._make_int_spin(1, 7, self.detection_params.baseline_savgol_polyorder)
        self.param_use_highpass_filter = QCheckBox("Enable high-pass preprocessing")
        self.param_use_highpass_filter.setChecked(bool(self.detection_params.use_highpass_filter))
        self.param_highpass_cutoff_hz = self._make_double_spin(0.001, 1000.0, self.detection_params.highpass_cutoff_hz, 0.1)
        self.param_highpass_order = self._make_int_spin(1, 10, self.detection_params.highpass_order)
        self.param_plateau_median_window_ms = self._make_double_spin(5.0, 500.0, self.detection_params.plateau_median_window_ms, 1.0)
        self.param_spike_noise_k = self._make_double_spin(0.1, 100.0, self.detection_params.spike_noise_k, 0.1)
        self.param_spike_prominence_k = self._make_double_spin(0.0, 100.0, self.detection_params.spike_prominence_k, 0.1)
        self.param_min_separation_ms = self._make_double_spin(1.0, 1000.0, self.detection_params.min_separation_ms, 1.0)
        self.param_spike_zoom_half_window_ms = self._make_double_spin(0.5, 500.0, self.detection_params.spike_zoom_half_window_ms, 0.5)

        params_layout.addRow("detection_mode", self.param_detection_mode)
        params_layout.addRow("startup_exclusion_ms", self.param_startup_exclusion_ms)
        params_layout.addRow("baseline_mode", self.param_baseline_mode)
        params_layout.addRow("baseline_rolling_window_ms", self.param_baseline_rolling_window_ms)
        params_layout.addRow("baseline_rolling_percentile", self.param_baseline_rolling_percentile)
        params_layout.addRow("baseline_savgol_window_ms", self.param_baseline_savgol_window_ms)
        params_layout.addRow("baseline_savgol_polyorder", self.param_baseline_savgol_polyorder)
        params_layout.addRow(self.param_use_highpass_filter)
        params_layout.addRow("highpass_cutoff_hz", self.param_highpass_cutoff_hz)
        params_layout.addRow("highpass_order", self.param_highpass_order)
        params_layout.addRow("plateau_median_window_ms", self.param_plateau_median_window_ms)
        params_layout.addRow("spike_threshold_k", self.param_spike_noise_k)
        params_layout.addRow("spike_prominence_k", self.param_spike_prominence_k)
        params_layout.addRow("min_separation_ms", self.param_min_separation_ms)
        params_layout.addRow("spike_zoom_half_window_ms", self.param_spike_zoom_half_window_ms)
        left_layout.addWidget(params_group)

        artifact_group = QGroupBox("Artifact Params")
        artifact_layout = QFormLayout(artifact_group)

        self.param_artifact_abs_low = self._make_double_spin(0.0, 100000.0, self.artifact_params.absolute_low, 1.0)
        self.param_artifact_min_depth = self._make_double_spin(0.0, 100000.0, self.artifact_params.min_depth, 0.5)
        self.param_artifact_drop_start = self._make_double_spin(0.0, 100000.0, self.artifact_params.drop_start_abs, 0.5)

        artifact_layout.addRow("Low Max", self.param_artifact_abs_low)
        artifact_layout.addRow("Min Depth", self.param_artifact_min_depth)
        artifact_layout.addRow("Drop Start", self.param_artifact_drop_start)
        left_layout.addWidget(artifact_group)

        overlay_group = QGroupBox("Signal Overlays")
        overlay_layout = QFormLayout(overlay_group)

        self.chk_show_raw = QCheckBox("Show raw")
        self.chk_show_corrected = QCheckBox("Show corrected")
        self.chk_show_baseline = QCheckBox("Show baseline")
        self.chk_show_highpass = QCheckBox("Show high-pass")
        self.chk_show_hybrid_denoised = QCheckBox("Show state-guided denoised")

        self.chk_show_raw.setChecked(True)
        self.chk_show_corrected.setChecked(True)
        self.chk_show_baseline.setChecked(True)
        self.chk_show_highpass.setChecked(False)
        self.chk_show_hybrid_denoised.setChecked(False)

        overlay_layout.addRow(self.chk_show_raw)
        overlay_layout.addRow(self.chk_show_corrected)
        overlay_layout.addRow(self.chk_show_baseline)
        overlay_layout.addRow(self.chk_show_highpass)
        overlay_layout.addRow(self.chk_show_hybrid_denoised)
        left_layout.addWidget(overlay_group)

        denoise_group = QGroupBox("State-Guided Denoising")
        denoise_layout = QFormLayout(denoise_group)
        self.param_low_state_denoiser = QComboBox()
        self.param_low_state_denoiser.addItem("Gonzalez adaptive wavelet", "gonzalez_adaptive_wavelet")
        self.param_low_state_denoiser.addItem("Gonzalez full trace (no plateau mask)", "gonzalez_full_trace")
        self.param_low_state_denoiser.addItem("Legacy Hybrid", "legacy_hybrid")
        self.param_low_state_denoiser.addItem("No low-state denoising", "none")
        low_state_index = self.param_low_state_denoiser.findData(str(self.detection_params.low_state_denoiser))
        self.param_low_state_denoiser.setCurrentIndex(max(0, low_state_index))
        self.param_gonzalez_threshold_sd = self._make_double_spin(
            0.0,
            10.0,
            self.detection_params.gonzalez_threshold_sd,
            0.1,
        )
        self.param_gonzalez_max_clusters = self._make_int_spin(
            1,
            100,
            int(self.detection_params.gonzalez_max_clusters),
        )
        self.param_gonzalez_attenuation_min = self._make_double_spin(
            0.0,
            1.0,
            self.detection_params.gonzalez_attenuation_min,
            0.05,
        )
        self.param_gonzalez_attenuation_max = self._make_double_spin(
            0.0,
            1.0,
            self.detection_params.gonzalez_attenuation_max,
            0.05,
        )
        self.param_gonzalez_noise_cluster_rejection = QCheckBox("Noise-cluster rejection")
        self.param_gonzalez_noise_cluster_rejection.setChecked(
            bool(self.detection_params.gonzalez_enable_noise_cluster_rejection)
        )
        self.param_gonzalez_event_template_rejection = QCheckBox("Event-template cleanup")
        self.param_gonzalez_event_template_rejection.setChecked(
            bool(self.detection_params.gonzalez_enable_event_template_rejection)
        )
        self.param_plateau_tv_weight = self._make_double_spin(
            0.0,
            100.0,
            self.detection_params.plateau_tv_weight,
            0.1,
        )
        self.param_plateau_tv_weight.setToolTip("Used by state-aware Gonzalez and Legacy Hybrid modes.")
        self.param_low_state_boundary_protect_ms = self._make_double_spin(
            0.0,
            500.0,
            self.detection_params.low_state_boundary_protect_ms,
            1.0,
        )
        denoise_layout.addRow("Low-state denoising method", self.param_low_state_denoiser)
        denoise_layout.addRow("Gonzalez threshold SD", self.param_gonzalez_threshold_sd)
        denoise_layout.addRow("Gonzalez max clusters", self.param_gonzalez_max_clusters)
        denoise_layout.addRow("Gonzalez attenuation min", self.param_gonzalez_attenuation_min)
        denoise_layout.addRow("Gonzalez attenuation max", self.param_gonzalez_attenuation_max)
        denoise_layout.addRow(self.param_gonzalez_noise_cluster_rejection)
        denoise_layout.addRow(self.param_gonzalez_event_template_rejection)
        denoise_layout.addRow("Plateau TV strength", self.param_plateau_tv_weight)
        denoise_layout.addRow("Boundary protection (ms)", self.param_low_state_boundary_protect_ms)
        left_layout.addWidget(denoise_group)

        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plot = TracePlotWidget()
        right_layout.addWidget(self.plot, stretch=5)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["idx", "time", "amp", "snr", "prom"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { gridline-color: #666666; }"
            "QTableWidget::item { color: #111111; }"
            "QHeaderView::section { background-color: #30343a; color: #f5f7fa; padding: 4px; border: 1px solid #4b4f56; }"
        )
        right_layout.addWidget(self.table, stretch=1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([280, 1200])

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

    def _connect_signals(self) -> None:
        self.btn_load_excel.clicked.connect(self.load_excel_dialog)
        self.btn_load_exported.clicked.connect(self.load_exported_dialog)
        self.btn_load_session.clicked.connect(self.load_session_dialog)
        self.btn_save_session.clicked.connect(self.save_session_dialog)
        self.btn_export.clicked.connect(self.export_all_dialog)
        self.btn_export_png_current.clicked.connect(self.export_current_dialog)
        self.btn_reload_params_cfg.clicked.connect(self.reload_detection_params_config)
        self.btn_rerun_current.clicked.connect(self.rerun_current)
        self.btn_rerun_all.clicked.connect(self.rerun_all)

        self.btn_prev_trace.clicked.connect(self.prev_trace)
        self.btn_next_trace.clicked.connect(self.next_trace)

        self.trace_list.currentRowChanged.connect(self.on_trace_selected)
        self.plot.clicked.connect(self.on_plot_clicked)
        self.table.cellClicked.connect(self.on_table_clicked)
        self.chk_show_raw.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_corrected.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_baseline.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_highpass.stateChanged.connect(self._on_overlay_changed)
        self.chk_show_hybrid_denoised.stateChanged.connect(self._on_overlay_changed)
        self.param_low_state_denoiser.currentIndexChanged.connect(self._update_denoise_control_state)
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
                "Ctrl+E": self.export_all_dialog,
                "Ctrl+Shift+E": self.export_current_dialog,
                "Q": lambda: self._toggle_checkbox(self.chk_show_raw),
                "W": lambda: self._toggle_checkbox(self.chk_show_corrected),
            },
        )

    def _toggle_checkbox(self, checkbox: QCheckBox) -> None:
        checkbox.setChecked(not checkbox.isChecked())

    def _pull_params_from_widgets(self) -> None:
        self.detection_params.detection_mode = str(self.param_detection_mode.currentData() or "mad")
        self.detection_params.baseline_mode = str(self.param_baseline_mode.currentData() or "percentile")
        self.detection_params.startup_exclusion_ms = float(self.param_startup_exclusion_ms.value())
        self.detection_params.baseline_rolling_window_ms = float(self.param_baseline_rolling_window_ms.value())
        self.detection_params.baseline_rolling_percentile = float(self.param_baseline_rolling_percentile.value())
        self.detection_params.baseline_savgol_window_ms = float(self.param_baseline_savgol_window_ms.value())
        self.detection_params.baseline_savgol_polyorder = int(self.param_baseline_savgol_polyorder.value())
        self.detection_params.use_highpass_filter = bool(self.param_use_highpass_filter.isChecked())
        self.detection_params.highpass_cutoff_hz = float(self.param_highpass_cutoff_hz.value())
        self.detection_params.highpass_order = int(self.param_highpass_order.value())
        self.detection_params.plateau_median_window_ms = float(self.param_plateau_median_window_ms.value())
        self.detection_params.spike_noise_k = float(self.param_spike_noise_k.value())
        self.detection_params.spike_prominence_k = float(self.param_spike_prominence_k.value())
        self.detection_params.min_separation_ms = float(self.param_min_separation_ms.value())
        self.detection_params.spike_zoom_half_window_ms = float(self.param_spike_zoom_half_window_ms.value())
        self.detection_params.low_state_denoiser = str(
            self.param_low_state_denoiser.currentData() or "gonzalez_adaptive_wavelet"
        )
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
        self.detection_params.plateau_tv_weight = float(self.param_plateau_tv_weight.value())
        self.detection_params.low_state_boundary_protect_ms = float(self.param_low_state_boundary_protect_ms.value())
        self.artifact_params.absolute_low = float(self.param_artifact_abs_low.value())
        self.artifact_params.min_depth = float(self.param_artifact_min_depth.value())
        self.artifact_params.drop_start_abs = float(self.param_artifact_drop_start.value())

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
        self.param_use_highpass_filter.setChecked(bool(self.detection_params.use_highpass_filter))
        self.param_highpass_cutoff_hz.setValue(float(self.detection_params.highpass_cutoff_hz))
        self.param_highpass_order.setValue(int(self.detection_params.highpass_order))
        self.param_plateau_median_window_ms.setValue(self.detection_params.plateau_median_window_ms)
        self.param_spike_noise_k.setValue(self.detection_params.spike_noise_k)
        self.param_spike_prominence_k.setValue(self.detection_params.spike_prominence_k)
        self.param_min_separation_ms.setValue(self.detection_params.min_separation_ms)
        self.param_spike_zoom_half_window_ms.setValue(self.detection_params.spike_zoom_half_window_ms)
        low_state_index = self.param_low_state_denoiser.findData(str(self.detection_params.low_state_denoiser))
        self.param_low_state_denoiser.setCurrentIndex(max(0, low_state_index))
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
        self.param_plateau_tv_weight.setValue(float(self.detection_params.plateau_tv_weight))
        self.param_low_state_boundary_protect_ms.setValue(float(self.detection_params.low_state_boundary_protect_ms))
        self._update_denoise_control_state()
        self.param_artifact_abs_low.setValue(self.artifact_params.absolute_low)
        self.param_artifact_min_depth.setValue(self.artifact_params.min_depth)
        self.param_artifact_drop_start.setValue(self.artifact_params.drop_start_abs)

    def _update_denoise_control_state(self, *_args: object) -> None:
        low_state_denoiser = str(self.param_low_state_denoiser.currentData() or "gonzalez_adaptive_wavelet")
        state_aware_mode = low_state_denoiser in {"gonzalez_adaptive_wavelet", "legacy_hybrid"}
        full_trace_mode = low_state_denoiser == "gonzalez_full_trace"
        self.param_plateau_tv_weight.setEnabled(state_aware_mode)
        self.param_plateau_tv_weight.setToolTip(
            "Used by state-aware Gonzalez and Legacy Hybrid modes."
            if state_aware_mode
            else (
                "Ignored by full-trace Gonzalez mode."
                if full_trace_mode
                else "Ignored when denoising is disabled."
            )
        )
        self.param_low_state_boundary_protect_ms.setEnabled(state_aware_mode)
        self.param_low_state_boundary_protect_ms.setToolTip(
            "Used by state-aware Gonzalez and Legacy Hybrid modes."
            if state_aware_mode
            else (
                "Ignored by full-trace Gonzalez mode."
                if full_trace_mode
                else "Ignored when denoising is disabled."
            )
        )

    def _result_root(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent / "Result"
        return Path(__file__).resolve().parents[2] / "Result"

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
        return Path(__file__).resolve().parents[1] / "detection_params.json"

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
        self._push_params_to_widgets()
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
            trace_cols = [col for col in corrected_df.columns if col != "time"]
            if not trace_cols:
                raise ValueError("No trace columns found in corrected_traces.csv")

            self.traces.clear()
            for trace_name in trace_cols:
                signal = pd.to_numeric(corrected_df[trace_name], errors="coerce").to_numpy(dtype=float)
                state = TraceCuration(
                    trace_name=trace_name,
                    raw=signal.copy(),
                    corrected_source=signal.copy(),
                    corrected=signal.copy(),
                    time=time,
                    sampling_rate_hz=_sampling_rate_hz(time),
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
            self._refresh_trace_list()
            if self.trace_names:
                self.trace_list.setCurrentRow(0)
            self.status_label.setText(f"Loaded exported result: {folder_path.name} | traces: {len(self.trace_names)}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Export Error", str(exc))

    def load_excel(self, file_path: str, pending_session: Optional[AppSession] = None) -> None:
        self._pull_params_from_widgets()
        self._analysis_generation += 1
        generation = self._analysis_generation
        self.excel_path = file_path
        self.traces.clear()
        self.trace_names = []
        self.current_trace_name = None
        self._table_spikes = []
        self._analysis_tasks = []
        self._pending_session = pending_session
        self._pending_session_generation = generation if pending_session is not None else None
        self.trace_list.clear()
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
        if self._pending_session is not None:
            apply_session_to_traces(self._pending_session, self.traces)
        self._refresh_trace_list()
        if self.trace_names:
            self.trace_list.setCurrentRow(0)
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
            self._push_params_to_widgets()

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
        save_session(file_path, self.excel_path, self.detection_params, self.artifact_params, self.traces)
        self.status_label.setText(f"Session saved: {Path(file_path).name}")

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
        spikes_dir = images_dir / "spikes"
        plateaus_dir = images_dir / "plateaus"
        summary_dir = images_dir / "summary"
        spikes_dir.mkdir(parents=True, exist_ok=True)
        plateaus_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        pg.setConfigOptions(antialias=True)
        exported_traces = 0
        all_trace_aligned_pixmaps: List[QtGui.QPixmap] = []
        all_trace_superimposed_pixmaps: List[QtGui.QPixmap] = []
        all_trace_plateau_superimposed_pixmaps: List[QtGui.QPixmap] = []
        all_trace_data_for_combined: List[Tuple[str, np.ndarray, np.ndarray, List[SpikeRecord]]] = []
        all_trace_plateau_data_for_combined: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for trace_name in self.trace_names:
            state = self.traces[trace_name]
            exported_count, aligned_pix, superimposed_pix = self._export_trace_spike_bundle(
                state,
                out_dir,
                spikes_dir=spikes_dir,
                plateaus_dir=plateaus_dir,
            )
            has_plateau_map = bool(state.plateau_mask is not None and np.any(np.asarray(state.plateau_mask, dtype=bool)))
            if exported_count <= 0 and not has_plateau_map:
                continue
            exported_traces += 1
            if not aligned_pix.isNull():
                all_trace_aligned_pixmaps.append(aligned_pix)
            if not superimposed_pix.isNull():
                all_trace_superimposed_pixmaps.append(superimposed_pix)
            if state.plateau_mask is not None and np.any(np.asarray(state.plateau_mask, dtype=bool)):
                plateau_superimposed_pix = _render_superimposed_plateaus_png(
                    trace_name=state.trace_name,
                    time=state.time,
                    corrected=state.corrected,
                    plateau_mask=np.asarray(state.plateau_mask, dtype=bool),
                    width=1400,
                    height=450,
                )
                if not plateau_superimposed_pix.isNull():
                    all_trace_plateau_superimposed_pixmaps.append(plateau_superimposed_pix)
            if state.spikes:
                all_trace_data_for_combined.append((trace_name, state.time, state.corrected, state.spikes))
            if state.plateau_mask is not None and np.any(np.asarray(state.plateau_mask, dtype=bool)):
                all_trace_plateau_data_for_combined.append(
                    (trace_name, state.time, state.corrected, np.asarray(state.plateau_mask, dtype=bool))
                )

        if exported_traces > 0:
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
            f"Exported all tables, reports, reload files{image_note} to {out_dir}"
        )

    def export_current_dialog(self) -> None:
        state = self._current_trace()
        if state is None:
            return
        if not self._analysis_ready_for_trace_names([state.trace_name], action="export current trace"):
            return
        self._pull_params_from_widgets()

        has_spikes = bool(state.spikes)
        has_plateau_map = bool(state.plateau_mask is not None and np.any(np.asarray(state.plateau_mask, dtype=bool)))

        safe_name = "".join(c if c.isalnum() else "_" for c in state.trace_name)
        out_dir = self._make_export_dir(f"export_current_{safe_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_export_tables_and_report(out_dir, [state.trace_name])

        images_dir = out_dir / "images"
        spikes_dir = images_dir / "spikes"
        plateaus_dir = images_dir / "plateaus"
        summary_dir = images_dir / "summary"
        spikes_dir.mkdir(parents=True, exist_ok=True)
        plateaus_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        pg.setConfigOptions(antialias=True)
        exported_count = 0
        if has_spikes or has_plateau_map:
            exported_count, _aligned_pix, _superimposed_pix = self._export_trace_spike_bundle(
                state,
                out_dir,
                spikes_dir=spikes_dir,
                plateaus_dir=plateaus_dir,
            )
        self._write_export_manifest(out_dir, "current", [state.trace_name])
        image_note = " with images" if exported_count > 0 or has_plateau_map else "; no spike/plateau images were available"
        self.status_label.setText(
            f"Exported current trace tables, reports, reload files{image_note} to {out_dir}"
        )

    def export_png_dialog(self) -> None:
        self.export_all_dialog()

    def export_png_current_dialog(self) -> None:
        self.export_current_dialog()

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
        spike_pixmaps: List[QtGui.QPixmap] = []
        half_window_ms = float(self.detection_params.spike_zoom_half_window_ms)
        single_w, single_h, aligned_row_h, _ = _spike_export_dimensions(half_window_ms)

        for spike_idx, spike in enumerate(state.spikes, 1):
            pix = _render_spike_png(
                time=state.time,
                corrected=state.corrected,
                hybrid_denoised=state.hybrid_denoised,
                baseline=state.baseline,
                threshold_line=state.debug_threshold,
                spike=spike,
                half_window_ms=half_window_ms,
                width=single_w,
                height=single_h * 2 if state.hybrid_denoised is not None else single_h,
            )
            if pix.isNull():
                continue
            spike_pixmaps.append(pix)
            pix.save(str(spike_out / f"{safe_name}_spike{spike_idx:04d}.png"))

        # Render superimposed (overlaid) spikes
        superimposed_pix = _render_superimposed_spikes_png(
            time=state.time,
            corrected=state.corrected,
            spikes=state.spikes,
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

        if state.plateau_mask is not None and np.any(np.asarray(state.plateau_mask, dtype=bool)):
            plateau_mask = np.asarray(state.plateau_mask, dtype=bool)
            long_plateau_mask = (
                np.asarray(state.long_plateau_mask, dtype=bool)
                if state.long_plateau_mask is not None
                else plateau_mask
            )
            short_plateau_mask = (
                np.asarray(state.short_plateau_mask, dtype=bool)
                if state.short_plateau_mask is not None
                else np.zeros_like(plateau_mask, dtype=bool)
            )
            full_trace_plateau_pix = _render_full_trace_plateau_png(
                trace_name=state.trace_name,
                time=state.time,
                corrected=state.corrected,
                plateau_mask=plateau_mask,
                long_plateau_mask=long_plateau_mask,
                short_plateau_mask=short_plateau_mask,
                spikes=state.spikes,
                width=1800,
                height=500,
            )
            if not full_trace_plateau_pix.isNull():
                full_trace_plateau_pix.save(str(plateau_out / f"{safe_name}_full_trace_plateau_codes.png"))

            plateau_superimposed_pix = _render_superimposed_plateaus_png(
                trace_name=state.trace_name,
                time=state.time,
                corrected=state.corrected,
                plateau_mask=plateau_mask,
                width=1400,
                height=450,
            )
            if not plateau_superimposed_pix.isNull():
                plateau_superimposed_pix.save(str(plateau_out / f"{safe_name}_plateaus_superimposed_onset0.png"))

            plateau_rows = _plateau_code_rows(
                trace_name=state.trace_name,
                time=state.time,
                plateau_mask=plateau_mask,
                long_plateau_mask=long_plateau_mask,
                short_plateau_mask=short_plateau_mask,
            )
            if plateau_rows:
                pd.DataFrame(plateau_rows).to_csv(str(plateau_out / f"{safe_name}_plateau_codes.csv"), index=False)

                regions = _contiguous_true_regions(plateau_mask)
                for idx, (start, end) in enumerate(regions, 1):
                    plateau_code = f"P{idx:03d}"
                    plateau_source = _plateau_source_for_region(start, end, long_plateau_mask, short_plateau_mask)
                    zoom_pix = _render_plateau_zoom_png(
                        trace_name=state.trace_name,
                        plateau_code=plateau_code,
                        time=state.time,
                        corrected=state.corrected,
                        start_idx=int(start),
                        end_idx_exclusive=int(end),
                        plateau_source=plateau_source,
                        spikes=state.spikes,
                        width=1400,
                        height=420,
                    )
                    if not zoom_pix.isNull():
                        zoom_pix.save(str(plateau_out / f"{safe_name}_{plateau_code}_zoom_onset0.png"))

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

        return len(spike_pixmaps), aligned_pix, superimposed_pix


    def _refresh_trace_list(self) -> None:
        selected = self.current_trace_name
        self.trace_list.blockSignals(True)
        self.trace_list.clear()
        for name in self.trace_names:
            state = self.traces.get(name)
            suffix = ""
            if state is not None:
                if state.analysis_status == "pending":
                    suffix = " [pending]"
                elif state.analysis_status == "running":
                    suffix = " [analyzing]"
                elif state.analysis_status == "error":
                    suffix = " [error]"
            item = QListWidgetItem(f"{name}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.trace_list.addItem(item)
        if selected in self.trace_names:
            self.trace_list.setCurrentRow(self.trace_names.index(selected))
        self.trace_list.blockSignals(False)

    def _refresh_trace_list_item(self, trace_name: str) -> None:
        try:
            row = self.trace_names.index(trace_name)
        except ValueError:
            return
        item = self.trace_list.item(row)
        state = self.traces.get(trace_name)
        if item is None or state is None:
            return
        suffix = ""
        if state.analysis_status == "pending":
            suffix = " [pending]"
        elif state.analysis_status == "running":
            suffix = " [analyzing]"
        elif state.analysis_status == "error":
            suffix = " [error]"
        item.setText(f"{trace_name}{suffix}")
        item.setData(Qt.ItemDataRole.UserRole, trace_name)

    def _make_analysis_request(self, state: TraceCuration) -> TraceAnalysisRequest:
        return TraceAnalysisRequest(
            trace_name=state.trace_name,
            time=np.asarray(state.time, dtype=float).copy(),
            raw=np.asarray(state.raw, dtype=float).copy(),
            corrected_source=np.asarray(state.corrected_source, dtype=float).copy(),
            sampling_rate_hz=_trace_sampling_rate_hz(state),
            detection_params=_clone_detection_params(self.detection_params),
            artifact_params=_clone_artifact_params(self.artifact_params),
            generation=int(self._analysis_generation),
        )

    def _start_analysis_for_trace(self, trace_name: str, priority: int = 0, force: bool = False) -> None:
        state = self.traces.get(trace_name)
        if state is None:
            return
        if not force and state.analysis_status in {"running", "done"}:
            return
        if force:
            state.corrected = np.asarray(state.corrected_source, dtype=float).copy()
            state.hybrid_denoised = None
            state.hybrid_denoised_cache_key = None
            state.hybrid_denoised_pending_key = None
            state.baseline = np.zeros_like(state.corrected, dtype=float)
            state.correction_baseline = None
            state.highpass_trace = None
            state.artifact_mask = None
            state.plateau_mask = None
            state.long_plateau_mask = None
            state.short_plateau_mask = None
            state.short_plateau_events = None
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
        request = self._make_analysis_request(state)
        task = _TraceAnalysisTask(request, self._hybrid_denoise_fn)
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
            trace_states={trace_name: self._pending_session.trace_states[trace_name]},
        )
        apply_session_to_traces(trace_session, {trace_name: state})

    def _apply_analysis_result(self, state: TraceCuration, result: TraceAnalysisResult) -> None:
        detection = result.detection
        state.spikes = detection.spikes
        state.corrected = result.corrected
        state.sampling_rate_hz = result.sampling_rate_hz
        state.hybrid_denoised = result.hybrid_denoised
        state.hybrid_denoised_cache_key = self._hybrid_cache_key()
        state.hybrid_denoised_pending_key = None
        state.smoothed = detection.smoothed
        state.baseline = detection.baseline
        state.correction_baseline = result.correction_baseline
        state.highpass_trace = _detection_highpass_trace(detection, self.detection_params.use_highpass_filter)
        state.artifact_mask = np.asarray(result.artifact_mask, dtype=bool)
        state.debug_threshold = detection.threshold
        state.candidate_windows = detection.candidate_windows
        state.plateau_mask = _detection_mask(detection, "baseline_mask")
        state.long_plateau_mask = _detection_mask(detection, "long_plateau_mask")
        state.short_plateau_mask = _detection_mask(detection, "short_plateau_mask", fallback_key="")
        state.short_plateau_events = _detection_short_events(detection)
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

    def on_trace_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.trace_names):
            return
        self.current_trace_name = self.trace_names[row]
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
            show_hybrid_denoised=self.chk_show_hybrid_denoised.isChecked(),
        )

        raw_for_view = np.asarray(state.raw, dtype=float)
        corrected_for_view = np.asarray(state.corrected, dtype=float)
        raw_baseline_for_view = _raw_baseline_for_display(raw_for_view, state.correction_baseline)
        hybrid_for_view = _as_corrected_units(
            state.hybrid_denoised,
            corrected_reference=corrected_for_view,
            correction_baseline=state.correction_baseline,
        )
        if hybrid_for_view is not None and state.hybrid_denoised is not None:
            state.hybrid_denoised = np.asarray(hybrid_for_view, dtype=float)
        corrected_baseline_for_view = _zero_baseline_for_corrected(corrected_for_view)

        self.plot.set_trace(
            time=state.time,
            raw=raw_for_view,
            corrected=corrected_for_view,
            hybrid_denoised=hybrid_for_view,
            highpass=state.highpass_trace,
            raw_baseline=raw_baseline_for_view,
            corrected_baseline=corrected_baseline_for_view,
            spikes=spikes_for_plot,
            plateau_mask=state.plateau_mask,
            long_plateau_mask=state.long_plateau_mask,
            short_plateau_mask=state.short_plateau_mask,
            artifact_mask=state.artifact_mask,
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
        short_count = 0
        if state.short_plateau_mask is not None:
            short_count = len(_contiguous_true_regions(np.asarray(state.short_plateau_mask, dtype=bool)))
        hybrid_status = ""
        if self.chk_show_hybrid_denoised.isChecked():
            hybrid_status = " | state-guided denoised: shown" if state.hybrid_denoised is not None else ""
        fs_status = f" | fs: {_trace_sampling_rate_hz(state):.3g} Hz"
        self.status_label.setText(
            f"Trace: {state.trace_name}{fs_status} | plateaus: {plateau_count} (short: {short_count}) | spikes: {len(state.spikes)}{hybrid_status}"
        )

    def _low_state_denoise_kwargs(self) -> Dict[str, object]:
        return {
            "threshold_sd": float(self.param_gonzalez_threshold_sd.value()),
            "max_clusters": int(self.param_gonzalez_max_clusters.value()),
            "attenuation_min": float(self.param_gonzalez_attenuation_min.value()),
            "attenuation_max": float(self.param_gonzalez_attenuation_max.value()),
            "enable_noise_cluster_rejection": bool(self.param_gonzalez_noise_cluster_rejection.isChecked()),
            "enable_event_template_rejection": bool(self.param_gonzalez_event_template_rejection.isChecked()),
            "enable_low_state_safety_gate": bool(self.detection_params.enable_low_state_safety_gate),
            "min_event_peak_preservation": float(self.detection_params.min_event_peak_preservation),
            "min_local_max_ratio": float(self.detection_params.min_local_max_ratio),
            "min_reliable_low_state_fraction": float(self.detection_params.min_reliable_low_state_fraction),
            "min_reliable_low_state_samples": int(self.detection_params.min_reliable_low_state_samples),
            "plateau_tv_min_duration_ms": float(self.detection_params.plateau_tv_min_duration_ms),
            "plateau_tv_max_weight_factor": float(self.detection_params.plateau_tv_max_weight_factor),
        }

    def _hybrid_cache_key(self) -> Tuple[object, ...]:
        low_state_denoiser = str(self.param_low_state_denoiser.currentData() or "gonzalez_adaptive_wavelet")
        state_aware_mode = low_state_denoiser in {"gonzalez_adaptive_wavelet", "legacy_hybrid"}
        plateau_tv_weight = (
            float(self.param_plateau_tv_weight.value()) if state_aware_mode else None
        )
        boundary_protect_ms = (
            float(self.param_low_state_boundary_protect_ms.value()) if state_aware_mode else None
        )
        return (
            low_state_denoiser,
            float(self.param_gonzalez_threshold_sd.value()),
            int(self.param_gonzalez_max_clusters.value()),
            float(self.param_gonzalez_attenuation_min.value()),
            float(self.param_gonzalez_attenuation_max.value()),
            bool(self.param_gonzalez_noise_cluster_rejection.isChecked()),
            bool(self.param_gonzalez_event_template_rejection.isChecked()),
            bool(self.detection_params.enable_low_state_safety_gate),
            float(self.detection_params.min_event_peak_preservation),
            float(self.detection_params.min_local_max_ratio),
            float(self.detection_params.min_reliable_low_state_fraction),
            int(self.detection_params.min_reliable_low_state_samples),
            float(self.detection_params.plateau_tv_min_duration_ms),
            float(self.detection_params.plateau_tv_max_weight_factor),
            plateau_tv_weight,
            boundary_protect_ms,
        )

    def _ensure_hybrid_denoised(self, state: TraceCuration) -> None:
        cache_key = self._hybrid_cache_key()
        if (
            state.hybrid_denoised is not None
            and len(state.hybrid_denoised) == len(state.corrected)
            and state.hybrid_denoised_cache_key == cache_key
        ):
            return
        if state.hybrid_denoised_pending_key == cache_key:
            return
        state.hybrid_denoised_pending_key = cache_key
        low_state_denoiser = str(self.param_low_state_denoiser.currentData() or "gonzalez_adaptive_wavelet")
        plateau_mask = state.long_plateau_mask if state.long_plateau_mask is not None else state.plateau_mask
        task = _HybridDenoiseTask(
            trace_name=state.trace_name,
            cache_key=cache_key,
            corrected=state.corrected,
            fs=_trace_sampling_rate_hz(state),
            baseline=state.baseline,
            plateau_mask=plateau_mask,
            low_state_denoiser=low_state_denoiser,
            low_state_kwargs=self._low_state_denoise_kwargs(),
            plateau_tv_weight=float(self.param_plateau_tv_weight.value()),
            boundary_protect_ms=float(self.param_low_state_boundary_protect_ms.value()),
            legacy_denoise_fn=self._hybrid_denoise_fn,
        )
        task.signals.finished.connect(self._on_hybrid_denoise_finished)
        self._hybrid_tasks.append(task)
        self._hybrid_thread_pool.start(task)

    def _detect_on_hybrid_denoised(
        self,
        trace_name: str,
        time: np.ndarray,
        corrected: np.ndarray,
        artifact_mask: np.ndarray,
        sampling_rate_hz: Optional[float] = None,
    ):
        """Detect plateaus on source data, then detect spikes on state-guided denoising."""
        params = _clone_detection_params(self.detection_params)
        params.low_state_denoiser = str(self.param_low_state_denoiser.currentData() or params.low_state_denoiser)
        params.gonzalez_threshold_sd = float(self.param_gonzalez_threshold_sd.value())
        params.gonzalez_max_clusters = int(self.param_gonzalez_max_clusters.value())
        params.gonzalez_attenuation_min = float(self.param_gonzalez_attenuation_min.value())
        params.gonzalez_attenuation_max = float(self.param_gonzalez_attenuation_max.value())
        params.gonzalez_enable_noise_cluster_rejection = bool(self.param_gonzalez_noise_cluster_rejection.isChecked())
        params.gonzalez_enable_event_template_rejection = bool(self.param_gonzalez_event_template_rejection.isChecked())
        params.plateau_tv_weight = float(self.param_plateau_tv_weight.value())
        params.low_state_boundary_protect_ms = float(self.param_low_state_boundary_protect_ms.value())
        request = TraceAnalysisRequest(
            trace_name=trace_name,
            time=np.asarray(time, dtype=float),
            raw=np.asarray(corrected, dtype=float),
            corrected_source=np.asarray(corrected, dtype=float),
            sampling_rate_hz=float(sampling_rate_hz) if sampling_rate_hz is not None else _sampling_rate_hz(time),
            detection_params=params,
            artifact_params=_clone_artifact_params(self.artifact_params),
            generation=int(self._analysis_generation),
        )
        result = analyze_trace_data(request, legacy_denoise_fn=self._hybrid_denoise_fn)
        return result.corrected, result.hybrid_denoised, result.correction_baseline, result.detection

    def _on_hybrid_denoise_finished(
        self,
        trace_name: str,
        cache_key: Tuple[object, ...],
        result: Optional[np.ndarray],
        error: Optional[str],
    ) -> None:
        self._hybrid_tasks = [task for task in self._hybrid_tasks if task.cache_key != cache_key or task.trace_name != trace_name]
        state = self.traces.get(trace_name)
        if state is None or state.hybrid_denoised_pending_key != cache_key:
            return
        if result is not None:
            result_array = np.asarray(result, dtype=float)
            if result_array.shape != np.asarray(state.corrected).shape or not np.all(np.isfinite(result_array)):
                result = None
                error = "State-guided denoising returned invalid data"
        state.hybrid_denoised_pending_key = None
        if error is not None or result is None:
            state.hybrid_denoised = None
            state.hybrid_denoised_cache_key = None
            if self.current_trace_name == trace_name:
                self.chk_show_hybrid_denoised.setChecked(False)
                QMessageBox.warning(self, "State-Guided Denoising Error", str(error or "Unknown error"))
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
        self.table.setRowCount(len(table_spikes))
        for row, spike in enumerate(table_spikes):
            values = [
                str(spike.spike_index),
                f"{spike.spike_time:.2f}",
                f"{spike.amplitude_above_baseline:.2f}",
                f"{spike.snr:.2f}",
                f"{spike.prominence:.2f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.table.setItem(row, col, item)

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
        if row < len(self.trace_names) - 1:
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
