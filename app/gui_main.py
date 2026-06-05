from __future__ import annotations

import json
import datetime
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor
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
from .hybrid_denoising import denoise_hybrid_fluorescence_trace, denoise_trace_hybrid_low_plateau_tv
from .io import estimate_sampling_rate_hz, load_excel_traces
from .models import ArtifactParams, DetectionParams, SpikeCluster, SpikeRecord, TraceCuration
from .models import cluster_spikes
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

    # Reference width: at ±5 ms (10 ms total), export around half of prior width.
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
        denoised_plot.setLabel("left", "Hybrid denoised")
        denoised_plot.setMinimumHeight(plot_height)
        denoised_plot.setMaximumHeight(plot_height)
        denoised_plot.plot(t_win_rescaled, denoised_win, pen=pg.mkPen("#8e24aa", width=2.0), name="Hybrid denoised")
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


def _render_cluster_png(
    time: np.ndarray,
    raw: np.ndarray,
    corrected: np.ndarray,
    baseline: Optional[np.ndarray],
    threshold_line: Optional[np.ndarray],
    cluster: SpikeCluster,
    half_window_ms: float = 5.0,
    width: int = 1200,
    height: int = 400,
) -> "QtGui.QPixmap":
    """Render a single spike cluster as a PNG pixmap (no display required)."""
    spikes = cluster.spikes
    n = len(spikes)

    # Center window on the middle spike.
    mid_spike = spikes[n // 2]
    t_center = mid_spike.spike_time

    # Target a configurable ±window around the center spike.
    half_ms = float(half_window_ms)
    half_window = _half_window_in_time_units(time, half_ms)
    zoom_factor = 1.05  # 5% breathing room so spikes aren't at the very edge
    half_zoomed = half_window * zoom_factor

    t_lo = t_center - half_zoomed
    t_hi = t_center + half_zoomed
    span_ms = (t_hi - t_lo) if _time_axis_is_ms(time) else (t_hi - t_lo) * 1000.0

    # Debug: verify window size in ms
    print(f"[DEBUG PNG] n={n}, spike_times={[s.spike_time for s in spikes]}, "
          f"half_ms={half_ms:.1f}, t_lo={t_lo:.6f}, t_hi={t_hi:.6f}, "
            f"span={span_ms:.1f}ms")

    idx_lo = int(np.searchsorted(time, t_lo))
    idx_hi = int(np.searchsorted(time, t_hi))
    idx_lo = max(0, idx_lo - 1)
    idx_hi = min(len(time), idx_hi + 1)

    t_win = time[idx_lo:idx_hi]
    raw_win = raw[idx_lo:idx_hi]
    corr_win = corrected[idx_lo:idx_hi]
    base_win = baseline[idx_lo:idx_hi] if baseline is not None else None
    thresh_win = threshold_line[idx_lo:idx_hi] if threshold_line is not None else None

    spike_xs = [float(time[s.spike_index]) for s in cluster.spikes if s.spike_index < len(time)]
    spike_ys = [float(corrected[s.spike_index]) for s in cluster.spikes if s.spike_index < len(time)]

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    overview_height = max(90, int(height * 0.25))
    total_height = height + overview_height

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    main_plot = pg.PlotWidget(background="#f7f9fc")
    main_plot.showGrid(x=True, y=True, alpha=0.2)
    main_plot.setLabel("bottom", _time_axis_label(time))
    main_plot.setLabel("left", "Signal")
    main_plot.setMinimumHeight(height)
    main_plot.setMaximumHeight(height)

    main_plot.plot(t_win, corr_win, pen=pg.mkPen("#2979ff", width=2.0))
    if base_win is not None:
        main_plot.plot(t_win, base_win, pen=pg.mkPen("#ff6d00", width=1.5))
    if thresh_win is not None:
        main_plot.plot(t_win, thresh_win, pen=pg.mkPen("#8e24aa", width=1.2, style=Qt.PenStyle.DashLine))

    if spike_xs:
        scatter = pg.ScatterPlotItem(x=spike_xs, y=spike_ys, size=10, brush=pg.mkBrush("#f1c40f"), pen=pg.mkPen("#ffffff", width=1.0))
        main_plot.addItem(scatter)

    # Add visible dashed vertical lines at window edges
    y_v = main_plot.viewRange()[1]
    main_plot.plot([t_lo, t_lo], y_v, pen=pg.mkPen("#999999", width=1.0, style=Qt.PenStyle.DashLine))
    main_plot.plot([t_hi, t_hi], y_v, pen=pg.mkPen("#999999", width=1.0, style=Qt.PenStyle.DashLine))
    # Add window label at top
    label = pg.TextItem(
        f"  {2.0 * half_ms:.0f}ms  ",
        color="#999999",
        fill=pg.mkColor("#ffffffcc"),
        border=pg.mkColor("#cccccc"),
        anchor=(0.5, 1.0),
    )
    main_plot.addItem(label)
    label.setPos((t_lo + t_hi) / 2.0, y_v[1])

    main_plot.setXRange(t_lo, t_hi, padding=0)
    # Compute Y range from the corrected data plus spike amplitudes so
    # the full spike shape is always visible (not clipped by percentile tails).
    all_values = [v for arr in [corr_win] if arr.size > 0 for v in np.asarray(arr, dtype=float).flat]
    all_values.extend(spike_ys)
    finite = np.array(all_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size > 0:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            pad = 1e-3
        else:
            pad = max(1e-3, (hi - lo) * 0.10)
        main_plot.setYRange(lo - pad, hi + pad, padding=0)

    overview_plot = pg.PlotWidget(background="#f7f9fc")
    overview_plot.showGrid(x=True, y=False, alpha=0.15)
    overview_plot.setLabel("bottom", "Overview")
    overview_plot.setMouseEnabled(x=False, y=False)
    overview_plot.setMinimumHeight(overview_height)
    overview_plot.setMaximumHeight(overview_height)

    axis_width = 52
    main_plot.getAxis("left").setWidth(axis_width)
    overview_plot.getAxis("left").setWidth(axis_width)
    overview_plot.getAxis("left").setStyle(showValues=False)

    overview_plot.plot(time, corrected, pen=pg.mkPen("#2979ff", width=1.1))
    if baseline is not None:
        overview_plot.plot(time, baseline, pen=pg.mkPen("#ff6d00", width=1.0))

    t_lo_over = max(float(time[0]), t_lo)
    t_hi_over = min(float(time[-1]), t_hi)
    region = pg.LinearRegionItem(
        values=(t_lo_over, t_hi_over),
        brush=pg.mkBrush(41, 121, 255, 50),
        pen=pg.mkPen("#2979ff", width=1.2),
        movable=False,
    )
    overview_plot.addItem(region)
    overview_plot.setXRange(float(time[0]), float(time[-1]), padding=0)

    layout.addWidget(main_plot)
    layout.addWidget(overview_plot)
    container.resize(width, total_height)

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

    # Build a color palette — one distinct color per trace.
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
    # Output filenames under Result/csv_YYYYMMDD_HHMMSS/
    "csv_name": "trace_summary_stats.csv",
    "events_csv_name": "plateau_event_stats.csv",
    "png_name": "trace_summary_stats.png",
    "pdf_name": "trace_summary_stats.pdf",
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


def _time_to_ms_scale(time: np.ndarray) -> float:
    t = np.asarray(time, dtype=float)
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    dt_med = float(np.median(dt))
    # If dt is >= 1e-3, this project convention treats axis as milliseconds.
    return 1.0 if dt_med >= 1e-3 else 1000.0


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

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 0.0
    dt_med = float(np.median(dt))
    scale = _time_to_ms_scale(t)
    return float(np.sum(mask) * dt_med * scale)


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
    """Return (rise_time_samples, decay_time_samples) using 10–90% of peak amplitude.

    Rise  = first crossing of 10% → first crossing of 90%, before the peak.
    Decay = last crossing of 90% → first crossing of 10%, after the peak.
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

    # Rise: scan pre-peak for first 10% crossing → first 90% crossing
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

    # Decay: scan post-peak for last 90% crossing → first 10% crossing
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

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    dt_med = float(np.median(dt))
    scale = _time_to_ms_scale(t)

    widths_ms: List[float] = []
    for start, end in _contiguous_true_regions(m):
        if end <= start or (end - start) < 3:
            continue
        amp = y[start:end] - b[start:end]
        width_samples = _event_fwhm_samples(amp)
        if np.isfinite(width_samples):
            widths_ms.append(float(width_samples * dt_med * scale))

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

        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            continue
        dt_med = float(np.median(dt))
        scale = _time_to_ms_scale(t)
        t0 = float(t[0]) if t.size > 0 and np.isfinite(t[0]) else 0.0

        regions = _contiguous_true_regions(m)
        for idx, (start, end) in enumerate(regions, 1):
            if end <= start:
                continue

            sig = y[start:end]
            amp = y[start:end] - b[start:end]
            sig_f = sig[np.isfinite(sig)]
            amp_f = amp[np.isfinite(amp)]

            onset_time_ms = float((t[start] - t0) * scale) if np.isfinite(t[start]) else float("nan")
            duration_ms = float((end - start) * dt_med * scale)

            fwhm_samples = _event_fwhm_samples(amp)
            fwhm_ms = float(fwhm_samples * dt_med * scale) if np.isfinite(fwhm_samples) else float("nan")

            rise_s, decay_s = _rise_decay_time_samples(amp)
            rise_time_ms = float(rise_s * dt_med * scale) if np.isfinite(rise_s) else float("nan")
            decay_time_ms = float(decay_s * dt_med * scale) if np.isfinite(decay_s) else float("nan")

            plateau_sd = float(np.std(sig_f, ddof=0)) if sig_f.size >= 2 else float("nan")
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
    spikes = [s for s in state.spikes if s.status in ("accepted", "manual")]
    if not spikes:
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
                    "mean_inter_plateau_interval_ms": float("nan"),
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
        else:
            mean_ipi = float("nan")

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
                "mean_inter_plateau_interval_ms": mean_ipi,
            }
        )
    return rows


def _validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _save_trace_summary_figure(csv_path: Path, png_path: Path, pdf_path: Optional[Path] = None) -> None:
    cfg = TRACE_SUMMARY_CONFIG
    # All summary statistics to display as bar charts
    _SUMMARY_STATS = [
        ("plateau_count", "Plateau count", "count"),
        ("first_onset_ms", "Latency to first plateau", "ms"),
        ("mean_duration_ms", "Mean plateau duration", "ms"),
        ("mean_fwhm_ms", "Mean FWHM", "ms"),
        ("mean_rise_time_ms", "Mean rise time", "ms"),
        ("mean_peak_amp_above_baseline", "Mean peak amplitude", "a.u."),
        ("mean_plateau_sd", "Mean plateau SD", "a.u."),
        ("mean_inter_plateau_interval_ms", "Mean inter-plateau interval", "ms"),
    ]

    df = pd.read_csv(csv_path)
    # Exclude the ALL-traces summary row from individual bar charts
    if "trace_id" in df.columns:
        df = df[df["trace_id"] != "ALL"].copy()
    df = df.sort_values("trace_id", kind="mergesort").reset_index(drop=True)

    # Local import keeps GUI startup light and makes plotting dependency explicit.
    import matplotlib.pyplot as plt

    plt.style.use("default")
    n_stats = len(_SUMMARY_STATS)
    n_cols = 4
    n_rows = (n_stats + n_cols - 1) // n_cols  # ceil(n_stats / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), constrained_layout=True)
    fig.patch.set_facecolor("white")
    
    # Flatten axes for iteration
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape((n_rows, n_cols))
    else:
        axes = axes

    for idx, (col, title, unit) in enumerate(_SUMMARY_STATS):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx] if n_rows > 1 else axes[0, col_idx]
        
        if col not in df.columns:
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#9ca3af")
            ax.axis("off")
            continue

        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        trace_names = df["trace_id"].to_numpy() if "trace_id" in df.columns else [f"T{i}" for i in range(len(vals))]
        
        # Keep only finite values and corresponding trace names
        finite_mask = np.isfinite(vals)
        vals = vals[finite_mask]
        trace_names = trace_names[finite_mask]

        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.grid(axis="x", visible=False)

        if vals.size > 0:
            # Create bar chart
            bars = ax.bar(range(vals.size), vals, color="#3b82f6", alpha=0.8, edgecolor="#1e40af", linewidth=1.2)
            
            # Color bars with slight gradient
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.Blues(0.5 + 0.3 * (i / max(1, vals.size - 1))))
            
            ax.set_xticks(range(vals.size))
            ax.set_xticklabels(trace_names, rotation=45, ha="right", fontsize=9)
        
        ylabel = f"{col} ({unit})" if unit else col
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)

    # Hide extra subplot panels
    for idx in range(n_stats, n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx] if n_rows > 1 else axes[0, col_idx]
        ax.axis("off")

    cfg = TRACE_SUMMARY_CONFIG
    fig.savefig(png_path, dpi=int(cfg["png_dpi"]), facecolor="white", bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
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


STATUS_ROW_COLORS = {
    "pending": QColor("#fff4cc"),
    "accepted": QColor("#dcfce7"),
    "rejected": QColor("#fee2e2"),
    "manual": QColor("#dbeafe"),
    "deleted": QColor("#e5e7eb"),
}


class _HybridWorkerSignals(QObject):
    finished = Signal(str, object, object, object)


class _HybridDenoiseTask(QRunnable):
    def __init__(
        self,
        trace_name: str,
        cache_key: Tuple[float, str],
        denoise_fn: Callable[..., np.ndarray],
        corrected: np.ndarray,
        fs: float,
        baseline: Optional[np.ndarray],
    ) -> None:
        super().__init__()
        self.trace_name = trace_name
        self.cache_key = cache_key
        self.denoise_fn = denoise_fn
        self.corrected = np.asarray(corrected, dtype=float).copy()
        self.fs = float(fs)
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=float).copy()
        self.signals = _HybridWorkerSignals()

    def run(self) -> None:
        try:
            result = np.asarray(
                self.denoise_fn(
                    self.corrected,
                    fs=self.fs,
                    baseline=self.baseline,
                    threshold_sd=self.cache_key[0],
                    event_mode=self.cache_key[1],
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
        self.setWindowTitle(f"Spike Curation App [{DETECTOR_BUILD_TAG}]")
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

        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)

        top_bar = QHBoxLayout()
        root_layout.addLayout(top_bar)

        self.btn_load_excel = QPushButton("Load Excel")
        self.btn_load_exported = QPushButton("Load Exported Result")
        self.btn_load_session = QPushButton("Load Session")
        self.btn_save_session = QPushButton("Save Session")
        self.btn_export = QPushButton("Export CSV")
        self.btn_export_png = QPushButton("Export PNG")
        self.btn_export_png_current = QPushButton("Export PNG (Current)")
        self.btn_reload_params_cfg = QPushButton("Reload Params Config")
        self.btn_rerun_current = QPushButton("Re-detect Current")
        self.btn_rerun_all = QPushButton("Re-detect All")

        for button in [
            self.btn_load_excel,
            self.btn_load_exported,
            self.btn_load_session,
            self.btn_save_session,
            self.btn_export,
            self.btn_export_png,
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
        self.param_startup_exclusion_ms = self._make_double_spin(0.0, 1000.0, self.detection_params.startup_exclusion_ms, 1.0)
        self.param_baseline_rolling_window_ms = self._make_double_spin(5.0, 5000.0, self.detection_params.baseline_rolling_window_ms, 5.0)
        self.param_plateau_median_window_ms = self._make_double_spin(5.0, 500.0, self.detection_params.plateau_median_window_ms, 1.0)
        self.param_spike_noise_k = self._make_double_spin(0.1, 100.0, self.detection_params.spike_noise_k, 0.1)
        self.param_spike_prominence_k = self._make_double_spin(0.0, 100.0, self.detection_params.spike_prominence_k, 0.1)
        self.param_min_separation_ms = self._make_double_spin(1.0, 1000.0, self.detection_params.min_separation_ms, 1.0)
        self.param_spike_zoom_half_window_ms = self._make_double_spin(0.5, 500.0, self.detection_params.spike_zoom_half_window_ms, 0.5)

        params_layout.addRow("detection_mode", self.param_detection_mode)
        params_layout.addRow("startup_exclusion_ms", self.param_startup_exclusion_ms)
        params_layout.addRow("baseline_rolling_window_ms", self.param_baseline_rolling_window_ms)
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
        self.chk_show_hybrid_denoised = QCheckBox("Show hybrid denoised")

        self.chk_show_raw.setChecked(True)
        self.chk_show_corrected.setChecked(True)
        self.chk_show_baseline.setChecked(True)
        self.chk_show_hybrid_denoised.setChecked(False)

        overlay_layout.addRow(self.chk_show_raw)
        overlay_layout.addRow(self.chk_show_corrected)
        overlay_layout.addRow(self.chk_show_baseline)
        overlay_layout.addRow(self.chk_show_hybrid_denoised)
        left_layout.addWidget(overlay_group)

        denoise_group = QGroupBox("Hybrid Denoising")
        denoise_layout = QFormLayout(denoise_group)
        self.param_hybrid_threshold_sd = self._make_double_spin(0.0, 10.0, 2.0, 0.1)
        self.param_plateau_tv_weight = self._make_double_spin(0.0, 100.0, 2.0, 0.1)
        self.param_hybrid_plateau_protect_ms = self._make_double_spin(0.0, 500.0, 20.0, 1.0)
        self.param_hybrid_event_mode = QComboBox()
        self.param_hybrid_event_mode.addItem("SVD", "svd")
        self.param_hybrid_event_mode.addItem("PCA", "pca")
        self.param_hybrid_event_mode.addItem("None", "none")
        denoise_layout.addRow("Hybrid threshold_sd", self.param_hybrid_threshold_sd)
        denoise_layout.addRow("Plateau TV strength", self.param_plateau_tv_weight)
        denoise_layout.addRow("Hybrid plateau protection (ms)", self.param_hybrid_plateau_protect_ms)
        denoise_layout.addRow("Hybrid event mode", self.param_hybrid_event_mode)
        left_layout.addWidget(denoise_group)

        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plot = TracePlotWidget()
        right_layout.addWidget(self.plot, stretch=5)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["idx", "time", "status", "source", "amp", "snr", "prom"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { gridline-color: #666666; }"
            "QTableWidget::item { color: #111111; }"
            "QHeaderView::section { background-color: #30343a; color: #f5f7fa; padding: 4px; border: 1px solid #4b4f56; }"
        )
        right_layout.addWidget(self.table, stretch=1)

        splitter.addWidget(left_panel)
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
        self.btn_export.clicked.connect(self.export_dialog)
        self.btn_export_png.clicked.connect(self.export_png_dialog)
        self.btn_export_png_current.clicked.connect(self.export_png_current_dialog)
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
        self.chk_show_hybrid_denoised.stateChanged.connect(self._on_overlay_changed)

    def _init_shortcuts(self) -> None:
        self.shortcuts = bind_shortcuts(
            self,
            {
                "Right": self.next_trace,
                "Left": self.prev_trace,
                "N": self.jump_next_spike,
                "B": self.jump_prev_spike,
                "A": lambda: self.set_selected_status("accepted"),
                "R": lambda: self.set_selected_status("rejected"),
                "P": lambda: self.set_selected_status("pending"),
                "Space": self.toggle_selected_status,
                "Delete": self.delete_selected_spike,
                "Ctrl+S": self.save_session_dialog,
                "Ctrl+E": self.export_dialog,
                "Q": lambda: self._toggle_checkbox(self.chk_show_raw),
                "W": lambda: self._toggle_checkbox(self.chk_show_corrected),
            },
        )

    def _toggle_checkbox(self, checkbox: QCheckBox) -> None:
        checkbox.setChecked(not checkbox.isChecked())

    def _pull_params_from_widgets(self) -> None:
        self.detection_params.detection_mode = str(self.param_detection_mode.currentData() or "mad")
        self.detection_params.startup_exclusion_ms = float(self.param_startup_exclusion_ms.value())
        self.detection_params.baseline_rolling_window_ms = float(self.param_baseline_rolling_window_ms.value())
        self.detection_params.plateau_median_window_ms = float(self.param_plateau_median_window_ms.value())
        self.detection_params.spike_noise_k = float(self.param_spike_noise_k.value())
        self.detection_params.spike_prominence_k = float(self.param_spike_prominence_k.value())
        self.detection_params.min_separation_ms = float(self.param_min_separation_ms.value())
        self.detection_params.spike_zoom_half_window_ms = float(self.param_spike_zoom_half_window_ms.value())
        self.artifact_params.absolute_low = float(self.param_artifact_abs_low.value())
        self.artifact_params.min_depth = float(self.param_artifact_min_depth.value())
        self.artifact_params.drop_start_abs = float(self.param_artifact_drop_start.value())

    def _push_params_to_widgets(self) -> None:
        # peak_window_ms removed (unused)
        mode_index = self.param_detection_mode.findData(str(self.detection_params.detection_mode))
        self.param_detection_mode.setCurrentIndex(max(0, mode_index))
        self.param_startup_exclusion_ms.setValue(self.detection_params.startup_exclusion_ms)
        self.param_baseline_rolling_window_ms.setValue(self.detection_params.baseline_rolling_window_ms)
        self.param_plateau_median_window_ms.setValue(self.detection_params.plateau_median_window_ms)
        self.param_spike_noise_k.setValue(self.detection_params.spike_noise_k)
        self.param_spike_prominence_k.setValue(self.detection_params.spike_prominence_k)
        self.param_min_separation_ms.setValue(self.detection_params.min_separation_ms)
        self.param_spike_zoom_half_window_ms.setValue(self.detection_params.spike_zoom_half_window_ms)
        self.param_artifact_abs_low.setValue(self.artifact_params.absolute_low)
        self.param_artifact_min_depth.setValue(self.artifact_params.min_depth)
        self.param_artifact_drop_start.setValue(self.artifact_params.drop_start_abs)

    # Fixed root for all exports.
    _RESULT_ROOT = Path(r"C:\Users\LM\Desktop\Spike detector\Result")

    def _make_export_dir(self, label: str) -> Path:
        """Create and return a timestamped subfolder under the Result directory.

        Pattern: Result/<label>_YYYYMMDD_HHMMSS/
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._RESULT_ROOT / f"{label}_{ts}"
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

    def _spike_from_row(self, row: pd.Series, trace_name: str) -> SpikeRecord:
        def _num(key: str, default: float = 0.0) -> float:
            return float(pd.to_numeric(row.get(key, default), errors="coerce"))

        spike_index = int(_num("spike_index", 0))
        return SpikeRecord(
            trace_name=trace_name,
            candidate_index=int(_num("candidate_index", spike_index)),
            spike_index=spike_index,
            spike_time=_num("spike_time", 0.0),
            spike_amplitude_raw_or_corrected=_num("spike_amplitude_raw_or_corrected", 0.0),
            baseline_at_spike=_num("baseline_at_spike", 0.0),
            amplitude_above_baseline=_num("amplitude_above_baseline", 0.0),
            prominence=_num("prominence", 0.0),
            width=_num("width", 0.0),
            local_noise_estimate=_num("local_noise_estimate", 0.0),
            snr=_num("snr", 0.0),
            detection_threshold_used=_num("detection_threshold_used", 0.0),
            fwhm_ms=_num("fwhm_ms", 0.0),
            status=str(row.get("status", "pending")),
            source=str(row.get("source", "auto")),
            notes=str(row.get("notes", "")),
        )

    def load_exported_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Export Folder")
        if not folder:
            return
        self.load_exported_result(folder)

    def load_exported_result(self, folder: str) -> None:
        try:
            folder_path = Path(folder)
            corrected_path = folder_path / "corrected_traces.csv"
            if not corrected_path.exists():
                raise FileNotFoundError("corrected_traces.csv not found in selected folder")

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
                    deleted_spikes=[],
                )
                self.traces[trace_name] = state

            review_path = folder_path / "spike_review_log.csv"
            curated_path = folder_path / "curated_spikes.csv"
            if review_path.exists():
                review_df = pd.read_csv(review_path)
                for _, row in review_df.iterrows():
                    trace_name = str(row.get("trace_name", ""))
                    state = self.traces.get(trace_name)
                    if state is None:
                        continue
                    spike = self._spike_from_row(row, trace_name)
                    if spike.status == "deleted":
                        state.deleted_spikes.append(spike)
                    else:
                        state.spikes.append(spike)
            elif curated_path.exists():
                curated_df = pd.read_csv(curated_path)
                for _, row in curated_df.iterrows():
                    trace_name = str(row.get("trace_name", ""))
                    state = self.traces.get(trace_name)
                    if state is None:
                        continue
                    state.spikes.append(self._spike_from_row(row, trace_name))

            for state in self.traces.values():
                state.sort_spikes()
                state.deleted_spikes.sort(key=lambda spike: spike.spike_index)

            self.trace_names = sorted(self.traces.keys())
            self._refresh_trace_list()
            if self.trace_names:
                self.trace_list.setCurrentRow(0)
            self.status_label.setText(f"Loaded exported result: {folder_path.name} | traces: {len(self.trace_names)}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Export Error", str(exc))

    def load_excel(self, file_path: str) -> None:
        try:
            self.excel_path = file_path
            time_map, raw_map, corrected_map, sampling_rate_map = load_excel_traces(file_path)
            self._pull_params_from_widgets()

            self.traces.clear()
            for trace_name, raw in raw_map.items():
                time = time_map[trace_name]
                corrected, artifact_mask = apply_artifact_interpolation(
                    time,
                    corrected_map[trace_name],
                    params=self.artifact_params.to_dict(),
                )
                sampling_rate_hz = float(sampling_rate_map.get(trace_name, _sampling_rate_hz(time)))
                corrected, hybrid_denoised, correction_baseline, detection = self._detect_on_hybrid_denoised(
                    trace_name, time, corrected, artifact_mask, sampling_rate_hz=sampling_rate_hz
                )
                baseline_for_view = detection.baseline
                threshold_for_view = detection.threshold
                trace_state = TraceCuration(
                    trace_name=trace_name,
                    raw=raw,
                    corrected_source=corrected_map[trace_name],
                    corrected=corrected,
                    time=time,
                    sampling_rate_hz=sampling_rate_hz,
                    spikes=detection.spikes,
                    smoothed=detection.smoothed,
                    baseline=baseline_for_view,
                    correction_baseline=correction_baseline,
                    artifact_mask=np.asarray(artifact_mask, dtype=bool),
                    plateau_mask=_detection_mask(detection, "baseline_mask"),
                    long_plateau_mask=_detection_mask(detection, "long_plateau_mask"),
                    short_plateau_mask=_detection_mask(detection, "short_plateau_mask", fallback_key=""),
                    short_plateau_events=_detection_short_events(detection),
                    debug_threshold=threshold_for_view,
                    candidate_windows=detection.candidate_windows,
                    hybrid_denoised=hybrid_denoised,
                    hybrid_denoised_cache_key=self._hybrid_cache_key(),
                )
                trace_state.sort_spikes()
                self.traces[trace_name] = trace_state

            self.trace_names = sorted(self.traces.keys())
            self._refresh_trace_list()
            if self.trace_names:
                self.trace_list.setCurrentRow(0)
            self.status_label.setText(f"Loaded {Path(file_path).name} | traces: {len(self.trace_names)}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

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
                self.load_excel(app_session.excel_path)
            if not self.traces:
                return

            apply_session_to_traces(app_session, self.traces)
            self._render_current_trace(reset_view=True)
            self.status_label.setText(f"Loaded session: {Path(file_path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Session Error", str(exc))

    def save_session_dialog(self) -> None:
        if not self.traces:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Session", "spike_session.json", "JSON Files (*.json)")
        if not file_path:
            return
        self._pull_params_from_widgets()
        save_session(file_path, self.excel_path, self.detection_params, self.artifact_params, self.traces)
        self.status_label.setText(f"Session saved: {Path(file_path).name}")

    def export_dialog(self) -> None:
        if not self.traces:
            return

        out_dir = Path(self._make_export_dir("csv"))
        cfg = TRACE_SUMMARY_CONFIG

        # ── Event-level CSV ──────────────────────────────────────────────────
        event_rows = _build_plateau_event_rows(self.trace_names, self.traces)
        event_cols = [
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
        events_df = pd.DataFrame(event_rows)
        if events_df.empty:
            events_df = pd.DataFrame(columns=event_cols)
        else:
            ordered_ev = [c for c in event_cols if c in events_df.columns]
            tail_ev = [c for c in events_df.columns if c not in ordered_ev]
            events_df = events_df[ordered_ev + tail_ev]
        events_df = events_df.sort_values(["trace_id", "plateau_id"], kind="mergesort").reset_index(drop=True)
        events_csv_path = out_dir / str(cfg.get("events_csv_name", "plateau_event_stats.csv"))
        events_df.to_csv(events_csv_path, index=False)

        # ── Per-trace (neuron) summary CSV ───────────────────────────────────
        rows = _build_trace_summary_rows(self.trace_names, self.traces, event_rows=event_rows)

        trace_df = pd.DataFrame(rows)

        # Build ALL-traces grand summary row from all event data
        if not events_df.empty:
            def _col_vals_all(col: str) -> np.ndarray:
                if col not in events_df.columns:
                    return np.array([], dtype=float)
                return pd.to_numeric(events_df[col], errors="coerce").to_numpy(dtype=float)

            def _agg(arr: np.ndarray) -> tuple[float, float]:
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return float("nan"), float("nan")
                return float(np.mean(arr)), float(np.median(arr))

            all_onsets = _col_vals_all("onset_time_ms")
            all_dur = _col_vals_all("duration_ms")
            all_fwhm = _col_vals_all("fwhm_ms")
            all_rise = _col_vals_all("rise_time_ms")
            all_peak = _col_vals_all("peak_amp_above_baseline")
            all_sd = _col_vals_all("plateau_sd")

            finite_onsets_all = all_onsets[np.isfinite(all_onsets)]
            sorted_onsets_all = np.sort(finite_onsets_all)
            if sorted_onsets_all.size > 1:
                gaps_all = np.diff(sorted_onsets_all)
                gaps_all = gaps_all[gaps_all > 0]
                mean_ipi_all = float(np.mean(gaps_all)) if gaps_all.size > 0 else float("nan")
            else:
                mean_ipi_all = float("nan")

            m_dur, med_dur = _agg(all_dur)
            m_fwhm, med_fwhm = _agg(all_fwhm)
            m_rise, med_rise = _agg(all_rise)
            m_peak, med_peak = _agg(all_peak)
            m_sd, med_sd = _agg(all_sd)

            all_row = {
                "trace_id": "ALL",
                "plateau_count": int(events_df.shape[0]),
                "first_onset_ms": float(np.min(finite_onsets_all)) if finite_onsets_all.size > 0 else float("nan"),
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
                "mean_inter_plateau_interval_ms": mean_ipi_all,
            }
            trace_df = pd.concat([trace_df, pd.DataFrame([all_row])], ignore_index=True)

        summary_col_order = [
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
            "mean_inter_plateau_interval_ms",
        ]
        ordered = [c for c in summary_col_order if c in trace_df.columns]
        tail = [c for c in trace_df.columns if c not in ordered]
        trace_df = trace_df[ordered + tail]

        # Reactivate export format consumed by "Load Exported Result".
        corrected_rows: Dict[str, np.ndarray] = {}
        time_ref: Optional[np.ndarray] = None
        for trace_name in self.trace_names:
            state = self.traces.get(trace_name)
            if state is None:
                continue
            if time_ref is None:
                time_ref = np.asarray(state.time, dtype=float)
            corrected_rows[str(trace_name)] = np.asarray(state.corrected, dtype=float)

        if time_ref is not None and corrected_rows:
            corrected_export = pd.DataFrame({"time": time_ref})
            for trace_name in self.trace_names:
                if trace_name in corrected_rows:
                    corrected_export[str(trace_name)] = corrected_rows[trace_name]
            corrected_export.to_csv(out_dir / "corrected_traces.csv", index=False)

        review_rows: List[Dict[str, object]] = []
        curated_rows: List[Dict[str, object]] = []
        for trace_name in self.trace_names:
            state = self.traces.get(trace_name)
            if state is None:
                continue

            all_spikes = sorted(state.spikes + state.deleted_spikes, key=lambda spike: spike.spike_index)
            for spike in all_spikes:
                row = spike.to_dict()
                row["trace_name"] = str(trace_name)
                review_rows.append(row)
                if str(spike.status) != "deleted":
                    curated_rows.append(row.copy())

        if review_rows:
            pd.DataFrame(review_rows).to_csv(out_dir / "spike_review_log.csv", index=False)
        if curated_rows:
            pd.DataFrame(curated_rows).to_csv(out_dir / "curated_spikes.csv", index=False)

        csv_path = out_dir / str(cfg["csv_name"])
        trace_df.to_csv(csv_path, index=False)

        png_path = out_dir / str(cfg["png_name"])
        pdf_path = out_dir / str(cfg["pdf_name"]) if bool(cfg.get("save_pdf", False)) else None
        _save_trace_summary_figure(csv_path=csv_path, png_path=png_path, pdf_path=pdf_path)

        extra = f", {pdf_path.name}" if pdf_path is not None else ""
        self.status_label.setText(
            f"Exported {csv_path.name}, {events_csv_path.name}, {png_path.name}{extra} to {out_dir}"
        )


    def export_png_dialog(self) -> None:
        if not self.traces:
            return
        self._pull_params_from_widgets()

        out_dir = self._make_export_dir("png_all")
        out_dir.mkdir(parents=True, exist_ok=True)
        spikes_dir = out_dir / "spikes"
        plateaus_dir = out_dir / "plateaus"
        summary_dir = out_dir / "summary"
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

        if exported_traces == 0:
            self.status_label.setText("No spikes available to export")
            return

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

        self.status_label.setText(
            f"Exported categorized PNGs for {exported_traces} traces to {out_dir} (spikes/plateaus/summary)"
        )

    def export_png_current_dialog(self) -> None:
        state = self._current_trace()
        if state is None:
            return
        self._pull_params_from_widgets()

        has_spikes = bool(state.spikes)
        has_plateau_map = bool(state.plateau_mask is not None and np.any(np.asarray(state.plateau_mask, dtype=bool)))
        if not has_spikes and not has_plateau_map:
            self.status_label.setText("No spikes or plateau labels in current trace to export")
            return

        safe_name = "".join(c if c.isalnum() else "_" for c in state.trace_name)
        out_dir = self._make_export_dir(f"png_{safe_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        spikes_dir = out_dir / "spikes"
        plateaus_dir = out_dir / "plateaus"
        summary_dir = out_dir / "summary"
        spikes_dir.mkdir(parents=True, exist_ok=True)
        plateaus_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        pg.setConfigOptions(antialias=True)
        self._export_trace_spike_bundle(
            state,
            out_dir,
            spikes_dir=spikes_dir,
            plateaus_dir=plateaus_dir,
        )
        self.status_label.setText(
            f"Exported categorized PNGs for '{state.trace_name}' to {out_dir} (spikes/plateaus/summary)"
        )

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
        if state.hybrid_denoised is None:
            raise RuntimeError(
                "Hybrid denoised detection trace is unavailable. Rerun detection before exporting PNGs."
            )

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
                        "plateau_height",
                        "internal_spikes",
                        "confidence",
                    ],
                )
                short_df.insert(0, "trace_name", state.trace_name)
                short_df["onset_index"] = short_df["onset_index"].astype(int)
                short_df["offset_index_exclusive"] = short_df["offset_index_exclusive"].astype(int)
                short_df["internal_spikes"] = short_df["internal_spikes"].astype(bool)
                short_df.to_csv(str(plateau_out / f"{safe_name}_short_plateau_detector_events.csv"), index=False)

        return len(spike_pixmaps), aligned_pix, superimposed_pix


    def _refresh_trace_list(self) -> None:
        self.trace_list.clear()
        for name in self.trace_names:
            item = QListWidgetItem(name)
            self.trace_list.addItem(item)

    def on_trace_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.trace_names):
            return
        self.current_trace_name = self.trace_names[row]
        self._render_current_trace(reset_view=True)

    def _current_trace(self) -> Optional[TraceCuration]:
        if self.current_trace_name is None:
            return None
        return self.traces.get(self.current_trace_name)

    def _render_current_trace(self, reset_view: bool = False, refit_y: bool = False) -> None:
        state = self._current_trace()
        if state is None:
            return

        spikes_for_plot = sorted(state.spikes + state.deleted_spikes, key=lambda spike: spike.spike_index)
        self.plot.set_overlay_visibility(
            show_raw=self.chk_show_raw.isChecked(),
            show_corrected=self.chk_show_corrected.isChecked(),
            show_baseline=self.chk_show_baseline.isChecked(),
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
        plateau_count = 0
        if state.plateau_mask is not None:
            plateau_count = len(_contiguous_true_regions(np.asarray(state.plateau_mask, dtype=bool)))
        short_count = 0
        if state.short_plateau_mask is not None:
            short_count = len(_contiguous_true_regions(np.asarray(state.short_plateau_mask, dtype=bool)))
        hybrid_status = ""
        if self.chk_show_hybrid_denoised.isChecked():
            hybrid_status = " | hybrid denoised: shown" if state.hybrid_denoised is not None else ""
        fs_status = f" | fs: {_trace_sampling_rate_hz(state):.3g} Hz"
        self.status_label.setText(
            f"Trace: {state.trace_name}{fs_status} | plateaus: {plateau_count} (short: {short_count}) | spikes: {len(state.spikes)}{hybrid_status}"
        )

    def _hybrid_cache_key(self) -> Tuple[float, str]:
        mode = str(self.param_hybrid_event_mode.currentData() or "svd")
        return float(self.param_hybrid_threshold_sd.value()), mode

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
        task = _HybridDenoiseTask(
            trace_name=state.trace_name,
            cache_key=cache_key,
            denoise_fn=self._hybrid_denoise_fn,
            corrected=state.corrected,
            fs=_trace_sampling_rate_hz(state),
            baseline=state.baseline,
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
        """Detect plateaus on source data, then detect spikes on hybrid denoising."""
        # Plateau labels and their interpolated fluorescence baseline must come
        # from the artifact-interpolated source trace before any denoising.
        baseline_result = detect_spikes(
            trace_name,
            time,
            corrected,
            self.detection_params,
            artifact_mask=artifact_mask,
            sampling_rate_hz=sampling_rate_hz,
        )
        baseline = np.asarray(baseline_result.baseline, dtype=float)
        plateau_mask = _detection_mask(baseline_result, "long_plateau_mask")
        threshold_sd, event_mode = self._hybrid_cache_key()
        fs = float(sampling_rate_hz) if sampling_rate_hz is not None and np.isfinite(sampling_rate_hz) and float(sampling_rate_hz) > 0.0 else _sampling_rate_hz(time)
        # Keep the original Hybrid output for low-state samples. Replace the
        # plateau and its protected margin with state-local TV denoising.
        denoised_uncorrected = np.asarray(
            denoise_trace_hybrid_low_plateau_tv(
                corrected,
                fs=fs,
                plateau_mask=plateau_mask,
                plateau_tv_weight=float(self.param_plateau_tv_weight.value()),
                boundary_protect_ms=float(self.param_hybrid_plateau_protect_ms.value()),
                hybrid_denoise_fn=self._hybrid_denoise_fn,
                threshold_sd=threshold_sd,
                event_mode=event_mode,
            ),
            dtype=float,
        )
        corrected_baseline_corrected = np.asarray(corrected, dtype=float) - baseline
        denoised_baseline_corrected = denoised_uncorrected - baseline
        detection = detect_spikes(
            trace_name,
            time,
            denoised_baseline_corrected,
            self.detection_params,
            artifact_mask=artifact_mask,
            baseline_override=np.zeros_like(baseline),
            plateau_mask_override=plateau_mask,
            sampling_rate_hz=sampling_rate_hz,
            short_plateau_trace_override=corrected_baseline_corrected,
        )
        return corrected_baseline_corrected, denoised_baseline_corrected, baseline, detection

    def _on_hybrid_denoise_finished(
        self,
        trace_name: str,
        cache_key: Tuple[float, str],
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
                error = "Hybrid denoising returned invalid data"
        state.hybrid_denoised_pending_key = None
        if error is not None or result is None:
            state.hybrid_denoised = None
            state.hybrid_denoised_cache_key = None
            if self.current_trace_name == trace_name:
                self.chk_show_hybrid_denoised.setChecked(False)
                QMessageBox.warning(self, "Hybrid Denoising Error", str(error or "Unknown error"))
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
        table_spikes = sorted(state.spikes + state.deleted_spikes, key=lambda spike: spike.spike_index)
        self._table_spikes = table_spikes
        self.table.setRowCount(len(table_spikes))
        for row, spike in enumerate(table_spikes):
            values = [
                str(spike.spike_index),
                f"{spike.spike_time:.2f}",
                spike.status,
                spike.source,
                f"{spike.amplitude_above_baseline:.2f}",
                f"{spike.snr:.2f}",
                f"{spike.prominence:.2f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(STATUS_ROW_COLORS.get(spike.status, QColor("white")))
                item.setForeground(QColor("#111111"))
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

    def _selected_spike(self) -> Optional[SpikeRecord]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._table_spikes):
            return None
        return self._table_spikes[row]

    def set_selected_status(self, status: str) -> None:
        state = self._current_trace()
        spike = self._selected_spike()
        if state is None or spike is None:
            return

        if status == "deleted":
            if spike in state.spikes:
                state.spikes.remove(spike)
                state.deleted_spikes.append(spike)
                state.deleted_spikes.sort(key=lambda item: item.spike_index)
            spike.status = "deleted"  # type: ignore[assignment]
            self._render_current_trace()
            return

        if spike in state.deleted_spikes:
            state.deleted_spikes.remove(spike)
            state.spikes.append(spike)
            state.sort_spikes()

        spike.status = status  # type: ignore[assignment]
        self._render_current_trace()

    def toggle_selected_status(self) -> None:
        state = self._current_trace()
        spike = self._selected_spike()
        if state is None or spike is None:
            return

        if spike in state.deleted_spikes:
            state.deleted_spikes.remove(spike)
            state.spikes.append(spike)
            state.sort_spikes()

        cycle = {"pending": "accepted", "accepted": "rejected", "rejected": "pending"}
        spike.status = cycle.get(spike.status, "pending")  # type: ignore[assignment]
        self._render_current_trace()

    def delete_selected_spike(self) -> None:
        state = self._current_trace()
        spike = self._selected_spike()
        if state is None or spike is None:
            return
        if spike in state.spikes:
            state.spikes.remove(spike)
            spike.status = "deleted"  # type: ignore[assignment]
            state.deleted_spikes.append(spike)
            state.deleted_spikes.sort(key=lambda item: item.spike_index)
        self._render_current_trace()

    def rerun_all(self) -> None:
        if not self.traces:
            return
        self._pull_params_from_widgets()
        for state in self.traces.values():
            corrected, artifact_mask = apply_artifact_interpolation(
                state.time,
                state.corrected_source,
                params=self.artifact_params.to_dict(),
            )
            sampling_rate_hz = _trace_sampling_rate_hz(state)
            corrected, hybrid_denoised, correction_baseline, detection = self._detect_on_hybrid_denoised(
                state.trace_name, state.time, corrected, artifact_mask, sampling_rate_hz=sampling_rate_hz
            )
            state.spikes = detection.spikes
            state.deleted_spikes = []
            state.corrected = corrected
            state.sampling_rate_hz = sampling_rate_hz
            state.hybrid_denoised = hybrid_denoised
            state.hybrid_denoised_cache_key = self._hybrid_cache_key()
            state.hybrid_denoised_pending_key = None
            state.smoothed = detection.smoothed
            state.baseline = detection.baseline
            state.correction_baseline = correction_baseline
            state.artifact_mask = np.asarray(artifact_mask, dtype=bool)
            state.debug_threshold = detection.threshold
            state.candidate_windows = detection.candidate_windows
            state.plateau_mask = _detection_mask(detection, "baseline_mask")
            state.long_plateau_mask = _detection_mask(detection, "long_plateau_mask")
            state.short_plateau_mask = _detection_mask(detection, "short_plateau_mask", fallback_key="")
            state.short_plateau_events = _detection_short_events(detection)
            state.sort_spikes()
        self._render_current_trace()

    def rerun_current(self) -> None:
        state = self._current_trace()
        if state is None:
            return
        self._pull_params_from_widgets()

        corrected, artifact_mask = apply_artifact_interpolation(
            state.time,
            state.corrected_source,
            params=self.artifact_params.to_dict(),
        )
        sampling_rate_hz = _trace_sampling_rate_hz(state)
        corrected, hybrid_denoised, correction_baseline, detection = self._detect_on_hybrid_denoised(
            state.trace_name, state.time, corrected, artifact_mask, sampling_rate_hz=sampling_rate_hz
        )
        state.spikes = detection.spikes
        state.deleted_spikes = []
        state.corrected = corrected
        state.sampling_rate_hz = sampling_rate_hz
        state.hybrid_denoised = hybrid_denoised
        state.hybrid_denoised_cache_key = self._hybrid_cache_key()
        state.hybrid_denoised_pending_key = None
        state.smoothed = detection.smoothed
        state.baseline = detection.baseline
        state.correction_baseline = correction_baseline
        state.artifact_mask = np.asarray(artifact_mask, dtype=bool)
        state.debug_threshold = detection.threshold
        state.candidate_windows = detection.candidate_windows
        state.plateau_mask = _detection_mask(detection, "baseline_mask")
        state.long_plateau_mask = _detection_mask(detection, "long_plateau_mask")
        state.short_plateau_mask = _detection_mask(detection, "short_plateau_mask", fallback_key="")
        state.short_plateau_events = _detection_short_events(detection)
        state.sort_spikes()
        self._render_current_trace()

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
