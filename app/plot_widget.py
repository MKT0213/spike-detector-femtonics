from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .models import SpikeRecord


class SmoothViewBox(pg.ViewBox):
    def __init__(self) -> None:
        super().__init__(enableMenu=False)
        self.setMouseMode(self.PanMode)
        self.setMouseEnabled(x=True, y=False)

    def wheelEvent(self, ev, axis=None) -> None:
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        if delta == 0:
            ev.ignore()
            return

        steps = abs(delta) / 120.0
        zoom_base = 1.08
        scale = zoom_base**steps
        x_factor = 1.0 / scale if delta > 0 else scale

        center = self.mapSceneToView(ev.scenePos())
        self.scaleBy((x_factor, 1.0), center=center)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            if ev.isStart() or ev.isFinish():
                ev.accept()
                return

            p_last = self.mapToView(ev.lastPos())
            p_now = self.mapToView(ev.pos())
            dx = float(p_now.x() - p_last.x())
            if dx != 0.0:
                self.translateBy(x=-dx, y=0.0)
            ev.accept()
            return

        super().mouseDragEvent(ev, axis=axis)


class TracePlotWidget(QWidget):
    clicked = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_plot = pg.PlotWidget(background="#f7f9fc", viewBox=SmoothViewBox())
        self.main_plot.showGrid(x=True, y=True, alpha=0.15)
        self.main_plot.setLabel("bottom", "Time")
        self.main_plot.setLabel("left", "Signal")
        self.main_plot.addLegend(offset=(10, 8))
        self.main_plot.setMouseEnabled(x=True, y=False)

        self.compare_plot = pg.PlotWidget(background="#f7f9fc", viewBox=SmoothViewBox())
        self.compare_plot.showGrid(x=True, y=True, alpha=0.12)
        self.compare_plot.setLabel("left", "Raw")
        self.compare_plot.addLegend(offset=(10, 8))
        self.compare_plot.setMouseEnabled(x=True, y=False)
        self.compare_plot.setXLink(self.main_plot)
        self.compare_plot.setMaximumHeight(230)

        self.overview_plot = pg.PlotWidget(background="#f7f9fc")
        self.overview_plot.setMaximumHeight(90)
        self.overview_plot.setMouseEnabled(x=False, y=False)
        self.overview_plot.hideAxis("left")
        self.overview_plot.setLabel("bottom", "Overview")

        self.region = pg.LinearRegionItem()
        self.overview_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)

        layout.addWidget(self.main_plot, stretch=4)
        layout.addWidget(self.compare_plot, stretch=1)
        layout.addWidget(self.overview_plot, stretch=0)

        self.main_raw = self.main_plot.plot(pen=pg.mkPen("#9aa0a6", width=0.9), name="Raw")
        self.main_corrected = self.main_plot.plot(pen=pg.mkPen("#2979ff", width=1.8), name="Corrected")
        self.main_highpass = self.main_plot.plot(
            pen=pg.mkPen("#00897b", width=1.3, style=Qt.PenStyle.DashLine),
            name="High-pass",
        )
        self.main_hybrid_denoised = self.main_plot.plot(pen=pg.mkPen("#8e24aa", width=1.8), name="State-Guided Denoised")
        self.main_plateau = self.main_plot.plot(pen=pg.mkPen("#c62828", width=2.2), name="Long Plateau")
        self.main_short_plateau = self.main_plot.plot(pen=pg.mkPen("#1565c0", width=2.2), name="Short Plateau")
        self.main_raw_baseline = self.main_plot.plot(pen=pg.mkPen("#ff6d00", width=1.6), name="Raw Baseline")
        self.main_corrected_baseline = self.main_plot.plot(
            pen=pg.mkPen("#00a884", width=1.4, style=Qt.PenStyle.DashLine),
            name="Corrected Baseline",
        )
        self.compare_raw = self.compare_plot.plot(pen=pg.mkPen("#5f6368", width=1.0), name="Raw")
        self.compare_raw_baseline = self.compare_plot.plot(
            pen=pg.mkPen("#ff6d00", width=1.2, style=Qt.PenStyle.DashLine),
            name="Raw Baseline",
        )
        for item in [
            self.main_raw,
            self.main_corrected,
            self.main_highpass,
            self.main_hybrid_denoised,
            self.main_plateau,
            self.main_short_plateau,
            self.main_raw_baseline,
            self.main_corrected_baseline,
            self.compare_raw,
            self.compare_raw_baseline,
        ]:
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")

        self.spike_scatter = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush("#f1c40f"),
            pen=pg.mkPen("#ffffff", width=1.0),
        )
        self.main_plot.addItem(self.spike_scatter)

        self.overview_curve = self.overview_plot.plot(pen=pg.mkPen("#777", width=1))
        self.overview_curve.setClipToView(True)
        self.overview_curve.setDownsampling(auto=True, method="peak")

        self.time = np.array([])
        self.corrected = np.array([])
        self._plateau_regions: List[pg.LinearRegionItem] = []
        self._long_plateau_regions: List[pg.LinearRegionItem] = []
        self._short_plateau_regions: List[pg.LinearRegionItem] = []
        self._artifact_regions: List[pg.LinearRegionItem] = []
        self._compare_artifact_regions: List[pg.LinearRegionItem] = []
        self._x_bounds: Optional[tuple[float, float]] = None
        self._view_initialized = False
        self._show_raw = True
        self._show_corrected = True
        self._show_baseline = True
        self._show_highpass = False
        self._show_hybrid_denoised = False
        self._has_raw_baseline = False
        self._has_corrected_baseline = False
        self._has_highpass = False
        self._has_hybrid_denoised = False
        self.main_highpass.setZValue(10)
        self.main_hybrid_denoised.setZValue(11)
        self._sync_legend()

        self.main_plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.main_plot.sigXRangeChanged.connect(self._sync_region_from_main)

    @staticmethod
    def _mask_regions(mask: np.ndarray) -> List[tuple[int, int]]:
        m = np.asarray(mask, dtype=bool)
        if m.size == 0 or not np.any(m):
            return []
        idx = np.flatnonzero(m)
        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        return [(int(group[0]), int(group[-1]) + 1) for group in groups]

    def _clear_artifact_regions(self) -> None:
        for region in self._artifact_regions:
            self.main_plot.removeItem(region)
        for region in self._compare_artifact_regions:
            self.compare_plot.removeItem(region)
        self._artifact_regions = []
        self._compare_artifact_regions = []

    def _clear_plateau_regions(self) -> None:
        for region in self._plateau_regions:
            self.main_plot.removeItem(region)
        self._plateau_regions = []
        self._long_plateau_regions = []
        self._short_plateau_regions = []

    def _region_time_bounds(self, time: np.ndarray, start: int, stop: int) -> Optional[tuple[float, float]]:
        n = int(time.size)
        if n <= 0 or start >= n:
            return None
        left = float(time[start])
        right_idx = min(n - 1, max(start, stop))
        right = float(time[right_idx])
        if not np.isfinite(left) or not np.isfinite(right):
            return None
        if right <= left:
            if n > 1:
                dt = np.nanmedian(np.diff(time[np.isfinite(time)]))
                right = left + (float(dt) if np.isfinite(dt) and dt > 0 else 1e-9)
            else:
                right = left + 1e-9
        return left, right

    def _add_plateau_regions(
        self,
        plateau_mask: Optional[np.ndarray],
        brush: pg.QtGui.QBrush,
        pen: pg.QtGui.QPen,
        target: List[pg.LinearRegionItem],
    ) -> None:
        if plateau_mask is None or self.time.size == 0:
            return
        mask = np.asarray(plateau_mask, dtype=bool)
        n = min(mask.size, self.time.size)
        if n <= 0:
            return
        time = np.asarray(self.time[:n], dtype=float)
        for start, stop in self._mask_regions(mask[:n]):
            bounds = self._region_time_bounds(time, start, stop)
            if bounds is None:
                continue
            region = pg.LinearRegionItem(
                values=bounds,
                brush=brush,
                pen=pen,
                movable=False,
            )
            region.setZValue(-8)
            self.main_plot.addItem(region)
            target.append(region)
            self._plateau_regions.append(region)

    def _set_plateau_regions(
        self,
        plateau_mask: Optional[np.ndarray],
        long_plateau_mask: Optional[np.ndarray] = None,
        short_plateau_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._clear_plateau_regions()
        self.main_plateau.setData([], [])
        self.main_short_plateau.setData([], [])
        if self.time.size == 0:
            return
        long_mask = long_plateau_mask if long_plateau_mask is not None else plateau_mask
        self._add_plateau_regions(
            long_mask,
            brush=pg.mkBrush(198, 40, 40, 45),
            pen=pg.mkPen(198, 40, 40, 0),
            target=self._long_plateau_regions,
        )
        self._add_plateau_regions(
            short_plateau_mask,
            brush=pg.mkBrush(21, 101, 192, 42),
            pen=pg.mkPen(21, 101, 192, 0),
            target=self._short_plateau_regions,
        )
        self._sync_legend()

    def _set_artifact_regions(self, artifact_mask: Optional[np.ndarray]) -> None:
        self._clear_artifact_regions()
        if artifact_mask is None or self.time.size == 0:
            return
        mask = np.asarray(artifact_mask, dtype=bool)
        n = min(mask.size, self.time.size)
        if n <= 0:
            return
        time = np.asarray(self.time[:n], dtype=float)
        for start, stop in self._mask_regions(mask[:n]):
            bounds = self._region_time_bounds(time, start, stop)
            if bounds is None:
                continue
            region = pg.LinearRegionItem(
                values=bounds,
                brush=pg.mkBrush(255, 193, 7, 55),
                pen=pg.mkPen("#d97706", width=1.0),
                movable=False,
            )
            region.setZValue(-5)
            self.main_plot.addItem(region)
            self._artifact_regions.append(region)
            compare_region = pg.LinearRegionItem(
                values=bounds,
                brush=pg.mkBrush(255, 193, 7, 45),
                pen=pg.mkPen("#d97706", width=1.0),
                movable=False,
            )
            compare_region.setZValue(-5)
            self.compare_plot.addItem(compare_region)
            self._compare_artifact_regions.append(compare_region)
        self._sync_legend()

    def _set_compare_y_range_from_data(self, raw: np.ndarray, raw_baseline: Optional[np.ndarray]) -> None:
        parts = [np.asarray(raw, dtype=float)]
        if raw_baseline is not None:
            parts.append(np.asarray(raw_baseline, dtype=float))
        finite_parts = [part[np.isfinite(part)] for part in parts if part.size > 0]
        if not finite_parts:
            return
        values = np.concatenate(finite_parts)
        if values.size == 0:
            return
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if values.size >= 200:
            q_lo, q_hi = np.percentile(values, [0.2, 99.8])
            if np.isfinite(q_lo) and np.isfinite(q_hi) and float(q_hi) > float(q_lo):
                full_span = y_max - y_min
                robust_span = float(q_hi) - float(q_lo)
                if robust_span > 0 and full_span > (6.0 * robust_span):
                    y_min = float(q_lo)
                    y_max = float(q_hi)
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return
        if y_max <= y_min:
            y_min -= 0.5
            y_max += 0.5
        span = max(1e-3, y_max - y_min)
        pad = span * 0.08
        self.compare_plot.setYRange(y_min - pad, y_max + pad, padding=0)

    def _sync_legend(self) -> None:
        legend = self.main_plot.plotItem.legend
        if legend is None:
            return
        legend.clear()
        entries = [
            ("Raw", self.main_raw, self._show_raw and self._has_raw_baseline),
            ("Corrected", self.main_corrected, self._show_corrected),
            ("High-pass", self.main_highpass, self._show_highpass and self._has_highpass),
            ("State-Guided Denoised", self.main_hybrid_denoised, self._show_hybrid_denoised and self._has_hybrid_denoised),
            ("Long Plateau", self.main_plateau, True),
            ("Short Plateau", self.main_short_plateau, True),
            ("Raw Baseline", self.compare_raw_baseline, self._show_baseline and self._show_raw and self._has_raw_baseline),
            ("Corrected Baseline", self.main_corrected_baseline, self._show_baseline and self._show_corrected and self._has_corrected_baseline),
        ]
        for label, item, visible in entries:
            if visible:
                legend.addItem(item, label)

    @staticmethod
    def _ranges_close(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
        scale = max(1.0, abs(a_min), abs(a_max), abs(b_min), abs(b_max))
        tol = 1e-6 * scale
        return abs(a_min - b_min) <= tol and abs(a_max - b_max) <= tol

    def _on_region_changed(self) -> None:
        min_x, max_x = self.region.getRegion()
        min_x, max_x = self._clamp_x_range(float(min_x), float(max_x))

        current = self.region.getRegion()
        if abs(float(current[0]) - min_x) > 1e-9 or abs(float(current[1]) - max_x) > 1e-9:
            self.region.blockSignals(True)
            self.region.setRegion((min_x, max_x))
            self.region.blockSignals(False)

        view = self.main_plot.viewRange()[0]
        if not self._ranges_close(float(view[0]), float(view[1]), min_x, max_x):
            self.main_plot.setXRange(min_x, max_x, padding=0)

    def _sync_region_from_main(self) -> None:
        view = self.main_plot.viewRange()[0]
        min_x, max_x = self._clamp_x_range(float(view[0]), float(view[1]))
        current = self.region.getRegion()
        if self._ranges_close(float(current[0]), float(current[1]), min_x, max_x):
            return
        self.region.blockSignals(True)
        self.region.setRegion((min_x, max_x))
        self.region.blockSignals(False)

    def _clamp_x_range(self, min_x: float, max_x: float) -> tuple[float, float]:
        bounds = self._x_bounds
        if bounds is None:
            return min_x, max_x

        lo, hi = bounds
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return min_x, max_x

        width = max(1e-9, max_x - min_x)
        full_span = hi - lo
        if width >= full_span:
            return lo, hi

        if min_x < lo:
            min_x = lo
            max_x = lo + width
        if max_x > hi:
            max_x = hi
            min_x = hi - width

        min_x = max(lo, min_x)
        max_x = min(hi, max_x)
        if max_x <= min_x:
            return lo, hi
        return min_x, max_x

    @staticmethod
    def _estimate_sampling_rate(time: np.ndarray) -> float:
        t = np.asarray(time, dtype=float)
        if t.size < 2:
            return 1.0
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return 1.0
        dt_med = float(np.median(dt))
        fs_from_seconds = 1.0 / dt_med
        # Mirror detector-side heuristic for traces whose time axis is in ms.
        if fs_from_seconds < 5.0 and dt_med >= 0.001:
            return float(1000.0 / dt_med)
        return float(fs_from_seconds)

    def _startup_ignore_samples(self, startup_ms: float = 50.0) -> int:
        if self.time.size < 2:
            return 0
        fs = self._estimate_sampling_rate(self.time)
        if not np.isfinite(fs) or fs <= 0.0:
            return 0
        count = int(round((float(startup_ms) / 1000.0) * fs))
        if count <= 0:
            return 0
        # Keep enough samples for robust percentile scaling.
        return max(0, min(int(self.time.size) - 1, count))

    def _set_main_y_range_from_data(
        self,
        raw: np.ndarray,
        corrected: np.ndarray,
        hybrid_denoised: Optional[np.ndarray],
        highpass: Optional[np.ndarray],
        raw_baseline: Optional[np.ndarray],
        corrected_baseline: Optional[np.ndarray],
    ) -> None:
        extent = self._y_extent_from_data(
            raw=raw,
            corrected=corrected,
            hybrid_denoised=hybrid_denoised,
            highpass=highpass,
            raw_baseline=raw_baseline,
            corrected_baseline=corrected_baseline,
        )
        if extent is None:
            return

        y_min, y_max = extent
        if not np.isfinite(y_max) or not np.isfinite(y_min) or y_max <= y_min:
            y_min, y_max = max(0.0, y_min), max(1e-3, y_max)
        span = max(1e-3, y_max - y_min)
        pad = span * 0.08
        self.main_plot.setYRange(y_min - pad, y_max + pad, padding=0)

    def _y_extent_from_data(
        self,
        raw: np.ndarray,
        corrected: np.ndarray,
        hybrid_denoised: Optional[np.ndarray],
        highpass: Optional[np.ndarray],
        raw_baseline: Optional[np.ndarray],
        corrected_baseline: Optional[np.ndarray],
    ) -> Optional[tuple[float, float]]:
        ignore_n = self._startup_ignore_samples(startup_ms=50.0)

        def _trim_start(arr: np.ndarray) -> np.ndarray:
            out = np.asarray(arr, dtype=float)
            if ignore_n > 0 and out.size > (ignore_n + 8):
                return out[ignore_n:]
            return out

        core_parts: List[np.ndarray] = []
        if self._show_corrected:
            core_parts.append(_trim_start(corrected))
        if self._show_hybrid_denoised and hybrid_denoised is not None:
            core_parts.append(_trim_start(hybrid_denoised))
        if self._show_highpass and highpass is not None:
            core_parts.append(_trim_start(highpass))
        if self._show_raw and self._show_baseline and raw_baseline is not None:
            core_parts.append(_trim_start(raw_baseline))
            core_parts.append(_trim_start(raw))
        if self._show_corrected and self._show_baseline and corrected_baseline is not None:
            core_parts.append(_trim_start(corrected_baseline))
        # Keep a sane default even if corrected is hidden: autoscale around
        # corrected-centric channels rather than raw artifact outliers.
        if not core_parts:
            core_parts.append(_trim_start(corrected))

        finite_core = [arr[np.isfinite(arr)] for arr in core_parts if arr.size > 0]
        values: np.ndarray
        if finite_core:
            values = np.concatenate(finite_core)
        elif self._show_raw:
            raw_finite = _trim_start(raw)
            raw_finite = raw_finite[np.isfinite(raw_finite)]
            if raw_finite.size == 0:
                return None
            values = raw_finite
        else:
            return None

        if values.size == 0:
            return None

        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if values.size >= 200:
            q_lo, q_hi = np.percentile(values, [0.2, 99.8])
            if np.isfinite(q_lo) and np.isfinite(q_hi) and float(q_hi) > float(q_lo):
                full_span = y_max - y_min
                robust_span = float(q_hi) - float(q_lo)
                # If one extreme segment dominates the range, fit around the bulk signal.
                if robust_span > 0 and full_span > (6.0 * robust_span):
                    y_min = float(q_lo)
                    y_max = float(q_hi)

        if not self._show_corrected and self._show_raw:
            # Only force y_min = 0 when showing raw data (non-negative).
            y_min = 0.0
            y_max = max(0.0, y_max)
        # Otherwise (corrected or baseline shown): use full data range including negatives
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return None
        
        # Add 5% padding on each side for better visualization
        span = y_max - y_min
        if span > 0:
            padding = 0.05 * span
            y_min -= padding
            y_max += padding
        elif span == 0:
            # If all values are the same, add a small symmetric range
            y_min -= 0.5
            y_max += 0.5
        
        return y_min, y_max

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        if not self.main_plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        self.clicked.emit(float(mouse_point.x()))

    def set_overlay_visibility(
        self,
        show_raw: bool,
        show_corrected: bool,
        show_baseline: bool,
        show_highpass: bool = False,
        show_hybrid_denoised: bool = False,
    ) -> None:
        self._show_raw = bool(show_raw)
        self._show_corrected = bool(show_corrected)
        self._show_baseline = bool(show_baseline)
        self._show_highpass = bool(show_highpass)
        self._show_hybrid_denoised = bool(show_hybrid_denoised)
        self.main_raw.setVisible(show_raw)
        self.main_corrected.setVisible(show_corrected)
        self.main_raw_baseline.setVisible(False)
        self.main_corrected_baseline.setVisible(show_baseline and show_corrected and self._has_corrected_baseline)
        self.main_highpass.setVisible(show_highpass and self._has_highpass)
        self.main_hybrid_denoised.setVisible(show_hybrid_denoised and self._has_hybrid_denoised)
        self._sync_legend()

    def set_trace(
        self,
        time: np.ndarray,
        raw: np.ndarray,
        corrected: np.ndarray,
        hybrid_denoised: Optional[np.ndarray],
        highpass: Optional[np.ndarray],
        raw_baseline: Optional[np.ndarray],
        corrected_baseline: Optional[np.ndarray],
        spikes: Iterable[SpikeRecord],
        plateau_mask: Optional[np.ndarray] = None,
        long_plateau_mask: Optional[np.ndarray] = None,
        short_plateau_mask: Optional[np.ndarray] = None,
        artifact_mask: Optional[np.ndarray] = None,
        reset_view: bool = False,
        refit_y: bool = False,
        startup_exclusion_ms: float = 0.0,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.corrected = np.asarray(corrected, dtype=float)
        startup_idx = 0
        
        # Exclude startup region from display if requested
        if startup_exclusion_ms > 0.0 and self.time.size > 0:
            # Compute startup samples from estimated fs so ms exclusion is unit-safe
            # for traces whose time axis may be in seconds or milliseconds.
            fs = self._estimate_sampling_rate(self.time)
            startup_idx = int(round((float(startup_exclusion_ms) / 1000.0) * max(1e-9, fs)))
            if startup_idx > 0 and startup_idx < len(self.time):
                # Slice all arrays to exclude startup region
                self.time = self.time[startup_idx:]
                raw = np.asarray(raw, dtype=float)[startup_idx:]
                corrected = np.asarray(corrected, dtype=float)[startup_idx:]
                if hybrid_denoised is not None:
                    hybrid_denoised = np.asarray(hybrid_denoised, dtype=float)[startup_idx:]
                if highpass is not None:
                    highpass = np.asarray(highpass, dtype=float)[startup_idx:]
                if raw_baseline is not None:
                    raw_baseline = np.asarray(raw_baseline, dtype=float)[startup_idx:]
                if corrected_baseline is not None:
                    corrected_baseline = np.asarray(corrected_baseline, dtype=float)[startup_idx:]
                if plateau_mask is not None:
                    plateau_mask = np.asarray(plateau_mask, dtype=bool)[startup_idx:]
                if long_plateau_mask is not None:
                    long_plateau_mask = np.asarray(long_plateau_mask, dtype=bool)[startup_idx:]
                if short_plateau_mask is not None:
                    short_plateau_mask = np.asarray(short_plateau_mask, dtype=bool)[startup_idx:]
                if artifact_mask is not None:
                    artifact_mask = np.asarray(artifact_mask, dtype=bool)[startup_idx:]
                self.corrected = corrected

        raw_array = np.asarray(raw, dtype=float)
        corrected_array = np.asarray(corrected, dtype=float)
        raw_baseline_array = None if raw_baseline is None else np.asarray(raw_baseline, dtype=float)
        corrected_baseline_array = None if corrected_baseline is None else np.asarray(corrected_baseline, dtype=float)
        hybrid_array = None if hybrid_denoised is None else np.asarray(hybrid_denoised, dtype=float)
        highpass_array = None if highpass is None else np.asarray(highpass, dtype=float)
        if (
            hybrid_array is not None
            and raw_baseline_array is not None
            and hybrid_array.shape == corrected_array.shape
            and raw_baseline_array.shape == corrected_array.shape
        ):
            current_offset = abs(float(np.nanmedian(hybrid_array)) - float(np.nanmedian(corrected_array)))
            candidate = hybrid_array - raw_baseline_array
            candidate_offset = abs(float(np.nanmedian(candidate)) - float(np.nanmedian(corrected_array)))
            if np.isfinite(candidate_offset) and (
                candidate_offset < current_offset
                and float(np.nanmedian(np.abs(candidate))) < float(np.nanmedian(np.abs(hybrid_array)))
            ):
                hybrid_array = candidate

        self._has_raw_baseline = bool(raw_baseline_array is not None and raw_baseline_array.shape == raw_array.shape)
        self._has_corrected_baseline = bool(
            corrected_baseline_array is not None and corrected_baseline_array.shape == corrected_array.shape
        )
        self._has_hybrid_denoised = bool(hybrid_array is not None and hybrid_array.shape == corrected_array.shape)
        self._has_highpass = bool(highpass_array is not None and highpass_array.shape == corrected_array.shape)

        # The main plot is corrected-unit only. Raw-unit data stays in the lower
        # raw panel; if raw appears in the main plot, it is baseline-subtracted.
        raw_main = raw_array - raw_baseline_array if self._has_raw_baseline else np.full_like(corrected_array, np.nan)

        self.main_raw.setData(self.time, raw_main)
        self.compare_raw.setData(self.time, raw)
        self.main_corrected.setData(self.time, corrected_array)
        if self._has_highpass:
            self.main_highpass.setData(self.time, highpass_array)
        else:
            self.main_highpass.setData([], [])
        if self._has_hybrid_denoised:
            self.main_hybrid_denoised.setData(self.time, hybrid_array)
        else:
            self.main_hybrid_denoised.setData([], [])

        self.main_raw_baseline.setData([], [])
        if self._has_raw_baseline:
            self.compare_raw_baseline.setData(self.time, raw_baseline_array)
        else:
            self.compare_raw_baseline.setData([], [])
        if self._has_corrected_baseline:
            self.main_corrected_baseline.setData(self.time, corrected_baseline_array)
        else:
            self.main_corrected_baseline.setData([], [])

        self.main_raw.setVisible(self._show_raw and self._has_raw_baseline)
        self.main_corrected_baseline.setVisible(
            self._show_baseline and self._show_corrected and self._has_corrected_baseline
        )
        self.main_highpass.setVisible(self._show_highpass and self._has_highpass)
        self.main_hybrid_denoised.setVisible(self._show_hybrid_denoised and self._has_hybrid_denoised)

        self._set_plateau_regions(
            plateau_mask=plateau_mask,
            long_plateau_mask=long_plateau_mask,
            short_plateau_mask=short_plateau_mask,
        )

        self._set_artifact_regions(artifact_mask)

        spike_points: List[tuple[float, float]] = []
        marker_signal = hybrid_array if self._has_hybrid_denoised else corrected_array
        for spike in spikes:
            local_idx = int(spike.spike_index) - int(startup_idx)
            if 0 <= local_idx < len(self.time) and 0 <= local_idx < len(marker_signal):
                spike_points.append((self.time[local_idx], marker_signal[local_idx]))

        if spike_points:
            xs = [p[0] for p in spike_points]
            ys = [p[1] for p in spike_points]
            self.spike_scatter.setData(xs, ys)
        else:
            self.spike_scatter.setData([], [])

        y_extent = self._y_extent_from_data(
            raw=raw_main,
            corrected=corrected_array,
            hybrid_denoised=hybrid_array if self._has_hybrid_denoised else None,
            highpass=highpass_array if self._has_highpass else None,
            raw_baseline=None,
            corrected_baseline=corrected_baseline_array if self._has_corrected_baseline else None,
        )

        if y_extent is not None and (reset_view or not self._view_initialized or refit_y):
            self._set_main_y_range_from_data(
                raw=raw_main,
                corrected=corrected_array,
                hybrid_denoised=hybrid_array if self._has_hybrid_denoised else None,
                highpass=highpass_array if self._has_highpass else None,
                raw_baseline=None,
                corrected_baseline=corrected_baseline_array if self._has_corrected_baseline else None,
            )
        if reset_view or not self._view_initialized or refit_y:
            self._set_compare_y_range_from_data(raw_array, raw_baseline_array if self._has_raw_baseline else None)

        finite_time = self.time[np.isfinite(self.time)]
        if finite_time.size >= 2:
            x_min = float(np.min(finite_time))
            x_max = float(np.max(finite_time))
            if x_max <= x_min:
                x_max = x_min + 1e-9
            self._x_bounds = (x_min, x_max)
            self.main_plot.setLimits(xMin=x_min, xMax=x_max)
            self.region.setBounds((x_min, x_max))
            self.overview_plot.setXRange(x_min, x_max, padding=0)
        else:
            self._x_bounds = None

        self.overview_curve.setData(self.time, corrected_array)
        if len(self.time) > 2 and (reset_view or not self._view_initialized):
            bounds = self._x_bounds if self._x_bounds is not None else (float(self.time[0]), float(self.time[-1]))
            start = bounds[0]
            span = max(1e-9, bounds[1] - bounds[0])
            stop = min(bounds[1], start + 0.2 * span)
            if stop <= start:
                stop = bounds[1]
            self.region.setRegion((start, stop))
            self.main_plot.setXRange(start, stop, padding=0)
            self._view_initialized = True

        # Recover from stale y-limits (e.g., default 0..1) where all signal data is off-screen.
        if y_extent is not None:
            view_y = self.main_plot.viewRange()[1]
            if y_extent[1] < float(view_y[0]) or y_extent[0] > float(view_y[1]):
                self._set_main_y_range_from_data(
                    raw=raw_main,
                    corrected=corrected_array,
                    hybrid_denoised=hybrid_array if self._has_hybrid_denoised else None,
                    highpass=highpass_array if self._has_highpass else None,
                    raw_baseline=None,
                    corrected_baseline=corrected_baseline_array if self._has_corrected_baseline else None,
                )
                self._set_compare_y_range_from_data(raw_array, raw_baseline_array if self._has_raw_baseline else None)
