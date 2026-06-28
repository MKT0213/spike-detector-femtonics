from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from detector.analysis.package import AnalysisPackage, load_analysis_package
from detector.core.render_normalization import NORMALIZATION_NATIVE
from detector.core.trace_identity import recording_groups, split_trace_identity

from .render_options import (
    CHANNEL_DEFINITIONS,
    DEFAULT_SELECTED_CHANNELS,
    MARKER_STYLE_DEFINITIONS,
    RenderOptions,
    TraceWindow,
)
from .rendering import (
    available_channel_ids,
    filtered_spikes,
    plateau_regions,
    prepare_trace_for_render,
    render_preview_pixmap,
    render_recording_event_raster,
    render_recording_aligned_trace_panels,
    render_recording_aligned_traces,
)
from .service import ExportOptions, default_export_dir, export_analysis_package


ALL_RECORDINGS_FILTER = "__all_recordings__"


class ExporterWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spike Analysis Exporter")
        self.resize(1280, 860)
        self.package: Optional[AnalysisPackage] = None
        self._syncing_window = False
        self.current_recording_id: Optional[str] = ALL_RECORDINGS_FILTER
        self._selected_trace_names: set[str] = set()

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        package_row = QHBoxLayout()
        self.package_path = QLineEdit()
        self.package_path.setReadOnly(True)
        self.btn_browse_package = QPushButton("Open Analysis Package")
        package_row.addWidget(QLabel("Package"))
        package_row.addWidget(self.package_path, stretch=1)
        package_row.addWidget(self.btn_browse_package)
        layout.addLayout(package_row)

        output_row = QHBoxLayout()
        self.output_path = QLineEdit()
        self.btn_browse_output = QPushButton("Choose Output Folder")
        output_row.addWidget(QLabel("Output"))
        output_row.addWidget(self.output_path, stretch=1)
        output_row.addWidget(self.btn_browse_output)
        layout.addLayout(output_row)

        middle = QHBoxLayout()
        layout.addLayout(middle, stretch=1)

        trace_group = QGroupBox("Traces")
        trace_layout = QVBoxLayout(trace_group)
        trace_buttons = QHBoxLayout()
        self.btn_select_all = QPushButton("All")
        self.btn_select_none = QPushButton("None")
        trace_buttons.addWidget(self.btn_select_all)
        trace_buttons.addWidget(self.btn_select_none)
        trace_layout.addLayout(trace_buttons)
        self.recording_selector = QComboBox()
        trace_layout.addWidget(QLabel("Recording"))
        trace_layout.addWidget(self.recording_selector)
        self.trace_list = QListWidget()
        self.trace_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        trace_layout.addWidget(self.trace_list, stretch=1)
        middle.addWidget(trace_group, stretch=1)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        middle.addWidget(controls_scroll, stretch=1)

        options_group = QGroupBox("Outputs")
        options_layout = QVBoxLayout(options_group)
        self.chk_tables = QCheckBox("Table CSVs")
        self.chk_reload = QCheckBox("Reload CSVs")
        self.chk_qc_report = QCheckBox("QC summary PNG/PDF")
        self.chk_spike_stats = QCheckBox("Spike statistics summary")
        self.chk_trace_images = QCheckBox("Trace images")
        self.chk_recording_event_raster = QCheckBox("Recording raster plot")
        self.chk_recording_aligned_traces = QCheckBox("Recording aligned traces")
        self.chk_recording_aligned_trace_panels = QCheckBox("Recording aligned trace panels")
        self.chk_recording_aligned_trace_panels_raw = QCheckBox("Recording aligned pure raw panels")
        self.chk_spike_images = QCheckBox("Spike images")
        self.chk_plateau_images = QCheckBox("Plateau images")
        self.chk_denoising_debug = QCheckBox("Denoising debug images")
        self.chk_long_plateau_debug = QCheckBox("Long-plateau debug images")
        options_layout.addWidget(
            self._checkbox_section(
                "Data and reports",
                [self.chk_tables, self.chk_reload, self.chk_qc_report, self.chk_spike_stats],
            )
        )
        options_layout.addWidget(
            self._checkbox_section(
                "Recording-level figures",
                [
                    self.chk_recording_event_raster,
                    self.chk_recording_aligned_traces,
                    self.chk_recording_aligned_trace_panels,
                    self.chk_recording_aligned_trace_panels_raw,
                ],
            )
        )
        options_layout.addWidget(
            self._checkbox_section(
                "Trace and event figures",
                [self.chk_trace_images, self.chk_spike_images, self.chk_plateau_images],
            )
        )
        options_layout.addWidget(
            self._checkbox_section(
                "Debug outputs",
                [self.chk_denoising_debug, self.chk_long_plateau_debug],
            )
        )
        self.chk_tables.setChecked(True)
        self.chk_reload.setChecked(True)
        self.chk_qc_report.setChecked(True)
        self.chk_spike_stats.setChecked(True)
        controls_layout.addWidget(options_group)

        render_group = QGroupBox("Render Settings")
        render_layout = QVBoxLayout(render_group)
        self.channel_checks: Dict[str, QCheckBox] = {}
        channels_group = QGroupBox("Trace signal channels")
        channels_grid = QGridLayout(channels_group)
        for idx, (channel, label) in enumerate(CHANNEL_DEFINITIONS):
            checkbox = QCheckBox(label)
            checkbox.setChecked(channel in DEFAULT_SELECTED_CHANNELS)
            self.channel_checks[channel] = checkbox
            channels_grid.addWidget(checkbox, idx // 2, idx % 2)
        render_layout.addWidget(channels_group)

        aligned_group = QGroupBox("Recording aligned panels")
        aligned_layout = QVBoxLayout(aligned_group)
        self.chk_aligned_panel_share_y = QCheckBox("Share y-axis across ROI panels")
        self.chk_aligned_panel_share_y.setChecked(False)
        self.chk_aligned_panel_share_y.setToolTip(
            "Use one y-axis range for all ROIs. Leave off to give each ROI an independent robust y-limit."
        )
        aligned_layout.addWidget(self.chk_aligned_panel_share_y)
        panel_row = QHBoxLayout()
        self.aligned_panel_channel = QComboBox()
        for channel, label in CHANNEL_DEFINITIONS:
            self.aligned_panel_channel.addItem(label, channel)
        denoised_index = self.aligned_panel_channel.findData("denoised")
        if denoised_index >= 0:
            self.aligned_panel_channel.setCurrentIndex(denoised_index)
        self.aligned_panel_y_min = QDoubleSpinBox()
        self.aligned_panel_y_min.setDecimals(1)
        self.aligned_panel_y_min.setRange(-100000.0, 100000.0)
        self.aligned_panel_y_min.setSingleStep(5.0)
        self.aligned_panel_y_min.setValue(-15.0)
        self.aligned_panel_y_max = QDoubleSpinBox()
        self.aligned_panel_y_max.setDecimals(1)
        self.aligned_panel_y_max.setRange(-100000.0, 100000.0)
        self.aligned_panel_y_max.setSingleStep(5.0)
        self.aligned_panel_y_max.setValue(30.0)
        panel_row.addWidget(QLabel("Panel source"))
        panel_row.addWidget(self.aligned_panel_channel)
        panel_row.addWidget(QLabel("Fallback Y min"))
        panel_row.addWidget(self.aligned_panel_y_min)
        panel_row.addWidget(QLabel("Fallback Y max"))
        panel_row.addWidget(self.aligned_panel_y_max)
        aligned_layout.addLayout(panel_row)
        render_layout.addWidget(aligned_group)

        event_group = QGroupBox("Event presentation")
        event_layout = QVBoxLayout(event_group)
        self.chk_shade_plateaus = QCheckBox("Debug plateau shading")
        self.chk_spike_markers = QCheckBox("Include spike markers")
        self.chk_label_spikes = QCheckBox("Include spike labels")
        self.chk_label_plateaus = QCheckBox("Include plateau labels")
        self.marker_style_combo = QComboBox()
        for style, label in MARKER_STYLE_DEFINITIONS:
            self.marker_style_combo.addItem(label, style)
        self.chk_shade_plateaus.setChecked(True)
        self.chk_spike_markers.setChecked(True)
        self.chk_shade_plateaus.setToolTip(
            "Used by Debug dots + shading mode. Plateau-bar mode uses colored plateau bars instead of shaded regions."
        )
        self.chk_spike_markers.setToolTip("Turn off to export spike plots without spike markers.")
        self.chk_label_spikes.setToolTip("Turn on to draw S0001-style labels at spike peaks.")
        self.chk_label_plateaus.setToolTip("Turn on to draw P0001-style labels next to plateau regions.")
        self.chk_recording_aligned_trace_panels_raw.setToolTip(
            "Export an additional recording-aligned panel image and CSV using raw native signal values only."
        )
        self.chk_recording_event_raster.setToolTip(
            "Export one event-only raster per recording: spike ticks by time, with colored plateau interval bars."
        )
        self.marker_style_combo.setToolTip(
            "Debug mode draws yellow spike dots with shaded plateaus. Plateau-bar mode draws hollow black spike circles and colored plateau interval bars."
        )
        marker_style_row = QHBoxLayout()
        marker_style_row.addWidget(QLabel("Event mode"))
        marker_style_row.addWidget(self.marker_style_combo, stretch=1)
        event_layout.addLayout(marker_style_row)
        event_layout.addWidget(self.chk_shade_plateaus)
        event_layout.addWidget(self.chk_spike_markers)
        event_layout.addWidget(self.chk_label_spikes)
        event_layout.addWidget(self.chk_label_plateaus)
        render_layout.addWidget(event_group)

        window_group = QGroupBox("Time window and zooms")
        window_layout = QVBoxLayout(window_group)
        self.chk_manual_window = QCheckBox("Manual trace window")
        window_layout.addWidget(self.chk_manual_window)
        window_row = QHBoxLayout()
        self.window_start = self._time_spin()
        self.window_end = self._time_spin()
        window_row.addWidget(QLabel("Start"))
        window_row.addWidget(self.window_start)
        window_row.addWidget(QLabel("End"))
        window_row.addWidget(self.window_end)
        window_layout.addLayout(window_row)

        self.overview_plot = pg.PlotWidget(background="#f7f9fc")
        self.overview_plot.setMaximumHeight(100)
        self.overview_plot.setMouseEnabled(x=False, y=False)
        self.overview_plot.hideAxis("left")
        self.overview_plot.setLabel("bottom", "Overview")
        self.overview_curve = self.overview_plot.plot(pen=pg.mkPen("#64748b", width=1))
        self.overview_region = pg.LinearRegionItem()
        self.overview_plot.addItem(self.overview_region)
        window_layout.addWidget(self.overview_plot)

        event_window_row = QHBoxLayout()
        self.spike_half_window = self._positive_spin(0.1, 100000.0, 5.0)
        self.plateau_pre_window = self._positive_spin(0.0, 1000000.0, 100.0)
        self.plateau_post_window = self._positive_spin(0.1, 1000000.0, 500.0)
        event_window_row.addWidget(QLabel("Spike +/- ms"))
        event_window_row.addWidget(self.spike_half_window)
        event_window_row.addWidget(QLabel("Plateau pre ms"))
        event_window_row.addWidget(self.plateau_pre_window)
        event_window_row.addWidget(QLabel("post ms"))
        event_window_row.addWidget(self.plateau_post_window)
        window_layout.addLayout(event_window_row)
        render_layout.addWidget(window_group)

        self.render_status = QLabel("")
        self.render_status.setWordWrap(True)
        render_layout.addWidget(self.render_status)
        controls_layout.addWidget(render_group, stretch=1)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_controls = QHBoxLayout()
        self.preview_trace_combo = QComboBox()
        self.preview_type_combo = QComboBox()
        for label, value in [
            ("Trace view", "trace_view"),
            ("Recording raster plot", "recording_event_raster"),
            ("Recording aligned traces", "recording_aligned_traces"),
            ("Recording aligned panels", "recording_aligned_trace_panels"),
            ("Recording aligned raw panels", "recording_aligned_trace_panels_raw"),
            ("Spike event", "spike_event"),
            ("Spike superimposed", "spike_superimposed"),
            ("Plateau zoom", "plateau_zoom"),
            ("Plateau superimposed", "plateau_superimposed"),
            ("Full trace plateau", "full_trace_plateau"),
            ("Debug preview", "debug_preview"),
        ]:
            self.preview_type_combo.addItem(label, value)
        self.event_label = QLabel("Event")
        self.event_combo = QComboBox()
        self.btn_refresh_preview = QPushButton("Refresh Preview")
        preview_controls.addWidget(QLabel("Trace"))
        preview_controls.addWidget(self.preview_trace_combo, stretch=1)
        preview_controls.addWidget(QLabel("Output"))
        preview_controls.addWidget(self.preview_type_combo)
        preview_controls.addWidget(self.event_label)
        preview_controls.addWidget(self.event_combo)
        preview_controls.addWidget(self.btn_refresh_preview)
        preview_layout.addLayout(preview_controls)

        self.preview_image = QLabel("Open an analysis package to preview renders")
        self.preview_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setWidget(self.preview_image)
        preview_layout.addWidget(self.preview_scroll, stretch=1)
        middle.addWidget(preview_group, stretch=2)

        action_row = QHBoxLayout()
        self.status_label = QLabel("Open an analysis package to begin")
        self.btn_export = QPushButton("Export Selected")
        self.btn_export.setEnabled(False)
        action_row.addWidget(self.status_label, stretch=1)
        action_row.addWidget(self.btn_export)
        layout.addLayout(action_row)

        self.btn_browse_package.clicked.connect(self.browse_package)
        self.btn_browse_output.clicked.connect(self.browse_output)
        self.btn_select_all.clicked.connect(self.select_all_traces)
        self.btn_select_none.clicked.connect(self.clear_all_traces)
        self.btn_export.clicked.connect(self.export_selected)
        self.recording_selector.currentIndexChanged.connect(self._on_recording_selected)
        self.trace_list.itemSelectionChanged.connect(self._on_trace_selection_changed)
        self.preview_trace_combo.currentIndexChanged.connect(self._on_preview_trace_changed)
        self.preview_type_combo.currentIndexChanged.connect(self._on_preview_type_changed)
        self.btn_refresh_preview.clicked.connect(self.refresh_preview)
        self.chk_manual_window.toggled.connect(self._on_manual_window_toggled)
        self.window_start.valueChanged.connect(self._on_window_spin_changed)
        self.window_end.valueChanged.connect(self._on_window_spin_changed)
        self.overview_region.sigRegionChanged.connect(self._on_region_changed)

        self._on_manual_window_toggled(False)
        self._on_preview_type_changed()

    def _time_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1.0e12, 1.0e12)
        spin.setSingleStep(1.0)
        return spin

    def _positive_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(float(minimum), float(maximum))
        spin.setSingleStep(1.0)
        spin.setValue(float(value))
        return spin

    def _checkbox_section(self, title: str, checkboxes: List[QCheckBox]) -> QGroupBox:
        group = QGroupBox(title)
        layout = QGridLayout(group)
        for row, checkbox in enumerate(checkboxes):
            layout.addWidget(checkbox, row, 0)
        return group

    def browse_package(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Analysis Package")
        if not folder:
            return
        self.load_package(folder)

    def load_package(self, folder: str | Path) -> None:
        try:
            package = load_analysis_package(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Open Package Error", str(exc))
            return
        self.package = package
        self.current_recording_id = ALL_RECORDINGS_FILTER
        self._selected_trace_names = set()
        self.package_path.setText(str(package.root))
        self.output_path.setText(str(default_export_dir(self._result_root(), package)))
        self._sync_recording_selector()
        self._refresh_trace_list()
        self.select_all_traces()
        self.btn_export.setEnabled(True)
        self._refresh_preview_traces()
        self._sync_controls_for_preview_trace(reset_window=True)
        recording_count = len(recording_groups(package.trace_names))
        self.status_label.setText(
            f"Loaded {package.root.name} | recordings: {recording_count} | traces: {len(package.trace_names)}"
        )

    def browse_output(self) -> None:
        start = self.output_path.text().strip() or str(self._result_root())
        folder = QFileDialog.getExistingDirectory(self, "Choose Output Folder", start)
        if folder:
            self.output_path.setText(folder)

    def _display_label_for_trace(self, trace_name: str) -> str:
        recording_id, roi_id = split_trace_identity(trace_name)
        if self.current_recording_id != ALL_RECORDINGS_FILTER and recording_id == self.current_recording_id:
            return roi_id
        return f"{recording_id} / {roi_id}"

    def _trace_names_for_current_recording(self) -> List[str]:
        if self.package is None:
            return []
        if self.current_recording_id in {None, ALL_RECORDINGS_FILTER}:
            return list(self.package.trace_names)
        return [
            name
            for name in self.package.trace_names
            if split_trace_identity(name)[0] == self.current_recording_id
        ]

    def _sync_recording_selector(self) -> None:
        if self.package is None:
            self.recording_selector.clear()
            return
        groups = recording_groups(self.package.trace_names)
        preferred = self.current_recording_id or ALL_RECORDINGS_FILTER

        self.recording_selector.blockSignals(True)
        self.recording_selector.clear()
        self.recording_selector.addItem(
            f"All recordings ({len(groups)} recordings, {len(self.package.trace_names)} ROIs)",
            ALL_RECORDINGS_FILTER,
        )
        for recording_id, names in groups.items():
            roi_count = len(names)
            self.recording_selector.addItem(
                f"{recording_id} ({roi_count} ROI{'s' if roi_count != 1 else ''})",
                recording_id,
            )
        index = self.recording_selector.findData(preferred)
        self.recording_selector.setCurrentIndex(index if index >= 0 else 0)
        self.current_recording_id = str(self.recording_selector.currentData() or ALL_RECORDINGS_FILTER)
        self.recording_selector.blockSignals(False)

    def _refresh_trace_list(self) -> None:
        selected = set(self._selected_trace_names)
        self.trace_list.blockSignals(True)
        self.trace_list.clear()
        for trace_name in self._trace_names_for_current_recording():
            item = QListWidgetItem(self._display_label_for_trace(trace_name))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, trace_name)
            item.setSelected(trace_name in selected)
            self.trace_list.addItem(item)
        self.trace_list.blockSignals(False)

    def _on_recording_selected(self, index: int) -> None:
        if index < 0:
            return
        self.current_recording_id = str(self.recording_selector.itemData(index) or ALL_RECORDINGS_FILTER)
        self._refresh_trace_list()
        self._refresh_preview_traces()
        self._sync_controls_for_preview_trace(reset_window=True)

    def select_all_traces(self) -> None:
        self.trace_list.blockSignals(True)
        for row in range(self.trace_list.count()):
            item = self.trace_list.item(row)
            item.setSelected(True)
        self.trace_list.blockSignals(False)
        self._sync_selected_trace_set_from_visible()
        self._refresh_preview_traces()
        self._sync_controls_for_preview_trace(reset_window=False)

    def clear_all_traces(self) -> None:
        self._selected_trace_names.clear()
        self.trace_list.blockSignals(True)
        self.trace_list.clearSelection()
        self.trace_list.blockSignals(False)
        self._refresh_preview_traces()
        self._sync_controls_for_preview_trace(reset_window=False)

    def selected_traces(self) -> List[str]:
        if self.package is None:
            return []
        return [
            name
            for name in self.package.trace_names
            if name in self._selected_trace_names and name in self.package.traces
        ]

    def _preview_trace_name(self) -> Optional[str]:
        value = self.preview_trace_combo.currentData()
        if value is not None:
            return str(value)
        names = self.selected_traces()
        return names[0] if names else None

    def _preview_state(self):
        if self.package is None:
            return None
        name = self._preview_trace_name()
        if not name:
            return None
        return self.package.traces.get(name)

    def _recording_preview_selection(self) -> tuple[str, List[str]]:
        if self.package is None:
            return "", []
        selected = self.selected_traces() or list(self.package.trace_names)
        preview_name = self._preview_trace_name()
        if self.current_recording_id not in {None, ALL_RECORDINGS_FILTER}:
            recording_id = str(self.current_recording_id)
        elif preview_name:
            recording_id = split_trace_identity(preview_name)[0]
        elif selected:
            recording_id = split_trace_identity(selected[0])[0]
        else:
            return "", []
        names = [
            name
            for name in selected
            if split_trace_identity(name)[0] == recording_id and name in self.package.traces
        ]
        if not names:
            names = [
                name
                for name in self.package.trace_names
                if split_trace_identity(name)[0] == recording_id and name in self.package.traces
            ]
        return recording_id, names

    def _refresh_preview_traces(self) -> None:
        if self.package is None:
            return
        current = self.preview_trace_combo.currentData()
        selected = self.selected_traces() or list(self.package.trace_names)
        self.preview_trace_combo.blockSignals(True)
        self.preview_trace_combo.clear()
        for name in selected:
            self.preview_trace_combo.addItem(self._display_label_for_trace(str(name)), name)
        if current in selected:
            self.preview_trace_combo.setCurrentIndex(selected.index(current))
        self.preview_trace_combo.blockSignals(False)

    def _render_options_from_ui(self) -> RenderOptions:
        selected_channels = tuple(
            channel
            for channel, checkbox in self.channel_checks.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        )
        return RenderOptions(
            shade_plateaus=self.chk_shade_plateaus.isChecked(),
            show_spike_markers=self.chk_spike_markers.isChecked(),
            label_spikes=self.chk_label_spikes.isChecked(),
            label_plateaus=self.chk_label_plateaus.isChecked(),
            marker_style=str(self.marker_style_combo.currentData() or "dot"),
            trace_window=TraceWindow.from_values(
                self.chk_manual_window.isChecked(),
                self.window_start.value(),
                self.window_end.value(),
            ),
            spike_half_window_ms=float(self.spike_half_window.value()),
            plateau_pre_ms=float(self.plateau_pre_window.value()),
            plateau_post_ms=float(self.plateau_post_window.value()),
            selected_channels=selected_channels,
            aligned_panel_channel=str(self.aligned_panel_channel.currentData() or "denoised"),
            aligned_panel_y_min=float(self.aligned_panel_y_min.value()),
            aligned_panel_y_max=float(self.aligned_panel_y_max.value()),
            aligned_panel_auto_y=True,
            aligned_panel_share_y=self.chk_aligned_panel_share_y.isChecked(),
        )

    def _on_trace_selection_changed(self) -> None:
        self._sync_selected_trace_set_from_visible()
        self._refresh_preview_traces()
        self._sync_controls_for_preview_trace(reset_window=False)

    def _sync_selected_trace_set_from_visible(self) -> None:
        if self.package is None:
            self._selected_trace_names.clear()
            return
        visible_names = set(self._trace_names_for_current_recording())
        self._selected_trace_names.difference_update(visible_names)
        for item in self.trace_list.selectedItems():
            trace_name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if trace_name is not None and str(trace_name) in self.package.traces:
                self._selected_trace_names.add(str(trace_name))

    def _on_preview_trace_changed(self) -> None:
        self._sync_controls_for_preview_trace(reset_window=True)

    def _on_preview_type_changed(self) -> None:
        preview_type = str(self.preview_type_combo.currentData() or "trace_view")
        needs_event = preview_type in {"spike_event", "plateau_zoom"}
        self.event_label.setVisible(needs_event)
        self.event_combo.setVisible(needs_event)
        self._update_event_selector()

    def _sync_controls_for_preview_trace(self, reset_window: bool = False) -> None:
        state = self._preview_state()
        if self.package is None or state is None:
            return
        available = set(available_channel_ids(state))
        unavailable = []
        for channel, checkbox in self.channel_checks.items():
            is_available = channel in available
            checkbox.setEnabled(is_available)
            if not is_available:
                checkbox.setChecked(False)
                unavailable.append(checkbox.text())
        if not any(checkbox.isChecked() and checkbox.isEnabled() for checkbox in self.channel_checks.values()):
            corrected = self.channel_checks.get("corrected")
            if corrected is not None and corrected.isEnabled():
                corrected.setChecked(True)
        self.render_status.setText(
            "Unavailable for this trace: " + ", ".join(unavailable)
            if unavailable
            else "All selected render channels are available for this trace."
        )

        data = prepare_trace_for_render(state, self.package.detection_params, RenderOptions(), apply_trace_window=False)
        corrected = data.array("corrected")
        if corrected is None or data.time.size == 0:
            self.overview_curve.setData([], [])
            return

        lo = float(data.time[0])
        hi = float(data.time[-1])
        self._syncing_window = True
        try:
            self.window_start.setRange(lo, hi)
            self.window_end.setRange(lo, hi)
            if reset_window or not self.chk_manual_window.isChecked():
                self.window_start.setValue(lo)
                self.window_end.setValue(hi)
            self.overview_region.setBounds((lo, hi))
            self.overview_region.setRegion((self.window_start.value(), self.window_end.value()))
            self.overview_plot.setXRange(lo, hi, padding=0)
            self.overview_curve.setData(data.time, corrected)
            finite = corrected[np.isfinite(corrected)]
            if finite.size:
                self.overview_plot.setYRange(float(np.min(finite)), float(np.max(finite)), padding=0.05)
        finally:
            self._syncing_window = False
        self._update_event_selector()

    def _on_manual_window_toggled(self, checked: bool) -> None:
        self.window_start.setEnabled(bool(checked))
        self.window_end.setEnabled(bool(checked))
        try:
            self.overview_region.setMovable(bool(checked))
        except AttributeError:
            pass
        self._update_event_selector()

    def _on_region_changed(self) -> None:
        if self._syncing_window:
            return
        start, end = self.overview_region.getRegion()
        self._syncing_window = True
        try:
            if not self.chk_manual_window.isChecked():
                self.chk_manual_window.setChecked(True)
            self.window_start.setValue(float(start))
            self.window_end.setValue(float(end))
        finally:
            self._syncing_window = False
        self._update_event_selector()

    def _on_window_spin_changed(self, *_args: object) -> None:
        if self._syncing_window:
            return
        self._syncing_window = True
        try:
            if not self.chk_manual_window.isChecked():
                self.chk_manual_window.setChecked(True)
            start = self.window_start.value()
            end = self.window_end.value()
            if end < start:
                start, end = end, start
            self.overview_region.setRegion((start, end))
        finally:
            self._syncing_window = False
        self._update_event_selector()

    def _update_event_selector(self) -> None:
        if self.package is None:
            return
        state = self._preview_state()
        if state is None:
            return
        preview_type = str(self.preview_type_combo.currentData() or "trace_view")
        options = self._render_options_from_ui()
        data = prepare_trace_for_render(state, self.package.detection_params, options, apply_trace_window=False)

        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        if preview_type == "spike_event":
            for idx, spike in enumerate(filtered_spikes(data, options), 1):
                self.event_combo.addItem(f"S{idx:04d} | t={float(spike.spike_time):.6g}", idx - 1)
        elif preview_type == "plateau_zoom":
            for idx, (start, _end) in enumerate(plateau_regions(data, options), 1):
                t = float(data.time[start]) if 0 <= start < data.time.size else float("nan")
                self.event_combo.addItem(f"P{idx:04d} | t={t:.6g}", idx - 1)
        self.event_combo.blockSignals(False)

    def refresh_preview(self) -> None:
        if self.package is None:
            return
        options = self._render_options_from_ui()
        preview_type = str(self.preview_type_combo.currentData() or "trace_view")
        if preview_type in {
            "recording_event_raster",
            "recording_aligned_traces",
            "recording_aligned_trace_panels",
            "recording_aligned_trace_panels_raw",
        }:
            recording_id, names = self._recording_preview_selection()
            states = [self.package.traces[name] for name in names if name in self.package.traces]
            if not recording_id or not states:
                self.status_label.setText("Select a recording to preview.")
                return
            try:
                if preview_type == "recording_event_raster":
                    pix, _metadata = render_recording_event_raster(
                        recording_id,
                        states,
                        self.package.detection_params,
                        options,
                    )
                elif preview_type in {"recording_aligned_trace_panels", "recording_aligned_trace_panels_raw"}:
                    panel_options = (
                        replace(options, normalization_mode=NORMALIZATION_NATIVE, aligned_panel_channel="raw")
                        if preview_type == "recording_aligned_trace_panels_raw"
                        else options
                    )
                    pix, _metadata = render_recording_aligned_trace_panels(
                        recording_id,
                        states,
                        self.package.detection_params,
                        panel_options,
                    )
                else:
                    pix, _metadata = render_recording_aligned_traces(
                        recording_id,
                        states,
                        self.package.detection_params,
                        options,
                    )
            except Exception as exc:
                self.preview_image.setText(str(exc))
                self.status_label.setText("Preview failed")
                return
            if pix.isNull():
                self.preview_image.setPixmap(QtGui.QPixmap())
                self.preview_image.setText("Recording preview unavailable for the selected ROIs.")
                self.status_label.setText("Preview unavailable")
                return
            self.preview_image.setText("")
            self.preview_image.setPixmap(pix)
            self.preview_image.adjustSize()
            self.status_label.setText(f"Preview refreshed for {recording_id}")
            return

        state = self._preview_state()
        if state is None:
            self.status_label.setText("Select a trace to preview.")
            return
        event_index = int(self.event_combo.currentData() or 0)
        try:
            pix = render_preview_pixmap(state, self.package.detection_params, options, preview_type, event_index)
        except Exception as exc:
            self.preview_image.setText(str(exc))
            self.status_label.setText("Preview failed")
            return
        if pix.isNull():
            self.preview_image.setPixmap(QtGui.QPixmap())
            self.preview_image.setText("Preview unavailable for the selected trace/output.")
            self.status_label.setText("Preview unavailable")
            return
        self.preview_image.setText("")
        self.preview_image.setPixmap(pix)
        self.preview_image.adjustSize()
        self.status_label.setText("Preview refreshed")

    def export_selected(self) -> None:
        if self.package is None:
            return
        trace_names = self.selected_traces()
        if not trace_names:
            QMessageBox.warning(self, "No Traces Selected", "Select at least one trace to export.")
            return
        out_dir = self.output_path.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No Output Folder", "Choose an output folder.")
            return
        options = ExportOptions(
            tables=self.chk_tables.isChecked(),
            reload_files=self.chk_reload.isChecked(),
            qc_report=self.chk_qc_report.isChecked(),
            spike_statistics=self.chk_spike_stats.isChecked(),
            trace_images=self.chk_trace_images.isChecked(),
            recording_event_raster=self.chk_recording_event_raster.isChecked(),
            recording_aligned_traces=self.chk_recording_aligned_traces.isChecked(),
            recording_aligned_trace_panels=self.chk_recording_aligned_trace_panels.isChecked(),
            recording_aligned_trace_panels_raw=self.chk_recording_aligned_trace_panels_raw.isChecked(),
            spike_images=self.chk_spike_images.isChecked(),
            plateau_images=self.chk_plateau_images.isChecked(),
            denoising_debug=self.chk_denoising_debug.isChecked(),
            long_plateau_debug=self.chk_long_plateau_debug.isChecked(),
            render_options=self._render_options_from_ui(),
        )
        if not options.has_any_output():
            QMessageBox.warning(self, "No Outputs Selected", "Select at least one export output.")
            return
        self.btn_export.setEnabled(False)
        self.status_label.setText("Exporting...")
        try:
            written = export_analysis_package(self.package, out_dir, trace_names, options)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            self.status_label.setText("Export failed")
        else:
            self.status_label.setText(f"Exported {len(written)} files to {out_dir}")
        finally:
            self.btn_export.setEnabled(True)

    def _result_root(self) -> Path:
        return Path(__file__).resolve().parents[3] / "Result"
