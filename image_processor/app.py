"""PySide6 UI for Femtonics TIFF ROI processing."""

from __future__ import annotations

import sys
import subprocess
import traceback
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

try:
    from .roi_core import (
        compute_roi_layout,
        iter_tiff_inputs,
        load_tiff_info,
        metadata_path_for_tiff,
        parse_metadata,
        split_tiff,
        validate_tiff_matches_metadata,
    )
    from .signal_export import default_export_output_dir, export_signal_documents, extract_roi_signals
    from .summary_images import upscale_export_image
    from .tiff_backend import (
        PreviewFrame,
        PreviewFrameCache,
        build_persistent_preview_cache,
        iter_tiff_frames,
        persistent_preview_cache_available,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from roi_core import (
        compute_roi_layout,
        iter_tiff_inputs,
        load_tiff_info,
        metadata_path_for_tiff,
        parse_metadata,
        split_tiff,
        validate_tiff_matches_metadata,
    )
    from signal_export import default_export_output_dir, export_signal_documents, extract_roi_signals
    from summary_images import upscale_export_image
    from tiff_backend import (
        PreviewFrame,
        PreviewFrameCache,
        build_persistent_preview_cache,
        iter_tiff_frames,
        persistent_preview_cache_available,
    )

try:
    from PySide6.QtCore import QObject, QProcess, QRunnable, QRectF, Qt, QThreadPool, Signal, Slot
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    HAS_QT = True
    QT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only on machines without Qt.
    HAS_QT = False
    QT_IMPORT_ERROR = exc


APP_STYLE = """
QWidget {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}
QGroupBox {
    font-weight: 600;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QTableWidget {
    gridline-color: #666666;
}
QTableWidget::item {
    color: #111111;
}
QHeaderView::section {
    background-color: #30343a;
    color: #f5f7fa;
    padding: 4px;
    border: 1px solid #4b4f56;
}
QPlainTextEdit {
    font-family: Consolas, "Courier New", monospace;
    font-size: 9pt;
}
"""


def finite_min_max(frame: np.ndarray) -> tuple[float, float]:
    values = np.asarray(frame, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def dtype_display_bounds(frame: np.ndarray) -> tuple[float, float]:
    dtype = np.asarray(frame).dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return finite_min_max(frame)


def auto_display_range(
    frame: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[float, float]:
    values = np.asarray(frame, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, lower_percentile))
    high = float(np.percentile(finite, upper_percentile))
    if high <= low:
        observed_low, observed_high = finite_min_max(frame)
        low, high = observed_low, observed_high
    if high <= low:
        high = low + 1.0
    return low, high


def scale_frame_to_uint8(frame: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    if white_level <= black_level:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - float(black_level)) * (255.0 / (float(white_level) - float(black_level)))
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def frame_to_rgb888(frame: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    scaled = scale_frame_to_uint8(frame, black_level, white_level)
    if scaled.ndim == 2:
        return np.repeat(scaled[:, :, None], 3, axis=2)
    if scaled.ndim == 3 and scaled.shape[2] >= 3:
        return scaled[:, :, :3]
    raise ValueError(f"Cannot render frame with shape {scaled.shape}.")


def average_tiff_frames(
    tiff_path: Path,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    accumulator: np.ndarray | None = None
    frame_shape: tuple[int, ...] | None = None

    frame_count = load_tiff_info(tiff_path).frame_count
    if frame_count <= 0:
        raise ValueError("Cannot average an empty TIFF stack.")

    for frame_number, raw_frame in enumerate(iter_tiff_frames(tiff_path)):
        if frame_number >= frame_count:
            break
        frame = np.asarray(raw_frame, dtype=np.float64)
        if accumulator is None:
            frame_shape = frame.shape
            accumulator = np.zeros(frame_shape, dtype=np.float64)
        elif frame.shape != frame_shape:
            raise ValueError("Cannot average TIFF frames with different shapes.")

        accumulator += frame
        if (
            progress is not None
            and (
                frame_number == 0
                or (frame_number + 1) % 500 == 0
                or frame_number + 1 == frame_count
            )
        ):
            progress(frame_number + 1, frame_count)

    if accumulator is None:
        raise ValueError("Cannot average an empty TIFF stack.")
    return accumulator / float(frame_count)


def default_average_png_path(tiff_path: Path, output_dir: Path | None = None) -> Path:
    return default_export_output_dir(tiff_path, output_dir) / f"{tiff_path.stem}_average_projection.png"


def save_tiff_average_png(
    tiff_path: Path,
    output_path: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    average_frame = average_tiff_frames(tiff_path, progress=progress)
    black_level, white_level = auto_display_range(average_frame)
    rgb = np.ascontiguousarray(frame_to_rgb888(average_frame, black_level, white_level))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upscale_export_image(Image.fromarray(rgb)).save(output_path)
    return output_path


def native_image_processor_executable(repo_root: Path | None = None) -> Path:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    return root / "image_processor" / "native_viewer" / "build" / "femtonics_image_processor.exe"


def native_viewer_executable(repo_root: Path | None = None) -> Path:
    return native_image_processor_executable(repo_root)


def resolved_native_image_processor_executable(repo_root: Path | None = None) -> Path:
    return native_image_processor_executable(repo_root)


def native_viewer_build_command(repo_root: Path | None = None) -> list[str]:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    env_script = root / "tools" / "native_dev_env.bat"
    source_dir = root / "image_processor" / "native_viewer"
    build_dir = source_dir / "build"
    command = (
        f'call "{env_script}" > nul '
        f'&& cmake -S "{source_dir}" -B "{build_dir}" -G Ninja '
        f'-DCMAKE_BUILD_TYPE=Release '
        f'-DCMAKE_PREFIX_PATH="!QT_ROOT!" '
        f'-DCMAKE_TOOLCHAIN_FILE="!VCPKG_ROOT!\\scripts\\buildsystems\\vcpkg.cmake" '
        f'-DVCPKG_TARGET_TRIPLET=x64-windows '
        f'&& cmake --build "{build_dir}"'
    )
    return ["cmd", "/v:on", "/c", command]


def build_native_viewer(repo_root: Path | None = None) -> int:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    env_script = root / "tools" / "native_dev_env.bat"
    if not env_script.exists():
        print(f"Native build environment script is missing:\n{env_script}", file=sys.stderr)
        return 1
    print("Native image processor UI is not built yet; attempting native build...", file=sys.stderr)
    return int(subprocess.call(native_viewer_build_command(root), cwd=str(root)))


def native_viewer_missing_message(executable: Path | None = None) -> str:
    exe = executable if executable is not None else native_image_processor_executable()
    return (
        "Native image processor UI is not built yet and could not be built automatically.\n\n"
        f"Expected executable:\n{exe}\n\n"
        "Build it from the repository root with:\n"
        "  call tools\\native_dev_env.bat\n"
        "  cmake -S image_processor\\native_viewer -B image_processor\\native_viewer\\build -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=\"%QT_ROOT%\" "
        "-DCMAKE_TOOLCHAIN_FILE=\"%VCPKG_ROOT%\\scripts\\buildsystems\\vcpkg.cmake\" "
        "-DVCPKG_TARGET_TRIPLET=x64-windows\n"
        "  cmake --build image_processor\\native_viewer\\build\n\n"
        "Use --legacy-python-viewer only if you intentionally want the older Python UI."
    )


def native_viewer_cli_args(argv: Sequence[str]) -> list[str]:
    args: list[str] = []
    for arg in argv:
        if arg.startswith("-"):
            args.append(arg)
            continue

        candidate = Path(arg).expanduser()
        if candidate.exists():
            args.append(str(candidate.resolve()))
        else:
            args.append(arg)
    return args


def launch_native_viewer(argv: Sequence[str] | None = None, repo_root: Path | None = None) -> int:
    args = native_viewer_cli_args(list(argv or []))
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    executable = resolved_native_image_processor_executable(repo_root)
    if not executable.exists():
        if build_native_viewer(root) != 0 or not executable.exists():
            executable = resolved_native_image_processor_executable(repo_root)
        if not executable.exists():
            print(native_viewer_missing_message(executable), file=sys.stderr)
            return 1

    try:
        return int(subprocess.call([str(executable), *args], cwd=str(executable.parent)))
    except OSError as exc:
        print(f"Could not launch native image processor UI:\n{executable}\n\n{exc}", file=sys.stderr)
        return 1


def _qt_required() -> None:
    if not HAS_QT:
        raise RuntimeError(
            "PySide6 is required for the image processor UI. "
            "Install the same environment used by the spike detector project."
        ) from QT_IMPORT_ERROR


if HAS_QT:

    class WorkerSignals(QObject):
        log = Signal(str)
        error = Signal(str)
        result = Signal(object)
        finished = Signal()


    class FunctionWorker(QRunnable):
        def __init__(self, fn: Callable[[WorkerSignals], object]) -> None:
            super().__init__()
            self.setAutoDelete(False)
            self.fn = fn
            self.signals = WorkerSignals()

        @Slot()
        def run(self) -> None:
            try:
                result = self.fn(self.signals)
                self.signals.result.emit(result)
            except Exception:
                self.signals.error.emit(traceback.format_exc())
            finally:
                self.signals.finished.emit()


    class RoiPreviewWidget(QFrame):
        def __init__(self) -> None:
            super().__init__()
            self.setFrameShape(QFrame.Shape.StyledPanel)
            self.setMinimumSize(720, 460)
            self.image: QImage | None = None
            self.layout_model = None
            self.source_to_preview_scale = 1.0

        def set_preview(self, image: QImage, layout_model, source_to_preview_scale: float = 1.0) -> None:
            self.image = image
            self.layout_model = layout_model
            self.source_to_preview_scale = float(source_to_preview_scale)
            self.update()

        def clear_preview(self) -> None:
            self.image = None
            self.layout_model = None
            self.source_to_preview_scale = 1.0
            self.update()

        def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming.
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.fillRect(self.rect(), QColor("#f7f9fc"))

            if self.image is None:
                painter.setPen(QPen(QColor("#6b7280")))
                font = QFont("Segoe UI", 12)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No TIFF loaded")
                return

            margin = 14
            available = self.rect().adjusted(margin, margin, -margin, -margin)
            image_size = self.image.size()
            scale = min(available.width() / image_size.width(), available.height() / image_size.height())
            draw_width = image_size.width() * scale
            draw_height = image_size.height() * scale
            left = available.left() + (available.width() - draw_width) / 2
            top = available.top() + (available.height() - draw_height) / 2
            target = QRectF(left, top, draw_width, draw_height)
            painter.drawImage(target, self.image)

            if self.layout_model is None:
                return

            pen = QPen(QColor("#111111"), 2)
            painter.setPen(pen)
            label_font = QFont("Segoe UI", 9)
            label_font.setBold(True)
            painter.setFont(label_font)
            roi_scale = self.source_to_preview_scale
            for box in self.layout_model.boxes:
                x = left + box.left * roi_scale * scale
                y = top + box.upper * roi_scale * scale
                width = box.width * roi_scale * scale
                height = box.height * roi_scale * scale
                painter.drawRect(QRectF(x, y, width, height))

                label = str(box.roi_index)
                metrics = painter.fontMetrics()
                text_rect = metrics.boundingRect(label).adjusted(-4, -2, 4, 2)
                text_rect.moveTopLeft((QRectF(x + 4, y + 4, text_rect.width(), text_rect.height())).topLeft().toPoint())
                painter.fillRect(text_rect, QColor("#f7f9fc"))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)


    class TiffProcessorMainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Femtonics Image Processor")
            self.resize(1500, 920)
            self.setStyleSheet(APP_STYLE)

            self.input_path = ""
            self.output_path = ""
            self.detected_sampling_rate_hz: float | None = None
            self.current_signal_table = None
            self.preview_tiff_path: Path | None = None
            self.preview_frame_count = 0
            self.preview_frame_index = 0
            self.preview_frame_array: np.ndarray | None = None
            self.preview_frame_source_shape: tuple[int, ...] | None = None
            self.preview_frame_downsample = 1
            self.preview_auto_range: tuple[float, float] = (0.0, 1.0)
            self.preview_layout_model = None
            self.display_full_range: tuple[float, float] = (0.0, 65535.0)
            self._updating_display_controls = False
            self._updating_frame_controls = False
            self._active_workers: list[FunctionWorker] = []
            self._active_preview_workers: list[FunctionWorker] = []
            self._active_prefetch_workers: list[FunctionWorker] = []
            self._active_cache_build_workers: list[FunctionWorker] = []
            self._preview_cache_build_paths: set[Path] = set()
            self._preview_worker_active = False
            self._pending_preview_frame_index: int | None = None
            self._pending_preview_request_id: int | None = None
            self._preview_request_id = 0
            self._preview_worker_request_id: int | None = None
            self.preview_cache = PreviewFrameCache(max_frames=6, max_edge=1536)
            self.thread_pool = QThreadPool(self)
            self.thread_pool.setMaxThreadCount(2)
            self.preview_thread_pool = QThreadPool(self)
            self.preview_thread_pool.setMaxThreadCount(1)
            self.prefetch_thread_pool = QThreadPool(self)
            self.prefetch_thread_pool.setMaxThreadCount(1)
            self.cache_build_thread_pool = QThreadPool(self)
            self.cache_build_thread_pool.setMaxThreadCount(1)

            root = QWidget()
            self.setCentralWidget(root)
            root_layout = QVBoxLayout(root)

            top_bar = QHBoxLayout()
            root_layout.addLayout(top_bar)

            self.btn_load_tiff = QPushButton("Load TIFF")
            self.btn_load_folder = QPushButton("Load Folder")
            self.btn_output_folder = QPushButton("Output Folder")
            self.btn_load_preview = QPushButton("Load Preview")
            self.btn_native_viewer = QPushButton("Native UI")
            self.btn_split = QPushButton("Split TIFFs")
            self.btn_calculate = QPushButton("Calculate Signals")
            self.btn_export = QPushButton("Export Excel")
            self.btn_average_png = QPushButton("Average PNG")

            for button in [
                self.btn_load_tiff,
                self.btn_load_folder,
                self.btn_output_folder,
                self.btn_load_preview,
                self.btn_native_viewer,
                self.btn_split,
                self.btn_calculate,
                self.btn_export,
                self.btn_average_png,
            ]:
                top_bar.addWidget(button)

            self.status_label = QLabel("No file loaded")
            top_bar.addWidget(self.status_label, stretch=1)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            root_layout.addWidget(splitter, stretch=1)

            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_layout.addWidget(QLabel("TIFF Queue"))
            self.tiff_list = QListWidget()
            self.tiff_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.tiff_list.setAlternatingRowColors(True)
            left_layout.addWidget(self.tiff_list, stretch=1)
            self.queue_status = QLabel("0 TIFFs")
            left_layout.addWidget(self.queue_status)

            left_buttons = QHBoxLayout()
            self.btn_select_all = QPushButton("Select All")
            self.btn_clear_selection = QPushButton("Clear Selection")
            left_buttons.addWidget(self.btn_select_all)
            left_buttons.addWidget(self.btn_clear_selection)
            left_layout.addLayout(left_buttons)

            center_panel = QWidget()
            center_layout = QVBoxLayout(center_panel)
            self.preview = RoiPreviewWidget()
            center_layout.addWidget(self.preview, stretch=1)

            frame_controls = QWidget()
            frame_layout = QHBoxLayout(frame_controls)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            self.frame_slider = QSlider(Qt.Orientation.Horizontal)
            self.frame_slider.setTracking(False)
            self.frame_slider.setEnabled(False)
            self.frame_spin = QSpinBox()
            self.frame_spin.setRange(1, 1)
            self.frame_spin.setEnabled(False)
            self.frame_total_label = QLabel("of 0")
            frame_layout.addWidget(QLabel("Frame"))
            frame_layout.addWidget(self.frame_slider, stretch=1)
            frame_layout.addWidget(self.frame_spin)
            frame_layout.addWidget(self.frame_total_label)
            center_layout.addWidget(frame_controls)

            self.metadata_table = QTableWidget(0, 2)
            self.metadata_table.setHorizontalHeaderLabels(["Field", "Value"])
            self.metadata_table.horizontalHeader().setStretchLastSection(True)
            self.metadata_table.setAlternatingRowColors(True)
            self.metadata_table.setMaximumHeight(190)
            center_layout.addWidget(self.metadata_table)

            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)

            source_group = QGroupBox("Source")
            source_layout = QFormLayout(source_group)
            self.input_label = QLineEdit()
            self.input_label.setReadOnly(True)
            self.input_label.setPlaceholderText("No input selected")
            self.output_label = QLineEdit()
            self.output_label.setReadOnly(True)
            self.output_label.setPlaceholderText("Default: source folder")
            self.recursive_check = QCheckBox("Recursive folder scan")
            self.recursive_check.setChecked(True)
            source_layout.addRow("Input", self.input_label)
            source_layout.addRow("Output", self.output_label)
            source_layout.addRow(self.recursive_check)
            right_layout.addWidget(source_group)

            processing_group = QGroupBox("Processing")
            processing_layout = QFormLayout(processing_group)
            self.use_metadata_sampling_check = QCheckBox("Use metadata sampling rate")
            self.use_metadata_sampling_check.setChecked(True)
            self.detected_sampling_label = QLabel("-")
            self.sampling_rate_spin = QDoubleSpinBox()
            self.sampling_rate_spin.setRange(0.0, 1_000_000.0)
            self.sampling_rate_spin.setDecimals(3)
            self.sampling_rate_spin.setSingleStep(1.0)
            self.sampling_rate_spin.setSpecialValueText("auto")
            self.sampling_rate_spin.setEnabled(False)
            sampling_mode_row = QWidget()
            sampling_mode_layout = QHBoxLayout(sampling_mode_row)
            sampling_mode_layout.setContentsMargins(0, 0, 0, 0)
            self.btn_reset_sampling = QPushButton("Reset to Metadata")
            sampling_mode_layout.addWidget(self.use_metadata_sampling_check)
            sampling_mode_layout.addWidget(self.btn_reset_sampling)
            self.write_csv_check = QCheckBox("Write CSV beside Excel")
            self.write_csv_check.setChecked(True)
            self.progress = QProgressBar()
            self.progress.setRange(0, 0)
            self.progress.hide()
            processing_layout.addRow(sampling_mode_row)
            processing_layout.addRow("Detected Hz", self.detected_sampling_label)
            processing_layout.addRow("Manual Hz", self.sampling_rate_spin)
            processing_layout.addRow(self.write_csv_check)
            processing_layout.addRow("Progress", self.progress)
            right_layout.addWidget(processing_group)

            self.display_group = QGroupBox("Brightness / Contrast")
            display_layout = QFormLayout(self.display_group)
            self.display_observed_label = QLabel("-")
            self.display_black_spin = QDoubleSpinBox()
            self.display_white_spin = QDoubleSpinBox()
            for spin in [self.display_black_spin, self.display_white_spin]:
                spin.setRange(0.0, 65535.0)
                spin.setDecimals(2)
                spin.setSingleStep(100.0)
            display_buttons = QWidget()
            display_button_layout = QHBoxLayout(display_buttons)
            display_button_layout.setContentsMargins(0, 0, 0, 0)
            self.btn_display_auto = QPushButton("Auto")
            self.btn_display_full = QPushButton("Full Range")
            display_button_layout.addWidget(self.btn_display_auto)
            display_button_layout.addWidget(self.btn_display_full)
            display_layout.addRow("Observed", self.display_observed_label)
            display_layout.addRow("Black", self.display_black_spin)
            display_layout.addRow("White", self.display_white_spin)
            display_layout.addRow(display_buttons)
            self.display_group.setEnabled(False)
            right_layout.addWidget(self.display_group)

            metadata_group = QGroupBox("Loaded Metadata")
            metadata_layout = QFormLayout(metadata_group)
            self.meta_size = QLabel("-")
            self.meta_frames = QLabel("-")
            self.meta_rois = QLabel("-")
            self.meta_grid = QLabel("-")
            self.meta_channel = QLabel("-")
            self.meta_sampling = QLabel("-")
            metadata_layout.addRow("Frame size", self.meta_size)
            metadata_layout.addRow("Frames", self.meta_frames)
            metadata_layout.addRow("ROIs", self.meta_rois)
            metadata_layout.addRow("Grid", self.meta_grid)
            metadata_layout.addRow("Channel", self.meta_channel)
            metadata_layout.addRow("Sampling", self.meta_sampling)
            right_layout.addWidget(metadata_group)

            log_group = QGroupBox("Run Log")
            log_layout = QVBoxLayout(log_group)
            self.log_text = QPlainTextEdit()
            self.log_text.setReadOnly(True)
            log_layout.addWidget(self.log_text)
            right_layout.addWidget(log_group, stretch=1)

            splitter.addWidget(left_panel)
            splitter.addWidget(center_panel)
            splitter.addWidget(right_panel)
            splitter.setSizes([260, 890, 350])

            self._connect_signals()

        def _connect_signals(self) -> None:
            self.btn_load_tiff.clicked.connect(self.load_tiff)
            self.btn_load_folder.clicked.connect(self.load_folder)
            self.btn_output_folder.clicked.connect(self.choose_output_folder)
            self.btn_load_preview.clicked.connect(self.load_preview)
            self.btn_native_viewer.clicked.connect(self.open_native_viewer)
            self.btn_split.clicked.connect(self.split_tiffs)
            self.btn_calculate.clicked.connect(self.calculate_signals)
            self.btn_export.clicked.connect(self.export_excel)
            self.btn_average_png.clicked.connect(self.export_average_png)
            self.btn_select_all.clicked.connect(self.tiff_list.selectAll)
            self.btn_clear_selection.clicked.connect(self.tiff_list.clearSelection)
            self.use_metadata_sampling_check.toggled.connect(self._on_sampling_mode_changed)
            self.btn_reset_sampling.clicked.connect(self.reset_sampling_to_metadata)
            self.recursive_check.toggled.connect(self.refresh_queue_if_loaded)
            self.tiff_list.itemSelectionChanged.connect(self._on_queue_selection_changed)
            self.tiff_list.currentItemChanged.connect(self._on_queue_current_changed)
            self.display_black_spin.valueChanged.connect(self._on_display_levels_changed)
            self.display_white_spin.valueChanged.connect(self._on_display_levels_changed)
            self.btn_display_auto.clicked.connect(self.set_display_auto)
            self.btn_display_full.clicked.connect(self.set_display_full_range)
            self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
            self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)

        def log(self, message: str) -> None:
            self.log_text.appendPlainText(message)
            self.status_label.setText(message)

        def log_background(self, message: str) -> None:
            self.log_text.appendPlainText(message)

        def load_tiff(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select TIFF or metadata file",
                "",
                "TIFF or metadata (*.tif *.tiff *.metadata.txt);;All files (*.*)",
            )
            if path:
                self.input_path = path
                self.input_label.setText(path)
                self.refresh_queue()

        def load_folder(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select folder containing TIFF files")
            if path:
                self.input_path = path
                self.input_label.setText(path)
                self.refresh_queue()

        def choose_output_folder(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Select output folder")
            if path:
                self.output_path = path
                self.output_label.setText(path)

        def open_native_viewer(self) -> None:
            try:
                tiff_path = self.current_tiff()
            except Exception as exc:
                QMessageBox.warning(self, "Native UI", str(exc))
                return

            executable = resolved_native_image_processor_executable()
            if not executable.exists():
                message = (
                    "Native image processor UI executable was not found.\n\n"
                    f"Expected:\n{executable}\n\n"
                    "Build it first with:\n"
                    "call tools\\native_dev_env.bat\n"
                    "cmake -S image_processor\\native_viewer -B image_processor\\native_viewer\\build -G Ninja "
                    "-DCMAKE_BUILD_TYPE=Release "
                    "-DCMAKE_PREFIX_PATH=\"%QT_ROOT%\" "
                    "-DCMAKE_TOOLCHAIN_FILE=\"%VCPKG_ROOT%\\scripts\\buildsystems\\vcpkg.cmake\" "
                    "-DVCPKG_TARGET_TRIPLET=x64-windows\n"
                    "cmake --build image_processor\\native_viewer\\build"
                )
                QMessageBox.information(self, "Native UI", message)
                self.log(f"Native image processor UI executable not found: {executable}")
                return

            result = QProcess.startDetached(str(executable), [str(tiff_path)], str(executable.parent))
            started = result[0] if isinstance(result, tuple) else bool(result)
            if not started:
                QMessageBox.critical(self, "Native UI", f"Could not launch:\n{executable}")
                self.log(f"Native image processor UI launch failed: {executable}")
                return

            self.log(f"Opened native image processor UI: {tiff_path.name}")

        def refresh_queue(self) -> None:
            self.detected_sampling_rate_hz = None
            self.detected_sampling_label.setText("-")
            self.meta_sampling.setText("-")
            self.metadata_table.setRowCount(0)
            self.preview.clear_preview()
            self.preview_tiff_path = None
            self.preview_frame_count = 0
            self.preview_frame_index = 0
            self.preview_frame_array = None
            self.preview_frame_source_shape = None
            self.preview_frame_downsample = 1
            self.preview_auto_range = (0.0, 1.0)
            self.preview_layout_model = None
            self._preview_request_id += 1
            self._pending_preview_frame_index = None
            self._pending_preview_request_id = None
            self._preview_worker_active = False
            self._preview_worker_request_id = None
            self.preview_cache.clear()
            self._configure_frame_controls(0, 0)
            self.display_group.setEnabled(False)
            self.tiff_list.blockSignals(True)
            self.tiff_list.clear()
            for tiff_path in self.discover_tiffs():
                item = QListWidgetItem(tiff_path.name)
                item.setToolTip(str(tiff_path))
                item.setData(Qt.ItemDataRole.UserRole, str(tiff_path))
                self.tiff_list.addItem(item)
            if self.tiff_list.count():
                self.tiff_list.setCurrentRow(0)
                self.tiff_list.selectAll()
            self.tiff_list.blockSignals(False)
            if self.tiff_list.count():
                self._update_queue_status()
                self.load_preview()
                self.log(f"Loaded {self.tiff_list.count()} TIFF file(s)")
            else:
                self._update_queue_status()
                self.log("No TIFF files found")

        def refresh_queue_if_loaded(self) -> None:
            if self.input_path:
                self.refresh_queue()

        def discover_tiffs(self) -> list[Path]:
            if not self.input_path:
                return []
            return iter_tiff_inputs(Path(self.input_path), self.recursive_check.isChecked())

        def selected_tiffs(self) -> list[Path]:
            selected_paths = {
                str(item.data(Qt.ItemDataRole.UserRole))
                for item in self.tiff_list.selectedItems()
            }
            if selected_paths:
                return [
                    Path(path)
                    for path in self._queue_paths()
                    if str(path) in selected_paths
                ]
            current = self.tiff_list.currentItem()
            if current is not None:
                return [Path(current.data(Qt.ItemDataRole.UserRole))]
            return self.discover_tiffs()

        def _queue_paths(self) -> list[Path]:
            return [
                Path(self.tiff_list.item(row).data(Qt.ItemDataRole.UserRole))
                for row in range(self.tiff_list.count())
            ]

        def current_tiff(self) -> Path:
            item = self.tiff_list.currentItem()
            if item is not None:
                return Path(item.data(Qt.ItemDataRole.UserRole))
            tiffs = self.selected_tiffs()
            if not tiffs:
                raise ValueError("No TIFF files found.")
            return tiffs[0]

        def output_dir(self) -> Path | None:
            return Path(self.output_path) if self.output_path else None

        def sampling_rate_override(self) -> float | None:
            if self.use_metadata_sampling_check.isChecked():
                return None
            value = float(self.sampling_rate_spin.value())
            if value <= 0:
                raise ValueError("Manual sampling rate must be greater than zero.")
            return value

        def _on_sampling_mode_changed(self, use_metadata: bool) -> None:
            self.sampling_rate_spin.setEnabled(not use_metadata)
            if use_metadata:
                self.sampling_rate_spin.setValue(0.0)
            elif self.sampling_rate_spin.value() <= 0:
                self.sampling_rate_spin.setValue(
                    float(self.detected_sampling_rate_hz)
                    if self.detected_sampling_rate_hz is not None
                    else 1.0
                )

        def reset_sampling_to_metadata(self) -> None:
            self.use_metadata_sampling_check.setChecked(True)
            self.sampling_rate_spin.setValue(0.0)
            self.status_label.setText("Sampling mode reset to metadata")

        def _on_queue_selection_changed(self) -> None:
            self._update_queue_status()

        def _on_queue_current_changed(self, *_args) -> None:
            self._update_queue_status()
            if self.tiff_list.currentItem() is None:
                return
            try:
                self.load_preview()
            except Exception:
                pass

        def _update_queue_status(self) -> None:
            total = self.tiff_list.count()
            selected = len(self.tiff_list.selectedItems())
            if total == 0:
                self.queue_status.setText("0 TIFFs")
            elif selected == total:
                self.queue_status.setText(f"{total} TIFFs | all selected")
            elif selected > 0:
                self.queue_status.setText(f"{total} TIFFs | {selected} selected")
            elif self.tiff_list.currentItem() is not None:
                self.queue_status.setText(f"{total} TIFFs | current only")
            else:
                self.queue_status.setText(f"{total} TIFFs | no target")

        def load_preview(self) -> None:
            try:
                tiff_path = self.current_tiff()
                metadata_path = metadata_path_for_tiff(tiff_path)
                metadata = parse_metadata(metadata_path)
                info = load_tiff_info(tiff_path)
                validate_tiff_matches_metadata(info, metadata)
                layout = compute_roi_layout(metadata)
                self.detected_sampling_rate_hz = metadata.sampling_rate_hz
                self.detected_sampling_label.setText(
                    f"{metadata.sampling_rate_hz:g}" if metadata.sampling_rate_hz is not None else "-"
                )

                self.preview_tiff_path = tiff_path
                self.preview_frame_count = info.frame_count
                self.preview_frame_index = 0
                self.preview_frame_array = None
                self.preview_frame_source_shape = None
                self.preview_frame_downsample = 1
                self.preview_auto_range = (0.0, 1.0)
                self.preview_layout_model = layout
                self._configure_frame_controls(info.frame_count, 0)
                self.display_group.setEnabled(False)
                self.preview.clear_preview()
                self._update_metadata(tiff_path, metadata_path, metadata, info, layout)
                self._request_preview_frame(0)
                self._start_persistent_preview_cache_build(tiff_path, info.frame_count)
                self.log(f"Loading preview: {tiff_path.name}")
            except Exception as exc:
                QMessageBox.critical(self, "Load preview", str(exc))
                self.log(f"Preview error: {exc}")

        def _start_persistent_preview_cache_build(self, tiff_path: Path, frame_count: int) -> None:
            if not persistent_preview_cache_available():
                return
            if frame_count <= 0:
                return
            if len(self._active_cache_build_workers) >= 1:
                return
            cache_key = Path(tiff_path).resolve()
            if cache_key in self._preview_cache_build_paths:
                return
            max_edge = self.preview_cache.max_edge
            self._preview_cache_build_paths.add(cache_key)

            def build_cache(_signals: WorkerSignals) -> Path:
                return build_persistent_preview_cache(tiff_path, frame_count, max_edge)

            worker = FunctionWorker(build_cache)

            def cache_ready(cache_path: Path, source_path: Path = Path(tiff_path)) -> None:
                self.log_background(f"Preview cache ready for {source_path.name}: {cache_path}")

            def cache_error(details: str, source_path: Path = Path(tiff_path)) -> None:
                last_line = details.splitlines()[-1] if details else "Unknown error"
                self.log_background(f"Preview cache unavailable for {source_path.name}: {last_line}")

            def finish_worker(
                done_worker: FunctionWorker = worker,
                source_key: Path = cache_key,
            ) -> None:
                if done_worker in self._active_cache_build_workers:
                    self._active_cache_build_workers.remove(done_worker)
                self._preview_cache_build_paths.discard(source_key)

            worker.signals.result.connect(cache_ready)
            worker.signals.error.connect(cache_error)
            worker.signals.finished.connect(finish_worker)
            self._active_cache_build_workers.append(worker)
            self.cache_build_thread_pool.start(worker)

        def _configure_frame_controls(self, frame_count: int, frame_index: int) -> None:
            frame_count = max(0, int(frame_count))
            frame_index = min(max(int(frame_index), 0), max(frame_count - 1, 0))
            enabled = frame_count > 0
            self._updating_frame_controls = True
            try:
                self.frame_slider.setEnabled(enabled)
                self.frame_spin.setEnabled(enabled)
                self.frame_slider.setRange(0, max(frame_count - 1, 0))
                self.frame_slider.setSingleStep(1)
                self.frame_slider.setPageStep(max(1, frame_count // 20) if enabled else 1)
                self.frame_slider.setValue(frame_index)
                self.frame_spin.setRange(1, max(frame_count, 1))
                self.frame_spin.setValue(frame_index + 1 if enabled else 1)
                self.frame_total_label.setText(f"of {frame_count}")
            finally:
                self._updating_frame_controls = False

        def _sync_frame_controls(self) -> None:
            self._updating_frame_controls = True
            try:
                self.frame_slider.setValue(self.preview_frame_index)
                self.frame_spin.setValue(self.preview_frame_index + 1)
            finally:
                self._updating_frame_controls = False

        def _on_frame_slider_changed(self, value: int) -> None:
            if self._updating_frame_controls:
                return
            self.set_preview_frame_index(int(value))

        def _on_frame_spin_changed(self, value: int) -> None:
            if self._updating_frame_controls:
                return
            self.set_preview_frame_index(int(value) - 1)

        def set_preview_frame_index(self, frame_index: int) -> None:
            if self.preview_tiff_path is None or self.preview_frame_count <= 0:
                return
            frame_index = min(max(int(frame_index), 0), self.preview_frame_count - 1)
            self._request_preview_frame(frame_index)

        def _request_preview_frame(self, frame_index: int) -> None:
            if self.preview_tiff_path is None or self.preview_frame_count <= 0:
                return
            frame_index = min(max(int(frame_index), 0), self.preview_frame_count - 1)
            tiff_path = self.preview_tiff_path
            frame_count = self.preview_frame_count
            self.preview_frame_index = frame_index
            self._sync_frame_controls()
            self.status_label.setText(f"Loading frame {frame_index + 1}/{frame_count}: {tiff_path.name}")
            self._preview_request_id += 1
            request_id = self._preview_request_id

            cached = self.preview_cache.get_cached(tiff_path, frame_index)
            if cached is not None:
                self._pending_preview_frame_index = None
                self._pending_preview_request_id = None
                self._apply_preview_frame(request_id, cached)
                self._prefetch_preview_neighbors(frame_index)
                return

            self._pending_preview_frame_index = frame_index
            self._pending_preview_request_id = request_id
            if not self._preview_worker_active:
                self._start_pending_preview_worker()

        def _start_pending_preview_worker(self) -> None:
            if self.preview_tiff_path is None or self.preview_frame_count <= 0:
                return
            if self._pending_preview_frame_index is None:
                return

            frame_index = min(
                max(int(self._pending_preview_frame_index), 0),
                self.preview_frame_count - 1,
            )
            request_id = self._pending_preview_request_id
            if request_id is None:
                return
            self._pending_preview_frame_index = None
            self._pending_preview_request_id = None
            tiff_path = self.preview_tiff_path
            frame_count = self.preview_frame_count
            self._preview_worker_active = True
            self._preview_worker_request_id = request_id

            def load_frame(_signals: WorkerSignals) -> PreviewFrame:
                return self.preview_cache.get(tiff_path, frame_index, frame_count)

            worker = FunctionWorker(load_frame)

            def apply_result(payload: PreviewFrame, expected_id: int = request_id) -> None:
                self._apply_preview_frame(expected_id, payload)

            def preview_error(details: str, expected_id: int = request_id) -> None:
                if expected_id != self._preview_request_id:
                    return
                self.log(details.rstrip())
                QMessageBox.critical(
                    self,
                    "Load frame",
                    details.splitlines()[-1] if details else "Unknown preview error",
                )

            def finish_worker(
                done_worker: FunctionWorker = worker,
                expected_id: int = request_id,
            ) -> None:
                if done_worker in self._active_preview_workers:
                    self._active_preview_workers.remove(done_worker)
                if self._preview_worker_request_id != expected_id:
                    return
                self._preview_worker_active = False
                self._preview_worker_request_id = None
                if self._pending_preview_frame_index is not None:
                    self._start_pending_preview_worker()

            worker.signals.result.connect(apply_result)
            worker.signals.error.connect(preview_error)
            worker.signals.finished.connect(finish_worker)
            self._active_preview_workers.append(worker)
            self.preview_thread_pool.start(worker)

        def _prefetch_preview_neighbors(self, frame_index: int) -> None:
            if self.preview_tiff_path is None or self.preview_frame_count <= 1:
                return
            candidates = [
                value
                for value in (frame_index + 1, frame_index - 1)
                if 0 <= value < self.preview_frame_count
            ]
            for candidate in candidates:
                if self.preview_cache.get_cached(self.preview_tiff_path, candidate) is not None:
                    continue
                self._start_preview_prefetch(candidate)

        def _start_preview_prefetch(self, frame_index: int) -> None:
            if self.preview_tiff_path is None or self.preview_frame_count <= 0:
                return
            if len(self._active_prefetch_workers) >= 2:
                return
            tiff_path = self.preview_tiff_path
            frame_count = self.preview_frame_count

            def prefetch(_signals: WorkerSignals) -> None:
                self.preview_cache.get(tiff_path, frame_index, frame_count)

            worker = FunctionWorker(prefetch)

            def finish_worker(done_worker: FunctionWorker = worker) -> None:
                if done_worker in self._active_prefetch_workers:
                    self._active_prefetch_workers.remove(done_worker)

            worker.signals.finished.connect(finish_worker)
            self._active_prefetch_workers.append(worker)
            self.prefetch_thread_pool.start(worker)

        def _apply_preview_frame(self, request_id: int, payload: PreviewFrame) -> None:
            if request_id != self._preview_request_id:
                return
            if self.preview_tiff_path is None or payload.tiff_path != self.preview_tiff_path:
                return
            self.preview_frame_index = payload.frame_index
            self.preview_frame_count = payload.frame_count
            self.preview_frame_array = payload.preview_frame
            self.preview_frame_source_shape = payload.source_shape
            self.preview_frame_downsample = payload.downsample
            self.preview_auto_range = payload.auto_range
            self._sync_frame_controls()
            self._configure_display_controls_from_ranges(
                payload.dtype_range,
                payload.auto_range,
                payload.observed_range,
            )
            self._refresh_preview_image()
            self._prefetch_preview_neighbors(payload.frame_index)
            source_h = payload.source_shape[0] if len(payload.source_shape) >= 1 else 0
            source_w = payload.source_shape[1] if len(payload.source_shape) >= 2 else 0
            self.status_label.setText(
                f"Preview frame {payload.frame_index + 1}/{payload.frame_count}: "
                f"{payload.tiff_path.name} ({source_w}x{source_h}, "
                f"{payload.backend}, {payload.elapsed_ms:.0f} ms)"
            )

        def _frame_qimage(self, frame: np.ndarray, black_level: float, white_level: float) -> QImage:
            scaled = scale_frame_to_uint8(frame, black_level, white_level)
            if scaled.ndim == 2:
                gray = np.ascontiguousarray(scaled)
                height, width = gray.shape
                return QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
            if scaled.ndim == 3 and scaled.shape[2] >= 3:
                rgb = np.ascontiguousarray(scaled[:, :, :3])
                height, width, _channels = rgb.shape
                bytes_per_line = 3 * width
                return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
            raise ValueError(f"Cannot render frame with shape {scaled.shape}.")

        def _update_observed_display_label(self, frame: np.ndarray) -> None:
            observed_low, observed_high = finite_min_max(frame)
            self.display_observed_label.setText(f"{observed_low:g} .. {observed_high:g}")

        def _configure_display_controls(self, frame: np.ndarray) -> None:
            self._configure_display_controls_from_ranges(
                dtype_display_bounds(frame),
                auto_display_range(frame),
                finite_min_max(frame),
            )

        def _configure_display_controls_from_ranges(
            self,
            dtype_range: tuple[float, float],
            auto_range: tuple[float, float],
            observed_range: tuple[float, float],
        ) -> None:
            full_low, full_high = dtype_range
            auto_low, auto_high = auto_range
            step = max((full_high - full_low) / 1000.0, 1.0)

            self.display_full_range = (full_low, full_high)
            observed_low, observed_high = observed_range
            self.display_observed_label.setText(f"{observed_low:g} .. {observed_high:g}")
            self._updating_display_controls = True
            try:
                for spin in [self.display_black_spin, self.display_white_spin]:
                    spin.setRange(full_low, full_high)
                    spin.setSingleStep(step)
                self.display_black_spin.setValue(auto_low)
                self.display_white_spin.setValue(auto_high)
            finally:
                self._updating_display_controls = False
            self.display_group.setEnabled(True)

        def _set_display_levels(self, black_level: float, white_level: float) -> None:
            full_low, full_high = self.display_full_range
            black_level = min(max(float(black_level), full_low), full_high)
            white_level = min(max(float(white_level), full_low), full_high)
            if white_level <= black_level:
                white_level = min(full_high, black_level + max((full_high - full_low) / 1000.0, 1.0))
                if white_level <= black_level:
                    black_level = max(full_low, white_level - 1.0)

            self._updating_display_controls = True
            try:
                self.display_black_spin.setValue(black_level)
                self.display_white_spin.setValue(white_level)
            finally:
                self._updating_display_controls = False

        def _on_display_levels_changed(self, *_args) -> None:
            if self._updating_display_controls:
                return
            self._set_display_levels(self.display_black_spin.value(), self.display_white_spin.value())
            self._refresh_preview_image()

        def set_display_auto(self) -> None:
            if self.preview_frame_array is None:
                return
            self._set_display_levels(*self.preview_auto_range)
            self._refresh_preview_image()

        def set_display_full_range(self) -> None:
            self._set_display_levels(*self.display_full_range)
            self._refresh_preview_image()

        def _refresh_preview_image(self) -> None:
            if self.preview_frame_array is None:
                self.preview.clear_preview()
                return
            image = self._frame_qimage(
                self.preview_frame_array,
                self.display_black_spin.value(),
                self.display_white_spin.value(),
            )
            self.preview.set_preview(
                image,
                self.preview_layout_model,
                source_to_preview_scale=1.0 / max(1, self.preview_frame_downsample),
            )

        def _update_metadata(self, tiff_path, metadata_path, metadata, info, layout) -> None:
            rows = [
                ("TIFF", str(tiff_path)),
                ("Metadata", str(metadata_path)),
                ("Frame size", f"{info.frame_width} x {info.frame_height}"),
                ("Frame count", str(info.frame_count)),
                ("Mode", info.mode),
                ("Dtype", info.dtype),
                ("ROI count", str(metadata.roi_count)),
                ("ROI size", f"{layout.roi_width} x {layout.roi_height}"),
                ("Grid", f"{layout.columns} columns x {layout.rows} rows"),
                ("Channel", metadata.channel or "-"),
                ("Data type", metadata.data_type or "-"),
                ("Sampling Hz", f"{metadata.sampling_rate_hz:g}" if metadata.sampling_rate_hz else "-"),
            ]
            self.metadata_table.setRowCount(len(rows))
            for row, (field, value) in enumerate(rows):
                self.metadata_table.setItem(row, 0, QTableWidgetItem(field))
                self.metadata_table.setItem(row, 1, QTableWidgetItem(value))
            self.metadata_table.resizeColumnsToContents()

            self.meta_size.setText(f"{info.frame_width} x {info.frame_height}")
            self.meta_frames.setText(str(info.frame_count))
            self.meta_rois.setText(str(metadata.roi_count))
            self.meta_grid.setText(f"{layout.columns} x {layout.rows}")
            self.meta_channel.setText(metadata.channel or "-")
            self.meta_sampling.setText(f"{metadata.sampling_rate_hz:g} Hz" if metadata.sampling_rate_hz else "-")

        def _set_busy(self, busy: bool) -> None:
            for button in [
                self.btn_load_tiff,
                self.btn_load_folder,
                self.btn_output_folder,
                self.btn_load_preview,
                self.btn_native_viewer,
                self.btn_split,
                self.btn_calculate,
                self.btn_export,
                self.btn_average_png,
            ]:
                button.setEnabled(not busy)
            for widget in [
                self.tiff_list,
                self.btn_select_all,
                self.btn_clear_selection,
                self.recursive_check,
                self.use_metadata_sampling_check,
                self.btn_reset_sampling,
                self.write_csv_check,
                self.frame_slider,
                self.frame_spin,
            ]:
                widget.setEnabled(not busy)
            if not busy and self.preview_frame_count <= 0:
                self.frame_slider.setEnabled(False)
                self.frame_spin.setEnabled(False)
            self.display_group.setEnabled((not busy) and self.preview_frame_array is not None)
            self.sampling_rate_spin.setEnabled(
                (not busy) and (not self.use_metadata_sampling_check.isChecked())
            )
            self.progress.setVisible(busy)

        def _run_worker(
            self,
            label: str,
            fn: Callable[[WorkerSignals], object],
            on_result: Callable[[object], None] | None = None,
        ) -> None:
            self._set_busy(True)
            self.log(label)
            worker = FunctionWorker(fn)
            worker.signals.log.connect(self.log)
            worker.signals.error.connect(self._worker_error)
            if on_result is not None:
                worker.signals.result.connect(on_result)
            self._active_workers.append(worker)

            def finish_worker(done_worker: FunctionWorker = worker) -> None:
                self._set_busy(False)
                if done_worker in self._active_workers:
                    self._active_workers.remove(done_worker)

            worker.signals.finished.connect(finish_worker)
            self.thread_pool.start(worker)

        def _worker_error(self, details: str) -> None:
            self.log(details.rstrip())
            QMessageBox.critical(self, "Processing error", details.splitlines()[-1] if details else "Unknown error")

        def split_tiffs(self) -> None:
            try:
                tiff_paths = self.selected_tiffs()
                if not tiff_paths:
                    raise ValueError("No TIFF files found.")
                output = self.output_dir()
                split_root = output / "split_rois" if output is not None else None
            except Exception as exc:
                QMessageBox.critical(self, "Split TIFFs", str(exc))
                return

            def worker(signals: WorkerSignals) -> None:
                successes = 0
                for tiff_path in tiff_paths:
                    result = split_tiff(
                        tiff_path,
                        debug=True,
                        output_root=split_root,
                        write_manifest=True,
                        log=signals.log.emit,
                    )
                    signals.log.emit(result.message)
                    successes += int(result.ok)
                signals.log.emit(f"Split complete: {successes}/{len(tiff_paths)} TIFF file(s) succeeded.")

            self._run_worker("Splitting TIFFs...", worker)

        def calculate_signals(self) -> None:
            try:
                tiff_path = self.current_tiff()
                rate = self.sampling_rate_override()
            except Exception as exc:
                QMessageBox.critical(self, "Calculate Signals", str(exc))
                return

            def worker(signals: WorkerSignals):
                def progress(done: int, total: int) -> None:
                    signals.log.emit(f"{tiff_path.name}: signal frames {done}/{total}")

                table = extract_roi_signals(tiff_path, sampling_rate_hz=rate, progress=progress)
                signals.log.emit(
                    f"Calculated signals for {tiff_path.name}: "
                    f"{table.tiff_info.frame_count} frames, {len(table.roi_labels)} ROI columns."
                )
                return table

            self._run_worker("Calculating signals...", worker, on_result=self._store_signal_table)

        def _store_signal_table(self, table) -> None:
            self.current_signal_table = table

        def export_excel(self) -> None:
            try:
                tiff_paths = self.selected_tiffs()
                if not tiff_paths:
                    raise ValueError("No TIFF files found.")
                rate = self.sampling_rate_override()
                output = self.output_dir()
                write_csv = self.write_csv_check.isChecked()
            except Exception as exc:
                QMessageBox.critical(self, "Export Excel", str(exc))
                return

            def worker(signals: WorkerSignals) -> None:
                for tiff_path in tiff_paths:
                    export_dir = output / tiff_path.stem if output is not None else None

                    def progress(done: int, total: int, name: str = tiff_path.name) -> None:
                        signals.log.emit(f"{name}: signal frames {done}/{total}")

                    result = export_signal_documents(
                        tiff_path,
                        output_dir=export_dir,
                        sampling_rate_hz=rate,
                        write_csv=write_csv,
                        progress=progress,
                    )
                    signals.log.emit(f"Excel: {result.xlsx_path}")
                    if result.csv_path is not None:
                        signals.log.emit(f"CSV: {result.csv_path}")
                signals.log.emit(f"Export complete: {len(tiff_paths)} TIFF file(s).")

            self._run_worker("Exporting Excel documents...", worker)

        def export_average_png(self) -> None:
            try:
                tiff_paths = self.selected_tiffs()
                if not tiff_paths:
                    raise ValueError("No TIFF files found.")
                output = self.output_dir()
            except Exception as exc:
                QMessageBox.critical(self, "Average PNG", str(exc))
                return

            def worker(signals: WorkerSignals) -> None:
                for tiff_path in tiff_paths:
                    export_dir = output / tiff_path.stem if output is not None else None
                    png_path = default_average_png_path(tiff_path, export_dir)

                    def progress(done: int, total: int, name: str = tiff_path.name) -> None:
                        signals.log.emit(f"{name}: averaged frames {done}/{total}")

                    save_tiff_average_png(tiff_path, png_path, progress=progress)
                    signals.log.emit(f"Average PNG: {png_path}")
                signals.log.emit(f"Average PNG export complete: {len(tiff_paths)} TIFF file(s).")

            self._run_worker("Exporting average PNG...", worker)


    TiffProcessorApp = TiffProcessorMainWindow


def run_legacy_python_viewer(argv: Sequence[str] | None = None) -> int:
    _qt_required()
    app_args = [sys.argv[0], *list(argv or [])]
    app = QApplication(app_args)
    window = TiffProcessorMainWindow()
    window.show()
    return app.exec()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--legacy-python-viewer" in args:
        args.remove("--legacy-python-viewer")
        return run_legacy_python_viewer(args)
    return launch_native_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
