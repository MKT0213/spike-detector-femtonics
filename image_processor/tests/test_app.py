import unittest
import io
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_processor import app, export_average_png
from image_processor.tiff_backend import build_persistent_preview_cache, persistent_preview_cache_available


class TiffProcessorAppTests(unittest.TestCase):
    def _close_window(self, window, qt_app) -> None:
        window.close()
        qt_app.processEvents()
        window.preview_thread_pool.waitForDone(5000)
        window.prefetch_thread_pool.waitForDone(5000)
        window.cache_build_thread_pool.waitForDone(5000)
        qt_app.processEvents()

    def _write_preview_fixture(self, root: Path, frame_count: int = 2) -> Path:
        tiff_path = root / "movie.tiff"
        metadata_path = root / "movie.tiff.metadata.txt"
        frames = [
            (np.arange(120 * 80, dtype=np.uint16).reshape(120, 80) + frame_index * 100)
            for frame_index in range(frame_count)
        ]
        Image.fromarray(frames[0]).save(
            tiff_path,
            save_all=True,
            append_images=[Image.fromarray(frame) for frame in frames[1:]],
        )
        metadata_path.write_text(
            """%Metadata for:
FileName: movie.mesc
Type: MultiROIChessBoard
Comment: 500hz synthetic preview test

>Context start
%SIZE
 Width: 80
 Height: 120
%PATTERN
 PatternNum: 1
 RoiCount: 1
 RoiSize (um): [[80, 120]]
 RoiIndex: [1]
 Channel: UG
 DataType: uint16
<Context end
""",
            encoding="utf-8",
        )
        return tiff_path

    def test_scale_frame_to_uint8_uses_black_and_white_levels(self) -> None:
        frame = np.array([[0, 50, 100, 150]], dtype=np.uint16)

        scaled = app.scale_frame_to_uint8(frame, black_level=50, white_level=150)

        np.testing.assert_array_equal(scaled, np.array([[0, 0, 127, 255]], dtype=np.uint8))

    def test_display_ranges_handle_uint16_frames(self) -> None:
        frame = np.array([[1000, 1200], [1400, 5000]], dtype=np.uint16)

        self.assertEqual(app.dtype_display_bounds(frame), (0.0, 65535.0))
        low, high = app.auto_display_range(frame)
        self.assertGreaterEqual(low, 1000.0)
        self.assertLessEqual(high, 5000.0)
        self.assertGreater(high, low)

    def test_average_tiff_frames_returns_mean_projection(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tiff_path = Path(temp_dir) / "movie.tiff"
            frames = [
                np.array([[0, 10], [20, 30]], dtype=np.uint16),
                np.array([[10, 20], [30, 40]], dtype=np.uint16),
                np.array([[20, 30], [40, 50]], dtype=np.uint16),
            ]
            Image.fromarray(frames[0]).save(
                tiff_path,
                save_all=True,
                append_images=[Image.fromarray(frame) for frame in frames[1:]],
            )

            averaged = app.average_tiff_frames(tiff_path)

            expected = np.array([[10, 20], [30, 40]], dtype=np.float64)
            np.testing.assert_array_equal(averaged, expected)

    def test_save_tiff_average_png_writes_single_png(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = root / "movie.tiff"
            png_path = root / "movie_average_projection.png"
            frames = [
                np.array([[0, 100], [200, 300]], dtype=np.uint16),
                np.array([[100, 200], [300, 400]], dtype=np.uint16),
            ]
            Image.fromarray(frames[0]).save(
                tiff_path,
                save_all=True,
                append_images=[Image.fromarray(frame) for frame in frames[1:]],
            )

            result = app.save_tiff_average_png(tiff_path, png_path)

            self.assertEqual(result, png_path)
            with Image.open(png_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.size[0], 2)
                self.assertGreater(image.size[1], 2)
                self.assertEqual(image.size[0], image.size[1])
                self.assertEqual(getattr(image, "n_frames", 1), 1)

    def test_export_average_png_cli_writes_projection(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = root / "movie.tiff"
            output_root = root / "native_exports"
            frames = [
                np.array([[0, 100], [200, 300]], dtype=np.uint16),
                np.array([[100, 200], [300, 400]], dtype=np.uint16),
            ]
            Image.fromarray(frames[0]).save(
                tiff_path,
                save_all=True,
                append_images=[Image.fromarray(frame) for frame in frames[1:]],
            )

            exit_code = export_average_png.main([str(tiff_path), "--output", str(output_root)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "movie" / "movie_average_projection.png").exists())

    def test_native_image_processor_executable_defaults_to_native_build_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            executable = app.native_image_processor_executable(root)

            self.assertEqual(
                executable,
                root / "image_processor" / "native_viewer" / "build" / "femtonics_image_processor.exe",
            )

    def test_resolved_native_image_processor_uses_single_native_ui_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            executable = app.resolved_native_image_processor_executable(root)

            self.assertEqual(
                executable,
                root / "image_processor" / "native_viewer" / "build" / "femtonics_image_processor.exe",
            )

    def test_main_launches_native_viewer_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "image_processor" / "native_viewer" / "build" / "femtonics_image_processor.exe"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")

            with (
                patch.object(app, "resolved_native_image_processor_executable", return_value=executable),
                patch.object(app.subprocess, "call", return_value=0) as call,
            ):
                exit_code = app.main(["movie.tiff"])

            self.assertEqual(exit_code, 0)
            call.assert_called_once_with([str(executable), "movie.tiff"], cwd=str(executable.parent))

    def test_main_builds_missing_native_viewer_then_launches(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "image_processor" / "native_viewer" / "build" / "femtonics_image_processor.exe"

            def build_native(repo_root: Path | None = None) -> int:
                self.assertEqual(repo_root, root)
                executable.parent.mkdir(parents=True)
                executable.write_text("", encoding="utf-8")
                return 0

            with (
                patch.object(app, "resolved_native_image_processor_executable", return_value=executable),
                patch.object(app, "build_native_viewer", side_effect=build_native) as build,
                patch.object(app.subprocess, "call", return_value=0) as call,
            ):
                exit_code = app.launch_native_viewer(["movie.tiff"], repo_root=root)

            self.assertEqual(exit_code, 0)
            build.assert_called_once_with(root)
            call.assert_called_once_with([str(executable), "movie.tiff"], cwd=str(executable.parent))

    def test_native_viewer_cli_args_resolves_existing_relative_paths(self) -> None:
        original_cwd = Path.cwd()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = root / "movie.tiff"
            tiff_path.write_text("", encoding="utf-8")
            os.chdir(root)
            try:
                args = app.native_viewer_cli_args(["--probe", "movie.tiff", "missing.tiff"])
            finally:
                os.chdir(original_cwd)

        self.assertEqual(args[0], "--probe")
        self.assertEqual(args[1], str(tiff_path.resolve()))
        self.assertEqual(args[2], "missing.tiff")

    def test_main_reports_missing_native_viewer_after_failed_build(self) -> None:
        with TemporaryDirectory() as temp_dir:
            executable = (
                Path(temp_dir)
                / "image_processor"
                / "native_viewer"
                / "build"
                / "femtonics_image_processor.exe"
            )
            stderr = io.StringIO()

            with (
                patch.object(app, "resolved_native_image_processor_executable", return_value=executable),
                patch.object(app, "build_native_viewer", return_value=1) as build,
                patch("sys.stderr", stderr),
            ):
                exit_code = app.main([])

            self.assertEqual(exit_code, 1)
            self.assertIn("Native image processor UI is not built yet", stderr.getvalue())
            self.assertIn(str(executable), stderr.getvalue())
            build.assert_called_once()

    def test_native_viewer_build_command_uses_native_env(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            command = app.native_viewer_build_command(root)

        self.assertEqual(command[:3], ["cmd", "/v:on", "/c"])
        self.assertIn("native_dev_env.bat", command[3])
        self.assertIn("-G Ninja", command[3])
        self.assertIn("cmake --build", command[3])

    def test_main_legacy_flag_opens_python_viewer_explicitly(self) -> None:
        with patch.object(app, "run_legacy_python_viewer", return_value=0) as run_legacy:
            exit_code = app.main(["--legacy-python-viewer"])

        self.assertEqual(exit_code, 0)
        run_legacy.assert_called_once_with([])

    def test_app_constructs_main_window(self) -> None:
        if not app.HAS_QT:
            self.skipTest(f"PySide6 is not available in this environment: {app.QT_IMPORT_ERROR}")

        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance() or QApplication([])
        window = app.TiffProcessorMainWindow()
        try:
            self.assertEqual(window.windowTitle(), "Femtonics Image Processor")
            self.assertEqual(window.tiff_list.count(), 0)
            self.assertEqual(window.status_label.text(), "No file loaded")
            self.assertEqual(window.btn_native_viewer.text(), "Native UI")
            self.assertEqual(window.btn_average_png.text(), "Average PNG")
        finally:
            self._close_window(window, qt_app)

    def test_frame_qimage_uses_grayscale_format_for_2d_frames(self) -> None:
        if not app.HAS_QT:
            self.skipTest(f"PySide6 is not available in this environment: {app.QT_IMPORT_ERROR}")

        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance() or QApplication([])
        window = app.TiffProcessorMainWindow()
        try:
            frame = np.array([[0, 128], [255, 512]], dtype=np.uint16)

            image = window._frame_qimage(frame, black_level=0, white_level=512)

            self.assertEqual(image.format(), QImage.Format.Format_Grayscale8)
            self.assertEqual(image.size().width(), 2)
            self.assertEqual(image.size().height(), 2)
        finally:
            self._close_window(window, qt_app)

    def test_preview_loads_frame_asynchronously_with_downsampled_display(self) -> None:
        if not app.HAS_QT:
            self.skipTest(f"PySide6 is not available in this environment: {app.QT_IMPORT_ERROR}")

        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = self._write_preview_fixture(root, frame_count=2)

            window = app.TiffProcessorMainWindow()
            try:
                window.preview_cache.max_edge = 40
                window.input_path = str(tiff_path)
                window.refresh_queue()

                deadline = time.monotonic() + 5.0
                while window.preview_frame_array is None and time.monotonic() < deadline:
                    qt_app.processEvents()
                    time.sleep(0.01)

                self.assertIsNotNone(window.preview_frame_array)
                self.assertEqual(window.preview_frame_source_shape, (120, 80))
                self.assertEqual(window.preview_frame_downsample, 3)
                self.assertLessEqual(max(window.preview_frame_array.shape[:2]), 40)
                self.assertIn("Preview frame 1/2", window.status_label.text())

                window.set_preview_frame_index(1)
                deadline = time.monotonic() + 5.0
                while "Preview frame 2/2" not in window.status_label.text() and time.monotonic() < deadline:
                    qt_app.processEvents()
                    time.sleep(0.01)

                self.assertEqual(window.preview_frame_index, 1)
                self.assertIn("Preview frame 2/2", window.status_label.text())
            finally:
                self._close_window(window, qt_app)

    def test_preview_scrubbing_coalesces_to_latest_requested_frame(self) -> None:
        if not app.HAS_QT:
            self.skipTest(f"PySide6 is not available in this environment: {app.QT_IMPORT_ERROR}")

        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = self._write_preview_fixture(root, frame_count=5)

            window = app.TiffProcessorMainWindow()
            try:
                window.preview_cache.max_edge = 40
                window.input_path = str(tiff_path)
                window.refresh_queue()

                deadline = time.monotonic() + 5.0
                while window.preview_frame_array is None and time.monotonic() < deadline:
                    qt_app.processEvents()
                    time.sleep(0.01)

                for frame_index in [1, 2, 3, 4]:
                    window.set_preview_frame_index(frame_index)

                deadline = time.monotonic() + 5.0
                while "Preview frame 5/5" not in window.status_label.text() and time.monotonic() < deadline:
                    qt_app.processEvents()
                    time.sleep(0.01)

                self.assertEqual(window.preview_frame_index, 4)
                self.assertIn("Preview frame 5/5", window.status_label.text())
            finally:
                self._close_window(window, qt_app)

    def test_preview_uses_persistent_zarr_cache_when_available(self) -> None:
        if not app.HAS_QT:
            self.skipTest(f"PySide6 is not available in this environment: {app.QT_IMPORT_ERROR}")
        if not persistent_preview_cache_available():
            self.skipTest("zarr is not available")

        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = self._write_preview_fixture(root, frame_count=2)
            build_persistent_preview_cache(tiff_path, frame_count=2, max_edge=40)

            window = app.TiffProcessorMainWindow()
            try:
                window.preview_cache.max_edge = 40
                window.input_path = str(tiff_path)
                window.refresh_queue()

                deadline = time.monotonic() + 5.0
                while "zarr-preview" not in window.status_label.text() and time.monotonic() < deadline:
                    qt_app.processEvents()
                    time.sleep(0.01)

                self.assertIn("zarr-preview", window.status_label.text())
                self.assertEqual(window.preview_frame_downsample, 3)
            finally:
                self._close_window(window, qt_app)


if __name__ == "__main__":
    unittest.main()
