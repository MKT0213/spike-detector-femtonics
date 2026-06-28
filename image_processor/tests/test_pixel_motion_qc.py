import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from image_processor import native_pipeline
from image_processor.pixel_motion_qc import motion_hook, pearson_correlation


class PixelMotionQcTests(unittest.TestCase):
    def test_pearson_correlation_handles_matching_and_different_frames(self) -> None:
        reference = np.array([[0, 1], [2, 3]], dtype=np.float32)
        self.assertAlmostEqual(pearson_correlation(reference, reference), 1.0)
        self.assertLess(pearson_correlation(reference, np.flipud(reference)), 1.0)

    def test_motion_hook_exports_frame_correlation_plot_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = root / "movie.tiff"
            self._write_shifted_frame_tiff(tiff_path)

            result = motion_hook(tiff_path, output_dir=root / "out")

            self.assertTrue(result["ok"])
            self.assertEqual(result["frame_count"], 5)
            self.assertEqual(result["min_correlation_frame"], 4)
            self.assertGreaterEqual(result["heavy_motion_count"], 1)

            plot_path = Path(result["correlation_plot_png"])
            scores_path = Path(result["correlation_scores_csv"])
            qc_path = Path(result["correlation_qc_json"])
            mean_path = Path(result["mean_image_png"])
            for path in [plot_path, scores_path, qc_path, mean_path]:
                self.assertTrue(path.exists(), path)

            with scores_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[4]["is_heavy_motion"], "true")
            self.assertLess(float(rows[4]["corr_to_mean"]), float(rows[0]["corr_to_mean"]))

            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["metric"], "pearson_correlation_to_mean_image")
            self.assertEqual(qc["min_correlation_frame"], 4)
            self.assertIn("correlation_plot_png", qc["outputs"])

            with Image.open(plot_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 720)

    def test_native_pipeline_can_run_pixel_correlation_motion_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = root / "movie.tiff"
            output_root = root / "pipeline"
            manifest_path = output_root / "manifest.json"
            self._write_shifted_frame_tiff(tiff_path)

            exit_code = native_pipeline.run(
                [tiff_path],
                recursive=False,
                output_root=output_root,
                sampling_rate_hz=None,
                write_csv=False,
                do_average=False,
                do_signals=False,
                do_split=False,
                motion_hook="image_processor.pixel_motion_qc:motion_hook",
                roi_hook=None,
                manifest_path=manifest_path,
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            action = manifest["tiffs"][0]["actions"]["motion_correction"]
            self.assertTrue(action["ok"])
            self.assertEqual(Path(manifest["tiffs"][0]["working_tiff"]), tiff_path)
            self.assertTrue(Path(action["correlation_plot_png"]).exists())
            self.assertTrue(Path(action["correlation_scores_csv"]).exists())

    @staticmethod
    def _write_shifted_frame_tiff(path: Path) -> None:
        rng = np.random.default_rng(7)
        base = rng.integers(0, 1200, size=(32, 32), dtype=np.uint16)
        base[8:18, 9:21] += np.uint16(8000)

        shifted = np.zeros_like(base)
        shifted[5:, 4:] = base[:-5, :-4]
        frames = [base, base, base, base, shifted]
        Image.fromarray(frames[0]).save(
            path,
            save_all=True,
            append_images=[Image.fromarray(frame) for frame in frames[1:]],
        )


if __name__ == "__main__":
    unittest.main()
