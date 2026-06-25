import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from image_processor.qc_images import (
    _difference_colorbar,
    _difference_display_limit,
    _difference_heatmap,
    _grayscale_colorbar,
    _motion_comparison_figure,
)
from image_processor.summary_images import compute_summary_images, write_summary_outputs


class SummaryImageTests(unittest.TestCase):
    def test_compute_summary_images_returns_mean_std_and_max(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
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

            summary = compute_summary_images(tiff_path)

            self.assertEqual(summary.frame_count, 3)
            self.assertEqual(summary.source_shape, (2, 2))
            self.assertEqual(summary.source_dtype, "uint16")
            np.testing.assert_allclose(summary.mean, np.array([[10, 20], [30, 40]], dtype=np.float32))
            np.testing.assert_allclose(summary.std, np.full((2, 2), 10, dtype=np.float32))
            np.testing.assert_allclose(summary.max_projection, np.array([[20, 30], [40, 50]], dtype=np.float32))

    def test_write_summary_outputs_writes_pngs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = root / "movie.tiff"
            frames = [
                np.array([[0, 100], [200, 300]], dtype=np.uint16),
                np.array([[100, 200], [300, 400]], dtype=np.uint16),
            ]
            Image.fromarray(frames[0]).save(
                tiff_path,
                save_all=True,
                append_images=[Image.fromarray(frame) for frame in frames[1:]],
            )

            summary = compute_summary_images(tiff_path)
            result = write_summary_outputs(summary, root / "out")

            for key in ["summary_mean_png", "summary_std_png", "summary_max_png", "summary_manifest_path"]:
                self.assertTrue(result[key].exists(), key)
            with Image.open(result["summary_mean_png"]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (2, 2))
            manifest = json.loads(result["summary_manifest_path"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(set(manifest["images"]), {"summary_mean_png", "summary_std_png", "summary_max_png"})

    def test_motion_qc_heatmap_uses_scientific_figure_layout(self) -> None:
        values = np.array([[-1.0, 0.0, 1.0]], dtype=float)
        limit = _difference_display_limit(values)
        heatmap = _difference_heatmap(values, limit)

        self.assertEqual(heatmap.shape, (1, 3, 3))
        self.assertGreater(int(heatmap[0, 0, 2]), int(heatmap[0, 0, 0]))
        self.assertTrue(np.allclose(heatmap[0, 1], np.array([255, 255, 255]), atol=1))
        self.assertGreater(int(heatmap[0, 2, 0]), int(heatmap[0, 2, 2]))

        grayscale_bar = _grayscale_colorbar(260, 10, 200)
        self.assertGreater(grayscale_bar.size[0], 240)
        self.assertGreater(grayscale_bar.size[1], 40)

        scale_bar = _difference_colorbar(220, limit)
        self.assertGreater(scale_bar.size[0], 200)
        self.assertGreater(scale_bar.size[1], 40)

        figure = _motion_comparison_figure(
            np.full((12, 16, 3), 40, dtype=np.uint8),
            np.full((12, 16, 3), 70, dtype=np.uint8),
            np.full((12, 16, 3), 255, dtype=np.uint8),
            grayscale_range=(10, 200),
            difference_limit=limit,
        )
        self.assertGreater(figure.size[0], figure.size[1])
        self.assertEqual(figure.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(figure.getpixel((32, 88)), (31, 41, 55))


if __name__ == "__main__":
    unittest.main()
