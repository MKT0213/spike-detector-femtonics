import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from image_processor.tiff_backend import (
    PreviewFrameCache,
    build_persistent_preview_cache,
    downsample_for_preview,
    iter_tiff_frames,
    load_preview_frame,
    persistent_preview_cache_available,
    read_tiff_frame,
    read_persistent_preview_frame,
)


class TiffBackendTests(unittest.TestCase):
    def test_iter_and_read_tiff_frames_round_trip_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "movie.tiff"
            frames = [
                np.full((4, 5), 10, dtype=np.uint16),
                np.full((4, 5), 20, dtype=np.uint16),
                np.full((4, 5), 30, dtype=np.uint16),
            ]
            Image.fromarray(frames[0]).save(
                path,
                save_all=True,
                append_images=[Image.fromarray(frame) for frame in frames[1:]],
            )

            loaded = list(iter_tiff_frames(path))

            self.assertEqual(len(loaded), 3)
            np.testing.assert_array_equal(read_tiff_frame(path, 1), frames[1])
            for actual, expected in zip(loaded, frames):
                np.testing.assert_array_equal(actual, expected)

    def test_downsample_for_preview_limits_largest_edge(self) -> None:
        frame = np.arange(120 * 80, dtype=np.uint16).reshape(120, 80)

        preview, step = downsample_for_preview(frame, max_edge=32)

        self.assertEqual(step, 4)
        self.assertLessEqual(max(preview.shape), 32)
        np.testing.assert_array_equal(preview, frame[::4, ::4])

    def test_load_preview_frame_reports_full_frame_ranges_but_returns_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "movie.tiff"
            frame = np.arange(100 * 50, dtype=np.uint16).reshape(100, 50)
            Image.fromarray(frame).save(path)

            payload = load_preview_frame(path, frame_index=0, frame_count=1, max_edge=25)

            self.assertEqual(payload.source_shape, (100, 50))
            self.assertEqual(payload.downsample, 4)
            self.assertEqual(payload.preview_frame.shape, (25, 13))
            self.assertEqual(payload.observed_range, (0.0, 4999.0))
            self.assertEqual(payload.dtype_range, (0.0, 65535.0))
            self.assertGreater(payload.auto_range[1], payload.auto_range[0])

    def test_preview_frame_cache_reuses_loaded_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "movie.tiff"
            frame = np.arange(10 * 10, dtype=np.uint16).reshape(10, 10)
            Image.fromarray(frame).save(path)
            cache = PreviewFrameCache(max_frames=2, max_edge=8)

            first = cache.get(path, 0, 1)
            second = cache.get(path, 0, 1)
            cached = cache.get_cached(path, 0)

            self.assertIs(first, second)
            self.assertIs(first, cached)

    def test_persistent_preview_cache_round_trips_zarr_frames(self) -> None:
        if not persistent_preview_cache_available():
            self.skipTest("zarr is not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "movie.tiff"
            frames = [
                np.arange(60 * 40, dtype=np.uint16).reshape(60, 40),
                np.arange(60 * 40, dtype=np.uint16).reshape(60, 40) + 100,
            ]
            Image.fromarray(frames[0]).save(
                path,
                save_all=True,
                append_images=[Image.fromarray(frame) for frame in frames[1:]],
            )

            cache_path = build_persistent_preview_cache(path, frame_count=2, max_edge=30)
            payload = read_persistent_preview_frame(path, frame_index=1, frame_count=2, max_edge=30)

            self.assertTrue(cache_path.exists())
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload.backend, "zarr-preview")
            self.assertEqual(payload.source_shape, (60, 40))
            self.assertEqual(payload.downsample, 2)
            np.testing.assert_array_equal(payload.preview_frame, frames[1][::2, ::2])

            memory_cache = PreviewFrameCache(max_frames=2, max_edge=30)
            from_memory = memory_cache.get(path, 1, 2)
            self.assertEqual(from_memory.backend, "zarr-preview")


if __name__ == "__main__":
    unittest.main()
