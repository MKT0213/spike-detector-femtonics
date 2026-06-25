import json
import gc
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from image_processor.roi_core import save_tiff_array_stack


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "split_multi_roi_tiff.py"


class SplitMultiRoiTiffTests(unittest.TestCase):
    def test_fast_array_stack_writer_round_trips_uint16_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "fast_stack.tiff"
            stack = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

            save_tiff_array_stack(output_path, stack)

            image = Image.open(output_path)
            try:
                self.assertEqual(image.size, (5, 4))
                self.assertEqual(image.n_frames, 3)
                for frame_number in range(3):
                    image.seek(frame_number)
                    actual = np.array(image, copy=True)
                    self.assertEqual(str(actual.dtype), "uint16")
                    np.testing.assert_array_equal(actual, stack[frame_number])
            finally:
                image.close()

    def test_recursive_debug_split_writes_row_major_roi_stacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tiff_path = root / "sample.tiff"
            metadata_path = root / "sample.tiff.metadata.txt"

            self._write_synthetic_tiff(tiff_path)
            metadata_path.write_text(self._metadata_text(), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(root), "--recursive", "--debug"],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Computed ROI grid: columns=3, rows=2", result.stdout)
            self.assertIn("ROI01: x=0..9, y=0..9", result.stdout)
            self.assertIn("ROI04: x=0..9, y=10..19", result.stdout)

            output_dir = root / "split_rois" / "sample"
            manifest_path = output_dir / "debug_manifest.json"
            self.assertTrue(manifest_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["tiff"]["frame_count"], 3)
            self.assertEqual(manifest["roi_grid"]["columns"], 3)
            self.assertEqual(manifest["roi_grid"]["rows"], 2)
            self.assertEqual(manifest["rois"][0]["display_range"]["x"], [0, 9])
            self.assertEqual(manifest["rois"][3]["display_range"]["y"], [10, 19])

            for roi_index in range(1, 7):
                output_path = output_dir / f"sample_ROI{roi_index:02d}.tiff"
                self.assertTrue(output_path.exists(), output_path)
                roi_stack = Image.open(output_path)
                try:
                    self.assertEqual(roi_stack.size, (10, 10))
                    self.assertEqual(roi_stack.n_frames, 3)
                    for frame_number in range(3):
                        roi_stack.seek(frame_number)
                        expected = frame_number * 100 + roi_index
                        actual = np.array(roi_stack, copy=True)
                        self.assertEqual(str(actual.dtype), "uint16")
                        self.assertTrue(np.all(actual == expected), output_path)
                        del actual
                finally:
                    roi_stack.close()
                    del roi_stack
                gc.collect()

    @staticmethod
    def _write_synthetic_tiff(path: Path) -> None:
        frames = []
        for frame_number in range(3):
            frame = np.zeros((20, 30), dtype=np.uint16)
            for roi_index in range(1, 7):
                zero_based = roi_index - 1
                row = zero_based // 3
                column = zero_based % 3
                left = column * 10
                upper = row * 10
                frame[upper : upper + 10, left : left + 10] = frame_number * 100 + roi_index
            frames.append(Image.fromarray(frame))

        frames[0].save(path, save_all=True, append_images=frames[1:])

    @staticmethod
    def _metadata_text() -> str:
        return """%Metadata for:
FileName: synthetic.mesc
MestagMemo: sample
Type: MultiROIChessBoard
Comment: synthetic test

>Context start
%SIZE
 Width: 30
 Height: 20
 Average: 2
%PATTERN
 PatternNum: 2
 RoiCount: 6
 RoiSize (um): [[10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [10, 10]]
 RoiIndex: [1, 2, 3, 4, 5, 6]
 Channel: UG
 DataType: uint16
<Context end
"""


if __name__ == "__main__":
    unittest.main()
