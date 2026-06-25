from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.core.io import is_ignored_trace_column, load_excel_traces  # noqa: E402


def test_load_excel_traces_ignores_frame_metadata_columns(tmp_path: Path) -> None:
    path = tmp_path / "traces.xlsx"
    df = pd.DataFrame(
        {
            "time": [0.0, 0.001, 0.002],
            "Frame": [1, 2, 3],
            "Frame Number": [100, 101, 102],
            "ROI A": [10.0, 12.0, 11.0],
            "ROI A_corrected": [0.0, 2.0, 1.0],
        }
    )
    df.to_excel(path, sheet_name="Recording 1", index=False)

    time_map, raw_map, corrected_map, sampling_rate_map = load_excel_traces(str(path))

    assert list(raw_map) == ["Recording 1 | ROI A"]
    np.testing.assert_allclose(time_map["Recording 1 | ROI A"], [0.0, 0.001, 0.002])
    np.testing.assert_allclose(raw_map["Recording 1 | ROI A"], [10.0, 12.0, 11.0])
    np.testing.assert_allclose(corrected_map["Recording 1 | ROI A"], [0.0, 2.0, 1.0])
    assert sampling_rate_map["Recording 1 | ROI A"] == 1000.0


def test_frame_column_name_detection_handles_common_labels() -> None:
    assert is_ignored_trace_column("frame")
    assert is_ignored_trace_column("Frame #")
    assert is_ignored_trace_column("frame_idx")
    assert is_ignored_trace_column("Frame Number")
    assert is_ignored_trace_column("Frame_corrected")
    assert not is_ignored_trace_column("ROI frame response")
