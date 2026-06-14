from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.gui_main import (  # noqa: E402
    SpikeCurationMainWindow,
    TraceAnalysisRequest,
    TraceAnalysisResult,
    analyze_trace_data,
    build_pending_trace_states,
)
from app.models import ArtifactParams, DetectionParams, TraceCuration  # noqa: E402


def test_analyze_trace_data_returns_completed_detection_payload() -> None:
    time = np.linspace(0.0, 0.199, 200)
    signal = np.zeros_like(time)
    signal[80] = 8.0
    params = DetectionParams(
        low_state_denoiser="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=2.0,
        spike_prominence_k=1.0,
        min_separation_ms=5.0,
    )
    request = TraceAnalysisRequest(
        trace_name="trace",
        time=time,
        raw=signal.copy(),
        corrected_source=signal.copy(),
        sampling_rate_hz=1000.0,
        detection_params=params,
        artifact_params=ArtifactParams(),
        generation=1,
    )

    result = analyze_trace_data(request)

    assert result.corrected.shape == signal.shape
    assert result.hybrid_denoised.shape == signal.shape
    assert result.correction_baseline.shape == signal.shape
    assert result.artifact_mask.shape == signal.shape
    assert np.isfinite(result.corrected).all()
    assert hasattr(result.detection, "spikes")


def test_build_pending_trace_states_does_not_analyze_traces() -> None:
    time = np.arange(5, dtype=float)
    states = build_pending_trace_states(
        time_map={"a": time, "b": time},
        raw_map={"a": np.ones(5), "b": np.full(5, 2.0)},
        corrected_map={"a": np.ones(5), "b": np.full(5, 3.0)},
        sampling_rate_map={"a": 10.0, "b": 20.0},
        generation=7,
    )

    assert set(states) == {"a", "b"}
    assert all(state.analysis_status == "pending" for state in states.values())
    assert all(state.spikes == [] for state in states.values())
    assert states["a"].analysis_generation == 7
    assert np.array_equal(states["b"].corrected, np.full(5, 3.0))


def test_analysis_status_transitions_to_done_and_error() -> None:
    app = QApplication.instance() or QApplication([])
    window = SpikeCurationMainWindow()
    time = np.arange(5, dtype=float)
    state = TraceCuration(
        trace_name="trace",
        raw=np.ones(5),
        corrected_source=np.ones(5),
        corrected=np.ones(5),
        time=time,
        sampling_rate_hz=1000.0,
        analysis_status="running",
        analysis_generation=3,
    )
    window.traces = {"trace": state}
    window.trace_names = ["trace"]
    window.current_trace_name = "trace"
    window._analysis_generation = 3
    detection = SimpleNamespace(
        spikes=[],
        smoothed=np.zeros(5),
        baseline=np.zeros(5),
        threshold=np.zeros(5),
        candidate_windows=[],
        qc_plot_data={
            "baseline_mask": np.zeros(5, dtype=bool),
            "long_plateau_mask": np.zeros(5, dtype=bool),
            "short_plateau_mask": np.zeros(5, dtype=bool),
            "short_plateau_event_table": np.zeros((0, 6), dtype=float),
        },
    )
    result = TraceAnalysisResult(
        corrected=np.zeros(5),
        hybrid_denoised=np.zeros(5),
        correction_baseline=np.zeros(5),
        detection=detection,
        artifact_mask=np.zeros(5, dtype=bool),
        sampling_rate_hz=1000.0,
    )

    window._on_trace_analysis_finished("trace", 3, result, None)
    assert state.analysis_status == "done"
    assert state.analysis_error == ""

    state.analysis_status = "running"
    window._on_trace_analysis_finished("trace", 3, None, "boom")
    assert state.analysis_status == "error"
    assert state.analysis_error == "boom"
    app.processEvents()
