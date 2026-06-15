from __future__ import annotations

import os
import json
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
    _write_long_plateau_debug_export,
    _write_gonzalez_cwt_debug_export,
    analyze_trace_data,
    build_pending_trace_states,
)
from app.hybrid_denoising import denoise_gonzalez_adaptive_wavelet, denoise_trace_state_guided_low_plateau_tv  # noqa: E402
from app.detection import detect_spikes  # noqa: E402
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
    assert result.gonzalez_cwt_debug["denoiser_used"] == "gonzalez_adaptive_wavelet"
    assert result.gonzalez_cwt_debug["input_mode"] == "corrected"
    assert "long_plateau_region_debug_rows" in result.long_plateau_debug
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


def test_default_analysis_uses_gonzalez_full_trace_wrapper_output() -> None:
    fs = 1000.0
    time = np.arange(260, dtype=float) / fs
    signal = 0.02 * np.sin(2.0 * np.pi * 5.0 * time)
    signal += 0.8 * np.exp(-0.5 * ((np.arange(time.size) - 120.0) / 3.0) ** 2)
    params = DetectionParams(
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=2.0,
        spike_prominence_k=1.0,
        min_separation_ms=5.0,
        gonzalez_max_clusters=4,
    )
    request = TraceAnalysisRequest(
        trace_name="trace",
        time=time,
        raw=signal.copy(),
        corrected_source=signal.copy(),
        sampling_rate_hz=fs,
        detection_params=params,
        artifact_params=ArtifactParams(),
        generation=1,
    )

    result = analyze_trace_data(request)
    expected = denoise_trace_state_guided_low_plateau_tv(
        result.corrected,
        fs=fs,
        plateau_mask=np.asarray(result.long_plateau_debug["long_plateau_final_mask"], dtype=bool),
        low_state_denoiser="gonzalez_full_trace",
        threshold_sd=float(params.gonzalez_threshold_sd),
        max_clusters=int(params.gonzalez_max_clusters),
        attenuation_min=float(params.gonzalez_attenuation_min),
        attenuation_max=float(params.gonzalez_attenuation_max),
        enable_noise_cluster_rejection=bool(params.gonzalez_enable_noise_cluster_rejection),
        enable_event_template_rejection=bool(params.gonzalez_enable_event_template_rejection),
        enable_low_state_safety_gate=bool(params.enable_low_state_safety_gate),
        min_event_peak_preservation=float(params.min_event_peak_preservation),
        min_local_max_ratio=float(params.min_local_max_ratio),
        min_reliable_low_state_fraction=float(params.min_reliable_low_state_fraction),
        min_reliable_low_state_samples=int(params.min_reliable_low_state_samples),
        plateau_tv_min_duration_ms=float(params.plateau_tv_min_duration_ms),
        plateau_tv_max_weight_factor=float(params.plateau_tv_max_weight_factor),
    )

    assert params.low_state_denoiser == "gonzalez_full_trace"
    assert np.allclose(result.hybrid_denoised, expected)
    assert result.gonzalez_cwt_debug["wrapper_output_mode"] == "gonzalez_full_trace"
    assert np.allclose(result.gonzalez_cwt_debug["wrapper_output"], result.hybrid_denoised)


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
        gonzalez_cwt_debug={"output": np.zeros(5), "denoiser_used": "gonzalez_adaptive_wavelet"},
        long_plateau_debug={"long_plateau_region_debug_rows": []},
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


def test_gonzalez_cwt_debug_export_writes_images_and_metrics(tmp_path: Path) -> None:
    fs = 1000.0
    time = np.arange(240, dtype=float) / fs
    trace = 0.03 * np.sin(2.0 * np.pi * 8.0 * time)
    trace += 0.18 * np.sin(2.0 * np.pi * 90.0 * time)
    trace += 0.7 * np.exp(-0.5 * ((np.arange(time.size) - 110.0) / 3.0) ** 2)
    debug = denoise_gonzalez_adaptive_wavelet(
        trace,
        fs=fs,
        input_mode="corrected",
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
        enable_noise_cluster_rejection=True,
    )
    wrapper = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=np.zeros(trace.shape, dtype=bool),
        low_state_denoiser="gonzalez_full_trace",
        return_debug=False,
        n_freqs=12,
        max_clusters=4,
        enable_noise_cluster_rejection=True,
    )
    debug = dict(debug)
    debug["wrapper_output"] = np.asarray(wrapper, dtype=float)
    debug["wrapper_correction_component"] = np.asarray(wrapper, dtype=float) - np.asarray(debug["output"], dtype=float)
    debug["wrapper_output_mode"] = "gonzalez_full_trace"

    written = _write_gonzalez_cwt_debug_export(
        trace_name="synthetic",
        time=time,
        corrected=trace,
        debug=debug,
        out_dir=tmp_path / "debug",
        startup_exclusion_ms=20.0,
    )
    names = {path.name for path in written}

    for name in [
        "02_morlet_cwt_scalogram_raw.png",
        "04_frequency_clusters.png",
        "07_attenuation_masks.png",
        "10_final_overlay.png",
        "11_removed_component.png",
        "12_wrapper_correction_component.png",
        "gonzalez_cwt_debug_metrics.json",
    ]:
        assert name in names
        assert (tmp_path / "debug" / name).exists()

    metrics = json.loads((tmp_path / "debug" / "gonzalez_cwt_debug_metrics.json").read_text(encoding="utf-8"))
    assert metrics["denoiser_used"] == "gonzalez_adaptive_wavelet"
    assert metrics["input_mode"] == "corrected"
    assert metrics["frequency_count"] == 12
    assert metrics["startup_exclusion_ms"] == 20.0
    assert metrics["startup_exclusion_samples"] == 20
    assert metrics["wrapper_output_mode"] == "gonzalez_full_trace"
    assert metrics["has_wrapper_output"] is True
    assert "rejected_frequency_clusters" in metrics
    assert "threshold_per_cluster" in metrics


def test_long_plateau_debug_records_regions_and_exports(tmp_path: Path) -> None:
    fs = 1000.0
    time = np.arange(900, dtype=float) / fs
    trace = 0.02 * np.sin(2.0 * np.pi * 3.0 * time)
    trace[180:380] += 1.0
    trace[560:595] += 0.9
    params = DetectionParams(
        low_state_denoiser="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        long_plateau_entry_noise_k=3.0,
        long_plateau_exit_noise_k=1.0,
        long_plateau_enter_sustain_ms=5.0,
        long_plateau_exit_sustain_ms=10.0,
    )

    result = detect_spikes(
        "synthetic",
        time,
        trace,
        params,
        sampling_rate_hz=fs,
    )
    debug = {
        str(key): value
        for key, value in result.qc_plot_data.items()
        if str(key).startswith("long_plateau_")
    }
    rows = debug["long_plateau_region_debug_rows"]

    assert any(bool(row["accepted"]) for row in rows)
    assert any(not bool(row["accepted"]) for row in rows)
    assert any(str(row["rejection_reason"]) for row in rows)

    written = _write_long_plateau_debug_export(
        trace_name="synthetic",
        time=time,
        corrected=trace,
        debug=debug,
        out_dir=tmp_path / "long_plateau",
        startup_exclusion_ms=100.0,
    )
    names = {path.name for path in written}
    for name in [
        "03_thresholds_overlay.png",
        "06_pipeline_masks.png",
        "07_final_overlap.png",
        "long_plateau_debug_regions.csv",
        "long_plateau_debug_metrics.json",
    ]:
        assert name in names
        assert (tmp_path / "long_plateau" / name).exists()
    metrics = json.loads((tmp_path / "long_plateau" / "long_plateau_debug_metrics.json").read_text(encoding="utf-8"))
    assert metrics["startup_exclusion_ms"] == 100.0
    assert metrics["startup_exclusion_samples"] == 100
