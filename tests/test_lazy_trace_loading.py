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

from detector.core import gui_main  # noqa: E402
from detector.core.gui_main import (  # noqa: E402
    SpikeCurationMainWindow,
    TraceAnalysisRequest,
    TraceAnalysisResult,
    _write_long_plateau_debug_export,
    _write_gonzalez_cwt_debug_export,
    _rolling_block_mean_1d,
    analyze_trace_data,
    build_pending_trace_states,
)
from detector.core.gonzalez_denoising import denoise_gonzalez_adaptive_wavelet, denoise_trace_gonzalez_full_trace  # noqa: E402
from detector.core.detection import detect_spikes  # noqa: E402
from detector.core.models import ArtifactParams, DetectionParams, TraceCuration  # noqa: E402
from detector.core.pipeline import PipelineConfig  # noqa: E402


def _current_pipeline() -> dict[str, object]:
    return PipelineConfig.from_preset("current").to_dict()


def test_analyze_trace_data_returns_completed_detection_payload() -> None:
    time = np.linspace(0.0, 0.199, 200)
    signal = np.zeros_like(time)
    signal[80] = 8.0
    params = DetectionParams(
        denoising_method="none",
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
        pipeline_config=_current_pipeline(),
        generation=1,
    )

    result = analyze_trace_data(request)

    assert result.corrected.shape == signal.shape
    assert result.hybrid_denoised.shape == signal.shape
    assert result.correction_baseline.shape == signal.shape
    assert result.gonzalez_cwt_debug["denoiser_used"] == "none"
    assert result.gonzalez_cwt_debug["input_normalization"] == "none"
    assert "long_plateau_region_debug_rows" in result.long_plateau_debug
    assert result.artifact_mask.shape == signal.shape
    assert np.isfinite(result.corrected).all()
    assert hasattr(result.detection, "spikes")
    assert isinstance(result.plateau_signal_classification, list)


def test_default_analysis_skips_artifact_interpolation() -> None:
    time = np.linspace(0.0, 0.199, 200)
    signal = np.zeros_like(time)
    signal[80] = 8.0
    params = DetectionParams(
        denoising_method="none",
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
        pipeline_config=_current_pipeline(),
        generation=1,
    )

    original = gui_main.apply_artifact_interpolation

    def fail_artifact_interpolation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("artifact interpolation should be opt-in")

    gui_main.apply_artifact_interpolation = fail_artifact_interpolation
    try:
        result = analyze_trace_data(request)
    finally:
        gui_main.apply_artifact_interpolation = original

    assert result.artifact_mask.shape == signal.shape
    assert not np.any(result.artifact_mask)


def test_optional_artifact_cleanup_step_runs_interpolation() -> None:
    time = np.linspace(0.0, 0.199, 200)
    signal = np.zeros_like(time)
    signal[80] = 8.0
    params = DetectionParams(
        denoising_method="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=2.0,
        spike_prominence_k=1.0,
        min_separation_ms=5.0,
    )
    pipeline = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "artifact_cleanup",
                "baseline_correction",
                "plateau_detection",
                "denoising",
                "spike_detection",
                "review_save",
                "signal_display",
            ],
            "enabled_steps": {
                "artifact_cleanup": True,
                "denoising": False,
            },
        }
    )
    request = TraceAnalysisRequest(
        trace_name="trace",
        time=time,
        raw=signal.copy(),
        corrected_source=signal.copy(),
        sampling_rate_hz=1000.0,
        detection_params=params,
        artifact_params=ArtifactParams(),
        pipeline_config=pipeline.to_dict(),
        generation=1,
    )
    calls: list[int] = []
    original = gui_main.apply_artifact_interpolation

    def fake_artifact_interpolation(
        time_values: np.ndarray,
        values: np.ndarray,
        params: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(1)
        mask = np.zeros(np.asarray(values).size, dtype=bool)
        mask[1] = True
        return np.asarray(values, dtype=float).copy(), mask

    gui_main.apply_artifact_interpolation = fake_artifact_interpolation
    try:
        result = analyze_trace_data(request)
    finally:
        gui_main.apply_artifact_interpolation = original

    assert calls == [1]
    assert result.artifact_mask[1]


def test_no_denoising_analysis_skips_gonzalez_cwt_debug(monkeypatch) -> None:
    def fail_cwt_debug(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Gonzalez CWT debug should not run when denoising is disabled")

    monkeypatch.setattr(gui_main, "denoise_gonzalez_adaptive_wavelet", fail_cwt_debug)
    time = np.linspace(0.0, 0.199, 200)
    signal = np.zeros_like(time)
    signal[80] = 8.0
    params = DetectionParams(
        denoising_method="none",
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
        pipeline_config=_current_pipeline(),
        generation=1,
    )

    result = analyze_trace_data(request)

    assert result.gonzalez_cwt_debug["denoiser_used"] == "none"
    assert np.allclose(result.hybrid_denoised, result.corrected)


def test_analyze_trace_data_detects_obvious_short_plateau_after_baseline_correction() -> None:
    fs = 1000.0
    n = 420
    time = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(11)
    signal = 200.0 + (0.05 * rng.standard_normal(n))
    signal[140:190] += 8.0
    params = DetectionParams(
        denoising_method="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=5.0,
        spike_prominence_k=3.0,
        min_separation_ms=10.0,
    )
    request = TraceAnalysisRequest(
        trace_name="short_plateau_trace",
        time=time,
        raw=signal.copy(),
        corrected_source=signal.copy(),
        sampling_rate_hz=fs,
        detection_params=params,
        artifact_params=ArtifactParams(),
        pipeline_config=_current_pipeline(),
        generation=1,
    )

    result = analyze_trace_data(request)
    short_mask = np.asarray(result.detection.qc_plot_data["short_plateau_mask"], dtype=bool)
    events = np.asarray(result.detection.qc_plot_data["short_plateau_event_table"], dtype=float)

    assert events.shape == (1, 6)
    assert np.count_nonzero(short_mask) >= 25
    assert events[0, 0] <= 145
    assert events[0, 1] >= 180


def test_analyze_trace_data_applies_pipeline_filter_step_before_spike_detection() -> None:
    fs = 1000.0
    time = np.arange(500, dtype=float) / fs
    signal = 0.4 * np.sin(2.0 * np.pi * 8.0 * time)
    signal += 0.25 * np.sin(2.0 * np.pi * 160.0 * time)
    signal[250] += 2.0
    params = DetectionParams(
        denoising_method="none",
        lowpass_cutoff_hz=30.0,
        lowpass_order=4,
        detection_mode="mad",
        spike_noise_k=5.0,
        spike_prominence_k=2.0,
    )
    pipeline = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "artifact_cleanup",
                "baseline_correction",
                "plateau_detection",
                "lowpass_filter",
                "denoising",
                "spike_detection",
                "review_save",
                "signal_display",
            ],
            "enabled_steps": {
                "lowpass_filter": True,
                "denoising": False,
            },
        }
    )
    request = TraceAnalysisRequest(
        trace_name="filtered",
        time=time,
        raw=signal.copy(),
        corrected_source=signal.copy(),
        sampling_rate_hz=fs,
        detection_params=params,
        artifact_params=ArtifactParams(),
        pipeline_config=pipeline.to_dict(),
        generation=1,
    )

    result = analyze_trace_data(request)

    assert result.lowpass_trace is not None
    assert np.allclose(result.detection.processed_signal, result.lowpass_trace)
    assert float(np.std(np.diff(result.lowpass_trace))) < float(np.std(np.diff(result.corrected)))


def test_rolling_block_mean_downsamples_numeric_values() -> None:
    result = _rolling_block_mean_1d(np.arange(1, 9, dtype=float), 4)

    assert np.allclose(result, np.array([2.5, 6.5]))


def test_analyze_trace_data_applies_rolling_downsample_before_baseline() -> None:
    fs = 1000.0
    time = np.arange(400, dtype=float) / fs
    signal = np.arange(400, dtype=float)
    params = DetectionParams(
        denoising_method="none",
        rolling_downsample_factor=4,
        startup_exclusion_ms=0.0,
    )
    pipeline = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "rolling_average_downsample",
                "baseline_correction",
                "plateau_detection",
                "denoising",
                "spike_detection",
                "review_save",
                "signal_display",
            ],
            "enabled_steps": {
                "rolling_average_downsample": True,
                "denoising": False,
            },
        }
    )
    request = TraceAnalysisRequest(
        trace_name="downsampled",
        time=time,
        raw=signal.copy(),
        corrected_source=signal.copy(),
        sampling_rate_hz=fs,
        detection_params=params,
        artifact_params=ArtifactParams(),
        pipeline_config=pipeline.to_dict(),
        generation=1,
    )

    result = analyze_trace_data(request)

    assert result.time.shape == (100,)
    assert result.raw.shape == (100,)
    assert result.corrected_source.shape == (100,)
    assert result.corrected.shape == (100,)
    assert result.artifact_mask.shape == (100,)
    assert result.sampling_rate_hz == 250.0
    assert np.allclose(result.time[:2], np.array([0.0015, 0.0055]))
    assert np.allclose(result.raw[:2], np.array([1.5, 5.5]))


def test_rolling_downsample_placement_before_and_after_spike_detection() -> None:
    fs = 1000.0
    time = np.arange(400, dtype=float) / fs
    signal = np.sin(2.0 * np.pi * 5.0 * time)
    params = DetectionParams(
        denoising_method="none",
        rolling_downsample_factor=4,
        startup_exclusion_ms=0.0,
    )
    before_spike = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "baseline_correction",
                "plateau_detection",
                "denoising",
                "rolling_average_downsample",
                "spike_detection",
                "review_save",
                "signal_display",
            ],
            "enabled_steps": {
                "rolling_average_downsample": True,
                "denoising": False,
            },
        }
    )
    after_spike = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "baseline_correction",
                "plateau_detection",
                "denoising",
                "spike_detection",
                "rolling_average_downsample",
                "review_save",
                "signal_display",
            ],
            "enabled_steps": {
                "rolling_average_downsample": True,
                "denoising": False,
            },
        }
    )

    before_result = analyze_trace_data(
        TraceAnalysisRequest(
            trace_name="before_spike",
            time=time,
            raw=signal.copy(),
            corrected_source=signal.copy(),
            sampling_rate_hz=fs,
            detection_params=params,
            artifact_params=ArtifactParams(),
            pipeline_config=before_spike.to_dict(),
            generation=1,
        )
    )
    after_result = analyze_trace_data(
        TraceAnalysisRequest(
            trace_name="after_spike",
            time=time,
            raw=signal.copy(),
            corrected_source=signal.copy(),
            sampling_rate_hz=fs,
            detection_params=params,
            artifact_params=ArtifactParams(),
            pipeline_config=after_spike.to_dict(),
            generation=1,
        )
    )

    assert before_result.time.shape == (100,)
    assert before_result.sampling_rate_hz == 250.0
    assert after_result.time.shape == signal.shape
    assert after_result.sampling_rate_hz == fs


def test_rerun_request_uses_original_arrays_after_downsampled_result() -> None:
    app = QApplication.instance() or QApplication([])
    window = SpikeCurationMainWindow()
    fs = 1000.0
    time = np.arange(400, dtype=float) / fs
    signal = np.arange(400, dtype=float)
    states = build_pending_trace_states(
        time_map={"trace": time},
        raw_map={"trace": signal.copy()},
        corrected_map={"trace": signal.copy()},
        sampling_rate_map={"trace": fs},
        generation=1,
    )
    state = states["trace"]
    window.traces = states
    window.trace_names = ["trace"]
    window.current_trace_name = "trace"
    params = DetectionParams(
        denoising_method="none",
        rolling_downsample_factor=4,
        startup_exclusion_ms=0.0,
    )
    pipeline = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "rolling_average_downsample",
                "baseline_correction",
                "plateau_detection",
                "denoising",
                "spike_detection",
            ],
            "enabled_steps": {
                "rolling_average_downsample": True,
                "denoising": False,
            },
        }
    )
    result = analyze_trace_data(
        TraceAnalysisRequest(
            trace_name="trace",
            time=time,
            raw=signal.copy(),
            corrected_source=signal.copy(),
            sampling_rate_hz=fs,
            detection_params=params,
            artifact_params=ArtifactParams(),
            pipeline_config=pipeline.to_dict(),
            generation=1,
        )
    )

    window._apply_analysis_result(state, result)
    rerun_request = window._make_analysis_request(state)

    assert state.time.shape == (100,)
    assert state.original_time is not None and state.original_time.shape == (400,)
    assert rerun_request.time.shape == (400,)
    assert rerun_request.raw.shape == (400,)
    assert rerun_request.corrected_source.shape == (400,)
    app.processEvents()


def test_analysis_detects_workbook_style_burst_plateaus_after_long_near_miss() -> None:
    fs = 2083.3333333333
    n = 3400
    time = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(12)
    signal = 200.0 + (0.04 * rng.standard_normal(n))
    signal[538:1192] += 60.0
    for start, stop in [(2229, 2336), (2959, 3105)]:
        signal[start:stop] += 40.0
        signal[np.arange(start + 3, stop - 3, 3)] += 24.0
        signal[start - 18 : start] += 3.0
        signal[stop : stop + 18] += 3.0
    params = DetectionParams(
        denoising_method="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=5.0,
        spike_prominence_k=3.0,
        min_separation_ms=10.0,
    )
    request = TraceAnalysisRequest(
        trace_name="12_0_16_style",
        time=time,
        raw=signal.copy(),
        corrected_source=signal.copy(),
        sampling_rate_hz=fs,
        detection_params=params,
        artifact_params=ArtifactParams(),
        pipeline_config=_current_pipeline(),
        generation=1,
    )

    result = analyze_trace_data(request)
    burst_mask = np.asarray(result.detection.qc_plot_data["burst_plateau_mask"], dtype=bool)
    burst_events = np.asarray(result.detection.qc_plot_data["burst_plateau_event_table"], dtype=float)

    assert burst_events.shape[0] >= 2
    assert np.count_nonzero(burst_mask[2229:2336]) >= 90
    assert np.count_nonzero(burst_mask[2959:3105]) >= 120
    rows = result.plateau_signal_classification
    burst_rows = [row for row in rows if row.get("detector_source") == "burst"]
    assert len(burst_rows) >= 2
    assert all(row.get("signal_type") != "unknown" for row in burst_rows)


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
        pipeline_config=_current_pipeline(),
        generation=1,
    )

    result = analyze_trace_data(request)
    expected = denoise_trace_gonzalez_full_trace(
        result.corrected,
        fs=fs,
        denoising_method="gonzalez_full_trace",
        normalization_baseline=result.correction_baseline,
        threshold_sd=float(params.gonzalez_threshold_sd),
        max_clusters=int(params.gonzalez_max_clusters),
        attenuation_min=float(params.gonzalez_attenuation_min),
        attenuation_max=float(params.gonzalez_attenuation_max),
        enable_noise_cluster_rejection=bool(params.gonzalez_enable_noise_cluster_rejection),
        enable_event_template_rejection=bool(params.gonzalez_enable_event_template_rejection),
        enable_denoise_safety_gate=bool(params.enable_denoise_safety_gate),
        min_event_peak_preservation=float(params.min_event_peak_preservation),
        min_local_max_ratio=float(params.min_local_max_ratio),
    )

    assert params.denoising_method == "gonzalez_full_trace"
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
        time=time.copy(),
        raw=np.ones(5),
        corrected_source=np.ones(5),
        corrected=np.zeros(5),
        hybrid_denoised=np.zeros(5),
        gonzalez_cwt_debug={"output": np.zeros(5), "denoiser_used": "gonzalez_adaptive_wavelet"},
        long_plateau_debug={"long_plateau_region_debug_rows": []},
        plateau_signal_classification=[],
        correction_baseline=np.zeros(5),
        detection=detection,
        artifact_mask=np.zeros(5, dtype=bool),
        sampling_rate_hz=1000.0,
        highpass_trace=None,
        lowpass_trace=None,
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
    wrapper = denoise_trace_gonzalez_full_trace(
        trace,
        fs=fs,
        denoising_method="gonzalez_full_trace",
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
        denoising_method="none",
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
