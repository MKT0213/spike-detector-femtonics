from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QFormLayout, QGroupBox  # noqa: E402

from detector.core import gonzalez_denoising  # noqa: E402
from detector.core import gui_main  # noqa: E402
from detector.core.gui_main import SpikeCurationMainWindow, TraceAnalysisRequest, analyze_trace_data  # noqa: E402
from detector.core.gonzalez_denoising import (  # noqa: E402
    denoise_trace_gonzalez_full_trace,
    normalize_gonzalez_denoising_method,
)
from detector.core.models import ArtifactParams, DetectionParams, TraceCuration  # noqa: E402
from detector.core.pipeline import PipelineConfig  # noqa: E402


def _current_pipeline() -> dict[str, object]:
    return PipelineConfig.from_preset("current").to_dict()


def _synthetic_trace(n: int = 240, fs: float = 1000.0) -> np.ndarray:
    time = np.arange(n, dtype=float) / fs
    trace = 0.03 * np.sin(2.0 * np.pi * 7.0 * time)
    trace += 0.16 * np.sin(2.0 * np.pi * 120.0 * time)
    trace += 0.9 * np.exp(-0.5 * ((np.arange(n, dtype=float) - 110.0) / 3.0) ** 2)
    return trace

def test_gonzalez_full_trace_and_none_modes_are_shape_stable() -> None:
    fs = 1000.0
    trace = _synthetic_trace(fs=fs)

    debug = denoise_trace_gonzalez_full_trace(
        trace,
        fs=fs,
        denoising_method="gonzalez_full_trace",
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )
    passthrough = denoise_trace_gonzalez_full_trace(trace, fs=fs, denoising_method="none")

    assert debug["denoiser_used"] == "gonzalez_full_trace"
    assert debug["denoising_method"] == "gonzalez_full_trace"
    assert np.asarray(debug["output"]).shape == trace.shape
    assert np.asarray(debug["gonzalez_input"]).shape == trace.shape
    assert "plateau_tv_output" not in debug
    assert np.allclose(passthrough, trace)
    assert normalize_gonzalez_denoising_method("gonzalez_adaptive_wavelet") == "gonzalez_full_trace"


def test_gonzalez_full_trace_dff_uses_supplied_baseline() -> None:
    fs = 1000.0
    trace = _synthetic_trace(fs=fs)
    baseline = 100.0 + 0.2 * np.sin(2.0 * np.pi * 2.0 * np.arange(trace.size) / fs)

    debug = denoise_trace_gonzalez_full_trace(
        trace,
        fs=fs,
        denoising_method="gonzalez_full_trace_dff",
        normalization_baseline=baseline,
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )

    assert debug["denoiser_used"] == "gonzalez_full_trace_dff"
    assert np.asarray(debug["output"]).shape == trace.shape
    assert debug["input_normalization"] == "dff_detector_baseline"
    assert debug["input_normalization_label"] == "dF/F0 using detector baseline"
    assert debug["normalization_source"] == "detector_baseline"
    assert np.allclose(debug["normalization_baseline"], baseline)
    assert np.allclose(debug["working_dff"], trace / baseline)


def test_full_trace_and_dff_use_same_core_after_input_preparation(monkeypatch) -> None:
    fs = 1000.0
    trace = _synthetic_trace(fs=fs)
    baseline = 100.0 + 0.2 * np.sin(2.0 * np.pi * 2.0 * np.arange(trace.size) / fs)
    calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def spy_core(values, fs, return_debug=False, **kwargs):
        calls.append((np.asarray(values, dtype=float).copy(), dict(kwargs)))
        output = np.asarray(values, dtype=float).copy()
        return {"output": output} if return_debug else output

    monkeypatch.setattr(gonzalez_denoising, "denoise_gonzalez_adaptive_wavelet", spy_core)

    full_trace = gonzalez_denoising.denoise_trace_gonzalez_full_trace(
        trace,
        fs=fs,
        denoising_method="gonzalez_full_trace",
        threshold_sd=2.0,
        max_clusters=4,
    )
    dff_trace = gonzalez_denoising.denoise_trace_gonzalez_full_trace(
        trace,
        fs=fs,
        denoising_method="gonzalez_full_trace_dff",
        normalization_baseline=baseline,
        threshold_sd=2.0,
        max_clusters=4,
    )

    assert len(calls) == 2
    assert np.allclose(calls[0][0], trace)
    assert np.allclose(calls[1][0], trace / baseline)
    assert calls[0][1] == calls[1][1]
    assert calls[0][1]["input_mode"] == "corrected"
    assert np.allclose(full_trace, trace)
    assert np.allclose(dff_trace, trace)


def test_gonzalez_s7_mode_uses_dff_and_s7_structural_steps() -> None:
    fs = 1000.0
    trace = _synthetic_trace(fs=fs)
    baseline = np.full(trace.shape, 100.0, dtype=float)

    debug = denoise_trace_gonzalez_full_trace(
        trace,
        fs=fs,
        denoising_method="gonzalez_s7_dff",
        normalization_baseline=baseline,
        return_debug=True,
        n_freqs=18,
        max_clusters=5,
    )
    core_debug = debug["gonzalez_debug"]

    assert debug["denoiser_used"] == "gonzalez_s7_dff"
    assert debug["input_normalization"] == "dff_detector_baseline"
    assert np.allclose(debug["working_dff"], trace / baseline)
    assert np.asarray(debug["output"]).shape == trace.shape
    assert core_debug["enable_noise_cluster_rejection"] is True
    assert core_debug["enable_event_template_rejection"] is True
    assert core_debug["event_template_rejection_mode"] == "pc1_negative"
    assert core_debug["low_frequency_gaussian_branch"]["enabled"] is True
    assert np.asarray(core_debug["s7_low_frequency_trace"]).shape == trace.shape
    assert debug["safety"]["level_preservation_merge"]["status"] == "skipped_for_s7_reproduction"
    assert debug["safety"]["amplitude_preservation_merge"]["status"] == "skipped_for_s7_reproduction"


def test_analysis_passes_first_pass_baseline_to_dff_denoiser(monkeypatch) -> None:
    fs = 1000.0
    time = np.arange(220, dtype=float) / fs
    raw = 120.0 + _synthetic_trace(n=time.size, fs=fs)
    params = DetectionParams(
        denoising_method="gonzalez_full_trace_dff",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=2.0,
        spike_prominence_k=1.0,
        min_separation_ms=5.0,
    )
    request = TraceAnalysisRequest(
        trace_name="dff_trace",
        time=time,
        raw=raw.copy(),
        corrected_source=raw.copy(),
        sampling_rate_hz=fs,
        detection_params=params,
        artifact_params=ArtifactParams(),
        pipeline_config=_current_pipeline(),
        generation=1,
    )
    captured: dict[str, np.ndarray | str | float] = {}

    def spy_denoise(corrected, fs, denoising_method, normalization_baseline=None, **kwargs):
        captured["corrected"] = np.asarray(corrected, dtype=float).copy()
        captured["fs"] = float(fs)
        captured["denoising_method"] = str(denoising_method)
        captured["normalization_baseline"] = np.asarray(normalization_baseline, dtype=float).copy()
        output = np.asarray(corrected, dtype=float)
        if kwargs.get("return_debug"):
            return {
                "output": output,
                "gonzalez_debug": {
                    "denoiser_used": "gonzalez_adaptive_wavelet",
                    "output": output,
                },
            }
        return output

    monkeypatch.setattr(gui_main, "denoise_trace_gonzalez_full_trace", spy_denoise)

    result = analyze_trace_data(request)

    assert captured["denoising_method"] == "gonzalez_full_trace_dff"
    assert captured["fs"] == fs
    assert np.allclose(captured["normalization_baseline"], result.correction_baseline)
    assert np.allclose(result.hybrid_denoised, captured["corrected"])
    assert result.gonzalez_cwt_debug["wrapper_output_mode"] == "gonzalez_full_trace_dff"
    assert result.gonzalez_cwt_debug["wrapper_input_normalization"] == "dF/F0 using detector baseline"


def test_detection_params_migrates_removed_denoising_keys() -> None:
    params = DetectionParams.from_dict(
        {
            "low_state_denoiser": "gonzalez_adaptive_wavelet",
            "plateau_tv_weight": 50.0,
            "low_state_boundary_protect_ms": 200.0,
        }
    )

    assert params.denoising_method == "gonzalez_full_trace"
    assert DetectionParams.from_dict({"low_state_denoiser": "none"}).denoising_method == "none"


def test_ui_only_exposes_active_gonzalez_denoising_methods() -> None:
    app = QApplication.instance() or QApplication([])
    window = SpikeCurationMainWindow()
    try:
        methods = [
            str(window.param_denoising_method.itemData(index))
            for index in range(window.param_denoising_method.count())
        ]

        assert DetectionParams().denoising_method == "gonzalez_full_trace"
        assert methods == ["gonzalez_full_trace", "gonzalez_full_trace_dff", "gonzalez_s7_dff", "none"]
        assert [
            str(window.param_denoising_method.itemText(index))
            for index in range(window.param_denoising_method.count())
        ] == ["On (corrected trace)", "On (dF/F0 input)", "S7 reproduction (dF/F0)", "Off"]
        assert window.param_denoising_method.currentData() == "gonzalez_full_trace"
        assert window.param_denoising_method.currentText() == "On (corrected trace)"
        assert window.param_denoising_input_normalization.text() == "Corrected trace"
        denoise_groups = [group for group in window.findChildren(QGroupBox) if group.title() == "Denoising"]
        assert len(denoise_groups) == 1
        denoise_layout = denoise_groups[0].layout()
        assert isinstance(denoise_layout, QFormLayout)
        assert denoise_layout.rowCount() == 8
        denoise_labels = [
            denoise_layout.itemAt(row, QFormLayout.LabelRole).widget().text()
            for row in range(denoise_layout.rowCount())
            if denoise_layout.itemAt(row, QFormLayout.LabelRole) is not None
        ]
        assert denoise_labels == [
            "Denoising",
            "Denoiser input",
            "Noise threshold SD",
            "Cluster count",
            "Min suppression",
            "Max suppression",
        ]
        assert window.chk_show_hybrid_denoised.text() == "Show selected overlay"
        assert "Hybrid" not in window.chk_show_hybrid_denoised.text()
        assert [
            str(window.display_render_source_combo.itemText(index))
            for index in range(window.display_render_source_combo.count())
        ] == [
            "Denoised trace",
            "Baseline corrected",
            "Raw baseline-subtracted",
            "Denoised dF/F0",
            "High-pass",
            "Low-pass",
        ]
        assert window.display_render_source_combo.currentData() == "denoised"
        assert "selected overlay" in window.display_render_source_combo.toolTip().lower()
        assert "residual overlay" in window.chk_show_denoise_residual.toolTip().lower()
        np.testing.assert_allclose(
            window._raw_overlay_signal(np.array([101.0, 102.0]), np.array([100.0, 100.0])),
            np.array([1.0, 2.0]),
        )
        assert window._raw_overlay_signal(np.array([101.0, 102.0]), None) is None
        window.param_denoising_method.setCurrentIndex(window.param_denoising_method.findData("gonzalez_full_trace_dff"))
        assert window.param_denoising_method.currentText() == "On (dF/F0 input)"
        assert window.param_denoising_input_normalization.text() == "dF/F0 using detector baseline"
        window.param_denoising_method.setCurrentIndex(window.param_denoising_method.findData("gonzalez_s7_dff"))
        assert window.param_denoising_method.currentText() == "S7 reproduction (dF/F0)"
        assert window.param_denoising_input_normalization.text() == "dF/F0 using detector baseline + S7 low-frequency branch"
        window.param_denoising_method.setCurrentIndex(window.param_denoising_method.findData("none"))
        assert window.param_denoising_input_normalization.text() == "Off"
        assert not hasattr(window, "param_plateau_tv_weight")
        assert not hasattr(window, "param_low_state_boundary_protect_ms")
    finally:
        window.close()
        app.processEvents()


def test_signal_overlay_controls_are_available_in_quick_panel_and_pipeline_page() -> None:
    app = QApplication.instance() or QApplication([])
    window = SpikeCurationMainWindow()
    try:
        assert window.overlay_controls_group.title() == "Signal Overlays"
        assert window.pipeline_display_render_source_combo.count() == window.display_render_source_combo.count()
        assert window.pipeline_display_render_source_combo.currentData() == window.display_render_source_combo.currentData()

        window.chk_show_raw.setChecked(False)
        assert not window.pipeline_chk_show_raw.isChecked()

        window.pipeline_chk_show_lowpass.setChecked(True)
        assert window.chk_show_lowpass.isChecked()

        window.pipeline_display_render_source_combo.setCurrentIndex(
            window.pipeline_display_render_source_combo.findData("highpass")
        )
        assert window.display_render_source_combo.currentData() == "highpass"

        window.display_units_combo.setCurrentIndex(
            window.display_units_combo.findData("percent_dff")
        )
        assert window.pipeline_display_units_combo.currentData() == "percent_dff"
    finally:
        window.close()
        app.processEvents()


def test_selected_overlay_can_render_corrected_without_changing_denoise_residual() -> None:
    app = QApplication.instance() or QApplication([])
    window = SpikeCurationMainWindow()
    try:
        n = 12
        corrected = np.linspace(-1.0, 1.0, n)
        denoised = corrected * 0.5
        state = TraceCuration(
            trace_name="trace",
            raw=100.0 + corrected,
            corrected_source=100.0 + corrected,
            corrected=corrected.copy(),
            time=np.arange(n, dtype=float) / 1000.0,
            sampling_rate_hz=1000.0,
            correction_baseline=np.full(n, 100.0),
            hybrid_denoised=denoised.copy(),
            analysis_status="done",
        )
        window.traces = {"trace": state}
        window.trace_names = ["trace"]
        window.current_trace_name = "trace"
        window.display_render_source_combo.setCurrentIndex(
            window.display_render_source_combo.findData("corrected")
        )
        window.chk_show_denoise_residual.setChecked(True)

        window._render_current_trace(refit_y=True)

        overlay_x, overlay_y = window.plot.main_hybrid_denoised.getOriginalDataset()
        residual_x, residual_y = window.plot.main_denoise_residual.getOriginalDataset()
        assert np.allclose(overlay_y, corrected)
        assert np.allclose(residual_y, corrected - denoised)
        assert window.plot._hybrid_denoised_label == "Baseline corrected"
        assert overlay_x.shape == residual_x.shape
    finally:
        window.close()
        app.processEvents()
