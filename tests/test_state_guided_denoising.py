from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gui_main import SpikeCurationMainWindow  # noqa: E402
from app.hybrid_denoising import (  # noqa: E402
    denoise_gonzalez_adaptive_wavelet,
    denoise_trace_hybrid_low_plateau_tv,
    denoise_trace_state_guided_low_plateau_tv,
    gonzalez_adaptive_wavelet_validation_metrics,
)
from app.models import DetectionParams  # noqa: E402


def _synthetic_trace() -> tuple[np.ndarray, np.ndarray, float]:
    fs = 1000.0
    n = 320
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=float) / fs
    trace = 0.03 * rng.standard_normal(n)
    for idx, amp in [(55, 0.8), (120, 0.7), (250, 0.9)]:
        trace += amp * np.exp(-0.5 * ((np.arange(n) - idx) / 2.0) ** 2)
    plateau = np.zeros(n, dtype=bool)
    plateau[170:220] = True
    trace[plateau] += 1.2 + (0.02 * rng.standard_normal(np.count_nonzero(plateau)))
    trace += 0.02 * np.sin(2.0 * np.pi * 4.0 * t)
    return trace, plateau, fs


def test_state_guided_default_gonzalez_uses_separate_plateau_path() -> None:
    trace, plateau, fs = _synthetic_trace()

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        boundary_protect_ms=2.0,
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )

    output = np.asarray(debug["output"], dtype=float)
    boundary = np.asarray(debug["boundary_mask"], dtype=bool)
    protected = np.asarray(debug["protected_plateau_mask"], dtype=bool)
    plateau_core = np.asarray(debug["plateau_core_mask"], dtype=bool)
    low_state_input = np.asarray(debug["low_state_input"], dtype=float)
    plateau_tv = np.asarray(debug["plateau_tv_output"], dtype=float)
    direct_gonzalez = np.asarray(debug["gonzalez_debug"]["output"], dtype=float)

    assert debug["denoiser_used"] == "state_guided_low_plateau_tv"
    assert debug["low_state_denoiser_used"] == "gonzalez_adaptive_wavelet"
    assert debug["gonzalez_debug"]["input_mode"] == "corrected"
    assert debug["gonzalez_debug"]["F0"] is None
    assert np.allclose(debug["gonzalez_debug"]["working_trace"], low_state_input)
    assert low_state_input.shape == trace.shape
    assert np.asarray(debug["low_state_output"]).shape == trace.shape
    assert np.asarray(debug["hybrid_input"]).shape == trace.shape
    assert np.asarray(debug["hybrid_output"]).shape == trace.shape
    assert debug["plateau_tv_status"] == "applied"
    assert plateau_tv.shape == trace.shape
    assert protected.shape == trace.shape
    assert plateau_core.shape == trace.shape
    assert output.shape == trace.shape
    assert np.all(np.isfinite(output))
    assert not np.allclose(low_state_input[protected], trace[protected])
    assert np.allclose(output[plateau_core], plateau_tv[plateau_core])
    assert np.allclose(output[boundary], trace[boundary])
    assert not np.allclose(output[plateau], direct_gonzalez[plateau])
    assert not np.allclose(output[boundary], direct_gonzalez[boundary])


def test_state_guided_legacy_and_none_modes_smoke() -> None:
    trace, plateau, fs = _synthetic_trace()

    legacy_debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        low_state_denoiser="legacy_hybrid",
        boundary_protect_ms=2.0,
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )
    none = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=np.zeros_like(plateau, dtype=bool),
        low_state_denoiser="none",
    )

    legacy = np.asarray(legacy_debug["output"], dtype=float)
    boundary = np.asarray(legacy_debug["boundary_mask"], dtype=bool)

    assert legacy.shape == trace.shape
    assert np.asarray(none).shape == trace.shape
    assert np.all(np.isfinite(legacy))
    assert np.all(np.isfinite(none))
    assert np.allclose(none, trace)
    assert legacy_debug["plateau_tv_status"] == "applied"
    assert np.asarray(legacy_debug["plateau_tv_output"]).shape == trace.shape
    assert np.allclose(legacy[boundary], trace[boundary])


def test_old_hybrid_wrapper_uses_state_guided_default() -> None:
    trace, plateau, fs = _synthetic_trace()

    debug = denoise_trace_hybrid_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )

    assert debug["wrapper_used"] == "denoise_trace_hybrid_low_plateau_tv"
    assert debug["denoiser_used"] == "state_guided_low_plateau_tv"
    assert debug["low_state_denoiser_used"] == "gonzalez_adaptive_wavelet"
    assert debug["plateau_tv_status"] == "applied"
    assert np.asarray(debug["output"]).shape == trace.shape


def test_gonzalez_full_trace_mode_ignores_plateau_mask() -> None:
    trace, plateau, fs = _synthetic_trace()

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        low_state_denoiser="gonzalez_full_trace",
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )
    direct = denoise_gonzalez_adaptive_wavelet(
        trace,
        fs=fs,
        input_mode="corrected",
        return_debug=False,
        n_freqs=12,
        max_clusters=4,
    )

    assert debug["denoiser_used"] == "gonzalez_full_trace"
    assert debug["low_state_denoiser_used"] == "gonzalez_full_trace"
    assert debug["plateau_tv_status"] == "not_applied_full_trace_mode"
    assert not np.any(np.asarray(debug["plateau_mask"], dtype=bool))
    assert not np.any(np.asarray(debug["protected_plateau_mask"], dtype=bool))
    assert np.allclose(np.asarray(debug["gonzalez_debug"]["output"], dtype=float), np.asarray(direct, dtype=float))
    assert debug["low_state_safety"]["amplitude_preservation_merge"]["status"] in {"evaluated", "no_events_evaluated"}
    assert np.asarray(debug["output"], dtype=float).shape == trace.shape


def test_state_guided_default_preserves_plateaus_boundaries_and_spikes() -> None:
    fs = 1000.0
    n = 1000
    rng = np.random.default_rng(7)
    idx = np.arange(n, dtype=float)
    t = idx / fs
    trace = 0.02 * np.sin(2.0 * np.pi * 3.0 * t) + (0.04 * rng.standard_normal(n))
    event = np.zeros(n, dtype=bool)
    for peak in [180, 420, 760]:
        pulse = 0.8 * np.exp(-0.5 * ((idx - peak) / 2.0) ** 2)
        trace += pulse
        event |= pulse > 0.05
    plateau = np.zeros(n, dtype=bool)
    plateau[520:650] = True
    trace[plateau] += 1.1 + (0.04 * rng.standard_normal(np.count_nonzero(plateau)))

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        boundary_protect_ms=20.0,
        return_debug=True,
        n_freqs=24,
        max_clusters=6,
    )

    output = np.asarray(debug["output"], dtype=float)
    boundary = np.asarray(debug["boundary_mask"], dtype=bool)
    metrics = gonzalez_adaptive_wavelet_validation_metrics(
        trace,
        output,
        fs,
        event_mask=event,
        plateau_mask=plateau,
    )

    assert output.shape == trace.shape
    assert np.all(np.isfinite(output))
    assert debug["low_state_denoiser_used"] == "gonzalez_adaptive_wavelet"
    assert debug["plateau_tv_status"] == "applied"
    assert np.allclose(output[boundary], trace[boundary])
    assert abs(metrics["plateau_median_bias"]) < 0.05
    assert metrics["jump_amplitude_preservation"] > 0.8
    assert metrics["event_peak_preservation"] > 0.8


def test_boundary_restoration_includes_inside_and_outside_plateau_edges() -> None:
    fs = 1000.0
    trace = np.zeros(80, dtype=float)
    plateau = np.zeros(80, dtype=bool)
    plateau[30:50] = True
    trace[plateau] = 1.0
    trace[29] = 0.7
    trace[30] = 1.8
    trace[49] = 1.6
    trace[50] = 0.6

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        low_state_denoiser="legacy_hybrid",
        low_state_denoise_fn=lambda values, fs, **_: np.zeros_like(values),
        boundary_protect_ms=2.0,
        return_debug=True,
    )

    output = np.asarray(debug["output"], dtype=float)
    boundary = np.asarray(debug["boundary_mask"], dtype=bool)
    core = np.asarray(debug["plateau_core_mask"], dtype=bool)

    assert boundary[28:32].all()
    assert boundary[48:52].all()
    assert not core[30]
    assert not core[49]
    assert core[33:47].all()
    assert np.allclose(output[boundary], trace[boundary])
    assert np.allclose(output[core], np.asarray(debug["plateau_tv_output"])[core])


def test_mostly_plateau_skips_low_state_denoising_instead_of_median_fill() -> None:
    fs = 1000.0
    trace = np.linspace(0.0, 1.0, 100)
    plateau = np.ones(100, dtype=bool)
    plateau[:5] = False

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        boundary_protect_ms=2.0,
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )

    assert debug["low_state_fallback_triggered"] is True
    assert "insufficient_reliable_low_state_samples" in debug["low_state_fallback_reasons"]
    assert np.allclose(debug["low_state_input"], trace)


def test_gonzalez_defaults_disable_optional_cleanup() -> None:
    trace, _plateau, fs = _synthetic_trace()

    debug = denoise_gonzalez_adaptive_wavelet(
        trace,
        fs=fs,
        input_mode="corrected",
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
        enable_low_state_safety_gate=False,
    )

    assert debug["enable_event_template_rejection"] is False
    assert debug["enable_noise_cluster_rejection"] is False
    assert all(not decision["rejected"] for decision in debug["rejected_frequency_clusters"])


def test_state_guided_gonzalez_does_not_reenable_noise_cluster_rejection_by_default() -> None:
    trace, plateau, fs = _synthetic_trace()

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )

    assert debug["gonzalez_debug"]["enable_noise_cluster_rejection"] is False


def test_collapsed_low_state_spikes_are_restored_without_safety_fallback() -> None:
    fs = 1000.0
    n = 220
    idx = np.arange(n, dtype=float)
    trace = 0.01 * np.sin(0.1 * idx)
    trace += 1.0 * np.exp(-0.5 * ((idx - 90) / 1.5) ** 2)
    plateau = np.zeros(n, dtype=bool)
    plateau[150:180] = True
    trace[plateau] += 0.6

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        low_state_denoiser="legacy_hybrid",
        low_state_denoise_fn=lambda values, fs, **_: np.zeros_like(values),
        boundary_protect_ms=2.0,
        return_debug=True,
    )

    restore = debug["low_state_safety"]["amplitude_preservation_merge"]

    assert debug["low_state_fallback_triggered"] is False
    assert debug["low_state_candidate_used"] is True
    assert restore["restored_event_count"] >= 1
    assert np.max(np.asarray(debug["low_state_output"], dtype=float)) > 0.75


def test_low_state_candidate_is_visible_by_default_without_safety_gate() -> None:
    fs = 1000.0
    n = 220
    idx = np.arange(n, dtype=float)
    trace = 0.01 * np.sin(0.1 * idx)
    trace += 1.0 * np.exp(-0.5 * ((idx - 90) / 1.5) ** 2)
    plateau = np.zeros(n, dtype=bool)
    plateau[150:180] = True
    trace[plateau] += 0.6

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        low_state_denoiser="legacy_hybrid",
        low_state_denoise_fn=lambda values, fs, **_: np.zeros_like(values),
        boundary_protect_ms=2.0,
        return_debug=True,
    )

    assert debug["low_state_safety_gate_enabled"] is False
    assert debug["low_state_candidate_used"] is True
    assert debug["low_state_fallback_triggered"] is False
    assert debug["low_state_safety"]["amplitude_preservation_merge"]["restored_event_count"] >= 1
    assert np.max(np.asarray(debug["low_state_output"], dtype=float)) > 0.75


def test_short_plateau_segment_is_not_over_smoothed_by_tv() -> None:
    fs = 1000.0
    trace = np.zeros(60, dtype=float)
    plateau = np.zeros(60, dtype=bool)
    plateau[20:23] = True
    trace[20:23] = [1.0, 1.4, 0.9]

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        low_state_denoiser="legacy_hybrid",
        low_state_denoise_fn=lambda values, fs, **_: values.copy(),
        boundary_protect_ms=0.0,
        plateau_tv_weight=50.0,
        plateau_tv_min_duration_ms=5.0,
        return_debug=True,
    )

    assert debug["plateau_tv_segments"][0]["skipped_reason"] == "short_plateau_segment"
    assert np.allclose(np.asarray(debug["output"])[plateau], trace[plateau])


def test_debug_contains_new_masks_and_safety_metrics() -> None:
    trace, plateau, fs = _synthetic_trace()

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )

    for key in [
        "protected_plateau_mask",
        "plateau_core_mask",
        "boundary_mask",
        "plateau_replace_mask",
    ]:
        assert np.asarray(debug[key]).shape == trace.shape
    assert "morphology_preservation" in debug["low_state_safety"]
    assert "plateau_tv_segments" in debug


def test_near_boundary_spike_is_preserved_in_boundary_band() -> None:
    fs = 1000.0
    trace = np.zeros(120, dtype=float)
    plateau = np.zeros(120, dtype=bool)
    plateau[50:85] = True
    trace[plateau] = 1.0
    trace[48] = 1.2
    trace[50] = 1.8

    debug = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        low_state_denoiser="legacy_hybrid",
        low_state_denoise_fn=lambda values, fs, **_: np.zeros_like(values),
        boundary_protect_ms=3.0,
        return_debug=True,
    )

    output = np.asarray(debug["output"], dtype=float)
    boundary = np.asarray(debug["boundary_mask"], dtype=bool)
    assert boundary[48]
    assert boundary[50]
    assert output[48] == trace[48]
    assert output[50] == trace[50]


def test_saved_config_migrates_risky_gonzalez_cleanup_off() -> None:
    params = DetectionParams.from_dict(
        {
            "gonzalez_enable_event_template_rejection": True,
            "gonzalez_enable_noise_cluster_rejection": True,
        }
    )

    assert params.gonzalez_enable_event_template_rejection is False
    assert params.gonzalez_enable_noise_cluster_rejection is False


def test_gonzalez_dff_alias_is_corrected_mode() -> None:
    trace, _plateau, fs = _synthetic_trace()

    debug = denoise_gonzalez_adaptive_wavelet(
        trace,
        fs=fs,
        input_mode="dff",
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
    )

    assert debug["input_mode"] == "corrected"
    assert debug["input_mode_alias"] == "dff"
    assert debug["F0"] is None
    assert np.asarray(debug["output"]).shape == trace.shape
    assert np.all(np.isfinite(debug["output"]))


def test_gonzalez_reconstructs_from_full_cleaned_coefficient_matrix() -> None:
    trace, _plateau, fs = _synthetic_trace()

    debug = denoise_gonzalez_adaptive_wavelet(
        trace,
        fs=fs,
        input_mode="corrected",
        return_debug=True,
        n_freqs=12,
        max_clusters=4,
        enable_low_state_safety_gate=False,
    )

    output = np.asarray(debug["output"], dtype=float)
    diagnostics = debug["amplitude_preservation_diagnostics"]

    assert output.shape == trace.shape
    assert np.all(np.isfinite(output))
    assert debug["safety_metrics"]["fallback_triggered"] is False
    assert diagnostics["max_abs_reconstruction_consistency_error"] < 1e-10
    assert "full_matrix_reconstruction" in diagnostics
    assert "integrated_reconstruction" not in diagnostics
    assert "cleaned_complex_cwt_coefficients" in debug


def test_ui_defaults_to_gonzalez_without_hybrid_default_label() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = SpikeCurationMainWindow()
    try:
        assert window.param_low_state_denoiser.currentData() == "gonzalez_adaptive_wavelet"
        assert window.param_low_state_denoiser.currentText() == "Gonzalez adaptive wavelet"
        assert not window.param_gonzalez_event_template_rejection.isChecked()
        assert window.param_plateau_tv_weight.isEnabled()
        assert window.param_low_state_boundary_protect_ms.isEnabled()
        window.param_low_state_denoiser.setCurrentIndex(window.param_low_state_denoiser.findData("gonzalez_full_trace"))
        assert window.param_low_state_denoiser.currentText() == "Gonzalez full trace (no plateau mask)"
        assert not window.param_plateau_tv_weight.isEnabled()
        assert not window.param_low_state_boundary_protect_ms.isEnabled()
        window.param_low_state_denoiser.setCurrentIndex(window.param_low_state_denoiser.findData("legacy_hybrid"))
        assert window.param_plateau_tv_weight.isEnabled()
        assert window.param_low_state_boundary_protect_ms.isEnabled()
        window.param_low_state_denoiser.setCurrentIndex(window.param_low_state_denoiser.findData("none"))
        assert not window.param_plateau_tv_weight.isEnabled()
        assert not window.param_low_state_boundary_protect_ms.isEnabled()
        assert "Hybrid" not in window.chk_show_hybrid_denoised.text()
        assert "hybrid" not in window.windowTitle().lower()
    finally:
        window.close()
        app.processEvents()
