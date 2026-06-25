from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.core.detection import _template_matching_log_likelihood_ratio_from_model, detect_spikes  # noqa: E402
from detector.core.models import DetectionParams  # noqa: E402


def _template_matching_params(template_indices: list[int] | None = None) -> DetectionParams:
    return DetectionParams(
        detection_mode="template_matching",
        denoising_method="none",
        startup_exclusion_ms=20.0,
        min_separation_ms=20.0,
        template_matching_template_indices=list(template_indices or []),
        template_matching_pre_ms=2.0,
        template_matching_post_ms=8.0,
        template_matching_false_positive_rate=0.05,
    )


def test_template_matching_detects_template_like_spikes() -> None:
    fs = 1000.0
    n = 4000
    time = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(42)
    trace = 0.08 * rng.standard_normal(n)
    spike_shape = np.array([-0.05, 0.0, 0.35, 1.0, 0.45, 0.12, -0.03, 0.0])
    spike_locs = np.array([250, 520, 840, 1500, 2300, 3100])
    for amplitude, loc in zip([2.4, 2.1, 2.2, 1.8, 2.5, 2.0], spike_locs):
        trace[loc : loc + spike_shape.size] += amplitude * spike_shape

    template_peak_indices = list((spike_locs[:3] + 3).astype(int))
    result = detect_spikes("trace", time, trace, _template_matching_params(template_peak_indices), sampling_rate_hz=fs)

    detected = np.array([spike.spike_index for spike in result.spikes], dtype=int)
    assert detected.size == spike_locs.size
    np.testing.assert_allclose(detected, spike_locs + 3, atol=1)
    assert np.all(np.isnan(result.threshold))
    assert "template_matching_likelihood_ratio" in result.qc_plot_data
    assert np.isfinite(result.qc_plot_data["template_matching_threshold"][0])
    np.testing.assert_array_equal(result.qc_plot_data["template_matching_requested_template_indices"], template_peak_indices)
    np.testing.assert_array_equal(result.qc_plot_data["template_matching_template_indices"], template_peak_indices)
    np.testing.assert_allclose(result.qc_plot_data["template_matching_autopick_used"], [0.0])
    assert all("template-matching" in spike.notes for spike in result.spikes)


def test_template_matching_requires_user_chosen_templates_when_autopick_is_off() -> None:
    fs = 1000.0
    rng = np.random.default_rng(43)
    trace = 0.08 * rng.standard_normal(3000)
    time = np.arange(trace.size, dtype=float) / fs

    result = detect_spikes("noise", time, trace, _template_matching_params(), sampling_rate_hz=fs)

    assert result.spikes == []
    assert result.qc_plot_data["template_matching_requested_template_indices"].size == 0
    assert result.qc_plot_data["template_matching_template_indices"].size == 0
    np.testing.assert_allclose(result.qc_plot_data["template_matching_autopick_used"], [0.0])


def test_template_matching_can_autopick_templates_with_conservative_std_candidates() -> None:
    fs = 1000.0
    n = 4000
    time = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(44)
    trace = 0.05 * rng.standard_normal(n)
    spike_shape = np.array([-0.05, 0.0, 0.35, 1.0, 0.45, 0.12, -0.03, 0.0])
    spike_locs = np.array([250, 520, 840, 1500, 2300, 3100])
    for amplitude, loc in zip([4.0, 3.8, 4.1, 3.6, 4.2, 3.7], spike_locs):
        trace[loc : loc + spike_shape.size] += amplitude * spike_shape

    params = _template_matching_params()
    params.template_matching_enable_autopick = True
    result = detect_spikes("trace", time, trace, params, sampling_rate_hz=fs)

    np.testing.assert_allclose(result.qc_plot_data["template_matching_autopick_used"], [1.0])
    np.testing.assert_array_equal(result.qc_plot_data["template_matching_autopick_indices"], spike_locs + 3)
    np.testing.assert_array_equal(result.qc_plot_data["template_matching_template_indices"], spike_locs + 3)
    detected = np.array([spike.spike_index for spike in result.spikes], dtype=int)
    assert detected.size == spike_locs.size
    np.testing.assert_allclose(detected, spike_locs + 3, atol=4)


def test_template_matching_tail_likelihood_rewards_stronger_same_direction_events() -> None:
    trace = np.zeros(30, dtype=float)
    trace[5:8] = np.array([0.0, 1.0, 0.0])
    trace[15:18] = np.array([0.0, 2.0, 0.0])

    ratio = _template_matching_log_likelihood_ratio_from_model(
        trace,
        template_mean=np.array([0.0, 1.0, 0.0]),
        signal_std=np.array([0.1, 0.1, 0.1]),
        noise_std=0.1,
        blocked=np.zeros(trace.size, dtype=bool),
        peak_offset=1,
    )

    assert ratio[16] > ratio[6]
