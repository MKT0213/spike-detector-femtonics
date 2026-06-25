from __future__ import annotations

import numpy as np

import detector.core.detection as detection
from detector.core.models import DetectionParams


def test_short_masks_are_padded_not_tiled_for_interpolation() -> None:
    values = np.array([0.0, 10.0, 100.0, 30.0, 400.0, 50.0])

    corrected = detection._interpolate_masked_series(values, np.array([True, False]))

    np.testing.assert_allclose(corrected, np.array([10.0, 10.0, 100.0, 30.0, 400.0, 50.0]))


def test_short_plateau_detector_keeps_stable_raised_floor_and_debug_arrays() -> None:
    fs = 1000.0
    n = 400
    rng = np.random.default_rng(1)
    trace = 0.05 * rng.standard_normal(n)
    trace[120:170] += 8.0

    mask, events = detection._detect_short_plateaus_baseline_corrected(trace, fs=fs)

    assert events.shape == (1, 6)
    assert events[0, 2] >= 25.0
    assert events[0, 4] >= 0.75
    assert events[0, 5] >= 0.65
    assert np.count_nonzero(mask) >= 25
    for key in (
        "short_floor_envelope",
        "short_step_up_score",
        "short_step_down_score",
        "short_support_mask",
        "short_refined_onset_score",
        "short_refined_offset_score",
    ):
        assert detection._LAST_SHORT_PLATEAU_DEBUG[key].shape == (n,)


def test_short_plateau_detector_rejects_noisy_and_spike_dominated_candidates() -> None:
    fs = 1000.0
    n = 400
    rng = np.random.default_rng(2)
    noisy = 0.05 * rng.standard_normal(n)
    noisy[90:210] += 1.4 * rng.standard_normal(120)
    noisy[120:170] += 4.5

    noisy_mask, noisy_events = detection._detect_short_plateaus_baseline_corrected(noisy, fs=fs)

    assert noisy_events.shape == (0, 6)
    assert not np.any(noisy_mask)

    rng = np.random.default_rng(2)
    spike_dominated = 0.05 * rng.standard_normal(n)
    spike_dominated[120:170] += 4.5
    gap_idx = np.linspace(123, 166, 6, dtype=int)
    spike_dominated[gap_idx] -= 4.5
    spike_dominated[140] += 25.0

    spike_mask, spike_events = detection._detect_short_plateaus_baseline_corrected(spike_dominated, fs=fs)

    assert spike_events.shape == (0, 6)
    assert not np.any(spike_mask)


def test_short_plateau_detector_parameters_can_relax_duration_gate() -> None:
    fs = 1000.0
    n = 300
    rng = np.random.default_rng(3)
    trace = 0.04 * rng.standard_normal(n)
    trace[120:142] += 8.0

    default_mask, default_events = detection._detect_short_plateaus_baseline_corrected(trace, fs=fs)

    tuned_mask, tuned_events = detection._detect_short_plateaus_baseline_corrected(
        trace,
        fs=fs,
        step_noise_k=2.0,
        elevated_noise_k=2.0,
        median_noise_k=2.5,
        pre_quiet_noise_k=2.5,
        min_support_fraction=0.55,
        min_confidence=0.25,
        min_duration_ms=12.0,
        min_support_ms=8.0,
        early_dwell_ms=18.0,
    )

    assert default_events.shape == (0, 6)
    assert not np.any(default_mask)
    assert tuned_events.shape == (1, 6)
    assert tuned_events[0, 2] >= 12.0
    assert np.count_nonzero(tuned_mask) >= 12


def test_short_plateau_detector_recenters_percentile_baseline_offset() -> None:
    fs = 1000.0
    n = 400
    rng = np.random.default_rng(4)
    trace = 0.10 + 0.04 * rng.standard_normal(n)
    trace[120:170] += 8.0

    mask, events = detection._detect_short_plateaus_baseline_corrected(trace, fs=fs)

    assert events.shape == (1, 6)
    assert events[0, 0] <= 125
    assert events[0, 1] >= 160
    assert np.count_nonzero(mask) >= 25
    assert detection._LAST_SHORT_PLATEAU_DEBUG["short_centering_baseline"].shape == (n,)
    assert abs(float(np.median(detection._LAST_SHORT_PLATEAU_DEBUG["short_centered_trace"][:100]))) < 0.15


def test_direct_detect_spikes_runs_short_plateau_detector_without_override() -> None:
    fs = 1000.0
    n = 400
    time = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(5)
    trace = 0.04 * rng.standard_normal(n)
    trace[120:170] += 8.0
    params = DetectionParams(
        denoising_method="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=5.0,
        spike_prominence_k=3.0,
        min_separation_ms=10.0,
    )

    result = detection.detect_spikes("trace", time, trace, params, sampling_rate_hz=fs)
    short_mask = np.asarray(result.qc_plot_data["short_plateau_mask"], dtype=bool)
    events = np.asarray(result.qc_plot_data["short_plateau_event_table"], dtype=float)

    assert events.shape == (1, 6)
    assert np.count_nonzero(short_mask) >= 25
    assert not np.any(np.asarray(result.qc_plot_data["long_plateau_mask"], dtype=bool))


def test_short_plateau_template_recovery_finds_weaker_matching_event() -> None:
    fs = 1000.0
    n = 900
    rng = np.random.default_rng(21)
    trace = 0.04 * rng.standard_normal(n)
    trace[120:170] += 8.0
    trace[360:410] += 8.2
    trace[640:690] += 5.5

    mask, events = detection._detect_short_plateaus_baseline_corrected(trace, fs=fs)

    assert events.shape[0] == 3
    assert np.count_nonzero(mask[640:690]) >= 40
    recovery = np.asarray(detection._LAST_SHORT_PLATEAU_DEBUG["short_template_recovery_mask"], dtype=bool)
    template_score = np.asarray(detection._LAST_SHORT_PLATEAU_DEBUG["short_template_score"], dtype=float)
    assert np.count_nonzero(recovery) > 0
    assert float(np.max(template_score)) > 0.8


def test_burst_associated_plateau_is_separate_from_clean_short_plateau() -> None:
    fs = 1000.0
    n = 500
    rng = np.random.default_rng(31)
    trace = 0.04 * rng.standard_normal(n)
    trace[120:190] += 8.0
    trace[np.arange(124, 188, 4)] += 24.0

    short_mask, short_events, burst_mask, burst_events = detection._detect_short_plateaus_baseline_corrected(
        trace,
        fs=fs,
        return_burst=True,
    )

    assert short_events.shape == (0, 6)
    assert not np.any(short_mask)
    assert burst_events.shape == (1, 8)
    assert burst_events[0, 4] >= 0.70
    assert burst_events[0, 5] >= 5.0
    assert np.count_nonzero(burst_mask[120:190]) >= 60


def test_spike_only_burst_without_raised_floor_is_not_burst_plateau() -> None:
    fs = 1000.0
    n = 500
    rng = np.random.default_rng(32)
    trace = 0.04 * rng.standard_normal(n)
    trace[[130, 145, 160, 175]] += [24.0, 30.0, 27.0, 22.0]

    short_mask, short_events, burst_mask, burst_events = detection._detect_short_plateaus_baseline_corrected(
        trace,
        fs=fs,
        return_burst=True,
    )

    assert short_events.shape == (0, 6)
    assert burst_events.shape == (0, 8)
    assert not np.any(short_mask)
    assert not np.any(burst_mask)


def test_direct_detect_spikes_reports_burst_plateau_source_without_long_overlap() -> None:
    fs = 1000.0
    n = 700
    time = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(33)
    trace = 200.0 + 0.04 * rng.standard_normal(n)
    trace[120:190] += 8.0
    trace[np.arange(124, 188, 4)] += 24.0
    trace[108:120] += 3.0
    trace[190:202] += 3.0
    params = DetectionParams(
        denoising_method="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=5.0,
        spike_prominence_k=3.0,
        min_separation_ms=10.0,
    )

    result = detection.detect_spikes("burst_trace", time, trace, params, sampling_rate_hz=fs)

    assert np.asarray(result.qc_plot_data["short_plateau_event_table"], dtype=float).shape == (0, 6)
    burst_events = np.asarray(result.qc_plot_data["burst_plateau_event_table"], dtype=float)
    assert burst_events.shape == (1, 8)
    assert np.count_nonzero(np.asarray(result.qc_plot_data["burst_plateau_mask"], dtype=bool)[120:190]) >= 60
    assert not np.any(np.asarray(result.qc_plot_data["long_plateau_mask"], dtype=bool)[120:190])


def test_long_plateau_still_controls_baseline_exclusion_not_burst_source() -> None:
    fs = 1000.0
    n = 700
    time = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(34)
    trace = 200.0 + 0.04 * rng.standard_normal(n)
    trace[120:360] += 20.0
    trace[np.arange(180, 280, 8)] += 18.0
    params = DetectionParams(
        denoising_method="none",
        use_auto_threshold_from_no_event=False,
        detection_mode="mad",
        spike_noise_k=5.0,
        spike_prominence_k=3.0,
        min_separation_ms=10.0,
    )

    result = detection.detect_spikes("long_trace", time, trace, params, sampling_rate_hz=fs)

    long_mask = np.asarray(result.qc_plot_data["long_plateau_mask"], dtype=bool)
    burst_mask = np.asarray(result.qc_plot_data["burst_plateau_mask"], dtype=bool)
    assert np.count_nonzero(long_mask[120:360]) >= 200
    assert not np.any(burst_mask[120:360])
