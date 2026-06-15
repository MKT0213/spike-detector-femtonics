from __future__ import annotations

import numpy as np

import app.detection as detection


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
