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
