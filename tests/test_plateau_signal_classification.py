from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.core.plateau_signal_classification import classify_plateau_signals  # noqa: E402


def _mask(n: int, start: int = 80, end: int = 180) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True
    return mask


def _base_trace(n: int = 260, start: int = 80, end: int = 180) -> tuple[np.ndarray, np.ndarray]:
    trace = np.zeros(n, dtype=float)
    trace[start:end] = 6.0
    return trace, _mask(n, start, end)


def test_clean_plateau_is_labeled_clean() -> None:
    corrected, plateau = _base_trace()
    denoised = corrected.copy()
    denoised[80:180] += 0.02 * np.sin(np.linspace(0.0, 2.0 * np.pi, 100))

    rows = classify_plateau_signals(
        corrected,
        denoised,
        plateau,
        sampling_rate_hz=1000.0,
        long_plateau_mask=plateau,
        local_noise=np.full(corrected.shape, 0.05),
    )

    assert len(rows) == 1
    assert rows[0]["detector_source"] == "long"
    assert rows[0]["signal_type"] == "clean"


def test_preserved_plateau_peaks_are_spike_like() -> None:
    corrected, plateau = _base_trace()
    for idx in [95, 120, 145, 170]:
        corrected[idx] += 8.0
    denoised = corrected.copy()

    rows = classify_plateau_signals(
        corrected,
        denoised,
        plateau,
        sampling_rate_hz=1000.0,
        short_plateau_mask=plateau,
        local_noise=np.full(corrected.shape, 0.05),
    )

    assert rows[0]["detector_source"] == "short"
    assert rows[0]["signal_type"] == "spike_like"
    assert rows[0]["preserved_peak_fraction"] >= 0.75


def test_suppressed_plateau_noise_is_noise_like() -> None:
    corrected, plateau = _base_trace()
    rng = np.random.default_rng(4)
    corrected[80:180] += 1.1 * rng.standard_normal(100)
    denoised = np.zeros_like(corrected)
    denoised[80:180] = 6.0

    rows = classify_plateau_signals(
        corrected,
        denoised,
        plateau,
        sampling_rate_hz=1000.0,
        burst_plateau_mask=plateau,
        local_noise=np.full(corrected.shape, 0.05),
    )

    assert rows[0]["detector_source"] == "burst"
    assert rows[0]["signal_type"] == "noise_like"
    assert rows[0]["noise_reduction_fraction"] > 0.5


def test_partly_preserved_peaks_and_removed_noise_are_mixed() -> None:
    corrected, plateau = _base_trace()
    denoised = np.zeros_like(corrected)
    denoised[80:180] = 6.0
    rng = np.random.default_rng(8)
    corrected[80:180] += 0.7 * rng.standard_normal(100)
    for idx in [96, 124, 152, 172]:
        corrected[idx] += 8.0
        denoised[idx] += 4.0

    rows = classify_plateau_signals(
        corrected,
        denoised,
        plateau,
        sampling_rate_hz=1000.0,
        burst_plateau_mask=plateau,
        local_noise=np.full(corrected.shape, 0.05),
    )

    assert rows[0]["signal_type"] == "mixed"
    assert rows[0]["denoised_peak_density_hz"] > 0.0
    assert rows[0]["preserved_peak_fraction"] > 0.0
    assert rows[0]["residual_fraction"] > 0.25


def test_missing_or_disabled_denoising_is_unknown() -> None:
    corrected, plateau = _base_trace()

    rows = classify_plateau_signals(
        corrected,
        None,
        plateau,
        sampling_rate_hz=1000.0,
        denoising_enabled=False,
    )

    assert rows[0]["signal_type"] == "unknown"
    assert np.isnan(float(rows[0]["signal_confidence"]))


def test_disabled_denoising_with_passthrough_trace_still_classifies_signal() -> None:
    corrected, plateau = _base_trace()
    for idx in [95, 120, 145, 170]:
        corrected[idx] += 8.0

    rows = classify_plateau_signals(
        corrected,
        corrected.copy(),
        plateau,
        sampling_rate_hz=1000.0,
        burst_plateau_mask=plateau,
        local_noise=np.full(corrected.shape, 0.05),
        denoising_enabled=False,
    )

    assert rows[0]["detector_source"] == "burst"
    assert rows[0]["signal_type"] != "unknown"
