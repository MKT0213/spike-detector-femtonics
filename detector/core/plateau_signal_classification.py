from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


SIGNAL_TYPE_COLUMNS = [
    "signal_type",
    "signal_confidence",
    "raw_highfreq_rms",
    "denoised_highfreq_rms",
    "residual_rms",
    "residual_fraction",
    "noise_reduction_fraction",
    "raw_peak_density_hz",
    "denoised_peak_density_hz",
    "preserved_peak_fraction",
]


def classify_plateau_signals(
    corrected: np.ndarray,
    denoised: Optional[np.ndarray],
    plateau_mask: Optional[np.ndarray],
    *,
    sampling_rate_hz: float,
    long_plateau_mask: Optional[np.ndarray] = None,
    short_plateau_mask: Optional[np.ndarray] = None,
    burst_plateau_mask: Optional[np.ndarray] = None,
    local_noise: Optional[np.ndarray] = None,
    denoising_enabled: bool = True,
) -> List[Dict[str, object]]:
    """Describe high-frequency signal content on accepted plateau regions.

    This is intentionally post-detection metadata. It never changes plateau
    acceptance or source masks.
    """
    corrected_arr = _as_1d_float(corrected)
    plateau = _as_mask(plateau_mask, corrected_arr.size)
    regions = _contiguous_true_regions(plateau)
    if not regions:
        return []

    fs = float(sampling_rate_hz) if np.isfinite(sampling_rate_hz) and float(sampling_rate_hz) > 0.0 else 1.0
    denoised_arr = _as_1d_float(denoised) if denoised is not None else None
    # A disabled denoiser can still pass through the corrected trace, which is
    # enough to classify preserved high-frequency content. Missing data remains
    # unknown.
    denoising_valid = (
        denoised_arr is not None
        and denoised_arr.shape == corrected_arr.shape
        and np.any(np.isfinite(denoised_arr))
    )
    local_noise_arr = _as_1d_float(local_noise) if local_noise is not None else None
    if local_noise_arr is not None and local_noise_arr.shape != corrected_arr.shape:
        local_noise_arr = None
    global_noise = _global_noise_estimate(corrected_arr, plateau)

    rows: List[Dict[str, object]] = []
    for start, end in regions:
        source = _source_for_region(start, end, long_plateau_mask, short_plateau_mask, burst_plateau_mask)
        base = {
            "start_index": int(start),
            "end_index_exclusive": int(end),
            "duration_ms": float((end - start) * 1000.0 / fs),
            "detector_source": source,
        }
        if not denoising_valid:
            row = dict(base)
            row.update(_unknown_metrics())
            rows.append(row)
            continue

        assert denoised_arr is not None
        row = dict(base)
        row.update(
            _classify_region(
                corrected_arr[start:end],
                denoised_arr[start:end],
                local_noise_arr[start:end] if local_noise_arr is not None else None,
                fs=fs,
                fallback_noise=global_noise,
            )
        )
        rows.append(row)
    return rows


def _as_1d_float(values: Optional[np.ndarray]) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def _as_mask(values: Optional[np.ndarray], size: int) -> np.ndarray:
    out = np.zeros(size, dtype=bool)
    if values is None:
        return out
    arr = np.asarray(values, dtype=bool).reshape(-1)
    n = min(size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _source_for_region(
    start: int,
    end: int,
    long_plateau_mask: Optional[np.ndarray],
    short_plateau_mask: Optional[np.ndarray],
    burst_plateau_mask: Optional[np.ndarray],
) -> str:
    for label, mask in (
        ("long", long_plateau_mask),
        ("short", short_plateau_mask),
        ("burst", burst_plateau_mask),
    ):
        if mask is None:
            continue
        arr = np.asarray(mask, dtype=bool).reshape(-1)
        if arr.size > start and np.any(arr[start : min(end, arr.size)]):
            return label
    return "unknown"


def _unknown_metrics() -> Dict[str, float | str]:
    metrics: Dict[str, float | str] = {
        "signal_type": "unknown",
        "signal_confidence": float("nan"),
    }
    for key in SIGNAL_TYPE_COLUMNS:
        if key not in metrics:
            metrics[key] = float("nan")
    return metrics


def _classify_region(
    raw_segment: np.ndarray,
    denoised_segment: np.ndarray,
    local_noise_segment: Optional[np.ndarray],
    *,
    fs: float,
    fallback_noise: float,
) -> Dict[str, float | str]:
    raw = _fill_finite(np.asarray(raw_segment, dtype=float))
    den = _fill_finite(np.asarray(denoised_segment, dtype=float))
    if raw.size < 3 or den.shape != raw.shape:
        return _unknown_metrics()

    raw_high = _high_frequency_component(raw, fs)
    den_high = _high_frequency_component(den, fs)
    residual = raw - den
    local_noise = _segment_noise(local_noise_segment, raw_high, fallback_noise)
    raw_hf_rms = _rms(raw_high)
    den_hf_rms = _rms(den_high)
    residual_rms = _rms(residual)
    residual_fraction = residual_rms / max(raw_hf_rms, local_noise, 1e-12)
    noise_reduction = (
        1.0 - (den_hf_rms / max(raw_hf_rms, 1e-12))
        if np.isfinite(raw_hf_rms) and raw_hf_rms > 1e-12
        else 0.0
    )
    raw_peaks = _positive_peaks(raw_high, local_noise, fs)
    den_peaks = _positive_peaks(den_high, local_noise, fs)
    match_samples = max(1, int(round(0.002 * fs)))
    preserved = _matched_fraction(raw_peaks, den_peaks, match_samples)
    duration_s = max(raw.size / max(fs, 1e-12), 1e-12)
    raw_density = float(raw_peaks.size / duration_s)
    den_density = float(den_peaks.size / duration_s)
    raw_hf_over_noise = raw_hf_rms / max(local_noise, 1e-12)
    den_hf_over_noise = den_hf_rms / max(local_noise, 1e-12)

    signal_type, confidence = _choose_signal_type(
        raw_hf_over_noise=raw_hf_over_noise,
        den_hf_over_noise=den_hf_over_noise,
        residual_fraction=residual_fraction,
        noise_reduction=noise_reduction,
        raw_peak_density=raw_density,
        denoised_peak_density=den_density,
        preserved_peak_fraction=preserved,
    )
    metrics = {
        "signal_type": signal_type,
        "signal_confidence": confidence,
        "raw_highfreq_rms": raw_hf_rms,
        "denoised_highfreq_rms": den_hf_rms,
        "residual_rms": residual_rms,
        "residual_fraction": residual_fraction,
        "noise_reduction_fraction": noise_reduction,
        "raw_peak_density_hz": raw_density,
        "denoised_peak_density_hz": den_density,
        "preserved_peak_fraction": preserved,
    }
    if not all(np.isfinite(float(metrics[key])) for key in metrics if key != "signal_type"):
        unknown = _unknown_metrics()
        unknown.update({key: metrics[key] for key in metrics if key != "signal_type" and np.isfinite(float(metrics[key]))})
        return unknown
    return metrics


def _fill_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr
    if not np.any(finite):
        return np.zeros_like(arr, dtype=float)
    idx = np.arange(arr.size, dtype=float)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def _high_frequency_component(values: np.ndarray, fs: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return arr - _finite_median(arr)
    target = max(3, int(round(0.100 * fs)))
    size = min(target, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if size < 3:
        return arr - _finite_median(arr)
    if size % 2 == 0:
        size -= 1
    trend = median_filter(arr, size=max(3, size), mode="nearest")
    return arr - trend


def _positive_peaks(values: np.ndarray, local_noise: float, fs: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return np.array([], dtype=int)
    sigma = _robust_sigma(arr)
    threshold = max(3.0 * float(local_noise), float(sigma), 1e-12)
    distance = max(1, int(round(0.002 * fs)))
    peaks, _props = find_peaks(arr, height=threshold, distance=distance)
    return np.asarray(peaks, dtype=int)


def _matched_fraction(raw_peaks: np.ndarray, denoised_peaks: np.ndarray, tolerance: int) -> float:
    raw = np.asarray(raw_peaks, dtype=int)
    den = np.asarray(denoised_peaks, dtype=int)
    if raw.size == 0:
        return 0.0
    if den.size == 0:
        return 0.0
    matched = 0
    used: set[int] = set()
    for peak in raw:
        candidates = np.flatnonzero(np.abs(den - int(peak)) <= int(tolerance))
        available = [int(idx) for idx in candidates if int(idx) not in used]
        if not available:
            continue
        used.add(available[0])
        matched += 1
    return float(matched / max(raw.size, 1))


def _choose_signal_type(
    *,
    raw_hf_over_noise: float,
    den_hf_over_noise: float,
    residual_fraction: float,
    noise_reduction: float,
    raw_peak_density: float,
    denoised_peak_density: float,
    preserved_peak_fraction: float,
) -> Tuple[str, float]:
    nr = max(0.0, float(noise_reduction)) if np.isfinite(noise_reduction) else 0.0
    clean_score = float(
        np.mean(
            [
                1.0 - _unit((raw_hf_over_noise - 1.5) / 2.5),
                1.0 - _unit(residual_fraction / 0.45),
                1.0 - _unit(raw_peak_density / 25.0),
            ]
        )
    )
    spike_score = float(
        np.mean(
            [
                _unit(raw_peak_density / 35.0),
                _unit(denoised_peak_density / 25.0),
                _unit(preserved_peak_fraction),
                1.0 - _unit(nr / 0.85),
            ]
        )
    )
    noise_score = float(
        np.mean(
            [
                _unit((raw_hf_over_noise - 2.0) / 6.0),
                _unit(nr / 0.70),
                1.0 - _unit(preserved_peak_fraction),
                1.0 - _unit(denoised_peak_density / 20.0),
            ]
        )
    )
    mixed_score = float(
        np.mean(
            [
                _unit(raw_hf_over_noise / 6.0),
                _unit(residual_fraction / 0.70),
                max(_unit(preserved_peak_fraction), _unit(denoised_peak_density / 25.0)),
                _unit(nr / 0.70),
            ]
        )
    )

    if raw_hf_over_noise < 2.0 and residual_fraction < 0.30 and raw_peak_density < 10.0:
        return "clean", _confidence(clean_score)
    if (
        mixed_score >= 0.45
        and residual_fraction >= 0.25
        and nr >= 0.20
        and (preserved_peak_fraction >= 0.25 or denoised_peak_density >= 10.0)
    ):
        return "mixed", _confidence(mixed_score)
    if preserved_peak_fraction >= 0.45 and raw_peak_density >= 10.0 and spike_score >= noise_score:
        return "spike_like", _confidence(spike_score)
    if noise_score >= 0.45 and nr >= 0.35 and preserved_peak_fraction <= 0.35:
        return "noise_like", _confidence(noise_score)
    if raw_hf_over_noise < 2.5 and residual_fraction < 0.35:
        return "clean", _confidence(clean_score)
    if preserved_peak_fraction >= 0.30 or denoised_peak_density >= 10.0:
        label = "mixed" if residual_fraction >= 0.25 and nr >= 0.15 else "spike_like"
        return label, _confidence(max(mixed_score if label == "mixed" else spike_score, 0.45))
    if nr >= 0.25:
        return "noise_like", _confidence(max(noise_score, 0.45))
    return "mixed", _confidence(max(mixed_score, 0.35))


def _unit(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _confidence(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")


def _robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - med)))
    return float(sigma) if np.isfinite(sigma) and sigma > 0.0 else 0.0


def _finite_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else 0.0


def _segment_noise(local_noise: Optional[np.ndarray], raw_high: np.ndarray, fallback_noise: float) -> float:
    if local_noise is not None:
        vals = np.asarray(local_noise, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size:
            return float(np.median(vals))
    for value in (fallback_noise, _robust_sigma(raw_high)):
        if np.isfinite(value) and float(value) > 0.0:
            return float(value)
    return 1e-9


def _global_noise_estimate(corrected: np.ndarray, plateau_mask: np.ndarray) -> float:
    arr = np.asarray(corrected, dtype=float)
    mask = np.asarray(plateau_mask, dtype=bool)
    values = arr[~mask] if arr.shape == mask.shape and np.any(~mask) else arr
    if values.size >= 3:
        diff_noise = _robust_sigma(np.diff(_fill_finite(values))) / np.sqrt(2.0)
        if np.isfinite(diff_noise) and diff_noise > 0.0:
            return float(diff_noise)
    return max(_robust_sigma(arr), 1e-9)
