"""Gonzalez-style denoising for baseline-corrected fluorescence traces.

The active denoising path is full-trace Gonzalez adaptive wavelet denoising.
It operates on the baseline-corrected trace by default, with an optional dF/F0
mode that uses the detector baseline only for normalization. The denoised trace
is a detection aid; raw corrected values remain preferred for quantitative
amplitude, width, area, plateau level, and summary measurements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pywt
from kneed import KneeLocator
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

_GONZALEZ_CWT_PRECISION = 12
_GONZALEZ_INVERSE_CWT_REGULARIZATION = 1e-3
_GONZALEZ_S7_WARD_MAX_CLUSTERS = 20
_GONZALEZ_S7_ATTENUATION_MIN = 0.5
_GONZALEZ_S7_ATTENUATION_MAX = 0.5
_GONZALEZ_S7_CLUSTER_SELECTION_RELATIVE_SCORE = 0.85
_GONZALEZ_S7_THRESHOLD_MASK_PADDING_PERIODS = 8.0
_GONZALEZ_S7_EVENT_ATTENUATION_MIN = 0.5
_GONZALEZ_S7_EVENT_ATTENUATION_MAX = 0.5
_GONZALEZ_S7_EVENT_DETECTION_Z = 12.0
_GONZALEZ_S7_EVENT_DETECTION_PROMINENCE_Z = 1.0
_GONZALEZ_S7_EVENT_DETECTION_MIN_DISTANCE_MS = 3.0
_GONZALEZ_S7_EVENT_PROTECTION_WINDOW_MS = 12.0
_GONZALEZ_S7_EVENT_PROTECTION_FLOOR = 0.98
_GONZALEZ_S7_COEFFICIENT_EVENT_PROTECTION_WINDOW_MS = 18.0
_GONZALEZ_GUARDED_DOWNWARD_REPAIR_WINDOW_MS = 25.0
_GONZALEZ_GUARDED_DOWNWARD_BASELINE_WINDOW_MS = 50.0
_GONZALEZ_GUARDED_DOWNWARD_MAX_WIDTH_MS = 12.0


def _validate_trace(values: np.ndarray, fs: float) -> np.ndarray:
    trace = np.asarray(values, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if trace.size == 0:
        return trace.copy()
    finite = np.isfinite(trace)
    if not np.any(finite):
        return np.zeros_like(trace)
    if not np.all(finite):
        idx = np.arange(trace.size)
        trace = trace.copy()
        trace[~finite] = np.interp(idx[~finite], idx[finite], trace[finite])
    return trace


def _event_windows(trace: np.ndarray, peaks: np.ndarray, half_width: int) -> np.ndarray:
    width = (2 * int(half_width)) + 1
    padded = np.pad(trace, (half_width, half_width), mode="reflect" if trace.size > 1 else "edge")
    return np.asarray([padded[int(peak) : int(peak) + width] for peak in peaks], dtype=float)


def _boolean_regions(mask: np.ndarray, merge_gap: int = 0) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0 or not np.any(values):
        return []
    padded = np.r_[False, values, False]
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    regions = [(int(start), int(stop)) for start, stop in zip(starts, stops)]
    gap = max(0, int(merge_gap))
    if gap <= 0 or len(regions) <= 1:
        return regions
    merged: list[tuple[int, int]] = []
    cur_start, cur_stop = regions[0]
    for start, stop in regions[1:]:
        if int(start) - int(cur_stop) <= gap:
            cur_stop = int(stop)
        else:
            merged.append((int(cur_start), int(cur_stop)))
            cur_start, cur_stop = int(start), int(stop)
    merged.append((int(cur_start), int(cur_stop)))
    return merged


def _peaks_from_threshold_regions(
    scores: np.ndarray,
    candidate_mask: np.ndarray,
    min_height: float,
    min_prominence: float,
    distance: int,
    merge_gap: int,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    regions = _boolean_regions(candidate_mask, merge_gap=merge_gap)
    peaks: list[int] = []
    for start, stop in regions:
        if stop <= start:
            continue
        segment = scores[start:stop]
        if segment.size == 0:
            continue
        local_peaks, _props = find_peaks(
            segment,
            height=max(float(min_height), 0.0),
            distance=max(1, int(distance)),
            prominence=max(float(min_prominence), 0.0),
        )
        if local_peaks.size == 0:
            local_peak = int(np.argmax(segment))
            if float(segment[local_peak]) >= float(min_height):
                peaks.append(int(start + local_peak))
        else:
            peaks.extend(int(start + peak) for peak in local_peaks)
    if not peaks:
        return np.asarray([], dtype=int)
    peaks_array = np.asarray(sorted(set(peaks)), dtype=int)
    if peaks_array.size <= 1:
        return peaks_array
    order = np.argsort(scores[peaks_array])[::-1]
    kept: list[int] = []
    for peak in peaks_array[order]:
        if all(abs(int(peak) - int(existing)) >= int(distance) for existing in kept):
            kept.append(int(peak))
    return np.asarray(sorted(kept), dtype=int)


def _dilate_boolean_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0 or not np.any(values):
        return values.copy()
    pad = max(0, int(radius))
    if pad <= 0:
        return values.copy()
    out = np.zeros(values.shape, dtype=bool)
    for start, stop in _boolean_regions(values):
        out[max(0, start - pad) : min(values.size, stop + pad)] = True
    return out


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    center = float(np.median(finite))
    mad_sigma = float(np.median(np.abs(finite - center)) / 0.6745)
    if np.isfinite(mad_sigma) and mad_sigma > 1e-12:
        return mad_sigma
    std = float(np.std(finite))
    return std if np.isfinite(std) else 0.0


def _odd_sample_window(seconds: float, fs: float, size: int | None = None) -> int:
    window = max(1, int(round(float(seconds) * float(fs))))
    if window % 2 == 0:
        window += 1
    if size is not None and size > 0:
        max_window = int(size) if int(size) % 2 == 1 else int(size) - 1
        window = max(1, min(window, max_window))
    return window


def _safe_debug_array(values: np.ndarray, max_bytes: int = 64 * 1024 * 1024) -> np.ndarray | dict:
    array = np.asarray(values)
    if array.nbytes <= int(max_bytes):
        return array.copy()
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.asarray([])
    summary: dict[str, Any] = {
        "shape": tuple(int(v) for v in array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "stored": "summary_only",
    }
    if finite.size:
        if np.iscomplexobj(finite):
            magnitude = np.abs(finite)
            summary.update(
                {
                    "finite_magnitude_min": float(np.min(magnitude)),
                    "finite_magnitude_max": float(np.max(magnitude)),
                    "finite_magnitude_mean": float(np.mean(magnitude)),
                }
            )
        else:
            summary.update(
                {
                    "finite_min": float(np.min(finite)),
                    "finite_max": float(np.max(finite)),
                    "finite_mean": float(np.mean(finite)),
                }
            )
    return summary


def _moving_average_reflect(values: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return data.copy()
    width = max(1, int(window))
    if width <= 1 or data.size == 1:
        return data.copy()
    if width % 2 == 0:
        width += 1
    width = min(width, data.size if data.size % 2 == 1 else max(1, data.size - 1))
    if width <= 1:
        return data.copy()
    radius = width // 2
    padded = np.pad(data, (radius, radius), mode="reflect" if data.size > 1 else "edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _gonzalez_frequency_vector(
    fs: float,
    min_freq: float,
    max_freq: float | None,
    n_freqs: int,
) -> np.ndarray:
    if int(n_freqs) < 2:
        raise ValueError("n_freqs must be at least 2")
    nyquist = 0.5 * float(fs)
    upper = 0.9 * nyquist if max_freq is None else min(float(max_freq), 0.95 * nyquist)
    lower = float(min_freq)
    if not np.isfinite(lower) or lower <= 0:
        raise ValueError("min_freq must be positive and finite")
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("max_freq must be positive and below Nyquist")
    if lower >= upper:
        lower = max(1e-6, upper / max(4.0, float(n_freqs)))
    return np.linspace(lower, upper, int(n_freqs), dtype=float)


def _gonzalez_dff_from_fluorescence(
    trace: np.ndarray,
    fs: float,
    rolling_baseline_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Paper-specified preprocessing: convert fluorescence to dF/F with a
    # rolling median baseline. The requested default window is 4 seconds.
    window = _odd_sample_window(rolling_baseline_seconds, fs, size=trace.size)
    f0 = median_filter(trace, size=window, mode="nearest")
    denominator = _safe_f0_denominator(f0)
    return (trace - f0) / denominator, f0


def _gonzalez_gaussian_low_frequency_branch(
    trace: np.ndarray,
    fs: float,
    cutoff_hz: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    values = np.asarray(trace, dtype=float)
    cutoff = max(float(cutoff_hz), 1e-9)
    # For a Gaussian low-pass, the -3 dB frequency is approximately
    # sqrt(log(2)) / (2*pi*sigma_seconds).
    sigma_seconds = float(np.sqrt(np.log(2.0)) / (2.0 * np.pi * cutoff))
    sigma_samples = max(float(sigma_seconds) * float(fs), 1e-12)
    if values.size <= 1:
        low = values.copy()
    else:
        low = gaussian_filter1d(values, sigma=sigma_samples, mode="nearest")
    return low, {
        "enabled": True,
        "cutoff_hz": float(cutoff),
        "sigma_seconds": sigma_seconds,
        "sigma_samples": sigma_samples,
        "note": "S7-style very-low-frequency branch using 1D Gaussian smoothing.",
    }


def _safe_f0_denominator(f0: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0, dtype=float)
    finite_scale = np.abs(f0[np.isfinite(f0)])
    reference = float(np.median(finite_scale)) if finite_scale.size else 1.0
    eps = max(1e-9, 1e-6 * reference)
    signs = np.where(f0 < 0.0, -1.0, 1.0)
    return np.where(np.abs(f0) < eps, signs * eps, f0)


def _dff_normalization_baseline_validation(f0: np.ndarray) -> dict:
    baseline = np.asarray(f0, dtype=float)
    finite = baseline[np.isfinite(baseline)]
    if finite.size == 0:
        return {
            "valid": False,
            "reason": "normalization_baseline_has_no_finite_values",
            "robust_dynamic_ratio": float("nan"),
        }
    abs_finite = np.abs(finite)
    reference = float(np.nanmedian(abs_finite))
    eps = max(1e-9, 1e-6 * reference)
    p01 = float(np.nanpercentile(abs_finite, 1.0))
    p99 = float(np.nanpercentile(abs_finite, 99.0))
    robust_ratio = float(p99 / max(p01, eps))
    valid = bool(np.isfinite(robust_ratio) and robust_ratio <= 50.0)
    reason = None if valid else "normalization_baseline_dynamic_range_too_large_for_dff"
    return {
        "valid": valid,
        "reason": reason,
        "robust_dynamic_ratio": robust_ratio,
        "abs_p01": p01,
        "abs_p99": p99,
        "abs_median": reference,
        "max_allowed_robust_dynamic_ratio": 50.0,
    }


def _gonzalez_pca_features(normalized_magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    # Paper-specified dimensionality reduction: each frequency band is one
    # sample, and its temporal coefficient pattern is the feature vector.
    data = np.nan_to_num(np.asarray(normalized_magnitude, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, n_features = data.shape
    n_components = max(1, min(n_samples, n_features))
    if n_samples <= 1 or n_features <= 1 or float(np.std(data)) <= 1e-12:
        return np.zeros((n_samples, 1), dtype=float), np.ones(1, dtype=float), 1
    pca = PCA(n_components=n_components, svd_solver="full")
    features = pca.fit_transform(data)
    explained = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0, posinf=0.0, neginf=0.0)
    if explained.size <= 1:
        return features, explained, 1
    # Paper-specified in spirit: choose PCA dimensionality from the elbow of
    # the explained-variance curve. KneeLocator is used here as an explicit,
    # reproducible elbow detector; the fallback keeps enough PCs to explain 90%.
    x = np.arange(1, explained.size + 1)
    elbow = KneeLocator(x, explained, curve="convex", direction="decreasing").elbow
    if elbow is None:
        cumulative = np.cumsum(explained)
        selected = int(np.searchsorted(cumulative, 0.90) + 1)
    else:
        selected = int(elbow)
    return features, explained, int(np.clip(selected, 1, explained.size))


def _gonzalez_ward_clusters(
    features: np.ndarray,
    selected_pc_count: int,
    max_clusters: int,
    selection_strategy: str = "max_silhouette",
    relative_score: float = 1.0,
) -> Tuple[np.ndarray, int, Dict[int, float]]:
    # Paper-specified clustering: Ward hierarchical clustering in PCA space.
    x = np.asarray(features[:, : max(1, int(selected_pc_count))], dtype=float)
    n_freqs = int(x.shape[0])
    if n_freqs <= 2 or float(np.std(x)) <= 1e-12:
        return np.zeros(n_freqs, dtype=int), 1, {}
    scores: Dict[int, float] = {}
    labels_by_k: Dict[int, np.ndarray] = {}
    max_k = min(int(max_clusters), n_freqs - 1)
    for k in range(2, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        try:
            score = float(silhouette_score(x, labels))
        except ValueError:
            score = float("-inf")
        scores[int(k)] = score
        labels_by_k[int(k)] = np.asarray(labels, dtype=int)
    finite_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite_scores:
        return np.zeros(n_freqs, dtype=int), 1, scores
    strategy = str(selection_strategy).strip().lower()
    if strategy in {"parsimonious", "parsimonious_silhouette", "broad_silhouette"}:
        best_score = max(finite_scores.values())
        rel = float(np.clip(relative_score, 0.0, 1.0))
        cutoff = best_score * rel if best_score >= 0.0 else best_score / max(rel, 1e-12)
        candidates = [k for k, score in finite_scores.items() if score >= cutoff]
        chosen = min(candidates) if candidates else max(finite_scores, key=finite_scores.get)
    else:
        chosen = max(finite_scores, key=finite_scores.get)
    return labels_by_k[chosen], int(chosen), scores


def _gonzalez_cluster_window(fs: float, max_frequency: float) -> int:
    # Under-specified in Gonzalez et al.: the paper states that the moving
    # average window depends on the maximum frequency in the cluster, but does
    # not give the formula. Use a conservative two-period window so fast bands
    # get short local averages and slow bands get longer local averages.
    period_samples = float(fs) / max(float(max_frequency), 1e-9)
    window = max(3, int(round(2.0 * period_samples)))
    if window % 2 == 0:
        window += 1
    return window


def _gonzalez_inverse_cwt(
    coeffs: np.ndarray,
    scales: np.ndarray,
    wavelet: str,
    *,
    filter_fft: np.ndarray | None = None,
    denominator_filter_fft: np.ndarray | None = None,
    precision: int = _GONZALEZ_CWT_PRECISION,
    regularization: float = _GONZALEZ_INVERSE_CWT_REGULARIZATION,
    method: str = "regularized_dual_frame",
) -> np.ndarray:
    if coeffs.size == 0:
        return np.asarray([], dtype=float)
    selected = np.asarray(coeffs, dtype=np.complex128)
    if selected.ndim == 1:
        selected = selected[None, :]
    if selected.ndim != 2:
        raise ValueError("coeffs must be a 2D array of CWT coefficients")
    if selected.shape[1] == 0:
        return np.asarray([], dtype=float)
    scale_values = np.maximum(np.asarray(scales, dtype=float), 1e-12)
    if selected.shape[0] != scale_values.size:
        raise ValueError("number of coefficient rows must match number of scales")
    if str(method).strip().lower() in {"weighted_sum", "legacy", "scale_weighted_sum"}:
        weighted = np.real(selected) / np.sqrt(scale_values)[:, None]
        weight_sum = float(np.sum(1.0 / np.sqrt(scale_values)))
        if weight_sum <= 1e-12:
            return np.zeros(selected.shape[1], dtype=float)
        return np.sum(weighted, axis=0) / weight_sum

    # PyWavelets exposes CWT but not an inverse CWT for this transform. The
    # reconstruction below inverts the same sampled analysis filters used by
    # pywt.cwt in a regularized least-squares sense. This is a numerical inverse
    # of the CWT operator, not an extra denoising rule.
    if filter_fft is None:
        filter_fft = _gonzalez_cwt_analysis_filter_fft(
            selected.shape[1],
            scale_values,
            wavelet,
            precision=precision,
        )
    analysis_fft = np.asarray(filter_fft, dtype=np.complex128)
    if analysis_fft.shape != selected.shape:
        raise ValueError("filter_fft shape must match coeffs shape")
    denominator_fft = analysis_fft if denominator_filter_fft is None else np.asarray(denominator_filter_fft, dtype=np.complex128)
    if denominator_fft.ndim != 2 or denominator_fft.shape[1] != selected.shape[1]:
        raise ValueError("denominator_filter_fft must be a 2D array with the same sample length as coeffs")

    coeff_fft = np.fft.fft(selected, axis=-1)
    denominator = np.sum(np.abs(denominator_fft) ** 2, axis=0)
    numerator = np.sum(np.conj(analysis_fft) * coeff_fft, axis=0)
    max_denominator = float(np.max(denominator)) if denominator.size else 0.0
    if max_denominator <= 1e-18:
        return np.zeros(selected.shape[1], dtype=float)
    lambda_value = max(0.0, float(regularization)) * max_denominator
    if lambda_value > 0.0:
        spectrum = numerator / (denominator + lambda_value)
    else:
        floor = max(1e-18, np.finfo(float).eps * max_denominator)
        spectrum = numerator / np.where(denominator > floor, denominator, np.inf)
    return np.real(np.fft.ifft(spectrum))


def _gonzalez_cwt_analysis_filter_fft(
    n_samples: int,
    scales: np.ndarray,
    wavelet: str,
    *,
    precision: int = _GONZALEZ_CWT_PRECISION,
) -> np.ndarray:
    n = int(n_samples)
    if n <= 0:
        return np.empty((0, 0), dtype=np.complex128)
    scale_values = np.maximum(np.asarray(scales, dtype=float), 1e-12)
    if scale_values.size == 0:
        return np.empty((0, n), dtype=np.complex128)

    if isinstance(wavelet, (pywt.ContinuousWavelet, pywt.Wavelet)):
        wavelet_obj = wavelet
    else:
        wavelet_obj = pywt.ContinuousWavelet(str(wavelet))
    int_psi, x = pywt.integrate_wavelet(wavelet_obj, precision=int(precision))
    if getattr(wavelet_obj, "complex_cwt", False):
        int_psi = np.conj(int_psi)
    int_psi = np.asarray(int_psi, dtype=np.complex128)
    x = np.asarray(x, dtype=float)
    if int_psi.size < 2 or x.size < 2:
        return np.zeros((scale_values.size, n), dtype=np.complex128)
    step = float(x[1] - x[0])
    if not np.isfinite(step) or abs(step) <= 1e-18:
        return np.zeros((scale_values.size, n), dtype=np.complex128)

    filters = np.zeros((scale_values.size, n), dtype=np.complex128)
    span = float(x[-1] - x[0])
    for row, scale in enumerate(scale_values):
        j = np.arange((float(scale) * span) + 1.0) / (float(scale) * step)
        j = j.astype(int)
        if j.size == 0:
            continue
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        if j.size < 2:
            continue
        int_psi_scale = int_psi[j][::-1]
        impulse_response = -np.sqrt(float(scale)) * np.diff(int_psi_scale)
        if impulse_response.size == 0:
            continue
        crop_start = int(np.floor((impulse_response.size - 1) / 2.0))
        circular_filter = np.zeros(n, dtype=np.complex128)
        positions = (np.arange(impulse_response.size, dtype=int) - crop_start) % n
        np.add.at(circular_filter, positions, impulse_response)
        filters[row] = np.fft.fft(circular_filter)
    return filters


def _gonzalez_inverse_cwt_info(
    regularization: float,
    precision: int,
) -> dict:
    return {
        "method": "regularized_dual_frame",
        "regularization": float(regularization),
        "wavelet_precision": int(precision),
        "note": (
            "Regularized least-squares inverse of the same sampled Morlet CWT "
            "analysis filters used by pywt.cwt."
        ),
    }


def _gonzalez_reconstruction_calibration(
    raw_reconstruction: np.ndarray,
    working_trace: np.ndarray,
) -> Tuple[float, float, dict]:
    raw = np.asarray(raw_reconstruction, dtype=float)
    target = np.asarray(working_trace, dtype=float)
    if raw.shape != target.shape or raw.size == 0:
        return 1.0, 0.0, {"calibration": "skipped_shape_mismatch"}
    raw_centered = raw - float(np.mean(raw))
    target_centered = target - float(np.mean(target))
    denominator = float(np.vdot(raw_centered, raw_centered).real)
    gain = float(np.vdot(raw_centered, target_centered).real / denominator) if denominator > 1e-12 else 1.0
    offset = float(np.mean(target) - (gain * np.mean(raw)))
    calibrated = offset + (gain * raw)
    residual = calibrated - target
    corr = float(np.corrcoef(calibrated, target)[0, 1]) if raw.size > 1 and np.std(calibrated) > 1e-12 else float("nan")
    return gain, offset, {
        "inverse_cwt_note": "Regularized dual-frame inverse CWT calibrated by least-squares gain/offset.",
        "gain": gain,
        "offset": offset,
        "raw_reconstruction_std": float(np.std(raw)),
        "target_working_trace_std": float(np.std(target)),
        "calibrated_reconstruction_std": float(np.std(calibrated)),
        "calibrated_rms_error": float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan"),
        "calibrated_correlation": corr,
    }


def _gonzalez_event_template_rejection(
    cluster_trace: np.ndarray,
    fs: float,
    cluster_id: int,
    max_frequency: float,
    attenuation_min: float,
    attenuation_max: float,
    enabled: bool,
    mode: str = "correlation",
    event_z_threshold: float = 3.0,
    event_prominence_z: float = 1.0,
    event_min_distance_ms: float = 3.0,
    event_detection_polarity: str = "both",
    event_candidate_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, dict]:
    # Under-specified in Gonzalez et al.: candidate event detection and the PC
    # cutoff are not given. This implementation detects both polarities from a
    # robust z-score, aligns event windows by polarity, uses PC1 as the event
    # template, and rejects low-template-correlation windows.
    trace = np.asarray(cluster_trace, dtype=float)
    candidate_mask = np.ones(trace.shape, dtype=bool)
    if event_candidate_mask is not None:
        candidate_mask = np.asarray(event_candidate_mask, dtype=bool)
        if candidate_mask.shape != trace.shape:
            candidate_mask = np.ones(trace.shape, dtype=bool)
    debug: dict[str, Any] = {
        "cluster_id": int(cluster_id),
        "enabled": bool(enabled),
        "event_indices": np.asarray([], dtype=int),
        "event_windows": np.empty((0, 0), dtype=float),
        "template_waveform": np.asarray([], dtype=float),
        "pc_coefficients": np.asarray([], dtype=float),
        "template_correlation": np.asarray([], dtype=float),
        "rejected_events": np.asarray([], dtype=int),
        "event_gain": np.ones(trace.shape, dtype=float),
        "cutoff": None,
        "polarity": "both",
        "mode": str(mode),
        "event_detection": {
            "z_threshold": float(event_z_threshold),
            "prominence_z": float(event_prominence_z),
            "min_distance_ms": float(event_min_distance_ms),
            "polarity": str(event_detection_polarity),
            "candidate_fraction": float(np.mean(candidate_mask)) if candidate_mask.size else 0.0,
        },
        "note": "Event detection and PC cutoff are under-specified by the paper.",
    }
    if not enabled or trace.size < 5:
        return trace.copy(), debug
    center = float(np.median(trace))
    sigma = _robust_sigma(trace - center)
    if sigma <= 1e-12:
        return trace.copy(), debug
    z = (trace - center) / sigma
    polarity_mode = str(event_detection_polarity).strip().lower()
    if polarity_mode not in {"positive", "negative", "both"}:
        polarity_mode = "both"
    if polarity_mode == "positive":
        event_scores = z
    elif polarity_mode == "negative":
        event_scores = -z
    else:
        event_scores = np.abs(z)
    distance = max(1, int(round(float(event_min_distance_ms) * float(fs) / 1000.0)))
    half_width = int(round(float(fs) / max(float(max_frequency), 1e-9)))
    half_width = int(np.clip(half_width, 2, max(2, int(round(0.05 * float(fs))))))
    eligible = np.asarray(candidate_mask, dtype=bool)
    if event_candidate_mask is not None:
        peaks = _peaks_from_threshold_regions(
            event_scores,
            eligible,
            min_height=max(float(event_z_threshold), 0.0),
            min_prominence=max(float(event_prominence_z), 0.0),
            distance=distance,
            merge_gap=max(distance, half_width),
        )
        detection_source = "adaptive_threshold_regions"
    else:
        peaks, _props = find_peaks(
            np.where(eligible, event_scores, 0.0),
            height=max(float(event_z_threshold), 0.0),
            distance=distance,
            prominence=max(float(event_prominence_z), 0.0),
        )
        peaks = np.asarray([int(peak) for peak in peaks if eligible[int(peak)]], dtype=int)
        detection_source = "robust_zscore_peaks"
    if peaks.size < 3:
        debug["event_indices"] = np.asarray(peaks, dtype=int)
        debug["event_detection"]["distance_samples"] = int(distance)
        debug["event_detection"]["half_width_samples"] = int(half_width)
        debug["event_detection"]["source"] = detection_source
        debug["event_detection"]["polarity"] = polarity_mode
        return trace.copy(), debug
    windows = _event_windows(trace - center, np.asarray(peaks, dtype=int), half_width)
    if polarity_mode == "positive":
        polarities = np.ones(peaks.shape, dtype=float)
    elif polarity_mode == "negative":
        polarities = -np.ones(peaks.shape, dtype=float)
    else:
        polarities = np.where(z[peaks] < 0.0, -1.0, 1.0)
    aligned = windows * polarities[:, None]
    # Preserve event amplitude in PC1 scores. Per-event variance normalization
    # makes weak noise wiggles look as important as real events, which defeats
    # the paper's PC1-coefficient cleanup of small/noisy events.
    normalized = aligned - aligned.mean(axis=1, keepdims=True)
    if normalized.shape[0] < 3 or normalized.shape[1] < 2:
        debug["event_indices"] = np.asarray(peaks, dtype=int)
        debug["event_windows"] = windows
        return trace.copy(), debug
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(normalized)
    template = np.asarray(pca.components_[0], dtype=float)
    if np.dot(template, normalized.mean(axis=0)) < 0.0:
        template = -template
    template /= np.linalg.norm(template) + 1e-12
    coefficients = normalized @ template
    correlations = (normalized @ template) / (np.linalg.norm(normalized, axis=1) + 1e-12)
    event_mode = str(mode).strip().lower()
    corr_cutoff = 0.25
    coeff_floor = np.percentile(coefficients, 5.0)
    if event_mode in {"pc1_negative", "s7_pc1_negative", "s7"}:
        rejected = np.flatnonzero(coefficients < 0.0)
        cutoff: dict[str, float] = {"pc1_coefficient": 0.0}
    else:
        rejected = np.flatnonzero((correlations < corr_cutoff) | ((coefficients < coeff_floor) & (correlations < 0.5)))
        cutoff = {"template_correlation": corr_cutoff, "pc_coefficient_5th_percentile": float(coeff_floor)}
    gain = np.ones(trace.shape, dtype=float)
    if rejected.size:
        if event_mode in {"pc1_negative", "s7_pc1_negative", "s7"}:
            rejected_coeffs = coefficients[rejected]
            denom = max(abs(float(np.min(rejected_coeffs))), 1e-12)
            severity_values = np.clip((-rejected_coeffs) / denom, 0.0, 1.0)
        else:
            min_corr = float(np.min(correlations[rejected]))
            denom = max(corr_cutoff - min_corr, 1e-12)
            severity_values = np.clip((corr_cutoff - correlations[rejected]) / denom, 0.0, 1.0)
        for event_row, severity in zip(rejected, severity_values):
            severity = float(severity)
            attenuation = float(attenuation_min) + severity * (float(attenuation_max) - float(attenuation_min))
            event_gain = 1.0 - float(np.clip(attenuation, 0.0, 1.0))
            peak = int(peaks[event_row])
            start = max(0, peak - half_width)
            stop = min(trace.size, peak + half_width + 1)
            gain[start:stop] = np.minimum(gain[start:stop], event_gain)
    debug.update(
        {
            "event_indices": np.asarray(peaks, dtype=int),
            "event_windows": windows,
            "template_waveform": template,
            "pc_coefficients": coefficients,
            "template_correlation": correlations,
            "rejected_events": np.asarray(rejected, dtype=int),
            "event_gain": gain,
            "cutoff": cutoff,
            "half_width_samples": int(half_width),
            "event_polarities": polarities,
            "pc1_input_normalization": "per_event_demean_preserve_amplitude",
            "pc1_coefficient_method": "uncentered_projection_onto_oriented_pc1_template",
            "event_detection": {
                "z_threshold": float(event_z_threshold),
                "prominence_z": float(event_prominence_z),
                "min_distance_ms": float(event_min_distance_ms),
                "polarity": polarity_mode,
                "distance_samples": int(distance),
                "half_width_samples": int(half_width),
                "candidate_fraction": float(np.mean(candidate_mask)) if candidate_mask.size else 0.0,
                "source": detection_source,
            },
        }
    )
    return trace * gain, debug


def _gonzalez_cluster_thresholds(
    magnitudes: np.ndarray,
    frequencies: np.ndarray,
    labels: np.ndarray,
    fs: float,
    threshold_sd: float,
) -> Dict[int, dict]:
    cluster_info: Dict[int, dict] = {}
    for cluster_id in np.unique(labels):
        indices = np.flatnonzero(labels == cluster_id)
        activity = np.mean(magnitudes[indices], axis=0)
        max_frequency = float(np.max(frequencies[indices]))
        window = _gonzalez_cluster_window(fs, max_frequency)
        moving_average = _moving_average_reflect(activity, window)
        residual = activity - moving_average
        residual_std = float(np.std(residual[np.isfinite(residual)])) if np.any(np.isfinite(residual)) else 0.0
        threshold = moving_average + (float(threshold_sd) * residual_std)
        peak_z = float(np.max(activity - moving_average) / max(residual_std, 1e-12)) if activity.size else 0.0
        event_fraction = float(np.mean(activity > threshold)) if activity.size else 0.0
        cluster_info[int(cluster_id)] = {
            "indices": indices,
            "activity": activity,
            "moving_average": moving_average,
            "residual": residual,
            "residual_std": residual_std,
            "threshold": threshold,
            "max_frequency": max_frequency,
            "mean_frequency": float(np.mean(frequencies[indices])),
            "moving_average_window_samples": int(window),
            "peak_z": peak_z,
            "event_fraction": event_fraction,
        }
    return cluster_info

def _gonzalez_rejected_frequency_clusters(
    cluster_info: Dict[int, dict],
    max_frequency: float,
    threshold_sd: float,
    enabled: bool,
) -> Tuple[set[int], List[dict]]:
    # Under-specified in Gonzalez et al.: clusters "primarily encoding noise"
    # were discarded, but no cutoff is provided. This conservative default only
    # allows at most one high-frequency, low-event cluster to be rejected.
    decisions: List[dict] = []
    candidates: List[Tuple[float, int]] = []
    for cluster_id, info in cluster_info.items():
        high_freq_ratio = float(info["mean_frequency"]) / max(float(max_frequency), 1e-12)
        peak_z = float(info["peak_z"])
        event_fraction = float(info["event_fraction"])
        score = high_freq_ratio * (1.0 - min(event_fraction, 1.0)) / (1.0 + max(peak_z, 0.0))
        candidate = bool(enabled and high_freq_ratio >= 0.75 and peak_z < float(threshold_sd) and event_fraction < 0.01)
        decisions.append(
            {
                "cluster_id": int(cluster_id),
                "score": float(score),
                "high_frequency_ratio": high_freq_ratio,
                "peak_z": peak_z,
                "event_fraction": event_fraction,
                "candidate": candidate,
                "rejected": False,
                "reason": "kept",
            }
        )
        if candidate:
            candidates.append((score, int(cluster_id)))
    rejected: set[int] = set()
    if candidates:
        _score, selected = max(candidates, key=lambda item: item[0])
        rejected.add(selected)
        for decision in decisions:
            if int(decision["cluster_id"]) == selected:
                decision["rejected"] = True
                decision["reason"] = "conservative_high_frequency_low_event_cluster"
            elif decision["candidate"]:
                decision["reason"] = "candidate_kept_by_single_cluster_default_cap"
    return rejected, decisions


def _gonzalez_event_protection_weights(
    trace: np.ndarray,
    fs: float,
    *,
    enabled: bool,
    window_ms: float = _GONZALEZ_S7_COEFFICIENT_EVENT_PROTECTION_WINDOW_MS,
    baseline_window_ms: float = 50.0,
    min_peak_distance_ms: float = 2.0,
) -> Tuple[np.ndarray, dict]:
    values = np.asarray(trace, dtype=float)
    weights = np.zeros(values.shape, dtype=float)
    debug: dict[str, Any] = {
        "enabled": bool(enabled),
        "status": "disabled" if not enabled else "not_evaluated",
        "window_ms": float(window_ms),
        "baseline_window_ms": float(baseline_window_ms),
        "protected_peak_count": 0,
        "protected_fraction": 0.0,
        "peak_indices": np.asarray([], dtype=int),
    }
    if not enabled:
        return weights, debug
    if values.size < 3:
        debug["status"] = "empty_or_too_short"
        return weights, debug
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        debug["status"] = "insufficient_finite_samples"
        return weights, debug

    hp_window = _odd_window_ms(baseline_window_ms, fs)
    hp_window = min(hp_window, values.size if values.size % 2 == 1 else max(1, values.size - 1))
    if hp_window < 3:
        high = values - float(np.nanmedian(values[finite]))
    else:
        high = values - median_filter(values, size=hp_window, mode="nearest")
    selected = high[finite]
    noise = max(_robust_sigma(selected), 1e-12)
    max_abs = float(np.max(np.abs(selected))) if selected.size else 0.0
    threshold = max(4.0 * noise, 0.20 * max_abs, 1e-12)
    distance = max(1, int(round(float(min_peak_distance_ms) * float(fs) / 1000.0)))
    masked_abs = np.where(finite, np.abs(high), 0.0)
    peaks, _props = find_peaks(masked_abs, height=threshold, distance=distance)
    peaks = np.asarray([int(peak) for peak in peaks if finite[int(peak)]], dtype=int)
    if peaks.size == 0:
        debug.update(
            {
                "status": "no_events_detected",
                "noise_sigma": float(noise),
                "peak_threshold": float(threshold),
                "baseline_window_samples": int(hp_window),
            }
        )
        return weights, debug

    half_window = max(1, int(round(float(window_ms) * float(fs) / 1000.0)))
    for peak in peaks:
        start = max(0, int(peak) - half_window)
        stop = min(values.size, int(peak) + half_window + 1)
        width = stop - start
        if width <= 1:
            weights[start:stop] = 1.0
            continue
        x = np.linspace(-1.0, 1.0, width)
        local_weights = 0.5 * (1.0 + np.cos(np.pi * np.clip(x, -1.0, 1.0)))
        weights[start:stop] = np.maximum(weights[start:stop], local_weights)

    debug.update(
        {
            "status": "evaluated",
            "protected_peak_count": int(peaks.size),
            "protected_fraction": float(np.mean(weights > 1e-6)),
            "peak_indices": peaks,
            "noise_sigma": float(noise),
            "peak_threshold": float(threshold),
            "baseline_window_samples": int(hp_window),
            "half_window_samples": int(half_window),
        }
    )
    return weights, debug


def denoise_gonzalez_adaptive_wavelet(
    trace,
    fs,
    input_mode="fluorescence",
    rolling_baseline_seconds=4.0,
    wavelet="cmor1.5-1.0",
    wavelet_precision=_GONZALEZ_CWT_PRECISION,
    min_freq=1.0,
    max_freq=None,
    n_freqs=100,
    max_clusters=20,
    cluster_selection_strategy="max_silhouette",
    cluster_selection_relative_score=1.0,
    threshold_sd=2.0,
    attenuation_min=0.5,
    attenuation_max=1.0,
    coefficient_threshold_domain="raw_magnitude",
    threshold_mask_padding_periods=0.0,
    enable_noise_cluster_rejection=False,
    enable_event_template_rejection=False,
    event_template_rejection_mode="correlation",
    event_attenuation_min=None,
    event_attenuation_max=None,
    event_rejection_application="coefficient",
    event_detection_z_threshold=3.0,
    event_detection_prominence_z=1.0,
    event_detection_min_distance_ms=3.0,
    event_detection_polarity="both",
    event_detection_use_threshold_crossings=False,
    enable_coefficient_event_protection=False,
    coefficient_event_protection_window_ms=_GONZALEZ_S7_COEFFICIENT_EVENT_PROTECTION_WINDOW_MS,
    enable_low_frequency_gaussian_branch=False,
    low_frequency_cutoff_hz=1.0,
    inverse_cwt_regularization=_GONZALEZ_INVERSE_CWT_REGULARIZATION,
    reconstruction_mode="residual_subtractive",
    removed_reconstruction_gain_mode="calibrated",
    removed_reconstruction_gain_scale=1.0,
    enable_denoise_safety_gate=False,
    min_event_peak_preservation=0.95,
    min_local_max_ratio=0.95,
    return_debug=False,
    **compat_kwargs,
):
    """Denoise with Gonzalez adaptive wavelet thresholding.

    ``input_mode="fluorescence"`` computes a rolling F0 and works in dF/F.
    ``input_mode="corrected"`` uses the provided baseline-corrected trace
    directly and returns corrected-trace units. The older ``"dff"`` spelling is
    accepted only as a compatibility alias for ``"corrected"``.
    PyWavelets does not provide a public inverse CWT here, so reconstruction
    uses a regularized dual-frame inverse of the same sampled CWT filters.
    Morphology safety gates should pass before denoised output is used as a
    detection aid.
    """
    if "enable_low_state_safety_gate" in compat_kwargs:
        enable_denoise_safety_gate = bool(compat_kwargs.pop("enable_low_state_safety_gate"))
    if compat_kwargs:
        unexpected = ", ".join(sorted(str(key) for key in compat_kwargs))
        raise TypeError(f"Unexpected Gonzalez denoising keyword argument(s): {unexpected}")

    values = _validate_trace(np.asarray(trace, dtype=float), fs)
    raw_mode = str(input_mode).strip().lower()
    mode_alias = raw_mode if raw_mode == "dff" else None
    mode = "corrected" if raw_mode == "dff" else raw_mode
    if mode not in {"fluorescence", "corrected"}:
        raise ValueError("input_mode must be 'fluorescence' or 'corrected'")
    if values.size == 0:
        metrics = {
            "input_min": float("nan"),
            "input_max": float("nan"),
            "input_std": float("nan"),
            "denoised_min": float("nan"),
            "denoised_max": float("nan"),
            "denoised_std": float("nan"),
            "output_input_std_ratio": float("nan"),
            "fallback_triggered": False,
            "warnings": [],
        }
        debug = {
            "denoiser_used": "gonzalez_adaptive_wavelet",
            "input_mode": mode,
            "input_mode_alias": mode_alias,
            "F0": None if mode == "corrected" else values.copy(),
            "working_trace": values.copy(),
            "corrected_signal": values.copy() if mode == "corrected" else None,
            "raw_dff": None if mode == "corrected" else values.copy(),
            "frequency_vector": np.asarray([], dtype=float),
            "wavelet_scales": np.asarray([], dtype=float),
            "original_complex_cwt_coefficients": np.empty((0, 0), dtype=complex),
            "normalized_coefficient_magnitudes": np.empty((0, 0), dtype=float),
            "pca_explained_variance": np.asarray([], dtype=float),
            "selected_pc_count": 0,
            "ward_cluster_labels": np.asarray([], dtype=int),
            "chosen_cluster_count": 0,
            "cluster_selection_strategy": str(cluster_selection_strategy),
            "cluster_selection_relative_score": float(cluster_selection_relative_score),
            "threshold_mask_padding_periods": float(threshold_mask_padding_periods),
            "silhouette_scores": {},
            "rejected_frequency_clusters": [],
            "threshold_per_cluster": {},
            "attenuation_mask_per_cluster": {},
            "reconstructed_cluster_traces": {},
            "event_template_rejection_results": {},
            "enable_noise_cluster_rejection": bool(enable_noise_cluster_rejection),
            "enable_event_template_rejection": bool(enable_event_template_rejection),
            "event_template_rejection_mode": str(event_template_rejection_mode),
            "event_attenuation_min": None if event_attenuation_min is None else float(event_attenuation_min),
            "event_attenuation_max": None if event_attenuation_max is None else float(event_attenuation_max),
            "event_rejection_application": str(event_rejection_application),
            "event_detection": {
                "z_threshold": float(event_detection_z_threshold),
                "prominence_z": float(event_detection_prominence_z),
                "min_distance_ms": float(event_detection_min_distance_ms),
                "polarity": str(event_detection_polarity),
                "use_threshold_crossings": bool(event_detection_use_threshold_crossings),
            },
            "coefficient_event_protection": {
                "enabled": bool(enable_coefficient_event_protection),
                "status": "empty_trace",
                "window_ms": float(coefficient_event_protection_window_ms),
            },
            "low_frequency_gaussian_branch": {"enabled": False, "status": "empty_trace"},
            "reconstruction_mode": str(reconstruction_mode),
            "inverse_cwt": _gonzalez_inverse_cwt_info(
                regularization=float(inverse_cwt_regularization),
                precision=int(wavelet_precision),
            ),
            "cwt_working_trace": values.copy(),
            "final_denoised_working_trace": values.copy(),
            "final_denoised_dff": None if mode == "corrected" else values.copy(),
            "final_denoised_fluorescence": values.copy() if mode == "fluorescence" else None,
            "output": values.copy(),
            "safety_metrics": metrics,
        }
        return debug if return_debug else values.copy()

    if mode == "fluorescence":
        working_trace, f0 = _gonzalez_dff_from_fluorescence(values, fs, rolling_baseline_seconds)
        working_label = "dff"
    else:
        # Direct working-trace mode: callers use this for baseline-subtracted
        # or baseline-corrected traces. Do not compute rolling F0 and do not
        # convert the result back to fluorescence units.
        working_trace = values.copy()
        f0 = None
        working_label = "corrected_signal"

    if bool(enable_low_frequency_gaussian_branch):
        low_frequency_branch, low_frequency_debug = _gonzalez_gaussian_low_frequency_branch(
            working_trace,
            fs=fs,
            cutoff_hz=float(low_frequency_cutoff_hz),
        )
        cwt_working_trace = np.asarray(working_trace - low_frequency_branch, dtype=float)
    else:
        low_frequency_branch = np.zeros_like(working_trace, dtype=float)
        low_frequency_debug = {"enabled": False, "status": "disabled"}
        cwt_working_trace = working_trace.copy()

    # Paper-specified CWT: Morlet continuous wavelet transform of the working
    # trace. For input_mode="corrected", this is simply the corrected input signal.
    frequencies = _gonzalez_frequency_vector(fs, min_freq, max_freq, n_freqs)
    scales = pywt.frequency2scale(wavelet, frequencies / float(fs))
    coeffs, returned_freqs = pywt.cwt(
        data=cwt_working_trace,
        scales=scales,
        wavelet=wavelet,
        sampling_period=1.0 / float(fs),
        method="fft",
        precision=int(wavelet_precision),
    )
    coeffs = np.asarray(coeffs)
    magnitudes = np.abs(coeffs)
    analysis_filter_fft = _gonzalez_cwt_analysis_filter_fft(
        coeffs.shape[1],
        scales,
        wavelet,
        precision=int(wavelet_precision),
    )

    # Paper-specified normalization: each frequency row is centered and scaled
    # by its own temporal coefficient-magnitude statistics.
    row_mean = magnitudes.mean(axis=1, keepdims=True)
    row_std = np.maximum(magnitudes.std(axis=1, keepdims=True), 1e-12)
    normalized = (magnitudes - row_mean) / row_std

    pca_features, explained_variance, selected_pc_count = _gonzalez_pca_features(normalized)
    labels, chosen_cluster_count, silhouette_scores = _gonzalez_ward_clusters(
        pca_features,
        selected_pc_count=selected_pc_count,
        max_clusters=max_clusters,
        selection_strategy=str(cluster_selection_strategy),
        relative_score=float(cluster_selection_relative_score),
    )
    threshold_domain = str(coefficient_threshold_domain).strip().lower()
    if threshold_domain in {"normalized", "normalized_magnitude", "zscore", "z_score"}:
        threshold_magnitudes = normalized
        threshold_domain = "normalized_magnitude"
    else:
        threshold_magnitudes = magnitudes
        threshold_domain = "raw_magnitude"
    cluster_info = _gonzalez_cluster_thresholds(threshold_magnitudes, frequencies, labels, fs, threshold_sd)
    rejected_clusters, rejection_decisions = _gonzalez_rejected_frequency_clusters(
        cluster_info,
        max_frequency=float(np.max(frequencies)),
        threshold_sd=threshold_sd,
        enabled=bool(enable_noise_cluster_rejection),
    )

    raw_reconstruction = _gonzalez_inverse_cwt(
        coeffs,
        scales,
        wavelet,
        filter_fft=analysis_filter_fft,
        precision=int(wavelet_precision),
        regularization=float(inverse_cwt_regularization),
    )
    raw_gain, raw_offset, amplitude_diagnostics = _gonzalez_reconstruction_calibration(
        raw_reconstruction,
        cwt_working_trace,
    )

    cleaned_cluster_traces: Dict[int, np.ndarray] = {}
    attenuation_debug: Dict[int, np.ndarray] = {}
    threshold_debug: Dict[int, dict] = {}
    event_debug: Dict[int, dict] = {}
    retained_cluster_ids: List[int] = []
    cleaned_coeffs_full = np.zeros_like(coeffs)
    identity_coeffs_full = np.zeros_like(coeffs)
    trace_domain_event_removal = np.zeros(values.shape, dtype=float)

    attenuation_min = float(np.clip(attenuation_min, 0.0, 1.0))
    attenuation_max = float(np.clip(attenuation_max, attenuation_min, 1.0))
    if event_attenuation_min is None:
        event_attenuation_min = attenuation_min
    if event_attenuation_max is None:
        event_attenuation_max = event_attenuation_min
    event_attenuation_min = float(np.clip(event_attenuation_min, 0.0, 1.0))
    event_attenuation_max = float(np.clip(event_attenuation_max, event_attenuation_min, 1.0))
    coefficient_event_weights, coefficient_event_debug = _gonzalez_event_protection_weights(
        working_trace,
        fs=fs,
        enabled=bool(enable_coefficient_event_protection),
        window_ms=float(coefficient_event_protection_window_ms),
    )
    event_rejection_application_normalized = str(event_rejection_application).strip().lower()
    use_trace_domain_event_rejection = event_rejection_application_normalized in {
        "trace",
        "trace_domain",
        "reconstructed_trace",
        "paper_trace",
    }
    for cluster_id, info in cluster_info.items():
        indices = np.asarray(info["indices"], dtype=int)
        identity_coeffs_full[indices] = coeffs[indices]
        threshold = np.asarray(info["threshold"], dtype=float)
        activity = np.asarray(info["activity"], dtype=float)
        padding_samples = max(
            0,
            int(
                round(
                    float(threshold_mask_padding_periods)
                    * float(fs)
                    / max(float(info["max_frequency"]), 1e-9)
                )
            ),
        )
        threshold_debug[int(cluster_id)] = {
            "activity": activity.copy(),
            "threshold": threshold,
            "moving_average": np.asarray(info["moving_average"], dtype=float),
            "residual_std": float(info["residual_std"]),
            "moving_average_window_samples": int(info["moving_average_window_samples"]),
            "max_frequency": float(info["max_frequency"]),
            "frequency_indices": indices.copy(),
            "threshold_mask_padding_periods": float(threshold_mask_padding_periods),
            "threshold_mask_padding_samples": int(padding_samples),
        }
        if int(cluster_id) in rejected_clusters:
            gain_mask = coefficient_event_weights.copy() if bool(enable_coefficient_event_protection) else np.zeros(values.shape, dtype=float)
            attenuation_debug[int(cluster_id)] = gain_mask.copy()
            cleaned_cluster_coeffs = coeffs[indices] * gain_mask[None, :]
            if np.any(gain_mask > 1e-12):
                cleaned_coeffs_full[indices] = cleaned_cluster_coeffs
                cleaned_cluster_traces[int(cluster_id)] = _gonzalez_inverse_cwt(
                    cleaned_cluster_coeffs,
                    scales[indices],
                    wavelet,
                    filter_fft=analysis_filter_fft[indices],
                    precision=int(wavelet_precision),
                    regularization=float(inverse_cwt_regularization),
                )
            else:
                cleaned_cluster_traces[int(cluster_id)] = np.zeros(values.shape, dtype=float)
            event_debug[int(cluster_id)] = {
                "enabled": False,
                "reason": "frequency_cluster_rejected_as_noise",
                "event_coefficients_preserved": bool(np.any(gain_mask > 1e-12)),
            }
            continue

        retained_cluster_ids.append(int(cluster_id))
        above = activity > threshold
        if padding_samples > 0:
            above = _dilate_boolean_mask(above, radius=padding_samples)
        below = ~above
        gap = np.maximum(threshold - activity, 0.0)
        denom = max(float(np.max(gap[below])) if np.any(below) else 0.0, 1e-12)
        severity = np.clip(gap / denom, 0.0, 1.0)
        attenuation_fraction = attenuation_min + severity * (attenuation_max - attenuation_min)
        gain_mask = np.where(below, 1.0 - attenuation_fraction, 1.0)
        if bool(enable_coefficient_event_protection):
            gain_mask = gain_mask + ((1.0 - gain_mask) * coefficient_event_weights)
        attenuation_debug[int(cluster_id)] = gain_mask.copy()

        # Paper-specified masking: attenuate the original complex coefficients
        # for the cluster, not the magnitude-only representation.
        cleaned_cluster_coeffs = coeffs[indices] * gain_mask[None, :]
        reconstructed = _gonzalez_inverse_cwt(
            cleaned_cluster_coeffs,
            scales[indices],
            wavelet,
            filter_fft=analysis_filter_fft[indices],
            denominator_filter_fft=analysis_filter_fft if use_trace_domain_event_rejection else None,
            precision=int(wavelet_precision),
            regularization=float(inverse_cwt_regularization),
        )
        event_processed, event_info = _gonzalez_event_template_rejection(
            reconstructed,
            fs=fs,
            cluster_id=int(cluster_id),
            max_frequency=float(info["max_frequency"]),
            attenuation_min=event_attenuation_min,
            attenuation_max=event_attenuation_max,
            enabled=bool(enable_event_template_rejection),
            mode=str(event_template_rejection_mode),
            event_z_threshold=float(event_detection_z_threshold),
            event_prominence_z=float(event_detection_prominence_z),
            event_min_distance_ms=float(event_detection_min_distance_ms),
            event_detection_polarity=str(event_detection_polarity),
            event_candidate_mask=(activity > threshold) if bool(event_detection_use_threshold_crossings) else None,
        )
        event_gain = np.asarray(event_info.get("event_gain", np.ones(values.shape, dtype=float)), dtype=float)
        if event_gain.shape != values.shape:
            event_gain = np.ones(values.shape, dtype=float)
            event_info["event_gain"] = event_gain
            event_info["event_gain_shape_warning"] = "event_gain shape mismatch; coefficient event attenuation skipped"
        if use_trace_domain_event_rejection:
            trace_domain_event_removal = trace_domain_event_removal + (reconstructed - event_processed)
            event_info["application"] = "trace_domain_after_cluster_iwt"
        else:
            cleaned_cluster_coeffs = cleaned_cluster_coeffs * event_gain[None, :]
            event_info["application"] = "coefficient_time_mask"
        cleaned_coeffs_full[indices] = cleaned_cluster_coeffs
        cleaned_cluster_traces[int(cluster_id)] = event_processed
        event_debug[int(cluster_id)] = event_info

    raw_full = _gonzalez_inverse_cwt(
        coeffs,
        scales,
        wavelet,
        filter_fft=analysis_filter_fft,
        precision=int(wavelet_precision),
        regularization=float(inverse_cwt_regularization),
    )
    identity_full = _gonzalez_inverse_cwt(
        identity_coeffs_full,
        scales,
        wavelet,
        filter_fft=analysis_filter_fft,
        precision=int(wavelet_precision),
        regularization=float(inverse_cwt_regularization),
    )
    consistency_error = (
        float(np.max(np.abs(raw_full - identity_full))) if raw_full.shape == identity_full.shape and raw_full.size else 0.0
    )
    denoised_raw = _gonzalez_inverse_cwt(
        cleaned_coeffs_full,
        scales,
        wavelet,
        filter_fft=analysis_filter_fft,
        precision=int(wavelet_precision),
        regularization=float(inverse_cwt_regularization),
    )
    removed_raw = _gonzalez_inverse_cwt(
        coeffs - cleaned_coeffs_full,
        scales,
        wavelet,
        filter_fft=analysis_filter_fft,
        precision=int(wavelet_precision),
        regularization=float(inverse_cwt_regularization),
    )
    reconstruction_mode_normalized = str(reconstruction_mode).strip().lower()
    if reconstruction_mode_normalized in {"residual_subtractive", "subtract_removed", "paper_safe"}:
        removed_gain_mode = str(removed_reconstruction_gain_mode).strip().lower()
        removed_gain_scale = float(removed_reconstruction_gain_scale)
        if removed_gain_mode in {"unit", "uncalibrated", "none", "raw_inverse"}:
            removed_gain = removed_gain_scale
            removed_gain_mode = "unit"
        else:
            removed_gain = raw_gain * removed_gain_scale
            removed_gain_mode = "calibrated"
        denoised_working_trace = cwt_working_trace - (removed_gain * removed_raw)
        reconstruction_note = (
            "Residual-subtractive reconstruction: regularized inverse CWT is applied to removed coefficients only, "
            "then subtracted from the original CWT input."
        )
    else:
        removed_gain_mode = "not_applicable"
        removed_gain = raw_gain
        removed_gain_scale = 1.0
        denoised_working_trace = raw_offset + (raw_gain * denoised_raw)
        reconstruction_note = "Direct regularized inverse CWT of cleaned coefficients."
    if use_trace_domain_event_rejection:
        denoised_working_trace = denoised_working_trace - (raw_gain * trace_domain_event_removal)
    if bool(enable_low_frequency_gaussian_branch):
        denoised_working_trace = denoised_working_trace + low_frequency_branch
    full_matrix_residual = denoised_working_trace - working_trace
    amplitude_diagnostics["max_abs_reconstruction_consistency_error"] = consistency_error
    amplitude_diagnostics["full_matrix_reconstruction"] = {
        "calibration": "raw_reconstruction_gain_offset_applied_to_full_cleaned_matrix",
        "gain": raw_gain,
        "offset": raw_offset,
        "raw_reconstruction_std": float(np.std(denoised_raw)),
        "removed_reconstruction_std": float(np.std(removed_raw)),
        "trace_domain_event_removal_std": float(np.std(trace_domain_event_removal)),
        "removed_reconstruction_gain_mode": removed_gain_mode,
        "removed_reconstruction_gain": float(removed_gain),
        "removed_reconstruction_gain_scale": float(removed_gain_scale),
        "target_working_trace_std": float(np.std(working_trace)),
        "calibrated_reconstruction_std": float(np.std(denoised_working_trace)),
        "calibrated_rms_error": (
            float(np.sqrt(np.mean(full_matrix_residual**2))) if full_matrix_residual.size else float("nan")
        ),
        "calibrated_correlation": (
            float(np.corrcoef(denoised_working_trace, working_trace)[0, 1])
            if denoised_raw.size > 1 and np.std(denoised_working_trace) > 1e-12
            else float("nan")
        ),
        "retained_cluster_count": int(len(retained_cluster_ids)),
        "rejected_cluster_count": int(len(rejected_clusters)),
        "reconstruction_mode": reconstruction_mode_normalized,
        "reconstruction_note": reconstruction_note,
        "inverse_cwt": _gonzalez_inverse_cwt_info(
            regularization=float(inverse_cwt_regularization),
            precision=int(wavelet_precision),
        ),
    }
    denoised_working_trace = np.asarray(denoised_working_trace, dtype=float)
    candidate_output = (
        denoised_working_trace
        if mode == "corrected"
        else np.asarray(f0 * (1.0 + denoised_working_trace), dtype=float)
    )

    input_std = float(np.std(working_trace)) if working_trace.size else float("nan")
    denoised_std = float(np.std(denoised_working_trace)) if denoised_working_trace.size else float("nan")
    std_ratio = denoised_std / max(input_std, 1e-12) if np.isfinite(input_std) else float("nan")
    warnings: List[str] = []
    fallback_triggered = False
    if np.isfinite(std_ratio) and std_ratio > 1.25:
        warnings.append(
            "Gonzalez denoising increased working-trace standard deviation above 1.25x."
        )
    output = np.asarray(candidate_output, dtype=float)
    morphology_metrics = _event_preservation_metrics(
        working_trace,
        denoised_working_trace,
        fs=fs,
        evaluation_mask=np.ones(working_trace.shape, dtype=bool),
        min_event_peak_preservation=float(min_event_peak_preservation),
        min_local_max_ratio=float(min_local_max_ratio),
    )
    morphology_reasons = list(morphology_metrics.get("fallback_reasons", []))
    if bool(enable_denoise_safety_gate) and bool(morphology_reasons):
        denoised_working_trace = working_trace.copy()
        output = values.copy()
        fallback_triggered = True
        warnings.extend(morphology_reasons)
        morphology_metrics["fallback_triggered"] = True
    elif bool(enable_denoise_safety_gate) and bool(enable_noise_cluster_rejection) and bool(morphology_reasons):
        denoised_working_trace = working_trace.copy()
        output = values.copy()
        fallback_triggered = True
        warnings.extend(
            f"noise_cluster_rejection_{reason}" for reason in morphology_reasons
        )
        morphology_metrics["fallback_triggered"] = True
    denoised_std = float(np.std(denoised_working_trace)) if denoised_working_trace.size else float("nan")
    std_ratio = denoised_std / max(input_std, 1e-12) if np.isfinite(input_std) else float("nan")
    denoised_fluorescence = None if mode == "corrected" else output.copy()
    safety_metrics = {
        "input_min": float(np.min(working_trace)) if working_trace.size else float("nan"),
        "input_max": float(np.max(working_trace)) if working_trace.size else float("nan"),
        "input_std": input_std,
        "denoised_min": float(np.min(denoised_working_trace)) if denoised_working_trace.size else float("nan"),
        "denoised_max": float(np.max(denoised_working_trace)) if denoised_working_trace.size else float("nan"),
        "denoised_std": denoised_std,
        "output_input_std_ratio": std_ratio,
        "fallback_triggered": fallback_triggered,
        "morphology_preservation": morphology_metrics,
        "warnings": warnings,
    }

    if not return_debug:
        return output

    debug = {
        "denoiser_used": "gonzalez_adaptive_wavelet",
        "input_mode": mode,
        "input_mode_alias": mode_alias,
        "wavelet": str(wavelet),
        "wavelet_precision": int(wavelet_precision),
        "threshold_sd": float(threshold_sd),
        "threshold_mask_padding_periods": float(threshold_mask_padding_periods),
        "coefficient_threshold_domain": threshold_domain,
        "attenuation_min": float(attenuation_min),
        "attenuation_max": float(attenuation_max),
        "event_attenuation_min": float(event_attenuation_min),
        "event_attenuation_max": float(event_attenuation_max),
        "event_rejection_application": (
            "trace_domain_after_cluster_iwt" if use_trace_domain_event_rejection else "coefficient_time_mask"
        ),
        "event_detection": {
            "z_threshold": float(event_detection_z_threshold),
            "prominence_z": float(event_detection_prominence_z),
            "min_distance_ms": float(event_detection_min_distance_ms),
            "polarity": str(event_detection_polarity),
            "use_threshold_crossings": bool(event_detection_use_threshold_crossings),
        },
        "max_clusters": int(max_clusters),
        "cluster_selection_strategy": str(cluster_selection_strategy),
        "cluster_selection_relative_score": float(cluster_selection_relative_score),
        "working_trace_label": working_label,
        "F0": None if f0 is None else f0.copy(),
        "working_trace": working_trace.copy(),
        "cwt_working_trace": cwt_working_trace.copy(),
        "s7_low_frequency_trace": low_frequency_branch.copy(),
        "corrected_signal": working_trace.copy() if mode == "corrected" else None,
        "raw_dff": working_trace.copy() if mode == "fluorescence" else None,
        "frequency_vector": frequencies.copy(),
        "pywt_returned_frequencies": np.asarray(returned_freqs, dtype=float),
        "wavelet_scales": np.asarray(scales, dtype=float),
        "original_complex_cwt_coefficients": _safe_debug_array(coeffs),
        "cleaned_complex_cwt_coefficients": _safe_debug_array(cleaned_coeffs_full),
        "normalized_coefficient_magnitudes": _safe_debug_array(normalized),
        "pca_explained_variance": explained_variance.copy(),
        "selected_pc_count": int(selected_pc_count),
        "ward_cluster_labels": labels.copy(),
        "chosen_cluster_count": int(chosen_cluster_count),
        "silhouette_scores": dict(silhouette_scores),
        "rejected_frequency_clusters": rejection_decisions,
        "retained_frequency_clusters": retained_cluster_ids,
        "threshold_per_cluster": threshold_debug,
        "attenuation_mask_per_cluster": attenuation_debug,
        "reconstructed_cluster_traces": cleaned_cluster_traces,
        "event_template_rejection_results": event_debug,
        "enable_noise_cluster_rejection": bool(enable_noise_cluster_rejection),
        "enable_event_template_rejection": bool(enable_event_template_rejection),
        "event_template_rejection_mode": str(event_template_rejection_mode),
        "coefficient_event_protection": coefficient_event_debug,
        "low_frequency_gaussian_branch": low_frequency_debug,
        "reconstruction_mode": reconstruction_mode_normalized,
        "removed_reconstruction_gain_mode": str(removed_reconstruction_gain_mode),
        "removed_reconstruction_gain_scale": float(removed_reconstruction_gain_scale),
        "inverse_cwt": _gonzalez_inverse_cwt_info(
            regularization=float(inverse_cwt_regularization),
            precision=int(wavelet_precision),
        ),
        "amplitude_preservation_diagnostics": amplitude_diagnostics,
        "final_denoised_working_trace": denoised_working_trace.copy(),
        "final_denoised_dff": denoised_working_trace.copy() if mode == "fluorescence" else None,
        "final_denoised_corrected": denoised_working_trace.copy() if mode == "corrected" else None,
        "final_denoised_fluorescence": None if denoised_fluorescence is None else np.asarray(denoised_fluorescence).copy(),
        "output": output.copy(),
        "safety_metrics": safety_metrics,
        "warnings": warnings,
        "algorithm_notes": {
            "method": "Gonzalez adaptive wavelet denoising",
            "active_pipeline": "Direct wavelet denoising of the provided working trace.",
            "no_post_smoothing": True,
            "corrected_input_mode": "Uses the input directly as a corrected working trace; no rolling F0 or fluorescence conversion.",
            "noise_cluster_rejection": "Optional and disabled by default because frequency-cluster cleanup can suppress true spike content.",
            "moving_average_window": "Two periods of the maximum cluster frequency; paper under-specifies exact formula.",
            "inverse_cwt": "Regularized dual-frame inverse of PyWavelets Morlet CWT filters with amplitude diagnostics.",
            "event_template_rejection": "Optional cleanup disabled by default because template rejection is under-specified.",
        },
    }
    return debug


def gonzalez_adaptive_wavelet_validation_metrics(
    raw_working_trace: np.ndarray,
    denoised_working_trace: np.ndarray,
    fs: float,
    event_mask: np.ndarray | None = None,
    plateau_mask: np.ndarray | None = None,
) -> dict:
    """Compute optional validation metrics, kept separate from denoising."""
    raw = _validate_trace(raw_working_trace, fs)
    denoised = _validate_trace(denoised_working_trace, fs)
    if raw.shape != denoised.shape:
        raise ValueError("raw_working_trace and denoised_working_trace must have the same shape")
    hp_window = _odd_sample_window(0.05, fs, size=raw.size)
    raw_high = raw - median_filter(raw, size=hp_window, mode="nearest")
    denoised_high = denoised - median_filter(denoised, size=hp_window, mode="nearest")
    metrics = {
        "high_frequency_noise_reduction": 1.0 - (float(np.std(denoised_high)) / max(float(np.std(raw_high)), 1e-12)),
        "rms_difference": float(np.sqrt(np.mean((denoised - raw) ** 2))) if raw.size else float("nan"),
    }
    if event_mask is not None:
        event = _validated_mask(event_mask, raw.shape, "event_mask")
        outside = ~event
        metrics["event_peak_preservation"] = (
            float(np.max(np.abs(denoised[event])) / max(float(np.max(np.abs(raw[event]))), 1e-12))
            if np.any(event)
            else float("nan")
        )
        metrics["rms_difference_outside_event_regions"] = (
            float(np.sqrt(np.mean((denoised[outside] - raw[outside]) ** 2))) if np.any(outside) else float("nan")
        )
        regions = _contiguous_true_regions(event)
        shifts: List[float] = []
        for start, stop in regions:
            raw_peak = start + int(np.argmax(np.abs(raw[start:stop])))
            denoised_peak = start + int(np.argmax(np.abs(denoised[start:stop])))
            shifts.append((denoised_peak - raw_peak) / float(fs))
        metrics["onset_offset_timing_shift_seconds"] = float(np.median(np.abs(shifts))) if shifts else float("nan")
    if plateau_mask is not None:
        plateau = _validated_mask(plateau_mask, raw.shape, "plateau_mask")
        metrics["plateau_median_bias"] = (
            float(np.median(denoised[plateau]) - np.median(raw[plateau])) if np.any(plateau) else float("nan")
        )
        regions = _contiguous_true_regions(plateau)
        jumps_raw: List[float] = []
        jumps_denoised: List[float] = []
        for start, stop in regions:
            before = slice(max(0, start - hp_window), start)
            inside = slice(start, min(stop, start + hp_window))
            if raw[before].size and raw[inside].size:
                jumps_raw.append(float(np.median(raw[inside]) - np.median(raw[before])))
                jumps_denoised.append(float(np.median(denoised[inside]) - np.median(denoised[before])))
        if jumps_raw:
            metrics["jump_amplitude_preservation"] = float(
                np.median(np.abs(jumps_denoised)) / max(float(np.median(np.abs(jumps_raw))), 1e-12)
            )
    return metrics


def _validated_mask(values: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must have the same shape as corrected")
    return mask


def _odd_window_ms(window_ms: float, fs: float) -> int:
    window = max(3, int(round(float(window_ms) * float(fs) / 1000.0)))
    return window + 1 if window % 2 == 0 else window


def _bounded_odd_window_ms(window_ms: float, fs: float, size: int) -> int:
    if size <= 1:
        return 1
    window = _odd_window_ms(window_ms, fs)
    max_window = size if size % 2 == 1 else size - 1
    return max(1, min(window, max_window))


def _contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return []
    starts = np.flatnonzero(values & np.concatenate(([True], ~values[:-1])))
    stops = np.flatnonzero(values & np.concatenate((~values[1:], [True]))) + 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _normalize_event_polarity(value: str) -> str:
    polarity = str(value).strip().lower()
    if polarity in {"absolute", "abs", "both", "either"}:
        return "absolute"
    if polarity in {"positive", "up", "upward", "pos"}:
        return "positive"
    if polarity in {"negative", "down", "downward", "neg"}:
        return "negative"
    raise ValueError("event polarity must be one of: absolute, positive, negative")


def _polarity_peak_signal(values: np.ndarray, finite: np.ndarray, polarity: str) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    mask = np.asarray(finite, dtype=bool)
    mode = _normalize_event_polarity(polarity)
    if mode == "positive":
        return np.where(mask, data, -np.inf)
    if mode == "negative":
        return np.where(mask, -data, -np.inf)
    return np.where(mask, np.abs(data), 0.0)


def _event_preservation_metrics(
    raw: np.ndarray,
    candidate: np.ndarray,
    fs: float,
    evaluation_mask: np.ndarray,
    min_event_peak_preservation: float = 0.95,
    min_local_max_ratio: float = 0.95,
    baseline_window_ms: float = 50.0,
    local_window_ms: float = 5.0,
    min_peak_distance_ms: float = 2.0,
    event_polarity: str = "absolute",
) -> dict:
    """Check that denoising preserves sharp morphology.

    The denoised trace is a detection aid; raw corrected samples remain the
    preferred source for quantitative amplitude, width, area, and plateau
    measurements.
    """
    source = np.asarray(raw, dtype=float)
    output = np.asarray(candidate, dtype=float)
    mask = np.asarray(evaluation_mask, dtype=bool)
    reasons: List[str] = []
    polarity_mode = _normalize_event_polarity(event_polarity)
    metrics: dict[str, Any] = {
        "enabled": True,
        "status": "not_evaluated",
        "fallback_triggered": False,
        "fallback_reasons": reasons,
        "evaluated_peak_count": 0,
        "event_count_before": 0,
        "event_count_after": 0,
        "median_peak_amplitude_ratio": float("nan"),
        "minimum_peak_amplitude_ratio": float("nan"),
        "median_local_max_ratio": float("nan"),
        "minimum_local_max_ratio": float("nan"),
        "polarity_flipped": False,
        "event_polarity": polarity_mode,
    }
    if source.shape != output.shape:
        reasons.append("shape_mismatch")
        metrics["fallback_triggered"] = True
        metrics["status"] = "failed"
        return metrics
    if source.size == 0:
        metrics["status"] = "empty"
        return metrics
    if mask.shape != source.shape:
        reasons.append("evaluation_mask_shape_mismatch")
        metrics["fallback_triggered"] = True
        metrics["status"] = "failed"
        return metrics
    finite = np.isfinite(source) & np.isfinite(output) & mask
    if np.count_nonzero(finite) < 3:
        metrics["status"] = "no_events_evaluated"
        return metrics

    hp_window = _odd_window_ms(baseline_window_ms, fs)
    hp_window = min(hp_window, source.size if source.size % 2 == 1 else max(1, source.size - 1))
    if hp_window < 3:
        raw_high = source - float(np.median(source[finite]))
        cand_high = output - float(np.median(output[finite]))
    else:
        raw_high = source - median_filter(source, size=hp_window, mode="nearest")
        cand_high = output - median_filter(output, size=hp_window, mode="nearest")

    selected = raw_high[finite]
    noise = max(_robust_sigma(selected), 1e-12)
    max_abs = float(np.max(np.abs(selected))) if selected.size else 0.0
    peak_threshold = max(4.0 * noise, 0.20 * max_abs, 1e-12)
    distance = max(1, int(round(float(min_peak_distance_ms) * float(fs) / 1000.0)))
    raw_peak_signal = _polarity_peak_signal(raw_high, finite, polarity_mode)
    raw_peaks, _ = find_peaks(raw_peak_signal, height=peak_threshold, distance=distance)
    raw_peaks = np.asarray([int(p) for p in raw_peaks if finite[int(p)]], dtype=int)
    candidate_peak_signal = _polarity_peak_signal(cand_high, finite, polarity_mode)
    candidate_peaks, _ = find_peaks(candidate_peak_signal, height=peak_threshold, distance=distance)
    candidate_peaks = np.asarray([int(p) for p in candidate_peaks if finite[int(p)]], dtype=int)
    metrics["event_count_before"] = int(raw_peaks.size)
    metrics["event_count_after"] = int(candidate_peaks.size)
    if raw_peaks.size == 0:
        metrics["status"] = "no_events_evaluated"
        return metrics

    half_window = max(1, int(round(float(local_window_ms) * float(fs) / 1000.0)))
    peak_ratios: List[float] = []
    local_ratios: List[float] = []
    polarity_flipped = False
    for peak in raw_peaks:
        start = max(0, int(peak) - half_window)
        stop = min(source.size, int(peak) + half_window + 1)
        local_mask = finite[start:stop]
        if not np.any(local_mask):
            continue
        raw_local = raw_high[start:stop][local_mask]
        cand_local = cand_high[start:stop][local_mask]
        raw_peak_amp = abs(float(raw_high[int(peak)]))
        cand_peak_amp = abs(float(cand_high[int(peak)]))
        raw_local_max = float(np.max(np.abs(raw_local))) if raw_local.size else 0.0
        cand_local_max = float(np.max(np.abs(cand_local))) if cand_local.size else 0.0
        if raw_peak_amp > 1e-12:
            peak_ratios.append(cand_peak_amp / raw_peak_amp)
            if (
                abs(float(cand_high[int(peak)])) > 0.25 * raw_peak_amp
                and np.sign(float(cand_high[int(peak)])) != np.sign(float(raw_high[int(peak)]))
            ):
                polarity_flipped = True
        if raw_local_max > 1e-12:
            local_ratios.append(cand_local_max / raw_local_max)

    metrics["evaluated_peak_count"] = int(len(peak_ratios))
    if not peak_ratios or not local_ratios:
        metrics["status"] = "no_events_evaluated"
        return metrics

    peak_arr = np.asarray(peak_ratios, dtype=float)
    local_arr = np.asarray(local_ratios, dtype=float)
    metrics.update(
        {
            "status": "evaluated",
            "median_peak_amplitude_ratio": float(np.median(peak_arr)),
            "minimum_peak_amplitude_ratio": float(np.min(peak_arr)),
            "median_local_max_ratio": float(np.median(local_arr)),
            "minimum_local_max_ratio": float(np.min(local_arr)),
            "polarity_flipped": bool(polarity_flipped),
        }
    )
    if float(metrics["minimum_peak_amplitude_ratio"]) < float(min_event_peak_preservation):
        reasons.append("event_peak_preservation_below_floor")
    if float(metrics["minimum_local_max_ratio"]) < float(min_local_max_ratio):
        reasons.append("local_max_ratio_below_floor")
    if polarity_flipped:
        reasons.append("event_polarity_flipped")
    metrics["fallback_triggered"] = bool(reasons)
    metrics["thresholds"] = {
        "min_event_peak_preservation": float(min_event_peak_preservation),
        "min_local_max_ratio": float(min_local_max_ratio),
        "peak_detection_threshold": float(peak_threshold),
        "baseline_window_samples": int(hp_window),
        "local_window_samples": int(half_window),
    }
    return metrics


def _restore_event_amplitudes(
    raw: np.ndarray,
    candidate: np.ndarray,
    fs: float,
    evaluation_mask: np.ndarray,
    min_event_peak_preservation: float = 0.95,
    min_local_max_ratio: float = 0.95,
    baseline_window_ms: float = 50.0,
    local_window_ms: float = 5.0,
    min_peak_distance_ms: float = 2.0,
    event_polarity: str = "absolute",
) -> Tuple[np.ndarray, dict]:
    source = np.asarray(raw, dtype=float)
    output = np.asarray(candidate, dtype=float).copy()
    mask = np.asarray(evaluation_mask, dtype=bool)
    polarity_mode = _normalize_event_polarity(event_polarity)
    debug: dict[str, Any] = {
        "enabled": True,
        "status": "not_evaluated",
        "event_polarity": polarity_mode,
        "restored_event_count": 0,
        "restored_events": [],
    }
    if source.shape != output.shape or mask.shape != source.shape or source.size == 0:
        debug["status"] = "shape_mismatch_or_empty"
        return output, debug
    finite = np.isfinite(source) & np.isfinite(output) & mask
    if np.count_nonzero(finite) < 3:
        debug["status"] = "no_events_evaluated"
        return output, debug

    hp_window = _odd_window_ms(baseline_window_ms, fs)
    hp_window = min(hp_window, source.size if source.size % 2 == 1 else max(1, source.size - 1))
    if hp_window < 3:
        raw_high = source - float(np.median(source[finite]))
        cand_high = output - float(np.median(output[finite]))
    else:
        raw_high = source - median_filter(source, size=hp_window, mode="nearest")
        cand_high = output - median_filter(output, size=hp_window, mode="nearest")

    selected = raw_high[finite]
    noise = max(_robust_sigma(selected), 1e-12)
    max_abs = float(np.max(np.abs(selected))) if selected.size else 0.0
    peak_threshold = max(4.0 * noise, 0.20 * max_abs, 1e-12)
    distance = max(1, int(round(float(min_peak_distance_ms) * float(fs) / 1000.0)))
    raw_peak_signal = _polarity_peak_signal(raw_high, finite, polarity_mode)
    raw_peaks, _ = find_peaks(raw_peak_signal, height=peak_threshold, distance=distance)
    raw_peaks = np.asarray([int(p) for p in raw_peaks if finite[int(p)]], dtype=int)
    if raw_peaks.size == 0:
        debug["status"] = "no_events_evaluated"
        return output, debug

    half_window = max(1, int(round(float(local_window_ms) * float(fs) / 1000.0)))
    restored_events: List[dict] = []
    for peak in raw_peaks:
        start = max(0, int(peak) - half_window)
        stop = min(source.size, int(peak) + half_window + 1)
        local_mask = finite[start:stop]
        if not np.any(local_mask):
            continue
        raw_local = raw_high[start:stop][local_mask]
        cand_local = cand_high[start:stop][local_mask]
        raw_peak_amp = abs(float(raw_high[int(peak)]))
        cand_peak_amp = abs(float(cand_high[int(peak)]))
        raw_local_max = float(np.max(np.abs(raw_local))) if raw_local.size else 0.0
        cand_local_max = float(np.max(np.abs(cand_local))) if cand_local.size else 0.0
        peak_ratio = cand_peak_amp / max(raw_peak_amp, 1e-12)
        local_ratio = cand_local_max / max(raw_local_max, 1e-12)
        polarity_flipped = (
            raw_peak_amp > 1e-12
            and cand_peak_amp > 0.25 * raw_peak_amp
            and np.sign(float(cand_high[int(peak)])) != np.sign(float(raw_high[int(peak)]))
        )
        if (
            peak_ratio >= float(min_event_peak_preservation)
            and local_ratio >= float(min_local_max_ratio)
            and not polarity_flipped
        ):
            continue

        width = stop - start
        if width <= 1:
            output[start:stop] = source[start:stop]
        else:
            x = np.linspace(-1.0, 1.0, width)
            weights = 0.5 * (1.0 + np.cos(np.pi * np.clip(x, -1.0, 1.0)))
            weights[~local_mask] = 0.0
            output[start:stop] = (weights * source[start:stop]) + ((1.0 - weights) * output[start:stop])
        restored_events.append(
            {
                "peak": int(peak),
                "start": int(start),
                "stop": int(stop),
                "peak_ratio_before": float(peak_ratio),
                "local_ratio_before": float(local_ratio),
                "raw_peak_sign": float(np.sign(float(raw_high[int(peak)]))),
                "polarity_flipped": bool(polarity_flipped),
            }
        )

    debug.update(
        {
            "status": "evaluated",
            "event_count_before": int(raw_peaks.size),
            "restored_event_count": int(len(restored_events)),
            "restored_events": restored_events,
            "thresholds": {
                "min_event_peak_preservation": float(min_event_peak_preservation),
                "min_local_max_ratio": float(min_local_max_ratio),
                "peak_detection_threshold": float(peak_threshold),
                "baseline_window_samples": int(hp_window),
                "local_window_samples": int(half_window),
            },
        }
    )
    return output, debug


def _repair_downward_denoising_artifacts(
    raw: np.ndarray,
    pre_restore: np.ndarray,
    restored: np.ndarray,
    fs: float,
    *,
    enabled: bool = True,
    baseline_window_ms: float = _GONZALEZ_GUARDED_DOWNWARD_BASELINE_WINDOW_MS,
    repair_window_ms: float = _GONZALEZ_GUARDED_DOWNWARD_REPAIR_WINDOW_MS,
    max_width_ms: float = _GONZALEZ_GUARDED_DOWNWARD_MAX_WIDTH_MS,
    min_peak_distance_ms: float = 2.0,
    raw_support_ratio: float = 0.60,
    pre_restore_support_ratio: float = 0.60,
) -> Tuple[np.ndarray, dict]:
    source = np.asarray(raw, dtype=float)
    pre = np.asarray(pre_restore, dtype=float)
    output = np.asarray(restored, dtype=float).copy()
    debug: dict[str, Any] = {
        "enabled": bool(enabled),
        "status": "not_evaluated",
        "repair_count": 0,
        "repairs": [],
    }
    if not bool(enabled):
        debug["status"] = "disabled"
        return output, debug
    if source.shape != pre.shape or source.shape != output.shape or source.size == 0:
        debug["status"] = "shape_mismatch_or_empty"
        return output, debug
    finite = np.isfinite(source) & np.isfinite(pre) & np.isfinite(output)
    if np.count_nonzero(finite) < 3:
        debug["status"] = "no_samples_evaluated"
        return output, debug

    hp_window = _odd_window_ms(float(baseline_window_ms), fs)
    hp_window = min(hp_window, source.size if source.size % 2 == 1 else max(1, source.size - 1))
    if hp_window < 3:
        raw_high = source - float(np.median(source[finite]))
        pre_high = pre - float(np.median(pre[finite]))
        out_high = output - float(np.median(output[finite]))
    else:
        raw_high = source - median_filter(source, size=hp_window, mode="nearest")
        pre_high = pre - median_filter(pre, size=hp_window, mode="nearest")
        out_high = output - median_filter(output, size=hp_window, mode="nearest")

    selected = out_high[finite]
    noise = max(_robust_sigma(selected), 1e-12)
    max_abs = float(np.max(np.abs(selected))) if selected.size else 0.0
    trough_threshold = max(4.0 * noise, 0.20 * max_abs, 1e-12)
    distance = max(1, int(round(float(min_peak_distance_ms) * float(fs) / 1000.0)))
    max_width_samples = max(1, int(round(float(max_width_ms) * float(fs) / 1000.0)))
    trough_signal = np.where(finite, -out_high, 0.0)
    troughs, props = find_peaks(
        trough_signal,
        height=trough_threshold,
        distance=distance,
        width=(1, max_width_samples),
    )
    troughs = np.asarray([int(p) for p in troughs if finite[int(p)]], dtype=int)
    if troughs.size == 0:
        debug.update(
            {
                "status": "evaluated",
                "threshold": float(trough_threshold),
                "baseline_window_samples": int(hp_window),
                "repair_window_samples": int(max(1, int(round(float(repair_window_ms) * float(fs) / 1000.0)))),
                "max_width_samples": int(max_width_samples),
            }
        )
        return output, debug

    repair_half_width = max(1, int(round(float(repair_window_ms) * float(fs) / 1000.0)))
    widths = np.asarray(props.get("widths", np.full(troughs.shape, np.nan)), dtype=float)
    repairs: List[dict] = []
    repaired_mask = np.zeros(output.shape, dtype=bool)
    for row, trough in enumerate(troughs):
        raw_drop = max(0.0, -float(raw_high[int(trough)]))
        pre_drop = max(0.0, -float(pre_high[int(trough)]))
        out_drop = max(0.0, -float(out_high[int(trough)]))
        if out_drop < trough_threshold:
            continue

        target: np.ndarray | None = None
        reason = ""
        raw_supports_trough = raw_drop >= float(raw_support_ratio) * out_drop
        pre_suppressed_trough = pre_drop <= float(pre_restore_support_ratio) * max(raw_drop, 1e-12)
        raw_lacks_trough = raw_drop <= float(raw_support_ratio) * out_drop
        if raw_supports_trough and pre_suppressed_trough:
            target = pre
            reason = "raw_negative_transient_not_restored"
        elif raw_lacks_trough:
            target = source
            reason = "denoised_only_negative_trough_repaired_to_raw"
        else:
            continue

        start = max(0, int(trough) - repair_half_width)
        stop = min(output.size, int(trough) + repair_half_width + 1)
        if stop <= start:
            continue
        width = stop - start
        if width <= 1:
            output[start:stop] = target[start:stop]
            weights = np.ones(width, dtype=float)
        else:
            x = np.linspace(-1.0, 1.0, width)
            weights = 0.5 * (1.0 + np.cos(np.pi * np.clip(x, -1.0, 1.0)))
            output[start:stop] = (weights * target[start:stop]) + ((1.0 - weights) * output[start:stop])
        repaired_mask[start:stop] = True
        repairs.append(
            {
                "trough": int(trough),
                "start": int(start),
                "stop": int(stop),
                "reason": reason,
                "target": "pre_restore" if target is pre else "raw",
                "raw_drop": float(raw_drop),
                "pre_restore_drop": float(pre_drop),
                "restored_drop": float(out_drop),
                "width_samples": float(widths[row]) if row < widths.size else float("nan"),
                "max_blend_weight": float(np.max(weights)) if weights.size else float("nan"),
            }
        )

    debug.update(
        {
            "status": "evaluated",
            "repair_count": int(len(repairs)),
            "repaired_sample_count": int(np.count_nonzero(repaired_mask)),
            "repairs": repairs,
            "threshold": float(trough_threshold),
            "baseline_window_samples": int(hp_window),
            "repair_window_samples": int(repair_half_width),
            "max_width_samples": int(max_width_samples),
        }
    )
    return output, debug


def _preserve_low_frequency_envelope(
    raw: np.ndarray,
    candidate: np.ndarray,
    fs: float,
    envelope_window_ms: float = 100.0,
) -> Tuple[np.ndarray, dict]:
    source = np.asarray(raw, dtype=float)
    output = np.asarray(candidate, dtype=float)
    debug: dict[str, Any] = {
        "enabled": True,
        "status": "not_evaluated",
        "envelope_window_ms": float(envelope_window_ms),
    }
    if source.shape != output.shape or source.size == 0:
        debug["status"] = "shape_mismatch_or_empty"
        return output.copy(), debug

    window = _bounded_odd_window_ms(float(envelope_window_ms), fs, source.size)
    raw_envelope = median_filter(source, size=window, mode="nearest")
    output_envelope = median_filter(output, size=window, mode="nearest")
    correction = raw_envelope - output_envelope
    restored = np.asarray(output + correction, dtype=float)
    debug.update(
        {
            "status": "applied",
            "envelope_window_samples": int(window),
            "median_abs_envelope_correction": float(np.median(np.abs(correction))) if correction.size else float("nan"),
            "max_abs_envelope_correction": float(np.max(np.abs(correction))) if correction.size else float("nan"),
            "raw_envelope_median": float(np.median(raw_envelope)) if raw_envelope.size else float("nan"),
            "output_envelope_median_before": float(np.median(output_envelope)) if output_envelope.size else float("nan"),
            "output_envelope_median_after": (
                float(np.median(median_filter(restored, size=window, mode="nearest"))) if restored.size else float("nan")
            ),
        }
    )
    return restored, debug


def normalize_gonzalez_denoising_method(value: str) -> str:
    method = str(value).strip().lower()
    if method in {"none", "off", "disabled"}:
        return "none"
    if method in {"gonzalez_s7_dff", "gonzalez_s7", "s7", "s7_dff", "gonzalez_s7_reproduction"}:
        return "gonzalez_s7_dff"
    if method in {
        "gonzalez_full_trace_guarded",
        "full_trace_guarded",
        "guarded",
        "artifact_guarded",
        "gonzalez_guarded",
    }:
        return "gonzalez_full_trace_guarded"
    if method in {"gonzalez_full_trace_dff", "full_trace_dff", "full_trace_gonzalez_dff", "gonzalez_dff"}:
        return "gonzalez_full_trace_dff"
    if method in {"gonzalez_full_trace", "full_trace_gonzalez", "full_gonzalez", "gonzalez_all_states", ""}:
        return "gonzalez_full_trace"
    if method in {"gonzalez_adaptive_wavelet", "legacy_hybrid", "hybrid", "legacy"}:
        # Removed modes are migrated to the current full-trace Gonzalez route
        # instead of being silently preserved.
        return "gonzalez_full_trace"
    raise ValueError(
        "denoising_method must be one of: gonzalez_full_trace, "
        "gonzalez_full_trace_guarded, gonzalez_full_trace_dff, gonzalez_s7_dff, none"
    )


def denoise_trace_gonzalez_full_trace(
    corrected: np.ndarray,
    fs: float,
    denoising_method: str = "gonzalez_full_trace",
    normalization_baseline: np.ndarray | None = None,
    enable_denoise_safety_gate: bool = False,
    min_event_peak_preservation: float = 0.95,
    min_local_max_ratio: float = 0.95,
    return_debug: bool = False,
    **gonzalez_kwargs: object,
) -> np.ndarray | dict:
    """Run the active full-trace Gonzalez denoiser.

    ``gonzalez_full_trace`` denoises the corrected trace directly.
    ``gonzalez_full_trace_guarded`` uses the same corrected-trace route, then
    applies positive-event restoration and a local downward-artifact guard.
    ``gonzalez_full_trace_dff`` first normalizes by the detector baseline and
    converts the denoised result back to corrected units. ``gonzalez_s7_dff``
    uses the same dF/F0 wrapper but enables the S7-style Gonzalez settings.
    ``none`` is a passthrough. State-guided low-state/plateau-TV denoising is
    intentionally not part of this active path.
    """
    trace = _validate_trace(corrected, fs)
    method = normalize_gonzalez_denoising_method(denoising_method)
    normalization_id = "dff_detector_baseline" if method in {"gonzalez_full_trace_dff", "gonzalez_s7_dff"} else "none"
    normalization_label = "dF/F0 using detector baseline" if normalization_id == "dff_detector_baseline" else "None"
    if "enable_low_state_safety_gate" in gonzalez_kwargs:
        enable_denoise_safety_gate = bool(gonzalez_kwargs.pop("enable_low_state_safety_gate"))
    if method == "none":
        if not return_debug:
            return trace.copy()
        return {
            "denoiser_used": "none",
            "denoising_method": "none",
            "candidate_used": False,
            "fallback_triggered": False,
            "fallback_reasons": [],
            "input_normalization": "none",
            "input_normalization_label": "None",
            "normalization_source": None,
            "output": trace.copy(),
            "gonzalez_input": trace.copy(),
            "gonzalez_output": trace.copy(),
            "gonzalez_debug": None,
            "normalization_baseline": None,
            "working_dff": None,
            "quantitative_measurement_note": (
                "Denoising is disabled; raw corrected trace is used for detection and measurements."
            ),
        }

    uses_s7 = method == "gonzalez_s7_dff"
    uses_guarded = method == "gonzalez_full_trace_guarded"
    uses_dff = method in {"gonzalez_full_trace_dff", "gonzalez_s7_dff"}
    normalization_baseline_validation: dict[str, Any] = {"valid": True, "reason": None}
    if uses_dff:
        if normalization_baseline is None:
            raise ValueError("normalization_baseline is required for gonzalez_full_trace_dff")
        f0 = _validate_trace(np.asarray(normalization_baseline, dtype=float), fs)
        if f0.shape != trace.shape:
            raise ValueError("normalization_baseline must have the same shape as corrected")
        normalization_baseline_validation = _dff_normalization_baseline_validation(f0)
        safe_f0 = _safe_f0_denominator(f0)
        gonzalez_input = trace / safe_f0
    else:
        f0 = None
        safe_f0 = None
        gonzalez_input = trace.copy()

    allowed = {
        "rolling_baseline_seconds",
        "wavelet",
        "wavelet_precision",
        "min_freq",
        "max_freq",
        "n_freqs",
        "max_clusters",
        "cluster_selection_strategy",
        "cluster_selection_relative_score",
        "threshold_sd",
        "coefficient_threshold_domain",
        "threshold_mask_padding_periods",
        "attenuation_min",
        "attenuation_max",
        "enable_noise_cluster_rejection",
        "enable_event_template_rejection",
        "event_template_rejection_mode",
        "event_attenuation_min",
        "event_attenuation_max",
        "event_rejection_application",
        "event_detection_z_threshold",
        "event_detection_prominence_z",
        "event_detection_min_distance_ms",
        "event_detection_polarity",
        "event_detection_use_threshold_crossings",
        "enable_coefficient_event_protection",
        "coefficient_event_protection_window_ms",
        "enable_low_frequency_gaussian_branch",
        "low_frequency_cutoff_hz",
        "inverse_cwt_regularization",
        "reconstruction_mode",
        "removed_reconstruction_gain_mode",
        "removed_reconstruction_gain_scale",
        "enable_denoise_safety_gate",
        "min_event_peak_preservation",
        "min_local_max_ratio",
        "event_restore_polarity",
        "enable_downward_artifact_guard",
    }
    kwargs = {key: value for key, value in gonzalez_kwargs.items() if key in allowed}
    event_restore_polarity = _normalize_event_polarity(str(kwargs.pop("event_restore_polarity", "absolute")))
    enable_downward_artifact_guard = bool(kwargs.pop("enable_downward_artifact_guard", False))
    if uses_s7:
        kwargs["min_freq"] = 3.0
        kwargs.setdefault("threshold_sd", 2.0)
        kwargs.setdefault("n_freqs", 80)
        kwargs["max_clusters"] = _GONZALEZ_S7_WARD_MAX_CLUSTERS
        kwargs["cluster_selection_strategy"] = "parsimonious_silhouette"
        kwargs["cluster_selection_relative_score"] = _GONZALEZ_S7_CLUSTER_SELECTION_RELATIVE_SCORE
        kwargs["attenuation_min"] = _GONZALEZ_S7_ATTENUATION_MIN
        kwargs["attenuation_max"] = _GONZALEZ_S7_ATTENUATION_MAX
        kwargs["coefficient_threshold_domain"] = "normalized_magnitude"
        kwargs["threshold_mask_padding_periods"] = _GONZALEZ_S7_THRESHOLD_MASK_PADDING_PERIODS
        kwargs["event_attenuation_min"] = _GONZALEZ_S7_EVENT_ATTENUATION_MIN
        kwargs["event_attenuation_max"] = _GONZALEZ_S7_EVENT_ATTENUATION_MAX
        kwargs["enable_coefficient_event_protection"] = False
        kwargs["enable_noise_cluster_rejection"] = True
        kwargs["enable_event_template_rejection"] = True
        kwargs["event_template_rejection_mode"] = "pc1_negative"
        kwargs["event_rejection_application"] = "coefficient"
        kwargs["event_detection_z_threshold"] = _GONZALEZ_S7_EVENT_DETECTION_Z
        kwargs["event_detection_prominence_z"] = _GONZALEZ_S7_EVENT_DETECTION_PROMINENCE_Z
        kwargs["event_detection_min_distance_ms"] = _GONZALEZ_S7_EVENT_DETECTION_MIN_DISTANCE_MS
        kwargs["event_detection_polarity"] = "positive"
        kwargs["event_detection_use_threshold_crossings"] = False
        kwargs["enable_low_frequency_gaussian_branch"] = True
        kwargs["low_frequency_cutoff_hz"] = 1.0
        kwargs["reconstruction_mode"] = "residual_subtractive"
        kwargs["removed_reconstruction_gain_mode"] = "unit"
    if uses_guarded:
        event_restore_polarity = "positive"
        enable_downward_artifact_guard = True
    kwargs.update(
        {
            "input_mode": "corrected",
            "enable_denoise_safety_gate": bool(enable_denoise_safety_gate),
            "min_event_peak_preservation": float(min_event_peak_preservation),
            "min_local_max_ratio": float(min_local_max_ratio),
        }
    )

    gonzalez_debug = None
    fallback_reasons: List[str] = []
    if uses_dff and not bool(normalization_baseline_validation.get("valid", True)):
        fallback_reasons.append(str(normalization_baseline_validation.get("reason") or "invalid_dff_normalization_baseline"))
        output = trace.copy()
        gonzalez_output = gonzalez_input.copy()
        if return_debug:
            gonzalez_debug = {
                "skipped": True,
                "reason": "invalid_dff_normalization_baseline",
                "normalization_baseline_validation": normalization_baseline_validation,
                "output": gonzalez_input.copy(),
            }
    else:
        try:
            result = denoise_gonzalez_adaptive_wavelet(
                gonzalez_input,
                fs=fs,
                return_debug=return_debug,
                **kwargs,
            )
            if return_debug:
                gonzalez_debug = dict(result)
                gonzalez_output = np.asarray(gonzalez_debug.get("output", gonzalez_input), dtype=float)
            else:
                gonzalez_output = np.asarray(result, dtype=float)
            output = np.asarray(gonzalez_output * safe_f0 if uses_dff else gonzalez_output, dtype=float)
            if output.shape != trace.shape:
                fallback_reasons.append("shape_mismatch")
            elif output.size and not np.all(np.isfinite(output)):
                fallback_reasons.append("non_finite_output")
        except Exception as exc:
            output = trace.copy()
            gonzalez_output = gonzalez_input.copy()
            fallback_reasons.append("gonzalez_exception")
            if return_debug:
                gonzalez_debug = {"error": str(exc)}

    level_restore: dict[str, Any] = {"enabled": False, "status": "not_run"}
    amplitude_restore: dict[str, Any] = {"enabled": False, "status": "not_run"}
    downward_artifact_guard: dict[str, Any] = {"enabled": False, "status": "not_run"}
    morphology = {"status": "not_evaluated", "fallback_reasons": []}
    if not fallback_reasons:
        if uses_s7:
            level_restore = {"enabled": False, "status": "skipped_for_s7_reproduction"}
            amplitude_restore = {"enabled": False, "status": "skipped_for_paper_s7_reproduction"}
            downward_artifact_guard = {"enabled": False, "status": "skipped_for_s7_reproduction"}
        else:
            output, level_restore = _preserve_low_frequency_envelope(trace, output, fs=fs)
            pre_amplitude_restore_output = np.asarray(output, dtype=float).copy()
            output, amplitude_restore = _restore_event_amplitudes(
                trace,
                output,
                fs=fs,
                evaluation_mask=np.ones(trace.shape, dtype=bool),
                min_event_peak_preservation=float(min_event_peak_preservation),
                min_local_max_ratio=float(min_local_max_ratio),
                event_polarity=event_restore_polarity,
            )
            output, downward_artifact_guard = _repair_downward_denoising_artifacts(
                trace,
                pre_amplitude_restore_output,
                output,
                fs=fs,
                enabled=bool(enable_downward_artifact_guard),
            )
        morphology = _event_preservation_metrics(
            trace,
            output,
            fs=fs,
            evaluation_mask=np.ones(trace.shape, dtype=bool),
            min_event_peak_preservation=float(min_event_peak_preservation),
            min_local_max_ratio=float(min_local_max_ratio),
            event_polarity=event_restore_polarity,
        )
        if bool(enable_denoise_safety_gate):
            fallback_reasons.extend(str(reason) for reason in morphology.get("fallback_reasons", []))

    fallback_triggered = bool(fallback_reasons)
    if fallback_triggered:
        output = trace.copy()
    if not return_debug:
        return output

    return {
        "denoiser_used": method,
        "denoising_method": method,
        "s7_reproduction_preset": (
            {
                "enabled": True,
                "frequency_cluster_mode": "ward_silhouette_up_to_20_clusters",
                "ward_max_clusters": _GONZALEZ_S7_WARD_MAX_CLUSTERS,
                "cluster_selection_strategy": "parsimonious_silhouette",
                "cluster_selection_relative_score": _GONZALEZ_S7_CLUSTER_SELECTION_RELATIVE_SCORE,
                "signal_attenuation_range": (
                    _GONZALEZ_S7_ATTENUATION_MIN,
                    _GONZALEZ_S7_ATTENUATION_MAX,
                ),
                "event_attenuation_range": (
                    _GONZALEZ_S7_EVENT_ATTENUATION_MIN,
                    _GONZALEZ_S7_EVENT_ATTENUATION_MAX,
                ),
                "event_protected_thresholding": False,
                "coefficient_threshold_domain": "normalized_magnitude",
                "threshold_mask_padding_periods": _GONZALEZ_S7_THRESHOLD_MASK_PADDING_PERIODS,
                "coefficient_event_protection": False,
                "noise_cluster_rejection": True,
                "event_template_rejection_mode": "pc1_negative",
                "event_rejection_application": "coefficient_time_mask",
                "event_detection": {
                    "z_threshold": _GONZALEZ_S7_EVENT_DETECTION_Z,
                    "prominence_z": _GONZALEZ_S7_EVENT_DETECTION_PROMINENCE_Z,
                    "min_distance_ms": _GONZALEZ_S7_EVENT_DETECTION_MIN_DISTANCE_MS,
                    "polarity": "positive",
                    "use_threshold_crossings": False,
                },
                "low_frequency_branch_hz": 1.0,
                "removed_reconstruction_gain_mode": "unit",
                "post_processing_merges": "disabled",
                "note": (
                    "Paper S7 reproduction: dF/F input, Ward/silhouette frequency clustering "
                    "up to 20 clusters with a broad-domain silhouette rule, 2 SD adaptive "
                    "thresholding at the paper's 0.5 attenuation endpoint, "
                    "PC1-negative noisy-event attenuation at the soft 0.5 endpoint, and "
                    "a 1 Hz low-frequency branch."
                ),
            }
            if uses_s7
            else {"enabled": False}
        ),
        "guarded_artifact_preset": (
            {
                "enabled": True,
                "base_method": "gonzalez_full_trace",
                "event_restore_polarity": "positive",
                "downward_artifact_guard": True,
                "note": (
                    "Original corrected-trace Gonzalez denoising with positive-event restoration "
                    "and a local guard for sharp downward artifacts."
                ),
            }
            if uses_guarded
            else {"enabled": False}
        ),
        "candidate_used": not fallback_triggered,
        "fallback_triggered": fallback_triggered,
        "fallback_reasons": fallback_reasons,
        "input_normalization": normalization_id,
        "input_normalization_label": normalization_label,
        "normalization_source": "detector_baseline" if uses_dff else None,
        "normalization_baseline_validation": normalization_baseline_validation,
        "shared_core_note": "Input normalization changes the denoiser input only; the Gonzalez core is shared.",
        "input_output_max_abs_diff": (
            float(np.max(np.abs(output - trace))) if output.shape == trace.shape and output.size else float("nan")
        ),
        "input_output_residual_std": (
            float(np.std(output - trace)) if output.shape == trace.shape and output.size else float("nan")
        ),
        "output": output,
        "gonzalez_input": gonzalez_input.copy(),
        "gonzalez_output": gonzalez_output.copy(),
        "gonzalez_debug": gonzalez_debug,
        "normalization_baseline": None if f0 is None else f0.copy(),
        "working_dff": gonzalez_input.copy() if uses_dff else None,
        "safety": {
            "fallback_triggered": fallback_triggered,
            "fallback_reasons": fallback_reasons,
            "safety_gate_enabled": bool(enable_denoise_safety_gate),
            "level_preservation_merge": level_restore,
            "amplitude_preservation_merge": amplitude_restore,
            "downward_artifact_guard": downward_artifact_guard,
            "morphology_preservation": morphology,
        },
        "quantitative_measurement_note": (
            "Denoised trace is a detection aid; raw corrected trace is preferred for quantitative "
            "amplitude, peak height, width, area, plateau level, and summary statistics."
        ),
    }
