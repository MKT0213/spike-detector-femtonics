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
) -> np.ndarray:
    # Under-specified implementation detail: PyWavelets has CWT but no fully
    # standard inverse CWT. This approximate synthesis follows the common
    # scale-normalized coefficient summation strategy and is calibrated later
    # against the original dF/F trace for amplitude diagnostics.
    if coeffs.size == 0:
        return np.asarray([], dtype=float)
    selected = np.asarray(coeffs)
    scale_values = np.maximum(np.asarray(scales, dtype=float), 1e-12)
    weighted = np.real(selected) / np.sqrt(scale_values)[:, None]
    weight_sum = float(np.sum(1.0 / np.sqrt(scale_values)))
    if weight_sum <= 1e-12:
        return np.zeros(selected.shape[1], dtype=float)
    return np.sum(weighted, axis=0) / weight_sum


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
        "inverse_cwt_note": "Approximate PyWavelets inverse CWT calibrated by least-squares gain/offset.",
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
) -> Tuple[np.ndarray, dict]:
    # Under-specified in Gonzalez et al.: candidate event detection and the PC
    # cutoff are not given. This implementation detects both polarities from a
    # robust z-score, aligns event windows by polarity, uses PC1 as the event
    # template, and rejects low-template-correlation windows.
    trace = np.asarray(cluster_trace, dtype=float)
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
        "note": "Event detection and PC cutoff are under-specified by the paper.",
    }
    if not enabled or trace.size < 5:
        return trace.copy(), debug
    center = float(np.median(trace))
    sigma = _robust_sigma(trace - center)
    if sigma <= 1e-12:
        return trace.copy(), debug
    z = (trace - center) / sigma
    distance = max(1, int(round(0.003 * float(fs))))
    peaks, _props = find_peaks(np.abs(z), height=3.0, distance=distance, prominence=1.0)
    if peaks.size < 3:
        debug["event_indices"] = np.asarray(peaks, dtype=int)
        return trace.copy(), debug
    half_width = int(round(float(fs) / max(float(max_frequency), 1e-9)))
    half_width = int(np.clip(half_width, 2, max(2, int(round(0.05 * float(fs))))))
    windows = _event_windows(trace - center, np.asarray(peaks, dtype=int), half_width)
    polarities = np.where(z[peaks] < 0.0, -1.0, 1.0)
    aligned = windows * polarities[:, None]
    scale = np.maximum(aligned.std(axis=1, keepdims=True), 1e-12)
    normalized = (aligned - aligned.mean(axis=1, keepdims=True)) / scale
    if normalized.shape[0] < 3 or normalized.shape[1] < 2:
        debug["event_indices"] = np.asarray(peaks, dtype=int)
        debug["event_windows"] = windows
        return trace.copy(), debug
    pca = PCA(n_components=1, svd_solver="full")
    coefficients = pca.fit_transform(normalized).reshape(-1)
    template = np.asarray(pca.components_[0], dtype=float)
    if np.dot(template, normalized.mean(axis=0)) < 0.0:
        template = -template
        coefficients = -coefficients
    template /= np.linalg.norm(template) + 1e-12
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


def denoise_gonzalez_adaptive_wavelet(
    trace,
    fs,
    input_mode="fluorescence",
    rolling_baseline_seconds=4.0,
    wavelet="cmor1.5-1.0",
    min_freq=1.0,
    max_freq=None,
    n_freqs=100,
    max_clusters=20,
    threshold_sd=2.0,
    attenuation_min=0.5,
    attenuation_max=1.0,
    enable_noise_cluster_rejection=False,
    enable_event_template_rejection=False,
    event_template_rejection_mode="correlation",
    enable_low_frequency_gaussian_branch=False,
    low_frequency_cutoff_hz=1.0,
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
    PyWavelets does not provide a standard inverse CWT here, so reconstruction
    is approximate plus calibration rather than mathematically exact.
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
            "silhouette_scores": {},
            "rejected_frequency_clusters": [],
            "threshold_per_cluster": {},
            "attenuation_mask_per_cluster": {},
            "reconstructed_cluster_traces": {},
            "event_template_rejection_results": {},
            "enable_noise_cluster_rejection": bool(enable_noise_cluster_rejection),
            "enable_event_template_rejection": bool(enable_event_template_rejection),
            "event_template_rejection_mode": str(event_template_rejection_mode),
            "low_frequency_gaussian_branch": {"enabled": False, "status": "empty_trace"},
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
    )
    coeffs = np.asarray(coeffs)
    magnitudes = np.abs(coeffs)

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
    )
    cluster_info = _gonzalez_cluster_thresholds(magnitudes, frequencies, labels, fs, threshold_sd)
    rejected_clusters, rejection_decisions = _gonzalez_rejected_frequency_clusters(
        cluster_info,
        max_frequency=float(np.max(frequencies)),
        threshold_sd=threshold_sd,
        enabled=bool(enable_noise_cluster_rejection),
    )

    raw_reconstruction = _gonzalez_inverse_cwt(coeffs, scales, wavelet)
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

    attenuation_min = float(np.clip(attenuation_min, 0.0, 1.0))
    attenuation_max = float(np.clip(attenuation_max, attenuation_min, 1.0))
    for cluster_id, info in cluster_info.items():
        indices = np.asarray(info["indices"], dtype=int)
        identity_coeffs_full[indices] = coeffs[indices]
        threshold = np.asarray(info["threshold"], dtype=float)
        activity = np.asarray(info["activity"], dtype=float)
        threshold_debug[int(cluster_id)] = {
            "activity": activity.copy(),
            "threshold": threshold,
            "moving_average": np.asarray(info["moving_average"], dtype=float),
            "residual_std": float(info["residual_std"]),
            "moving_average_window_samples": int(info["moving_average_window_samples"]),
            "max_frequency": float(info["max_frequency"]),
            "frequency_indices": indices.copy(),
        }
        if int(cluster_id) in rejected_clusters:
            attenuation_debug[int(cluster_id)] = np.zeros(values.shape, dtype=float)
            cleaned_cluster_traces[int(cluster_id)] = np.zeros(values.shape, dtype=float)
            event_debug[int(cluster_id)] = {"enabled": False, "reason": "frequency_cluster_rejected_as_noise"}
            continue

        retained_cluster_ids.append(int(cluster_id))
        below = activity <= threshold
        gap = np.maximum(threshold - activity, 0.0)
        denom = max(float(np.max(gap[below])) if np.any(below) else 0.0, 1e-12)
        severity = np.clip(gap / denom, 0.0, 1.0)
        attenuation_fraction = attenuation_min + severity * (attenuation_max - attenuation_min)
        gain_mask = np.where(below, 1.0 - attenuation_fraction, 1.0)
        attenuation_debug[int(cluster_id)] = gain_mask.copy()

        # Paper-specified masking: attenuate the original complex coefficients
        # for the cluster, not the magnitude-only representation.
        cleaned_cluster_coeffs = coeffs[indices] * gain_mask[None, :]
        reconstructed = _gonzalez_inverse_cwt(cleaned_cluster_coeffs, scales[indices], wavelet)
        event_processed, event_info = _gonzalez_event_template_rejection(
            reconstructed,
            fs=fs,
            cluster_id=int(cluster_id),
            max_frequency=float(info["max_frequency"]),
            attenuation_min=attenuation_min,
            attenuation_max=attenuation_max,
            enabled=bool(enable_event_template_rejection),
            mode=str(event_template_rejection_mode),
        )
        event_gain = np.asarray(event_info.get("event_gain", np.ones(values.shape, dtype=float)), dtype=float)
        if event_gain.shape != values.shape:
            event_gain = np.ones(values.shape, dtype=float)
            event_info["event_gain"] = event_gain
            event_info["event_gain_shape_warning"] = "event_gain shape mismatch; coefficient event attenuation skipped"
        cleaned_cluster_coeffs = cleaned_cluster_coeffs * event_gain[None, :]
        cleaned_coeffs_full[indices] = cleaned_cluster_coeffs
        cleaned_cluster_traces[int(cluster_id)] = event_processed
        event_debug[int(cluster_id)] = event_info

    raw_full = _gonzalez_inverse_cwt(coeffs, scales, wavelet)
    identity_full = _gonzalez_inverse_cwt(identity_coeffs_full, scales, wavelet)
    consistency_error = (
        float(np.max(np.abs(raw_full - identity_full))) if raw_full.shape == identity_full.shape and raw_full.size else 0.0
    )
    denoised_raw = _gonzalez_inverse_cwt(cleaned_coeffs_full, scales, wavelet)
    denoised_working_trace = raw_offset + (raw_gain * denoised_raw)
    if bool(enable_low_frequency_gaussian_branch):
        denoised_working_trace = denoised_working_trace + low_frequency_branch
    full_matrix_residual = denoised_working_trace - working_trace
    amplitude_diagnostics["max_abs_reconstruction_consistency_error"] = consistency_error
    amplitude_diagnostics["full_matrix_reconstruction"] = {
        "calibration": "raw_reconstruction_gain_offset_applied_to_full_cleaned_matrix",
        "gain": raw_gain,
        "offset": raw_offset,
        "raw_reconstruction_std": float(np.std(denoised_raw)),
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
        "threshold_sd": float(threshold_sd),
        "attenuation_min": float(attenuation_min),
        "attenuation_max": float(attenuation_max),
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
        "low_frequency_gaussian_branch": low_frequency_debug,
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
            "inverse_cwt": "Approximate PyWavelets inverse CWT with amplitude diagnostics; not guaranteed to be quantitative amplitude-preserving reconstruction.",
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
    masked_abs_raw = np.where(finite, np.abs(raw_high), 0.0)
    raw_peaks, _ = find_peaks(masked_abs_raw, height=peak_threshold, distance=distance)
    raw_peaks = np.asarray([int(p) for p in raw_peaks if finite[int(p)]], dtype=int)
    masked_abs_candidate = np.where(finite, np.abs(cand_high), 0.0)
    candidate_peaks, _ = find_peaks(masked_abs_candidate, height=peak_threshold, distance=distance)
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
) -> Tuple[np.ndarray, dict]:
    source = np.asarray(raw, dtype=float)
    output = np.asarray(candidate, dtype=float).copy()
    mask = np.asarray(evaluation_mask, dtype=bool)
    debug: dict[str, Any] = {
        "enabled": True,
        "status": "not_evaluated",
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
    masked_abs_raw = np.where(finite, np.abs(raw_high), 0.0)
    raw_peaks, _ = find_peaks(masked_abs_raw, height=peak_threshold, distance=distance)
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
    if method in {"gonzalez_full_trace_dff", "full_trace_dff", "full_trace_gonzalez_dff", "gonzalez_dff"}:
        return "gonzalez_full_trace_dff"
    if method in {"gonzalez_full_trace", "full_trace_gonzalez", "full_gonzalez", "gonzalez_all_states", ""}:
        return "gonzalez_full_trace"
    if method in {"gonzalez_adaptive_wavelet", "legacy_hybrid", "hybrid", "legacy"}:
        # Removed modes are migrated to the current full-trace Gonzalez route
        # instead of being silently preserved.
        return "gonzalez_full_trace"
    raise ValueError("denoising_method must be one of: gonzalez_full_trace, gonzalez_full_trace_dff, gonzalez_s7_dff, none")


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
    uses_dff = method in {"gonzalez_full_trace_dff", "gonzalez_s7_dff"}
    if uses_dff:
        if normalization_baseline is None:
            raise ValueError("normalization_baseline is required for gonzalez_full_trace_dff")
        f0 = _validate_trace(np.asarray(normalization_baseline, dtype=float), fs)
        if f0.shape != trace.shape:
            raise ValueError("normalization_baseline must have the same shape as corrected")
        safe_f0 = _safe_f0_denominator(f0)
        gonzalez_input = trace / safe_f0
    else:
        f0 = None
        safe_f0 = None
        gonzalez_input = trace.copy()

    allowed = {
        "rolling_baseline_seconds",
        "wavelet",
        "min_freq",
        "max_freq",
        "n_freqs",
        "max_clusters",
        "threshold_sd",
        "attenuation_min",
        "attenuation_max",
        "enable_noise_cluster_rejection",
        "enable_event_template_rejection",
        "event_template_rejection_mode",
        "enable_low_frequency_gaussian_branch",
        "low_frequency_cutoff_hz",
        "enable_denoise_safety_gate",
        "min_event_peak_preservation",
        "min_local_max_ratio",
    }
    kwargs = {key: value for key, value in gonzalez_kwargs.items() if key in allowed}
    if uses_s7:
        kwargs.setdefault("min_freq", 3.0)
        kwargs.setdefault("max_clusters", 20)
        kwargs.setdefault("threshold_sd", 2.0)
        kwargs.setdefault("attenuation_min", 0.5)
        kwargs.setdefault("attenuation_max", 1.0)
        kwargs["enable_noise_cluster_rejection"] = True
        kwargs["enable_event_template_rejection"] = True
        kwargs["event_template_rejection_mode"] = "pc1_negative"
        kwargs["enable_low_frequency_gaussian_branch"] = True
        kwargs["low_frequency_cutoff_hz"] = 1.0
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
    morphology = {"status": "not_evaluated", "fallback_reasons": []}
    if not fallback_reasons:
        if uses_s7:
            level_restore = {"enabled": False, "status": "skipped_for_s7_reproduction"}
            amplitude_restore = {"enabled": False, "status": "skipped_for_s7_reproduction"}
        else:
            output, level_restore = _preserve_low_frequency_envelope(trace, output, fs=fs)
            output, amplitude_restore = _restore_event_amplitudes(
                trace,
                output,
                fs=fs,
                evaluation_mask=np.ones(trace.shape, dtype=bool),
                min_event_peak_preservation=float(min_event_peak_preservation),
                min_local_max_ratio=float(min_local_max_ratio),
            )
        morphology = _event_preservation_metrics(
            trace,
            output,
            fs=fs,
            evaluation_mask=np.ones(trace.shape, dtype=bool),
            min_event_peak_preservation=float(min_event_peak_preservation),
            min_local_max_ratio=float(min_local_max_ratio),
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
        "candidate_used": not fallback_triggered,
        "fallback_triggered": fallback_triggered,
        "fallback_reasons": fallback_reasons,
        "input_normalization": normalization_id,
        "input_normalization_label": normalization_label,
        "normalization_source": "detector_baseline" if uses_dff else None,
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
            "morphology_preservation": morphology,
        },
        "quantitative_measurement_note": (
            "Denoised trace is a detection aid; raw corrected trace is preferred for quantitative "
            "amplitude, peak height, width, area, plateau level, and summary statistics."
        ),
    }
