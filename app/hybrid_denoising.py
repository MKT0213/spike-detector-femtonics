"""State-guided denoising for baseline-corrected fluorescence traces.

The active plateau-heavy denoising path is
``denoise_trace_state_guided_low_plateau_tv``: baseline-corrected input stays
in corrected units, protected plateau regions are bridged before low-state
Gonzalez adaptive wavelet denoising, plateau cores are TV-denoised separately,
and boundary samples are restored from the raw corrected trace. Legacy Hybrid
denoising remains available as a fallback and comparison method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pywt
from kneed import KneeLocator
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


@dataclass
class HybridWaveletConfig:
    wavelet: str = "cmor1.5-1.0"
    min_freq: float = 1.0
    max_freq: float = 500.0
    n_freqs: int = 100
    max_clusters: int = 4
    threshold_sd: float = 2.0
    mask_attenuation_min: float = 0.5
    mask_attenuation_max: float = 1.0
    event_mode: str = "svd"
    event_merge_gap_ms: float = 10.0
    event_min_count: int = 3
    event_attenuation_min: float = 0.5
    event_attenuation_max: float = 1.0


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


def _normalize_event_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"svd", "pca", "none"}:
        raise ValueError("event_mode must be one of: svd, pca, none")
    return mode


def _frequency_vector(fs: float, cfg: HybridWaveletConfig) -> np.ndarray:
    nyquist = 0.5 * float(fs)
    upper = min(float(cfg.max_freq), 0.95 * nyquist)
    lower = float(cfg.min_freq)
    if upper <= 0:
        raise ValueError("Hybrid sampling rate is too low for wavelet denoising")
    if lower <= 0 or upper <= lower:
        lower = max(1e-6, upper / max(4.0, float(cfg.n_freqs)))
    if int(cfg.n_freqs) < 2:
        raise ValueError("n_freqs must be at least 2")
    return np.geomspace(lower, upper, int(cfg.n_freqs))


def _morlet_cwt(trace: np.ndarray, fs: float, cfg: HybridWaveletConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = _frequency_vector(fs, cfg)
    scales = pywt.frequency2scale(cfg.wavelet, freqs / float(fs))
    coeffs, returned_freqs = pywt.cwt(
        data=trace,
        scales=scales,
        wavelet=cfg.wavelet,
        sampling_period=1.0 / float(fs),
        method="fft",
    )
    magnitude = np.abs(coeffs)
    scale = np.maximum(magnitude.std(axis=1, keepdims=True), 1e-8)
    normalized = (magnitude - magnitude.mean(axis=1, keepdims=True)) / scale
    return coeffs, normalized, np.asarray(returned_freqs, dtype=float)


def _frequency_features(normalized: np.ndarray) -> Tuple[np.ndarray, int]:
    data = np.nan_to_num(np.asarray(normalized, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if data.shape[0] <= 1:
        return np.zeros((data.shape[0], 1), dtype=float), 1
    pca = PCA(n_components=None, svd_solver="full")
    features = pca.fit_transform(data)
    n_pc = int(features.shape[1])
    if n_pc <= 1:
        return features, 1
    variance = np.nan_to_num(pca.explained_variance_, nan=0.0)
    elbow = KneeLocator(np.arange(1, n_pc + 1), variance, curve="convex", direction="decreasing").elbow
    return features, int(np.clip(int(elbow) if elbow is not None else n_pc, 2, n_pc))


def _frequency_domains(features: np.ndarray, n_pc: int, max_clusters: int) -> List[np.ndarray]:
    x = np.asarray(features[:, : max(1, int(n_pc))], dtype=float)
    count = int(x.shape[0])
    if count <= 2 or float(np.std(x)) <= 1e-12:
        return [np.arange(count, dtype=int)]
    labels_by_k: Dict[int, np.ndarray] = {}
    scores: List[float] = []
    max_k = min(int(max_clusters), count - 1)
    for k in range(2, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        labels_by_k[k] = labels
        scores.append(float(silhouette_score(x, labels)))
    if not scores:
        return [np.arange(count, dtype=int)]
    labels = labels_by_k[int(np.argmax(scores)) + 2]
    return [np.flatnonzero(labels == cluster_id) for cluster_id in np.unique(labels)]


def _inverse_cwt(coeffs: np.ndarray, scales: np.ndarray, cfg: HybridWaveletConfig, mask: np.ndarray | None = None) -> np.ndarray:
    selected = coeffs if mask is None else coeffs * mask
    psi, time = pywt.ContinuousWavelet(cfg.wavelet).wavefun()
    psi0 = psi[np.argmin(np.abs(time))]
    return np.real(np.sum(np.real(selected) / np.sqrt(scales)[:, None], axis=0) * (1.0 / psi0))


def _event_regions(activity: np.ndarray, threshold: np.ndarray, fs: float, cfg: HybridWaveletConfig) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    above = np.asarray(activity) > np.asarray(threshold)
    if not np.any(above):
        return [], np.asarray([], dtype=int)
    starts = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    stops = np.flatnonzero(above & np.concatenate((~above[1:], [True]))) + 1
    gap = max(0, int(round(float(cfg.event_merge_gap_ms) * float(fs) / 1000.0)))
    regions: List[Tuple[int, int]] = []
    start, stop = int(starts[0]), int(stops[0])
    for next_start, next_stop in zip(starts[1:], stops[1:]):
        if int(next_start) - stop <= gap:
            stop = int(next_stop)
        else:
            regions.append((start, stop))
            start, stop = int(next_start), int(next_stop)
    regions.append((start, stop))
    peaks = np.asarray([start + int(np.argmax(activity[start:stop])) for start, stop in regions], dtype=int)
    return regions, peaks


def _event_windows(trace: np.ndarray, peaks: np.ndarray, half_width: int) -> np.ndarray:
    width = (2 * int(half_width)) + 1
    padded = np.pad(trace, (half_width, half_width), mode="reflect" if trace.size > 1 else "edge")
    return np.asarray([padded[int(peak) : int(peak) + width] for peak in peaks], dtype=float)


def _attenuate_noise_events(
    trace: np.ndarray,
    regions: List[Tuple[int, int]],
    peaks: np.ndarray,
    fs: float,
    cfg: HybridWaveletConfig,
    protected_mask: np.ndarray | None = None,
) -> np.ndarray:
    mode = _normalize_event_mode(cfg.event_mode)
    if mode == "none" or peaks.size < int(cfg.event_min_count):
        return trace
    lengths = [stop - start for start, stop in regions if stop > start]
    half_width = max(1, int(round(float(np.mean(lengths))))) if lengths else max(1, int(round(0.025 * fs)))
    windows = _event_windows(trace, peaks, half_width)
    means = windows.mean(axis=1, keepdims=True)
    scales = windows.std(axis=1, keepdims=True)
    valid = scales[:, 0] > 1e-8
    if np.count_nonzero(valid) < int(cfg.event_min_count):
        return trace
    normalized = (windows[valid] - means[valid]) / scales[valid]
    if mode == "svd":
        _, _, vt = np.linalg.svd(normalized, full_matrices=False)
        template = vt[0]
        if np.dot(template, normalized.mean(axis=0)) < 0:
            template = -template
        template /= np.linalg.norm(template) + 1e-12
        scores_valid = (normalized @ template) / (np.linalg.norm(normalized, axis=1) + 1e-12)
        cutoff = 0.0
    else:
        pca = PCA(n_components=1, svd_solver="full")
        scores_valid = pca.fit_transform(normalized).reshape(-1)
        if np.dot(pca.components_[0], normalized.mean(axis=0)) < 0:
            scores_valid = -scores_valid
        cutoff = 0.0
    scores = np.full(peaks.shape, np.nan, dtype=float)
    scores[valid] = scores_valid
    noise = np.isfinite(scores) & (scores <= cutoff)
    if not np.any(noise):
        return trace
    severity = cutoff - scores[noise]
    severity /= max(float(np.max(severity)), 1e-12)
    attenuation = float(cfg.event_attenuation_min) + severity * (
        float(cfg.event_attenuation_max) - float(cfg.event_attenuation_min)
    )
    gains = np.ones(scores.shape, dtype=float)
    gains[noise] = 1.0 - attenuation
    mask = np.ones(trace.shape, dtype=float)
    for (start, stop), is_noise, gain in zip(regions, noise, gains):
        # Event-template rejection can mistake plateau edges for noise events.
        # Keep it disabled wherever a plateau or its padded boundary is present.
        protected = protected_mask is not None and np.any(protected_mask[start:stop])
        if is_noise and not protected:
            mask[start:stop] = np.minimum(mask[start:stop], float(gain))
    return trace * mask


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
    finite_scale = np.abs(f0[np.isfinite(f0)])
    reference = float(np.median(finite_scale)) if finite_scale.size else 1.0
    eps = max(1e-9, 1e-6 * reference)
    signs = np.where(f0 < 0.0, -1.0, 1.0)
    denominator = np.where(np.abs(f0) < eps, signs * eps, f0)
    return (trace - f0) / denominator, f0


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
    corr_cutoff = 0.25
    coeff_floor = np.percentile(coefficients, 5.0)
    rejected = np.flatnonzero((correlations < corr_cutoff) | ((coefficients < coeff_floor) & (correlations < 0.5)))
    gain = np.ones(trace.shape, dtype=float)
    if rejected.size:
        min_corr = float(np.min(correlations[rejected]))
        denom = max(corr_cutoff - min_corr, 1e-12)
        for event_row in rejected:
            severity = float(np.clip((corr_cutoff - correlations[event_row]) / denom, 0.0, 1.0))
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
            "cutoff": {"template_correlation": corr_cutoff, "pc_coefficient_5th_percentile": float(coeff_floor)},
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
    enable_low_state_safety_gate=True,
    min_event_peak_preservation=0.80,
    min_local_max_ratio=0.80,
    return_debug=False,
):
    """Denoise with Gonzalez adaptive wavelet thresholding.

    ``input_mode="fluorescence"`` computes a rolling F0 and works in dF/F.
    ``input_mode="corrected"`` uses the provided baseline-corrected trace
    directly and returns corrected-trace units. The older ``"dff"`` spelling is
    accepted only as a compatibility alias for ``"corrected"``. Direct
    Gonzalez is not recommended for full plateau-heavy corrected traces; use
    ``denoise_trace_state_guided_low_plateau_tv`` as the safer default.
    PyWavelets does not provide a standard inverse CWT here, so reconstruction
    is approximate plus calibration rather than mathematically exact.
    Morphology safety gates should pass before denoised output is used as a
    detection aid.
    """
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

    # Paper-specified CWT: Morlet continuous wavelet transform of the working
    # trace. For input_mode="corrected", this is simply the corrected input signal.
    frequencies = _gonzalez_frequency_vector(fs, min_freq, max_freq, n_freqs)
    scales = pywt.frequency2scale(wavelet, frequencies / float(fs))
    coeffs, returned_freqs = pywt.cwt(
        data=working_trace,
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
        working_trace,
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
    morphology_metrics = _low_state_event_preservation_metrics(
        working_trace,
        denoised_working_trace,
        fs=fs,
        evaluation_mask=np.ones(working_trace.shape, dtype=bool),
        min_event_peak_preservation=float(min_event_peak_preservation),
        min_local_max_ratio=float(min_local_max_ratio),
    )
    morphology_reasons = list(morphology_metrics.get("fallback_reasons", []))
    if bool(enable_low_state_safety_gate) and bool(morphology_reasons):
        denoised_working_trace = working_trace.copy()
        output = values.copy()
        fallback_triggered = True
        warnings.extend(morphology_reasons)
        morphology_metrics["fallback_triggered"] = True
    elif bool(enable_noise_cluster_rejection) and bool(morphology_reasons):
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
        "working_trace_label": working_label,
        "F0": None if f0 is None else f0.copy(),
        "working_trace": working_trace.copy(),
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


def _hybrid_wavelet_denoise_impl(
    dff: np.ndarray,
    fs: float,
    cfg: HybridWaveletConfig,
    noise_reference_mask: np.ndarray | None = None,
    protected_plateau_mask: np.ndarray | None = None,
    plateau_mask_floor: float | None = None,
) -> np.ndarray:
    """Run continuous Hybrid denoising with optional plateau-aware thresholds."""
    trace = _validate_trace(dff, fs)
    if trace.size == 0:
        return trace.copy()
    if noise_reference_mask is not None:
        noise_reference_mask = np.asarray(noise_reference_mask, dtype=bool)
        if noise_reference_mask.shape != trace.shape:
            raise ValueError("noise_reference_mask must have the same shape as dff")
    if protected_plateau_mask is not None:
        protected_plateau_mask = np.asarray(protected_plateau_mask, dtype=bool)
        if protected_plateau_mask.shape != trace.shape:
            raise ValueError("protected_plateau_mask must have the same shape as dff")
    coeffs, normalized, _freqs = _morlet_cwt(trace, fs, cfg)
    features, n_pc = _frequency_features(normalized)
    domains = _frequency_domains(features, n_pc=n_pc, max_clusters=cfg.max_clusters)
    frequencies = _frequency_vector(fs, cfg)
    scales = pywt.frequency2scale(cfg.wavelet, frequencies / float(fs))
    rebuilt_domains: List[np.ndarray] = []
    for indices in domains:
        domain_coeffs = coeffs[indices]
        domain_scales = scales[indices]
        activity = np.abs(domain_coeffs).mean(axis=0)
        max_freq = float(np.max(frequencies[indices]))
        window = max(1, int(2 * ((round(fs / max_freq) * 2) // 2) + 1))
        if activity.size:
            max_window = activity.size if activity.size % 2 == 1 else max(1, activity.size - 1)
            window = min(window, max_window)
        local_mean = np.convolve(activity, np.ones(window) / float(window), mode="same")
        activity_residual = activity - local_mean
        if noise_reference_mask is None:
            # Preserve the legacy public API's original threshold behavior.
            sigma = float(np.std(activity_residual))
        else:
            sigma = _robust_sigma(activity_residual[noise_reference_mask])
            if sigma <= 1e-12:
                sigma = float(np.std(activity_residual))
        threshold = local_mean + (float(cfg.threshold_sd) * sigma)
        delta = np.maximum(-(activity - local_mean), 0.0)
        delta_max = float(np.max(delta)) if delta.size else 0.0
        below = np.zeros_like(delta) if delta_max <= 1e-12 else np.clip(
            delta / delta_max,
            float(cfg.mask_attenuation_min),
            float(cfg.mask_attenuation_max),
        )
        mask = np.where(activity > threshold, 1.0, 1.0 - below)
        if protected_plateau_mask is not None and plateau_mask_floor is not None:
            floor = float(np.clip(plateau_mask_floor, 0.0, 1.0))
            mask[protected_plateau_mask] = np.maximum(mask[protected_plateau_mask], floor)
        rebuilt = _inverse_cwt(domain_coeffs, domain_scales, cfg, mask=mask)
        regions, peaks = _event_regions(activity, threshold, fs, cfg)
        rebuilt_domains.append(
            _attenuate_noise_events(rebuilt, regions, peaks, fs, cfg, protected_mask=protected_plateau_mask)
        )
    rebuilt = np.sum(rebuilt_domains, axis=0) if rebuilt_domains else np.zeros_like(trace)
    denominator = float(np.vdot(rebuilt, rebuilt).real)
    gain = float(np.vdot(rebuilt, trace).real / denominator) if denominator > 1e-12 else 0.0
    return (gain * rebuilt) + float(np.mean(trace))


def _legacy_hybrid_corrected_denoise(corrected, fs, **kwargs):
    """Run the original Hybrid wavelet denoiser on baseline-corrected units."""
    cfg_kwargs = dict(kwargs)
    return_debug = bool(cfg_kwargs.pop("return_debug", False))
    cfg_kwargs.setdefault("event_mode", "none")
    cfg_kwargs.setdefault("threshold_sd", 1.5)
    cfg_kwargs.setdefault("mask_attenuation_min", 0.0)
    cfg_kwargs.setdefault("mask_attenuation_max", 0.4)

    allowed = {
        "wavelet",
        "min_freq",
        "max_freq",
        "n_freqs",
        "max_clusters",
        "threshold_sd",
        "mask_attenuation_min",
        "mask_attenuation_max",
        "event_mode",
        "event_merge_gap_ms",
        "event_min_count",
        "event_attenuation_min",
        "event_attenuation_max",
    }

    cfg = HybridWaveletConfig(
        **{key: value for key, value in cfg_kwargs.items() if key in allowed}
    )

    output = _hybrid_wavelet_denoise_impl(
        corrected,
        fs,
        cfg,
        noise_reference_mask=None,
        protected_plateau_mask=None,
        plateau_mask_floor=None,
    )
    if not return_debug:
        return output
    return {
        "denoiser_used": "legacy_hybrid",
        "output": output,
        "input": _validate_trace(corrected, fs),
        "config": cfg,
    }


def hybrid_wavelet_denoise(dff: np.ndarray, fs: float, **kwargs: object) -> np.ndarray:
    """Compatibility wrapper: run the legacy Hybrid denoiser on corrected units."""
    return _legacy_hybrid_corrected_denoise(dff, fs=fs, **kwargs)


def denoise_hybrid_fluorescence_trace(
    corrected: np.ndarray,
    fs: float,
    baseline: np.ndarray | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Compatibility wrapper: run legacy Hybrid on a baseline-corrected trace.

    ``baseline`` is accepted for the historical API but is intentionally not
    used by the main method. The incoming ``corrected`` signal is already
    baseline-subtracted/corrected, so no dF/F or rolling F0 conversion is done.
    """
    _ = baseline
    return _legacy_hybrid_corrected_denoise(corrected, fs=fs, **kwargs)


def _validated_mask(values: np.ndarray, shape: Tuple[int, ...], name: str) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must have the same shape as corrected")
    return mask


def _dilate_boolean_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if radius <= 0 or values.size == 0:
        return values.copy()
    kernel = np.ones((2 * int(radius)) + 1, dtype=int)
    padded = np.pad(values.astype(int), (int(radius), int(radius)), mode="constant")
    return np.convolve(padded, kernel, mode="valid") > 0


def _erode_boolean_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if radius <= 0 or values.size == 0:
        return values.copy()
    kernel = np.ones((2 * int(radius)) + 1, dtype=int)
    padded = np.pad(values.astype(int), (int(radius), int(radius)), mode="constant")
    return np.convolve(padded, kernel, mode="valid") == kernel.size


def _interpolate_masked_trace(values: np.ndarray, reliable_mask: np.ndarray) -> np.ndarray:
    trace = np.asarray(values, dtype=float)
    reliable = np.asarray(reliable_mask, dtype=bool) & np.isfinite(trace)
    if np.count_nonzero(reliable) < 2:
        fill = float(np.median(trace[np.isfinite(trace)])) if np.any(np.isfinite(trace)) else 0.0
        return np.full(trace.shape, fill, dtype=float)
    indices = np.arange(trace.size)
    return np.interp(indices, indices[reliable], trace[reliable])


def _odd_window_ms(window_ms: float, fs: float) -> int:
    window = max(3, int(round(float(window_ms) * float(fs) / 1000.0)))
    return window + 1 if window % 2 == 0 else window


def _contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return []
    starts = np.flatnonzero(values & np.concatenate(([True], ~values[:-1])))
    stops = np.flatnonzero(values & np.concatenate((~values[1:], [True]))) + 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _low_state_event_preservation_metrics(
    raw: np.ndarray,
    candidate: np.ndarray,
    fs: float,
    evaluation_mask: np.ndarray,
    min_event_peak_preservation: float = 0.80,
    min_local_max_ratio: float = 0.80,
    baseline_window_ms: float = 50.0,
    local_window_ms: float = 5.0,
    min_peak_distance_ms: float = 2.0,
) -> dict:
    """Check that low-state denoising preserves sharp morphology.

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


def _apply_plateau_level_correction(output: np.ndarray, corrected: np.ndarray, plateau_mask: np.ndarray, fs: float) -> np.ndarray:
    """Blend median-matching offsets into plateaus for optional visualization."""
    result = np.asarray(output, dtype=float).copy()
    source = np.asarray(corrected, dtype=float)
    n = result.size
    taper = max(1, int(round(0.05 * float(fs))))
    offset_curve = np.zeros(n, dtype=float)
    weight_curve = np.zeros(n, dtype=float)
    for start, stop in _contiguous_true_regions(plateau_mask):
        offset = float(np.median(source[start:stop]) - np.median(result[start:stop]))
        core = np.ones(stop - start, dtype=float)
        left = max(0, start - taper)
        right = min(n, stop + taper)
        weights = np.ones(right - left, dtype=float)
        if start > left:
            weights[: start - left] = np.linspace(0.0, 1.0, start - left, endpoint=False)
        if right > stop:
            weights[stop - left :] = np.linspace(1.0, 0.0, right - stop, endpoint=False)
        weights[start - left : stop - left] = core
        offset_curve[left:right] += offset * weights
        weight_curve[left:right] += weights
    active = weight_curve > 0
    result[active] += offset_curve[active] / weight_curve[active]
    return result


def score_denoising_result(
    corrected: np.ndarray,
    denoised: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    non_event_mask: np.ndarray | None = None,
) -> dict:
    """Score bias and high-frequency noise changes without judging smoothness alone."""
    source = _validate_trace(corrected, fs)
    output = _validate_trace(denoised, fs)
    if output.shape != source.shape:
        raise ValueError("denoised must have the same shape as corrected")
    plateau = _validated_mask(plateau_mask, source.shape, "plateau_mask")
    nonplateau = ~plateau
    if non_event_mask is not None:
        nonplateau &= _validated_mask(non_event_mask, source.shape, "non_event_mask")
    hp_window = _odd_window_ms(100.0, fs)
    source_residual = source - median_filter(source, size=hp_window, mode="nearest")
    output_residual = output - median_filter(output, size=hp_window, mode="nearest")
    difference = output - source

    def _metric(values: np.ndarray, mask: np.ndarray, fn) -> float:
        selected = np.asarray(values, dtype=float)[mask]
        return float(fn(selected)) if selected.size else float("nan")

    std_input_plateau = _metric(source_residual, plateau, np.std)
    std_output_plateau = _metric(output_residual, plateau, np.std)
    std_input_nonplateau = _metric(source_residual, nonplateau, np.std)
    std_output_nonplateau = _metric(output_residual, nonplateau, np.std)

    def _reduction(before: float, after: float) -> float:
        return float(1.0 - (after / before)) if np.isfinite(before) and before > 1e-12 else float("nan")

    return {
        "median_output_minus_input_plateau": _metric(difference, plateau, np.median),
        "mean_output_minus_input_plateau": _metric(difference, plateau, np.mean),
        "median_output_minus_input_nonplateau": _metric(difference, nonplateau, np.median),
        "mean_output_minus_input_nonplateau": _metric(difference, nonplateau, np.mean),
        "std_input_plateau": std_input_plateau,
        "std_output_plateau": std_output_plateau,
        "std_input_nonplateau": std_input_nonplateau,
        "std_output_nonplateau": std_output_nonplateau,
        "plateau_noise_reduction_fraction": _reduction(std_input_plateau, std_output_plateau),
        "nonplateau_noise_reduction_fraction": _reduction(std_input_nonplateau, std_output_nonplateau),
    }


def denoise_hybrid_fluorescence_trace_plateau_aware(
    corrected: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    non_event_mask: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
    f0_window_ms: float = 2000.0,
    slow_window_ms: float = 500.0,
    mask_dilation_ms: float = 100.0,
    threshold_sd: float = 2.0,
    plateau_mask_floor: float = 0.7,
    remove_residual_drift: bool = True,
    residual_drift_window_ms: float | None = None,
    plateau_level_correction: bool = False,
    return_debug: bool = False,
    **kwargs: object,
) -> np.ndarray | dict:
    """Compatibility wrapper for the active state-guided denoiser.

    Historical arguments related to dF/F baseline estimation and slow/fast
    splitting are accepted for API stability, but the active route calls
    ``denoise_trace_state_guided_low_plateau_tv`` and keeps the signal in
    baseline-corrected units. Low-state samples use Gonzalez adaptive wavelet
    denoising by default; plateau cores use TV denoising and boundary samples
    remain raw corrected values.
    """
    _ = baseline, f0_window_ms, slow_window_ms, plateau_mask_floor, remove_residual_drift
    _ = residual_drift_window_ms, plateau_level_correction
    trace = _validate_trace(corrected, fs)
    plateau = _validated_mask(plateau_mask, trace.shape, "plateau_mask")
    supplied_non_event = None if non_event_mask is None else _validated_mask(non_event_mask, trace.shape, "non_event_mask")
    # Compatibility wrapper: keep this older public name on the state-guided
    # path. The signal stays in corrected units throughout this call.
    cfg_kwargs = dict(kwargs)
    cfg_kwargs["threshold_sd"] = float(threshold_sd)
    result = denoise_trace_state_guided_low_plateau_tv(
        trace,
        fs=fs,
        plateau_mask=plateau,
        boundary_protect_ms=mask_dilation_ms,
        return_debug=return_debug,
        **cfg_kwargs,
    )
    if not return_debug:
        return result
    debug = dict(result)
    debug["wrapper_used"] = "denoise_hybrid_fluorescence_trace_plateau_aware"
    debug["non_event_mask"] = supplied_non_event
    return debug


def sweep_plateau_denoising_parameters(
    corrected: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    non_event_mask: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
) -> List[dict]:
    """Evaluate a compact plateau-aware tuning grid without plotting.

    Prefer low plateau bias together with reduced plateau high-frequency noise.
    Reject settings that strongly shift plateau amplitude or create substantial
    nonplateau bias. Visual smoothness alone is not a sufficient tuning target.
    """
    results: List[dict] = []
    for threshold_sd in [1.0, 1.2, 1.5]:
        for mask_attenuation_max in [0.3, 0.4, 0.5]:
            for plateau_mask_floor in [0.6, 0.7, 0.8]:
                for slow_window_ms in [300.0, 500.0, 800.0]:
                    output = denoise_hybrid_fluorescence_trace_plateau_aware(
                        corrected=corrected,
                        fs=fs,
                        plateau_mask=plateau_mask,
                        non_event_mask=non_event_mask,
                        baseline=baseline,
                        threshold_sd=threshold_sd,
                        plateau_mask_floor=plateau_mask_floor,
                        slow_window_ms=slow_window_ms,
                        mask_attenuation_max=mask_attenuation_max,
                    )
                    score = score_denoising_result(corrected, output, fs, plateau_mask, non_event_mask)
                    results.append(
                        {
                            "threshold_sd": threshold_sd,
                            "mask_attenuation_max": mask_attenuation_max,
                            "plateau_mask_floor": plateau_mask_floor,
                            "slow_window_ms": slow_window_ms,
                            **score,
                        }
                    )
    return results


def _bounded_odd_window_ms(window_ms: float, fs: float, size: int) -> int:
    if size <= 1:
        return 1
    window = _odd_window_ms(window_ms, fs)
    max_window = size if size % 2 == 1 else size - 1
    return max(1, min(window, max_window))


def _plateau_boundary_blend_weight(plateau_mask: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return smoothing weights that taper to raw values at plateau transitions."""
    plateau = np.asarray(plateau_mask, dtype=bool)
    weight = np.ones(plateau.shape, dtype=float)
    boundary_mask = np.zeros(plateau.shape, dtype=bool)
    if plateau.size == 0 or radius <= 0:
        return weight, boundary_mask
    transitions = np.flatnonzero(plateau[1:] != plateau[:-1]) + 1
    if transitions.size == 0:
        return weight, boundary_mask
    indices = np.arange(plateau.size)
    distance = np.min(np.abs(indices[:, None] - transitions[None, :]), axis=1)
    boundary_mask = distance <= int(radius)
    weight[boundary_mask] = np.clip(distance[boundary_mask] / float(radius), 0.0, 1.0)
    return weight, boundary_mask


def score_label_guided_denoising(
    corrected: np.ndarray,
    denoised: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    non_event_mask: np.ndarray | None = None,
) -> dict:
    """Measure plateau bias and high-frequency noise reduction."""
    source = _validate_trace(corrected, fs)
    output = _validate_trace(denoised, fs)
    if output.shape != source.shape:
        raise ValueError("denoised must have the same shape as corrected")
    plateau = _validated_mask(plateau_mask, source.shape, "plateau_mask")
    nonplateau = ~plateau
    if non_event_mask is not None:
        nonplateau &= _validated_mask(non_event_mask, source.shape, "non_event_mask")

    hp_window = _bounded_odd_window_ms(100.0, fs, source.size)
    source_highpass = source - median_filter(source, size=hp_window, mode="nearest")
    output_highpass = output - median_filter(output, size=hp_window, mode="nearest")
    difference = output - source

    def _metric(values: np.ndarray, mask: np.ndarray, fn) -> float:
        selected = np.asarray(values, dtype=float)[mask]
        return float(fn(selected)) if selected.size else float("nan")

    std_input_plateau = _metric(source_highpass, plateau, np.std)
    std_output_plateau = _metric(output_highpass, plateau, np.std)
    std_input_nonplateau = _metric(source_highpass, nonplateau, np.std)
    std_output_nonplateau = _metric(output_highpass, nonplateau, np.std)

    def _reduction(before: float, after: float) -> float:
        return float(1.0 - (after / before)) if np.isfinite(before) and before > 1e-12 else float("nan")

    return {
        "median_output_minus_input_plateau": _metric(difference, plateau, np.median),
        "mean_output_minus_input_plateau": _metric(difference, plateau, np.mean),
        "median_output_minus_input_nonplateau": _metric(difference, nonplateau, np.median),
        "std_highpass_input_plateau": std_input_plateau,
        "std_highpass_output_plateau": std_output_plateau,
        "plateau_noise_reduction_fraction": _reduction(std_input_plateau, std_output_plateau),
        "std_highpass_input_nonplateau": std_input_nonplateau,
        "std_highpass_output_nonplateau": std_output_nonplateau,
        "nonplateau_noise_reduction_fraction": _reduction(std_input_nonplateau, std_output_nonplateau),
    }


def denoise_fluorescence_trace_label_guided(
    corrected: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    non_event_mask: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
    method: str = "savgol",
    f0_window_ms: float = 2000.0,
    smooth_window_ms: float = 80.0,
    polyorder: int = 2,
    boundary_protect_ms: float = 150.0,
    return_debug: bool = False,
) -> np.ndarray | dict:
    """Denoise plateau-heavy traces continuously without CWT reconstruction.

    This is the preferred path for plateau-heavy recordings. The full dF/F
    trace is smoothed in one pass. Plateau labels guide F0 estimation and apply
    a smooth raw-to-denoised crossfade around transitions so plateau level,
    onset, offset, and duration remain stable.
    """
    trace = _validate_trace(corrected, fs)
    plateau = _validated_mask(plateau_mask, trace.shape, "plateau_mask")
    non_event = None if non_event_mask is None else _validated_mask(non_event_mask, trace.shape, "non_event_mask")
    if trace.size == 0:
        debug = {
            "output": trace.copy(),
            "f0": trace.copy(),
            "dff": trace.copy(),
            "denoised_dff": trace.copy(),
            "plateau_mask": plateau,
            "protected_plateau_mask": plateau.copy(),
            "boundary_mask": plateau.copy(),
            "smoothing_method": str(method).strip().lower(),
        }
        debug.update(score_label_guided_denoising(trace, trace, fs, plateau, non_event))
        return debug if return_debug else trace.copy()

    boundary_radius = max(0, int(round(float(boundary_protect_ms) * float(fs) / 1000.0)))
    protected_plateau = _dilate_boolean_mask(plateau, radius=boundary_radius)
    reliable_baseline = ~protected_plateau
    if non_event is not None:
        reliable_baseline &= non_event

    if baseline is None:
        f0_source = _interpolate_masked_trace(trace, reliable_baseline)
        f0_window = _bounded_odd_window_ms(f0_window_ms, fs, trace.size)
        f0 = median_filter(f0_source, size=f0_window, mode="nearest")
    else:
        f0 = _validate_trace(np.asarray(baseline, dtype=float), fs)
        if f0.shape != trace.shape:
            raise ValueError("baseline must have the same shape as corrected")

    scale = np.maximum(np.abs(f0), 1e-9)
    dff = (trace - f0) / scale
    smooth_window = _bounded_odd_window_ms(smooth_window_ms, fs, trace.size)
    normalized_method = str(method).strip().lower()
    if normalized_method == "savgol":
        order = int(polyorder)
        if order < 0:
            raise ValueError("polyorder must be non-negative")
        if smooth_window <= order:
            required_window = order + 1 if (order + 1) % 2 == 1 else order + 2
            max_window = trace.size if trace.size % 2 == 1 else trace.size - 1
            smooth_window = min(max_window, required_window)
        if smooth_window <= order or smooth_window < 3:
            smoothed_dff = dff.copy()
        else:
            smoothed_dff = savgol_filter(dff, window_length=smooth_window, polyorder=order, mode="interp")
    elif normalized_method == "median":
        smoothed_dff = median_filter(dff, size=smooth_window, mode="nearest")
    elif normalized_method == "tv":
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError as exc:
            raise ValueError("method='tv' requires scikit-image") from exc
        smoothed_dff = np.asarray(denoise_tv_chambolle(dff, weight=0.1), dtype=float)
    else:
        raise ValueError("method must be one of: savgol, median, tv")

    smoothing_weight, boundary_mask = _plateau_boundary_blend_weight(plateau, radius=boundary_radius)
    denoised_dff = dff + (smoothing_weight * (smoothed_dff - dff))
    output = f0 + (denoised_dff * scale)
    if not return_debug:
        return output
    debug = {
        "output": output,
        "f0": f0,
        "dff": dff,
        "denoised_dff": denoised_dff,
        "plateau_mask": plateau,
        "protected_plateau_mask": protected_plateau,
        "boundary_mask": boundary_mask,
        "smoothing_method": normalized_method,
    }
    debug.update(score_label_guided_denoising(trace, output, fs, plateau, non_event))
    return debug


def _boundary_mask_from_state_changes(state_mask: np.ndarray, radius: int) -> np.ndarray:
    """Mark samples near binary-state transitions."""
    state = np.asarray(state_mask, dtype=bool)
    boundary = np.zeros(state.shape, dtype=bool)
    if state.size <= 1 or radius < 0:
        return boundary
    transitions = np.flatnonzero(state[1:] != state[:-1]) + 1
    for transition in transitions:
        start = max(0, int(transition) - int(radius))
        stop = min(state.size, int(transition) + int(radius) + 1)
        boundary[start:stop] = True
    return boundary


def _median_filter_segment(segment: np.ndarray, requested_window: int) -> np.ndarray:
    """Median-filter one segment with reflection padding and an odd local window."""
    values = np.asarray(segment, dtype=float)
    if values.size <= 1:
        return values.copy()
    window = min(int(requested_window), int(values.size))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="reflect")
    filtered = median_filter(padded, size=window, mode="nearest")
    return np.asarray(filtered[radius : radius + values.size], dtype=float)


def denoise_trace_by_binary_state_median(
    corrected: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    plateau_window_ms: float = 60.0,
    low_window_ms: float = 120.0,
    boundary_protect_ms: float = 50.0,
    return_debug: bool = False,
) -> np.ndarray | dict:
    """Median-filter binary-state segments without crossing state boundaries.

    This is a simple alternative for plateau-heavy recordings. It works
    directly on the corrected trace and preserves raw onset/offset samples.
    """
    trace = _validate_trace(corrected, fs)
    plateau = _validated_mask(plateau_mask, trace.shape, "plateau_mask")
    low = ~plateau
    plateau_window = _odd_window_ms(plateau_window_ms, fs)
    low_window = _odd_window_ms(low_window_ms, fs)
    boundary_radius = max(0, int(round(float(boundary_protect_ms) * float(fs) / 1000.0)))
    boundary = _boundary_mask_from_state_changes(plateau, radius=boundary_radius)

    output = trace.copy()
    for state_mask, window in [(plateau, plateau_window), (low, low_window)]:
        for start, stop in _contiguous_true_regions(state_mask):
            output[start:stop] = _median_filter_segment(trace[start:stop], requested_window=window)
    output[boundary] = trace[boundary]

    if not return_debug:
        return output
    corrected_minus_output = trace - output

    def _std(values: np.ndarray, mask: np.ndarray) -> float:
        selected = np.asarray(values, dtype=float)[mask]
        return float(np.std(selected)) if selected.size else float("nan")

    def _reduction(before: float, after: float) -> float:
        return float(1.0 - (after / before)) if np.isfinite(before) and before > 1e-12 else float("nan")

    std_raw_plateau = _std(trace, plateau)
    std_denoised_plateau = _std(output, plateau)
    std_raw_low = _std(trace, low)
    std_denoised_low = _std(output, low)
    return {
        "output": output,
        "plateau_mask": plateau,
        "low_mask": low,
        "boundary_mask": boundary,
        "plateau_window_samples": plateau_window,
        "low_window_samples": low_window,
        "corrected_minus_output": corrected_minus_output,
        "std_raw_plateau": std_raw_plateau,
        "std_denoised_plateau": std_denoised_plateau,
        "std_raw_low": std_raw_low,
        "std_denoised_low": std_denoised_low,
        "plateau_noise_reduction_fraction": _reduction(std_raw_plateau, std_denoised_plateau),
        "low_noise_reduction_fraction": _reduction(std_raw_low, std_denoised_low),
    }


def _tv_denoise_1d(values: np.ndarray, weight: float, max_iterations: int = 200) -> np.ndarray:
    """Denoise one segment with 1D total variation using a stable dual update."""
    signal = np.asarray(values, dtype=float)
    if signal.size <= 1 or float(weight) <= 0.0:
        return signal.copy()
    dual = np.zeros(signal.size - 1, dtype=float)
    step = 0.24
    scale = max(float(weight), 1e-12)
    for _ in range(max(1, int(max_iterations))):
        divergence = np.empty(signal.size, dtype=float)
        divergence[0] = -dual[0]
        divergence[-1] = dual[-1]
        if signal.size > 2:
            divergence[1:-1] = dual[:-1] - dual[1:]
        estimate = signal - (scale * divergence)
        gradient = np.diff(estimate)
        updated = np.clip(dual + (step * gradient / scale), -1.0, 1.0)
        if float(np.max(np.abs(updated - dual))) < 1e-5:
            dual = updated
            break
        dual = updated
    divergence = np.empty(signal.size, dtype=float)
    divergence[0] = -dual[0]
    divergence[-1] = dual[-1]
    if signal.size > 2:
        divergence[1:-1] = dual[:-1] - dual[1:]
    return signal - (scale * divergence)


def _normalize_low_state_denoiser(value: str) -> str:
    method = str(value).strip().lower()
    allowed = {"gonzalez_adaptive_wavelet", "legacy_hybrid", "none"}
    if method not in allowed:
        raise ValueError("low_state_denoiser must be one of: gonzalez_adaptive_wavelet, legacy_hybrid, none")
    return method


def denoise_trace_state_guided_low_plateau_tv(
    corrected: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    plateau_tv_weight: float = 2.0,
    plateau_tv_min_duration_ms: float = 5.0,
    plateau_tv_max_weight_factor: float = 0.25,
    boundary_protect_ms: float = 20.0,
    low_state_denoiser: str = "gonzalez_adaptive_wavelet",
    low_state_denoise_fn=None,
    hybrid_denoise_fn=None,
    enable_low_state_safety_gate: bool = True,
    min_event_peak_preservation: float = 0.80,
    min_local_max_ratio: float = 0.80,
    min_reliable_low_state_fraction: float = 0.10,
    min_reliable_low_state_samples: int = 20,
    return_debug: bool = False,
    **low_state_kwargs: object,
) -> np.ndarray | dict:
    """Run a state-aware corrected-trace denoiser.

    Gonzalez adaptive wavelet and Legacy Hybrid both denoise a low-state input
    where protected plateau samples have been bridged from neighboring
    low-state values. Plateau cores are conservatively TV-denoised separately
    from the raw corrected trace, and both inside and outside state-boundary
    samples are restored from the raw corrected trace. ``low_state_denoiser="none"``
    remains a passthrough. The denoised output is a detection aid; raw corrected
    trace values remain preferred for quantitative amplitude/width/area/level
    measurements.
    """
    trace = _validate_trace(corrected, fs)
    plateau = _validated_mask(plateau_mask, trace.shape, "plateau_mask")
    boundary_radius = max(0, int(round(float(boundary_protect_ms) * float(fs) / 1000.0)))
    protected_plateau = _dilate_boolean_mask(plateau, radius=boundary_radius)
    plateau_core_mask = plateau & _erode_boolean_mask(plateau, radius=boundary_radius)
    boundary_mask = protected_plateau & (~plateau_core_mask)
    reliable_low = ~protected_plateau
    minimum_reliable = max(
        int(min_reliable_low_state_samples),
        int(float(min_reliable_low_state_fraction) * trace.size),
    )
    low_state_method = _normalize_low_state_denoiser(low_state_denoiser)

    alias_kwargs = low_state_kwargs.pop("hybrid_kwargs", None)
    if isinstance(alias_kwargs, dict):
        merged_kwargs = dict(alias_kwargs)
        merged_kwargs.update(low_state_kwargs)
        low_state_kwargs = merged_kwargs
    low_state_kwargs.pop("return_debug", None)
    if low_state_denoise_fn is None and hybrid_denoise_fn is not None:
        low_state_denoise_fn = hybrid_denoise_fn

    apply_plateau_tv = low_state_method != "none" and bool(np.any(plateau_core_mask))
    low_state_safety: dict[str, Any] = {
        "fallback_triggered": False,
        "fallback_reasons": [],
    }
    reliable_low_count = int(np.count_nonzero(reliable_low))
    skip_low_state_denoising = (
        low_state_method != "none"
        and reliable_low_count < minimum_reliable
    )
    if skip_low_state_denoising:
        low_state_input = trace.copy()
        low_state_safety = {
            "fallback_triggered": True,
            "fallback_reasons": ["insufficient_reliable_low_state_samples"],
            "reliable_low_state_samples": reliable_low_count,
            "minimum_reliable_low_state_samples": int(minimum_reliable),
        }
    elif low_state_method != "none":
        # The low-state denoiser should not decide the plateau level. Bridge
        # plateaus and their edge neighborhood from reliable low-state values.
        low_state_input = _interpolate_masked_trace(trace, reliable_mask=reliable_low)
    else:
        low_state_input = trace.copy()
    low_state_output: np.ndarray
    gonzalez_debug = None
    low_state_exception: str | None = None
    low_state_candidate_used = False

    low_state_eval_mask = ~(plateau_core_mask | boundary_mask)

    def _accept_low_state_candidate(candidate: np.ndarray, source_label: str) -> Tuple[np.ndarray, dict, bool]:
        candidate_arr = np.asarray(candidate, dtype=float)
        fallback_reasons: List[str] = []
        if candidate_arr.shape != low_state_input.shape:
            fallback_reasons.append("shape_mismatch")
        if candidate_arr.size and not np.all(np.isfinite(candidate_arr)):
            fallback_reasons.append("non_finite_output")
        input_std = float(np.std(low_state_input)) if low_state_input.size else float("nan")
        output_std = float(np.std(candidate_arr)) if candidate_arr.shape == low_state_input.shape else float("nan")
        std_ratio = output_std / max(input_std, 1e-12) if np.isfinite(input_std) else float("nan")
        morphology = (
            _low_state_event_preservation_metrics(
                trace,
                candidate_arr,
                fs=fs,
                evaluation_mask=low_state_eval_mask,
                min_event_peak_preservation=float(min_event_peak_preservation),
                min_local_max_ratio=float(min_local_max_ratio),
            )
            if candidate_arr.shape == trace.shape
            else {"status": "not_evaluated", "fallback_reasons": []}
        )
        if bool(enable_low_state_safety_gate):
            fallback_reasons.extend(str(reason) for reason in morphology.get("fallback_reasons", []))
        valid = not fallback_reasons
        safety = {
            "source": source_label,
            "shape_matches_input": bool(candidate_arr.shape == low_state_input.shape),
            "output_is_finite": bool(candidate_arr.size == 0 or np.all(np.isfinite(candidate_arr))),
            "input_std": input_std,
            "output_std": output_std,
            "output_input_std_ratio": std_ratio,
            "morphology_preservation": morphology,
            "fallback_triggered": not valid,
            "fallback_reasons": fallback_reasons,
            "evaluation_mask_samples": int(np.count_nonzero(low_state_eval_mask)),
            "safety_gate_enabled": bool(enable_low_state_safety_gate),
        }
        if candidate_arr.shape != low_state_input.shape:
            safety["output_shape"] = tuple(int(v) for v in candidate_arr.shape)
            safety["input_shape"] = tuple(int(v) for v in low_state_input.shape)
        return (candidate_arr if valid else low_state_input.copy(), safety, valid)

    if low_state_method == "none" or skip_low_state_denoising:
        low_state_output = low_state_input.copy()
    elif low_state_method == "legacy_hybrid":
        denoise_fn = _legacy_hybrid_corrected_denoise if low_state_denoise_fn is None else low_state_denoise_fn
        candidate = np.asarray(denoise_fn(low_state_input, fs=fs, **low_state_kwargs), dtype=float)
        low_state_output, low_state_safety, low_state_candidate_used = _accept_low_state_candidate(
            candidate,
            source_label="legacy_hybrid",
        )
    else:
        gonzalez_allowed = {
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
            "enable_low_state_safety_gate",
            "min_event_peak_preservation",
            "min_local_max_ratio",
        }
        gonzalez_kwargs = {key: value for key, value in low_state_kwargs.items() if key in gonzalez_allowed}
        gonzalez_kwargs.update(
            {
                "input_mode": "corrected",
                "enable_noise_cluster_rejection": bool(
                    gonzalez_kwargs.get("enable_noise_cluster_rejection", False)
                ),
                "enable_event_template_rejection": bool(
                    gonzalez_kwargs.get("enable_event_template_rejection", False)
                ),
                "enable_low_state_safety_gate": bool(enable_low_state_safety_gate),
                "min_event_peak_preservation": float(min_event_peak_preservation),
                "min_local_max_ratio": float(min_local_max_ratio),
                "attenuation_min": float(gonzalez_kwargs.get("attenuation_min", 0.5)),
                "attenuation_max": float(gonzalez_kwargs.get("attenuation_max", 1.0)),
            }
        )
        try:
            gonzalez_result = denoise_gonzalez_adaptive_wavelet(
                low_state_input,
                fs=fs,
                return_debug=return_debug,
                **gonzalez_kwargs,
            )
            if return_debug:
                gonzalez_debug = dict(gonzalez_result)
                candidate = np.asarray(gonzalez_debug.get("output", low_state_input), dtype=float)
            else:
                candidate = np.asarray(gonzalez_result, dtype=float)
            low_state_output, low_state_safety, low_state_candidate_used = _accept_low_state_candidate(
                candidate,
                source_label="gonzalez_adaptive_wavelet",
            )
            if isinstance(gonzalez_debug, dict):
                gonzalez_safety = gonzalez_debug.get("safety_metrics", {})
                if isinstance(gonzalez_safety, dict) and bool(gonzalez_safety.get("fallback_triggered", False)):
                    morphology = gonzalez_safety.get("morphology_preservation", {})
                    reasons = []
                    if isinstance(morphology, dict):
                        reasons = [str(reason) for reason in morphology.get("fallback_reasons", [])]
                    if not reasons:
                        reasons = ["gonzalez_internal_safety_fallback"]
                    existing = list(low_state_safety.get("fallback_reasons", []))
                    for reason in reasons:
                        if reason not in existing:
                            existing.append(reason)
                    low_state_safety["fallback_reasons"] = existing
                    low_state_safety["fallback_triggered"] = True
                    low_state_safety["gonzalez_internal_safety_fallback"] = True
                    low_state_candidate_used = False
        except Exception as exc:
            low_state_exception = str(exc)
            low_state_safety = {
                "fallback_triggered": True,
                "fallback_reasons": ["gonzalez_exception"],
                "exception": low_state_exception,
            }
            if return_debug:
                gonzalez_debug = {"error": low_state_exception}
            low_state_output = low_state_input.copy()

    output = low_state_output.copy()
    replace_mask = np.zeros_like(plateau, dtype=bool)
    plateau_tv: np.ndarray | str
    plateau_tv_status = "not_applied"
    plateau_tv_segments: List[dict] = []
    if apply_plateau_tv:
        plateau_tv = trace.copy()
        min_plateau_tv_samples = max(1, int(round(float(plateau_tv_min_duration_ms) * float(fs) / 1000.0)))
        for start, stop in _contiguous_true_regions(plateau_core_mask):
            segment = trace[start:stop]
            noise_sigma = _robust_sigma(np.diff(segment)) / np.sqrt(2.0) if segment.size > 1 else 0.0
            segment_sigma = _robust_sigma(segment)
            raw_weight = float(plateau_tv_weight) * float(noise_sigma)
            max_weight = float(plateau_tv_max_weight_factor) * max(float(segment_sigma), float(noise_sigma), 1e-12)
            final_weight = min(raw_weight, max_weight)
            segment_debug = {
                "start": int(start),
                "stop": int(stop),
                "length": int(segment.size),
                "noise_sigma": float(noise_sigma),
                "segment_sigma": float(segment_sigma),
                "raw_weight": float(raw_weight),
                "final_weight": float(final_weight),
            }
            if segment.size < min_plateau_tv_samples:
                final_weight = 0.0
                segment_debug["final_weight"] = 0.0
                segment_debug["skipped_reason"] = "short_plateau_segment"
            elif not np.isfinite(final_weight) or final_weight <= 1e-12 or noise_sigma <= 1e-12:
                final_weight = 0.0
                segment_debug["final_weight"] = 0.0
                segment_debug["skipped_reason"] = "near_zero_noise_or_weight"
            else:
                plateau_tv[start:stop] = _tv_denoise_1d(segment, weight=final_weight)
            plateau_tv_segments.append(segment_debug)
        replace_mask = plateau_core_mask
        output[replace_mask] = plateau_tv[replace_mask]
        output[boundary_mask] = trace[boundary_mask]
        plateau_tv_status = "applied"
    else:
        plateau_tv = plateau_tv_status
    if not return_debug:
        return output

    hp_window = _bounded_odd_window_ms(100.0, fs, trace.size)
    trace_high = trace - median_filter(trace, size=hp_window, mode="nearest")
    output_high = output - median_filter(output, size=hp_window, mode="nearest")
    nonplateau = ~plateau

    def _masked_std(values: np.ndarray, mask: np.ndarray) -> float:
        selected = np.asarray(values, dtype=float)[mask]
        return float(np.std(selected)) if selected.size else float("nan")

    def _reduction(before: float, after: float) -> float:
        return float(1.0 - (after / before)) if np.isfinite(before) and before > 1e-12 else float("nan")

    plateau_noise_before = _masked_std(trace_high, plateau)
    plateau_noise_after = _masked_std(output_high, plateau)
    nonplateau_noise_before = _masked_std(trace_high, nonplateau)
    nonplateau_noise_after = _masked_std(output_high, nonplateau)
    low_state_diff = low_state_output - low_state_input
    low_state_residual_std = float(np.std(low_state_diff)) if low_state_diff.size else float("nan")
    low_state_noise_before = _masked_std(trace_high, nonplateau)
    low_state_output_high = low_state_output - median_filter(low_state_output, size=hp_window, mode="nearest")
    low_state_noise_after = _masked_std(low_state_output_high, nonplateau)
    plateau_median_bias = (
        float(np.median(output[plateau]) - np.median(trace[plateau])) if np.any(plateau) else float("nan")
    )

    return {
        "denoiser_used": "state_guided_low_plateau_tv",
        "low_state_denoiser_used": low_state_method,
        "low_state_candidate_used": bool(low_state_candidate_used),
        "low_state_fallback_triggered": bool(low_state_safety.get("fallback_triggered", False)),
        "low_state_fallback_reasons": list(low_state_safety.get("fallback_reasons", [])),
        "low_state_input_output_max_abs_diff": (
            float(np.max(np.abs(low_state_diff))) if low_state_diff.size else float("nan")
        ),
        "low_state_input_output_residual_std": low_state_residual_std,
        "low_state_noise_reduction_fraction": _reduction(low_state_noise_before, low_state_noise_after),
        "output": output,
        "low_state_output": low_state_output,
        "low_state_input": low_state_input,
        "hybrid_output": low_state_output,
        "hybrid_input": low_state_input,
        "low_state_custom_fn_used": bool(low_state_denoise_fn is not None),
        "low_state_safety": low_state_safety,
        "low_state_exception": low_state_exception,
        "gonzalez_debug": gonzalez_debug,
        "plateau_tv_output": plateau_tv,
        "plateau_tv_status": plateau_tv_status,
        "plateau_tv_segments": plateau_tv_segments,
        "plateau_mask": plateau,
        "protected_plateau_mask": protected_plateau,
        "plateau_core_mask": plateau_core_mask,
        "low_mask": ~plateau,
        "boundary_mask": boundary_mask,
        "plateau_replace_mask": replace_mask,
        "plateau_tv_weight": float(plateau_tv_weight),
        "plateau_tv_min_duration_ms": float(plateau_tv_min_duration_ms),
        "plateau_tv_max_weight_factor": float(plateau_tv_max_weight_factor),
        "reliable_low_state_samples": reliable_low_count,
        "minimum_reliable_low_state_samples": int(minimum_reliable),
        "low_state_safety_gate_enabled": bool(enable_low_state_safety_gate),
        "quantitative_measurement_note": (
            "Denoised trace is a detection aid; raw corrected trace is preferred for quantitative "
            "amplitude, peak height, width, area, plateau level, and summary statistics."
        ),
        "plateau_high_frequency_noise_before": plateau_noise_before,
        "plateau_high_frequency_noise_after": plateau_noise_after,
        "plateau_noise_reduction_fraction": _reduction(plateau_noise_before, plateau_noise_after),
        "nonplateau_high_frequency_noise_before": nonplateau_noise_before,
        "nonplateau_high_frequency_noise_after": nonplateau_noise_after,
        "nonplateau_noise_reduction_fraction": _reduction(nonplateau_noise_before, nonplateau_noise_after),
        "plateau_median_bias": plateau_median_bias,
    }


def denoise_trace_hybrid_low_plateau_tv(
    corrected: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    plateau_tv_weight: float = 2.0,
    boundary_protect_ms: float = 20.0,
    hybrid_denoise_fn=None,
    return_debug: bool = False,
    low_state_denoiser: str | None = None,
    low_state_denoise_fn=None,
    **hybrid_kwargs: object,
) -> np.ndarray | dict:
    """Compatibility wrapper for the state-guided low-state/plateau-TV path."""
    explicit_method = low_state_denoiser is not None or "low_state_denoiser" in hybrid_kwargs
    if "low_state_denoiser" in hybrid_kwargs:
        low_state_denoiser = str(hybrid_kwargs.pop("low_state_denoiser"))
    if low_state_denoiser is None:
        low_state_denoiser = "gonzalez_adaptive_wavelet"
    if low_state_denoise_fn is None and hybrid_denoise_fn is not None:
        low_state_denoise_fn = hybrid_denoise_fn
        if not explicit_method and low_state_denoiser == "gonzalez_adaptive_wavelet":
            low_state_denoiser = "legacy_hybrid"
    result = denoise_trace_state_guided_low_plateau_tv(
        corrected,
        fs=fs,
        plateau_mask=plateau_mask,
        plateau_tv_weight=plateau_tv_weight,
        boundary_protect_ms=boundary_protect_ms,
        low_state_denoiser=low_state_denoiser,
        low_state_denoise_fn=low_state_denoise_fn,
        return_debug=return_debug,
        **hybrid_kwargs,
    )
    if return_debug:
        debug = dict(result)
        debug["wrapper_used"] = "denoise_trace_hybrid_low_plateau_tv"
        return debug
    return result
