"""Hybrid-style PCA-wavelet denoising for fluorescence traces.

This is an app-owned, one-dimensional adaptation of Hybrid cal_wavelet.py.
It keeps the original processing stages while exposing a fluorescence-unit
wrapper for direct comparison with the existing denoising overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pywt
from kneed import KneeLocator
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
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


def hybrid_wavelet_denoise(dff: np.ndarray, fs: float, **kwargs: object) -> np.ndarray:
    """Run the Hybrid PCA-wavelet pipeline on one dF/F trace."""
    return _hybrid_wavelet_denoise_impl(dff, fs, HybridWaveletConfig(**kwargs))


def denoise_hybrid_fluorescence_trace(
    corrected: np.ndarray,
    fs: float,
    baseline: np.ndarray | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Run Hybrid internally in dF/F and return fluorescence-unit output."""
    trace = _validate_trace(corrected, fs)
    if trace.size == 0:
        return trace.copy()
    if baseline is None:
        window = max(3, int(round(2.0 * fs)))
        if window % 2 == 0:
            window += 1
        f0 = median_filter(trace, size=window, mode="nearest")
    else:
        f0 = _validate_trace(np.asarray(baseline, dtype=float), fs)
        if f0.shape != trace.shape:
            raise ValueError("baseline must have the same shape as corrected")
    scale = np.maximum(np.abs(f0), 1e-9)
    dff = (trace - f0) / scale
    denoised_dff = hybrid_wavelet_denoise(dff, fs=fs, **kwargs)
    return f0 + (denoised_dff * scale)


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
    threshold_sd: float = 1.5,
    plateau_mask_floor: float = 0.7,
    remove_residual_drift: bool = True,
    residual_drift_window_ms: float | None = None,
    plateau_level_correction: bool = False,
    return_debug: bool = False,
    **kwargs: object,
) -> np.ndarray | dict:
    """Denoise continuously while preserving long plateau signals.

    Plateaus are treated as slow signal, not artifacts. Noise inside plateaus is
    still removed because the full-length fast residual is denoised continuously.
    """
    trace = _validate_trace(corrected, fs)
    plateau = _validated_mask(plateau_mask, trace.shape, "plateau_mask")
    supplied_non_event = None
    if non_event_mask is not None:
        supplied_non_event = _validated_mask(non_event_mask, trace.shape, "non_event_mask")

    dilation_samples = max(0, int(round(float(mask_dilation_ms) * float(fs) / 1000.0)))
    protected_plateau = _dilate_boolean_mask(plateau, radius=dilation_samples)
    reliable_baseline = ~protected_plateau
    if supplied_non_event is not None:
        reliable_baseline &= supplied_non_event

    if baseline is None:
        f0_source = _interpolate_masked_trace(trace, reliable_baseline)
        f0_window = _odd_window_ms(f0_window_ms, fs)
        f0 = median_filter(f0_source, size=f0_window, mode="nearest")
    else:
        f0 = _validate_trace(np.asarray(baseline, dtype=float), fs)
        if f0.shape != trace.shape:
            raise ValueError("baseline must have the same shape as corrected")

    scale = np.maximum(np.abs(f0), 1e-9)
    dff = (trace - f0) / scale
    slow_window = _odd_window_ms(slow_window_ms, fs)
    slow = median_filter(dff, size=slow_window, mode="nearest")
    fast = dff - slow

    # Non-event regions provide wavelet noise estimates. Plateau samples remain
    # in the continuous transform, but do not inflate the threshold estimate.
    noise_reference = supplied_non_event.copy() if supplied_non_event is not None else ~protected_plateau
    noise_reference &= np.isfinite(fast)
    if np.count_nonzero(noise_reference) < 2:
        noise_reference = np.isfinite(fast) & (~protected_plateau)
    if np.count_nonzero(noise_reference) < 2:
        noise_reference = np.isfinite(fast)

    # Template rejection defaults to off: SVD/PCA can misclassify plateau edges.
    cfg_kwargs = dict(kwargs)
    cfg_kwargs.setdefault("event_mode", "none")
    cfg_kwargs.setdefault("mask_attenuation_min", 0.0)
    cfg_kwargs.setdefault("mask_attenuation_max", 0.3)
    cfg_kwargs["threshold_sd"] = float(threshold_sd)
    cfg = HybridWaveletConfig(**cfg_kwargs)
    fast_denoised = _hybrid_wavelet_denoise_impl(
        fast,
        fs,
        cfg,
        noise_reference_mask=noise_reference,
        protected_plateau_mask=protected_plateau,
        plateau_mask_floor=plateau_mask_floor,
    )
    fast_denoised_before_drift_removal = fast_denoised.copy()
    if np.any(noise_reference):
        fast_denoised += np.median(fast[noise_reference]) - np.median(fast_denoised[noise_reference])
    if remove_residual_drift:
        drift_window_ms = slow_window_ms if residual_drift_window_ms is None else residual_drift_window_ms
        drift_window = _odd_window_ms(drift_window_ms, fs)
        fast_denoised_slow = median_filter(fast_denoised, size=drift_window, mode="nearest")
        fast_denoised = fast_denoised - fast_denoised_slow
    denoised_dff = slow + fast_denoised
    output_before_optional_level_correction = f0 + (denoised_dff * scale)
    output = output_before_optional_level_correction
    if plateau_level_correction:
        output = _apply_plateau_level_correction(output, trace, plateau, fs)

    if not return_debug:
        return output
    debug = {
        "output": output,
        "output_before_optional_level_correction": output_before_optional_level_correction,
        "f0": f0,
        "dff": dff,
        "slow": slow,
        "fast": fast,
        "fast_denoised_before_drift_removal": fast_denoised_before_drift_removal,
        "fast_denoised": fast_denoised,
        "denoised_dff": denoised_dff,
        "plateau_mask": plateau,
        "protected_plateau_mask": protected_plateau,
        "noise_reference_mask": noise_reference,
    }
    debug.update(score_denoising_result(trace, output, fs, plateau, supplied_non_event))
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


def denoise_trace_hybrid_low_plateau_tv(
    corrected: np.ndarray,
    fs: float,
    plateau_mask: np.ndarray,
    plateau_tv_weight: float = 2.0,
    boundary_protect_ms: float = 20.0,
    hybrid_denoise_fn=None,
    return_debug: bool = False,
    **hybrid_kwargs: object,
) -> np.ndarray | dict:
    """Use Hybrid denoising outside plateaus and TV denoising inside them."""
    trace = _validate_trace(corrected, fs)
    plateau = _validated_mask(plateau_mask, trace.shape, "plateau_mask")
    boundary_radius = max(0, int(round(float(boundary_protect_ms) * float(fs) / 1000.0)))
    protected_plateau = _dilate_boolean_mask(plateau, radius=boundary_radius)
    # Hybrid must not process plateau samples or their edge neighborhood.
    # Bridge those samples from reliable low-state values before denoising.
    hybrid_input = _interpolate_masked_trace(trace, reliable_mask=~protected_plateau)
    denoise_fn = denoise_hybrid_fluorescence_trace if hybrid_denoise_fn is None else hybrid_denoise_fn
    hybrid_output = np.asarray(denoise_fn(hybrid_input, fs=fs, **hybrid_kwargs), dtype=float)
    if hybrid_output.shape != trace.shape:
        raise ValueError("Hybrid denoising output must have the same shape as corrected")

    plateau_tv = trace.copy()
    for start, stop in _contiguous_true_regions(protected_plateau):
        plateau_tv[start:stop] = _tv_denoise_1d(trace[start:stop], weight=float(plateau_tv_weight))

    output = hybrid_output.copy()
    replace_mask = protected_plateau
    output[replace_mask] = plateau_tv[replace_mask]
    if not return_debug:
        return output
    return {
        "output": output,
        "hybrid_output": hybrid_output,
        "hybrid_input": hybrid_input,
        "plateau_tv_output": plateau_tv,
        "plateau_mask": plateau,
        "protected_plateau_mask": protected_plateau,
        "low_mask": ~plateau,
        "boundary_mask": protected_plateau & (~plateau),
        "plateau_replace_mask": replace_mask,
        "plateau_tv_weight": float(plateau_tv_weight),
    }
