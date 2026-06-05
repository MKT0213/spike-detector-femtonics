from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, percentile_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

from .models import DetectionParams, DetectionResult, SpikeRecord


DETECTOR_BUILD_TAG = "hybrid-detectors-2026-06-04"


@dataclass(frozen=True)
class BaselineConfig:
    # Rolling lower-percentile baseline seed used before plateau interpolation.
    rolling_pct_window_ms: float = 300.0
    rolling_pct_percentile: float = 10.0


BASELINE_CFG = BaselineConfig()


def _odd_geq(value: int, minimum: int = 5) -> int:
    out = max(minimum, int(value))
    if out % 2 == 0:
        out += 1
    return out


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    # Use a C-level median filter to keep baseline/noise estimation fast on long traces.
    return median_filter(np.asarray(x, dtype=float), size=_odd_geq(window), mode="nearest")


def _rolling_median_centered(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median - zero time shift."""
    y = np.asarray(x, dtype=float)
    if y.size == 0:
        return y.copy()
    win = _odd_geq(int(window), minimum=3)
    half = win // 2
    padded = np.pad(y, (half, half), mode="reflect")
    result = median_filter(padded, size=win, mode="constant", cval=0.0)
    return result[half : half + y.size]


def _estimate_sampling_rate(time: np.ndarray) -> float:
    t = np.asarray(time, dtype=float)
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    dt_med = float(np.median(dt))
    fs_from_seconds = 1.0 / dt_med

    # Heuristic: many recordings store time in milliseconds.
    # If direct Hz estimate is implausibly low for spike traces, reinterpret dt as ms.
    # Use a conservative dt threshold (>= 0.001 s = 1 ms) so sub-ms dt values like 0.48 ms
    # are also caught and converted to Hz.
    if fs_from_seconds < 5.0 and dt_med >= 0.001:
        return float(1000.0 / dt_med)

    return float(fs_from_seconds)


def _to_samples(milliseconds: float, fs: float, minimum: int = 1) -> int:
    return max(minimum, int(round((milliseconds / 1000.0) * fs)))


def _interpolate_invalid(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float).copy()
    valid = np.isfinite(y) & np.isfinite(t)
    if valid.sum() < 2:
        return np.nan_to_num(y, nan=0.0)
    bad = ~valid
    if np.any(bad):
        y[bad] = np.interp(t[bad], t[valid], y[valid])
    return y


def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _merge_close_regions(regions: List[Tuple[int, int]], max_gap_samples: int) -> List[Tuple[int, int]]:
    if not regions:
        return []
    merged: List[Tuple[int, int]] = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_samples:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _rolling_percentile(
    signal: np.ndarray,
    fs: float,
    window_ms: float,
    percentile: float,
) -> np.ndarray:
    """Rolling percentile with reflective padding.

    Pipeline:
        1) Reflective pad (prevents edge transients at window boundaries).
        2) Rolling percentile via scipy C-level filter.
        3) Trim padded regions.

    Args:
        signal:     Input signal (1-D).
        fs:         Sampling rate in Hz.
        window_ms:  Rolling window in milliseconds.
        percentile: Percentile to compute (0-100).

    Returns:
        Rolling percentile array of same length as signal.
    """
    y = np.asarray(signal, dtype=float)
    if y.size == 0:
        return y.copy()

    win = _odd_geq(_to_samples(window_ms, fs, minimum=5))

    # Step 1: reflective padding to prevent edge transients.
    pad = min(int(win // 2), max(0, y.size - 1))
    if pad > 0:
        y_pad = np.pad(y, (pad, pad), mode="reflect")
    else:
        y_pad = y

    # Step 2: rolling percentile (C-level, fast).
    pct = percentile_filter(y_pad, percentile=float(percentile), size=win, mode="nearest")

    # Step 3: trim padded regions.
    if pad > 0:
        pct = pct[pad : -pad]

    return pct


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Compute Otsu threshold for a 1-D array of values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        return 0.0
    if v_max <= v_min:
        return v_min

    hist, edges = np.histogram(arr, bins=max(16, int(bins)), range=(v_min, v_max))
    hist = hist.astype(float)
    total = float(np.sum(hist))
    if total <= 0.0:
        return v_min

    prob = hist / total
    omega = np.cumsum(prob)
    centers = (edges[:-1] + edges[1:]) * 0.5
    mu = np.cumsum(prob * centers)
    mu_total = float(mu[-1])

    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros_like(denom)
    valid = denom > 0.0
    sigma_b2[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]

    return float(centers[int(np.argmax(sigma_b2))])


def _refine_plateau_onset_foot(
    raw_signal: np.ndarray,
    core_mask: np.ndarray,
    excluded: np.ndarray,
    fs: float,
    noise_estimate: float,  # ignored
) -> np.ndarray:
    """Refine plateau onset to the first sustained positive slope.

    Workflow:
      - Compute raw derivative (lag 2 ms).
      - Estimate derivative noise (MAD) from the full trace (robust).
      - Slope quiet threshold = 3 x MAD_deriv.
      - From rough onset (Otsu), scan backwards to find the latest stretch
        of length `quiet_len` (3 ms) where |deriv| <= threshold.
      - Candidate onset = sample after that quiet stretch.
      - Confirmation: from candidate onset, check that deriv is positive
        and stays above threshold for at least `sustained_len` (2 ms).
        If not, move forward to the first point where a sustained positive
        run begins. This locks onto the actual start of the rise.
      - Offset refined symmetrically.
    """
    raw = np.asarray(raw_signal, dtype=float)
    core = np.asarray(core_mask, dtype=bool)
    blocked = np.asarray(excluded, dtype=bool)
    n = min(raw.size, core.size, blocked.size)
    if n <= 2:
        return core[:n].copy()

    raw = raw[:n]
    core = core[:n]
    blocked = blocked[:n]

    # Derivative
    lag = _to_samples(2.0, fs, minimum=1)
    deriv = np.zeros_like(raw)
    if lag < n:
        deriv[lag:] = raw[lag:] - raw[:-lag]

    # Estimate derivative noise from entire trace (plateaus are sparse, won't inflate MAD much)
    d_finite = deriv[np.isfinite(deriv) & ~blocked]
    if d_finite.size < 2:
        deriv_noise = 1e-9
    else:
        deriv_noise = float(np.median(np.abs(d_finite - np.median(d_finite))))
    deriv_noise = max(deriv_noise, 1e-9)

    slope_thresh = 3.0 * deriv_noise
    quiet_len = _to_samples(3.0, fs, minimum=2)      # must be quiet for 3 ms
    sustained_len = _to_samples(2.0, fs, minimum=1)  # must stay positive after onset for 2 ms

    # Raw baseline/noise for amplitude-based offset (computed once, outside the regions loop)
    raw_finite = raw[np.isfinite(raw) & ~blocked]
    if raw_finite.size > 0:
        raw_baseline = float(np.median(raw_finite))
        raw_noise = float(np.median(np.abs(raw_finite - raw_baseline)))
        raw_noise = max(raw_noise, 1e-9)
    else:
        raw_baseline = 0.0
        raw_noise = 1e-9

    refined = np.zeros(n, dtype=bool)
    regions = _contiguous_regions(core)

    for start, end in regions:
        # --- Onset refinement ---
        onset = start
        search_back = _to_samples(50.0, fs, minimum=5)
        first_idx = max(0, start - search_back)

        # Find last quiet stretch before start
        quiet_found = False
        for i in range(start - quiet_len, first_idx - 1, -1):
            if i + quiet_len > n:
                continue
            seg = deriv[i : i + quiet_len]
            if np.all(np.isfinite(seg)) and np.all(np.abs(seg) <= slope_thresh) and not np.any(blocked[i : i + quiet_len]):
                onset_candidate = i + quiet_len   # first sample after quiet
                quiet_found = True
                break
        if not quiet_found:
            onset_candidate = start

        # Confirm sustained positive slope from onset_candidate onward.
        # Move forward if needed to find the first sample where deriv stays
        # > slope_thresh for at least sustained_len.
        for j in range(onset_candidate, min(n, onset_candidate + _to_samples(20.0, fs, minimum=10))):
            if j + sustained_len > n:
                break
            window = deriv[j : j + sustained_len]
            if np.all(np.isfinite(window)) and np.all(window > slope_thresh) and not np.any(blocked[j : j + sustained_len]):
                onset = j
                break
        else:
            # Fallback: if no sustained positive run found, keep original candidate
            onset = onset_candidate

        # Ensure onset not blocked
        while onset < end and blocked[onset]:
            onset += 1
        onset = max(0, min(onset, n - 1))

        # --- Offset refinement (amplitude quiet band) ---
        amp_quiet_thresh = raw_baseline + 2.0 * raw_noise
        amp_quiet_len = _to_samples(5.0, fs, minimum=3)  # 5 ms sustained quiet

        offset = end
        search_forward = _to_samples(50.0, fs, minimum=5)
        last_idx = min(n, end + search_forward)
        offset_found = False
        for i in range(end, last_idx - amp_quiet_len + 1):
            if i + amp_quiet_len > n:
                continue
            seg = raw[i : i + amp_quiet_len]
            if np.all(np.isfinite(seg)) and np.all(seg <= amp_quiet_thresh) and not np.any(blocked[i : i + amp_quiet_len]):
                offset = i  # start of sustained quiet stretch
                offset_found = True
                break
        if not offset_found:
            offset = end
            while offset > onset and blocked[offset - 1]:
                offset -= 1
        offset = max(onset + 1, min(offset, n))

        refined[onset:offset] = True

    refined &= ~blocked
    return refined


def _interpolate_masked_series(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float).copy()
    mask_arr = np.asarray(mask, dtype=bool)
    if y.size == 0:
        return y
    if mask_arr.size != y.size:
        mask_arr = np.resize(mask_arr, y.size)

    valid = np.isfinite(y) & ~mask_arr
    if int(np.sum(valid)) < 2:
        if int(np.sum(valid)) == 1:
            y[mask_arr] = float(y[valid][0])
            return y
        return np.nan_to_num(y, nan=0.0)

    valid_idx = np.flatnonzero(valid)
    masked_idx = np.flatnonzero(mask_arr)
    if masked_idx.size == 0:
        return y

    y[masked_idx] = np.interp(masked_idx, valid_idx, y[valid])
    return y


def _estimate_local_noise_mad(
    signal: np.ndarray,
    baseline_reference: np.ndarray,
    window_samples: int,
    excluded_mask: np.ndarray,
) -> np.ndarray:
    """Estimate local noise as rolling MAD of residual using a shared window."""
    y = np.asarray(signal, dtype=float)
    b = np.asarray(baseline_reference, dtype=float)
    n = min(y.size, b.size)
    if n <= 0:
        return np.zeros(0, dtype=float)

    y = y[:n]
    b = b[:n]
    excluded = np.asarray(excluded_mask, dtype=bool)
    if excluded.size != n:
        excluded = np.resize(excluded, n)

    resid = np.abs(y - b)
    resid[~np.isfinite(resid)] = np.nan
    resid_filled = _interpolate_masked_series(
        np.nan_to_num(resid, nan=0.0),
        mask=(excluded | ~np.isfinite(resid)),
    )
    local_noise = _rolling_median(resid_filled, window=max(3, int(window_samples)))
    local_noise = np.asarray(local_noise, dtype=float)
    local_noise[~np.isfinite(local_noise)] = 0.0
    return np.maximum(local_noise, 1e-9)


def _mad_unscaled(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1e-9
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center)))
    if (not np.isfinite(mad)) or mad <= 1e-12:
        mad = float(np.std(arr))
    if (not np.isfinite(mad)) or mad <= 1e-12:
        mad = 1e-9
    return mad


def _estimate_drift_corrected_proxy(
    plateau_proxy: np.ndarray,
    valid_mask: np.ndarray,
    fs: float,
    drift_window_ms: float,
    drift_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    proxy = np.asarray(plateau_proxy, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    n = int(proxy.size)
    if valid.size != n:
        valid = np.resize(valid, n)
    if n == 0:
        return proxy.copy(), proxy.copy()

    proxy_filled = _interpolate_masked_series(
        np.nan_to_num(proxy, nan=0.0),
        mask=(~(valid & np.isfinite(proxy))),
    )
    drift = _rolling_percentile(
        proxy_filled,
        fs=fs,
        window_ms=float(drift_window_ms),
        percentile=float(np.clip(drift_percentile, 5.0, 50.0)),
    )
    return drift, proxy_filled - drift


def _detrended_noise_scale(detrended_proxy: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(valid_mask, dtype=bool)
    values = np.asarray(detrended_proxy, dtype=float)
    if valid.size != values.size:
        valid = np.resize(valid, values.size)
    finite = values[valid & np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1e-9

    center = float(np.median(finite))
    low_state = finite[finite <= center]
    if low_state.size < 8:
        low_state = finite
    noise = max(_mad_unscaled(finite), _mad_unscaled(low_state))
    return center, max(noise, 1e-9)


def _hysteresis_plateau_mask(
    signal: np.ndarray,
    high_threshold: np.ndarray,
    low_threshold: np.ndarray,
    valid_mask: np.ndarray,
    enter_len: int,
    exit_len: int,
) -> np.ndarray:
    score = np.asarray(signal, dtype=float)
    high = np.asarray(high_threshold, dtype=float)
    low = np.asarray(low_threshold, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    n = int(score.size)
    if high.size != n:
        high = np.resize(high, n)
    if low.size != n:
        low = np.resize(low, n)
    if valid.size != n:
        valid = np.resize(valid, n)
    if n == 0:
        return np.zeros(0, dtype=bool)

    finite = np.isfinite(score) & np.isfinite(high) & np.isfinite(low) & valid
    above_high = (score > high) & finite
    below_low = (score <= low) & finite
    min_enter = int(max(1, enter_len))
    min_exit = int(max(1, exit_len))
    enter_starts = _rolling_run_starts(above_high, min_enter, min_fraction=1.0)
    exit_starts = _rolling_run_starts(below_low, min_exit, min_fraction=1.0)

    mask = np.zeros(n, dtype=bool)
    cursor = 0
    while cursor < n:
        possible_entries = np.flatnonzero(enter_starts[cursor:])
        if possible_entries.size == 0:
            break
        start = cursor + int(possible_entries[0])
        exit_search_start = min(n, start + min_enter)
        possible_exits = np.flatnonzero(exit_starts[exit_search_start:])
        if possible_exits.size == 0:
            end = n
            cursor = n
        else:
            end = exit_search_start + int(possible_exits[0])
            cursor = max(end + min_exit, start + min_enter)
        if end > start:
            mask[start:end] = True

    return mask & valid


def _filter_otsu_hysteresis_candidates(
    mask: np.ndarray,
    detrended_proxy: np.ndarray,
    high_threshold: np.ndarray,
    low_threshold: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    candidate_mask = np.asarray(mask, dtype=bool)
    signal = np.asarray(detrended_proxy, dtype=float)
    high = np.asarray(high_threshold, dtype=float)
    low = np.asarray(low_threshold, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    n = int(signal.size)
    if candidate_mask.size != n:
        candidate_mask = np.resize(candidate_mask, n)
    if high.size != n:
        high = np.resize(high, n)
    if low.size != n:
        low = np.resize(low, n)
    if valid.size != n:
        valid = np.resize(valid, n)

    kept = np.zeros(n, dtype=bool)
    finite_valid = valid & np.isfinite(signal) & np.isfinite(high) & np.isfinite(low)
    for start, end in _contiguous_regions(candidate_mask & valid):
        region_valid = finite_valid[start:end]
        region_signal = signal[start:end][region_valid]
        if region_signal.size < 3:
            continue
        entry_level = float(np.median(high[start:end][region_valid]))
        exit_level = float(np.median(low[start:end][region_valid]))
        if float(np.median(region_signal)) <= entry_level:
            continue
        if float(np.percentile(region_signal, 25.0)) <= exit_level:
            continue
        kept[start:end] = True
    return kept & valid


def _estimate_baseline_simple(
    signal: np.ndarray,
    fs: float,
    excluded_mask: np.ndarray | None = None,
    plateau_median_window_ms: float = 25.0,
    baseline_rolling_window_ms: float = 300.0,
    baseline_rolling_percentile: float = 10.0,
    long_plateau_drift_window_ms: float = 2000.0,
    long_plateau_drift_percentile: float = 20.0,
    long_plateau_entry_noise_k: float = 5.0,
    long_plateau_exit_noise_k: float = 2.0,
    long_plateau_exit_fraction: float = 0.5,
    long_plateau_enter_sustain_ms: float = 10.0,
    long_plateau_exit_sustain_ms: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Plateau mask estimation from a centered rolling-median proxy (configurable window).

    The returned baseline is used for display and for the downstream
    baseline-corrected voltage trace.
    """
    y = np.asarray(signal, dtype=float)
    if y.size == 0:
        empty_mask = np.zeros(y.shape, dtype=bool)
        empty_debug = {
            "long_plateau_slow_drift": y.copy(),
            "long_plateau_detrended_proxy": y.copy(),
            "long_plateau_high_threshold_line": y.copy(),
            "long_plateau_low_threshold_line": y.copy(),
            "long_plateau_otsu_threshold_line": y.copy(),
            "long_plateau_hysteresis_mask": empty_mask,
            "long_plateau_otsu_candidate_mask": empty_mask,
        }
        return y.copy(), empty_mask, y.copy(), y.copy(), y.copy(), empty_debug

    excluded = np.zeros(y.shape, dtype=bool)
    if excluded_mask is not None:
        ex = np.asarray(excluded_mask, dtype=bool)
        n_ex = min(ex.size, excluded.size)
        if n_ex > 0:
            excluded[:n_ex] = ex[:n_ex]

    baseline_window = _to_samples(float(baseline_rolling_window_ms), fs, minimum=5)
    baseline_percentile = float(np.clip(float(baseline_rolling_percentile), 1.0, 50.0))
    baseline_input = _interpolate_masked_series(
        np.nan_to_num(y, nan=0.0),
        mask=(excluded | ~np.isfinite(y)),
    )
    baseline_seed = _rolling_percentile(
        baseline_input,
        fs=fs,
        window_ms=float(baseline_rolling_window_ms),
        percentile=baseline_percentile,
    )

    # --- plateau proxy: centered median with configurable window ---
    plateau_win_odd = _odd_geq(_to_samples(plateau_median_window_ms, fs, minimum=3))
    plateau_proxy = _rolling_median_centered(signal, window=plateau_win_odd)

    valid_for_detection = np.isfinite(plateau_proxy) & (~excluded)
    proxy_for_detection = _interpolate_masked_series(
        np.nan_to_num(plateau_proxy, nan=0.0),
        mask=(~valid_for_detection),
    )
    slow_drift, detrended_proxy = _estimate_drift_corrected_proxy(
        plateau_proxy=proxy_for_detection,
        valid_mask=valid_for_detection,
        fs=fs,
        drift_window_ms=float(long_plateau_drift_window_ms),
        drift_percentile=float(long_plateau_drift_percentile),
    )
    threshold_values = detrended_proxy[valid_for_detection & np.isfinite(detrended_proxy)]
    baseline_median, detrended_noise = _detrended_noise_scale(detrended_proxy, valid_for_detection)
    otsu_threshold = _otsu_threshold(threshold_values) if threshold_values.size else baseline_median
    entry_threshold_value = max(
        float(otsu_threshold),
        float(baseline_median) + (max(0.0, float(long_plateau_entry_noise_k)) * detrended_noise),
    )
    exit_by_noise = float(baseline_median) + (max(0.0, float(long_plateau_exit_noise_k)) * detrended_noise)
    exit_by_fraction = float(baseline_median) + (
        float(np.clip(float(long_plateau_exit_fraction), 0.0, 1.0)) * (entry_threshold_value - float(baseline_median))
    )
    exit_threshold_value = min(entry_threshold_value, max(exit_by_noise, exit_by_fraction))
    high_threshold_line = np.full(detrended_proxy.shape, entry_threshold_value, dtype=float)
    low_threshold_line = np.full(detrended_proxy.shape, exit_threshold_value, dtype=float)
    otsu_threshold_line = np.full(detrended_proxy.shape, float(otsu_threshold), dtype=float)
    enter_len = _to_samples(float(long_plateau_enter_sustain_ms), fs, minimum=1)
    exit_len = _to_samples(float(long_plateau_exit_sustain_ms), fs, minimum=1)
    hysteresis_mask = _hysteresis_plateau_mask(
        detrended_proxy,
        high_threshold_line,
        low_threshold_line,
        valid_mask=valid_for_detection,
        enter_len=enter_len,
        exit_len=exit_len,
    )
    otsu_candidate_mask = _filter_otsu_hysteresis_candidates(
        hysteresis_mask,
        detrended_proxy,
        high_threshold_line,
        low_threshold_line,
        valid_mask=valid_for_detection,
    )
    plateau_debug = {
        "long_plateau_slow_drift": slow_drift,
        "long_plateau_detrended_proxy": detrended_proxy,
        "long_plateau_high_threshold_line": high_threshold_line,
        "long_plateau_low_threshold_line": low_threshold_line,
        "long_plateau_otsu_threshold_line": otsu_threshold_line,
        "long_plateau_hysteresis_mask": hysteresis_mask,
        "long_plateau_otsu_candidate_mask": otsu_candidate_mask,
    }

    if not np.any(valid_for_detection):
        local_noise = _estimate_local_noise_mad(
            signal=y,
            baseline_reference=baseline_seed,
            window_samples=baseline_window,
            excluded_mask=excluded,
        )
        return baseline_seed, np.zeros(y.shape, dtype=bool), detrended_proxy, high_threshold_line, local_noise, plateau_debug

    plateau_mask = otsu_candidate_mask.copy()
    plateau_mask &= ~excluded
    long_plateau_min_len = _to_samples(25.0, fs, minimum=1)
    plateau_mask = _filter_short_regions(plateau_mask, min_len=long_plateau_min_len)

    if not np.any(plateau_mask):
        local_noise = _estimate_local_noise_mad(
            signal=y,
            baseline_reference=baseline_seed,
            window_samples=baseline_window,
            excluded_mask=excluded,
        )
        return baseline_seed, plateau_mask, detrended_proxy, high_threshold_line, local_noise, plateau_debug

    # --- Refine plateau edges with a foot-based onset/offset pass ---
    plateau_mask = _refine_plateau_onset_foot(
        raw_signal=y,
        core_mask=plateau_mask,
        excluded=excluded,
        fs=fs,
        noise_estimate=0.0,
    )

    plateau_mask = _suppress_short_impulse_regions(y, plateau_mask, fs=fs)
    plateau_mask = _merge_split_long_plateaus(plateau_mask, fs=fs)
    plateau_mask = _rescue_overmasked_plateau(y, plateau_mask)
    plateau_mask = _filter_short_regions(plateau_mask, min_len=long_plateau_min_len)

    # Pad plateau interpolation window so baseline stitching does not start/end too close
    # to the detected edges when onset/offset are slightly early/late.
    interp_pad = _to_samples(20.0, fs, minimum=0)
    interp_mask = _dilate_mask(plateau_mask, radius=interp_pad)
    baseline_for_display = _interpolate_masked_series(
        baseline_seed,
        mask=(interp_mask & (~excluded)),
    )

    if not np.any(plateau_mask):
        local_noise = _estimate_local_noise_mad(
            signal=y,
            baseline_reference=baseline_for_display,
            window_samples=baseline_window,
            excluded_mask=excluded,
        )
        return baseline_for_display, plateau_mask, detrended_proxy, high_threshold_line, local_noise, plateau_debug

    local_noise = _estimate_local_noise_mad(
        signal=y,
        baseline_reference=baseline_for_display,
        window_samples=baseline_window,
        excluded_mask=excluded,
    )
    return baseline_for_display, plateau_mask, detrended_proxy, high_threshold_line, local_noise, plateau_debug


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    r = int(max(0, radius))
    if r == 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((2 * r) + 1, dtype=int)
    return np.convolve(np.asarray(mask, dtype=int), kernel, mode="same") > 0


def _mask_from_regions(regions: List[Tuple[int, int]], n: int) -> np.ndarray:
    out = np.zeros(int(max(0, n)), dtype=bool)
    for start, end in regions:
        s = max(0, int(start))
        e = min(out.size, int(end))
        if e > s:
            out[s:e] = True
    return out


def _filter_short_regions(mask: np.ndarray, min_len: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or int(min_len) <= 1:
        return m
    regions = _contiguous_regions(m)
    kept = [(s, e) for s, e in regions if (e - s) >= int(min_len)]
    return _mask_from_regions(kept, m.size)


def _longest_true_run(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not np.any(m):
        return 0
    return max((end - start) for start, end in _contiguous_regions(m))


def _rolling_run_starts(mask: np.ndarray, run_len: int, min_fraction: float = 1.0) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(m.shape, dtype=bool)
    length = int(max(1, run_len))
    if m.size < length:
        return out
    kernel = np.ones(length, dtype=np.int16)
    counts = np.convolve(m.astype(np.int16), kernel, mode="valid")
    required = int(math.ceil(float(np.clip(min_fraction, 0.0, 1.0)) * float(length)))
    out[: counts.size] = counts >= max(1, required)
    return out


def _detect_short_plateaus_baseline_corrected(
    corrected_trace: np.ndarray,
    fs: float,
    excluded_mask: np.ndarray | None = None,
    long_plateau_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect short plateau events on a denoised, baseline-corrected trace.

    The signal itself decides whether the event remains above baseline. Slope is
    used to place the onset, while offset requires a sustained return near zero.
    """
    y_in = np.asarray(corrected_trace, dtype=float)
    n = int(y_in.size)
    if n <= 2:
        return np.zeros(n, dtype=bool), np.zeros((0, 6), dtype=float)

    y = y_in.copy()
    finite = np.isfinite(y)
    if int(np.sum(finite)) < 3:
        return np.zeros(n, dtype=bool), np.zeros((0, 6), dtype=float)
    if not np.all(finite):
        idx = np.arange(n, dtype=float)
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])

    blocked = np.zeros(n, dtype=bool)
    if excluded_mask is not None:
        ex = np.asarray(excluded_mask, dtype=bool)
        blocked[: min(n, ex.size)] = ex[: min(n, ex.size)]

    long_mask = np.zeros(n, dtype=bool)
    if long_plateau_mask is not None:
        lm = np.asarray(long_plateau_mask, dtype=bool)
        long_mask[: min(n, lm.size)] = lm[: min(n, lm.size)]

    valid = finite & (~blocked)
    reference = y[valid & (~long_mask)]
    if reference.size < 16:
        reference = y[valid]
    reference = reference[np.isfinite(reference)]
    if reference.size < 8:
        return np.zeros(n, dtype=bool), np.zeros((0, 6), dtype=float)

    ref_center = float(np.median(reference))
    noise = float(np.median(np.abs(reference - ref_center)))
    if (not np.isfinite(noise)) or noise <= 1e-12:
        noise = float(np.std(reference))
    if (not np.isfinite(noise)) or noise <= 1e-12:
        noise = 1e-9

    positive_ref = reference[reference > 0.0]
    if positive_ref.size >= 8:
        positive_scale = float(np.percentile(positive_ref, 90.0))
    else:
        positive_scale = float(np.percentile(np.abs(reference), 95.0))
    if (not np.isfinite(positive_scale)) or positive_scale <= 0.0:
        positive_scale = noise

    close_thresh = max(1.5 * noise, 0.03 * positive_scale, 1e-9)
    rise_thresh = max(3.0 * noise, 0.10 * positive_scale, close_thresh * 1.25)
    min_height = max(4.0 * noise, 0.14 * positive_scale, rise_thresh)

    lag = _to_samples(1.0, fs, minimum=1)
    deriv = np.zeros(n, dtype=float)
    if lag < n:
        deriv[lag:] = y[lag:] - y[:-lag]
    deriv_reference = deriv[valid & (~long_mask)]
    deriv_reference = deriv_reference[np.isfinite(deriv_reference)]
    if deriv_reference.size < 8:
        deriv_reference = deriv[np.isfinite(deriv)]
    d_center = float(np.median(deriv_reference)) if deriv_reference.size else 0.0
    d_noise = float(np.median(np.abs(deriv_reference - d_center))) if deriv_reference.size else 0.0
    if (not np.isfinite(d_noise)) or d_noise <= 1e-12:
        d_noise = 1e-9
    slope_thresh = max(3.0 * d_noise, 1e-9)

    hard_min_duration_ms = 20.0
    min_duration = _to_samples(hard_min_duration_ms, fs, minimum=3)
    return_len = _to_samples(10.0, fs, minimum=3)
    min_sustain = _to_samples(8.0, fs, minimum=2)
    onset_back = _to_samples(25.0, fs, minimum=3)
    onset_forward = _to_samples(8.0, fs, minimum=2)
    max_onset_lead = _to_samples(12.0, fs, minimum=1)
    min_high_run = _to_samples(3.0, fs, minimum=2)
    pre_quiet_window = _to_samples(25.0, fs, minimum=3)
    triangular_duration = _to_samples(90.0, fs, minimum=min_duration)
    internal_edge = _to_samples(4.0, fs, minimum=1)

    close_to_baseline = (y <= close_thresh) & valid
    close_run_starts = _rolling_run_starts(close_to_baseline, return_len, min_fraction=0.80)
    high = (y > rise_thresh) & valid & (~long_mask)

    mask = np.zeros(n, dtype=bool)
    events: List[List[float]] = []
    next_available = 0

    for high_start, high_end in _contiguous_regions(high):
        if high_start < next_available:
            continue
        if high_start <= 0 or high_start >= n:
            continue
        if (high_end - high_start) < min_high_run:
            continue
        confirm_stop = min(n, high_start + min_sustain)
        if confirm_stop <= high_start:
            continue
        if float(np.mean(y[high_start:confirm_stop] > close_thresh)) < 0.60:
            continue

        pre_lo = max(0, high_start - pre_quiet_window)
        if not np.any(close_to_baseline[pre_lo:high_start]):
            continue

        search_lo = max(0, high_start - onset_back)
        search_hi = min(n, high_start + onset_forward)
        strong_window = (deriv[search_lo:search_hi] > slope_thresh) & valid[search_lo:search_hi]
        strong_candidates = np.flatnonzero(strong_window)
        if strong_candidates.size == 0:
            continue

        quiet_candidates = np.flatnonzero(close_to_baseline[search_lo : high_start + 1])
        if quiet_candidates.size > 0:
            baseline_exit = search_lo + int(quiet_candidates[-1]) + 1
        else:
            baseline_exit = high_start

        strong_after_exit = np.flatnonzero(
            (deriv[baseline_exit:search_hi] > slope_thresh) & valid[baseline_exit:search_hi]
        )
        if strong_after_exit.size > 0:
            onset = baseline_exit + int(strong_after_exit[0])
        else:
            onset = search_lo + int(strong_candidates[0])
        onset = max(onset, high_start - max_onset_lead)
        onset = max(0, min(onset, n - 1))
        if blocked[onset] or long_mask[onset]:
            continue

        offset_search = max(onset + 1, high_end)
        if offset_search >= n:
            continue
        close_offsets = np.flatnonzero(close_run_starts[offset_search:])
        if close_offsets.size == 0:
            continue
        offset = offset_search + int(close_offsets[0])
        offset = max(onset + 1, min(offset, n))

        if offset - onset < min_duration:
            continue
        if np.any(blocked[onset:offset]) or np.any(long_mask[onset:offset]):
            continue

        seg = y[onset:offset]
        seg = seg[np.isfinite(seg)]
        if seg.size < min_duration:
            continue

        peak_search_stop = min(seg.size, max(min_sustain * 3, _to_samples(25.0, fs, minimum=3)))
        if peak_search_stop >= 3:
            early = seg[:peak_search_stop]
            early_peaks, _ = find_peaks(
                early,
                height=rise_thresh,
                prominence=max(noise, 1e-9),
                distance=_to_samples(1.0, fs, minimum=1),
            )
            if early_peaks.size > 0:
                first_peak_idx = int(early_peaks[0])
                first_peak_value = float(seg[first_peak_idx])
                if np.isfinite(first_peak_value) and first_peak_value > close_thresh:
                    drop_threshold = max(3.0 * noise, 0.30 * (first_peak_value - close_thresh), 1e-9)
                    post_peak = seg[first_peak_idx + 1 :]
                    first_drop_offsets = np.flatnonzero(post_peak <= (first_peak_value - drop_threshold))
                    if first_drop_offsets.size > 0:
                        first_drop_idx = first_peak_idx + 1 + int(first_drop_offsets[0])
                        return_check_len = max(return_len, _to_samples(3.0, fs, minimum=2))
                        min_valid_in_return = max(1, int(math.ceil(0.70 * float(return_check_len))))
                        post_drop_start = onset + first_drop_idx
                        return_scan_horizon = max(6 * return_len, _to_samples(80.0, fs, minimum=return_check_len))
                        check_stop = min(n, offset + return_check_len, post_drop_start + return_scan_horizon)
                        baseline_like_return = False
                        for win_start in range(post_drop_start, check_stop - return_check_len + 1):
                            win_stop = win_start + return_check_len
                            window = y[win_start:win_stop]
                            window_valid = valid[win_start:win_stop] & (~long_mask[win_start:win_stop]) & np.isfinite(window)
                            if int(np.sum(window_valid)) < min_valid_in_return:
                                continue
                            window_values = window[window_valid]
                            close_fraction = float(np.mean(window_values <= close_thresh))
                            median_level = float(np.median(window_values))
                            p75_level = float(np.percentile(window_values, 75.0))
                            level_score = 1.0 - float(
                                np.clip((median_level - close_thresh) / max(close_thresh, 1e-9), 0.0, 1.0)
                            )
                            baseline_return_confidence = (0.70 * close_fraction) + (0.30 * level_score)
                            if (
                                baseline_return_confidence >= 0.80
                                and close_fraction >= 0.70
                                and p75_level <= (2.0 * close_thresh)
                            ):
                                baseline_like_return = True
                                break
                        if baseline_like_return:
                            continue

        above_close = seg > close_thresh
        occupancy = float(np.mean(above_close)) if above_close.size else 0.0
        if occupancy < 0.45:
            continue
        if _longest_true_run(above_close) < min_sustain:
            continue

        height_values = seg[above_close]
        if height_values.size == 0:
            continue
        plateau_height = float(np.median(height_values))
        peak = float(np.max(seg))
        if (not np.isfinite(plateau_height)) or plateau_height < min_height or peak <= 0.0:
            continue

        height_ratio = plateau_height / max(peak, 1e-9)
        if (offset - onset) <= triangular_duration and height_ratio < 0.28:
            continue

        inner_start = min(seg.size, internal_edge)
        inner_stop = max(inner_start, seg.size - internal_edge)
        inner = seg[inner_start:inner_stop]
        internal_spikes = False
        if inner.size >= 3:
            peaks, _ = find_peaks(
                inner,
                height=plateau_height + (3.0 * noise),
                prominence=max(3.0 * noise, 1e-9),
                distance=_to_samples(2.0, fs, minimum=1),
            )
            internal_spikes = bool(peaks.size > 0)

        duration_ms = float((offset - onset) * 1000.0 / max(float(fs), 1e-9))
        if duration_ms < hard_min_duration_ms:
            continue
        snr_score = min(1.0, max(0.0, (plateau_height / max(noise, 1e-9)) / 8.0))
        duration_score = min(1.0, max(0.0, duration_ms / 80.0))
        occupancy_score = min(1.0, max(0.0, (occupancy - 0.45) / 0.45))
        shape_score = min(1.0, max(0.0, height_ratio / 0.65))
        confidence = float(
            min(1.0, max(0.0, (0.35 * snr_score) + (0.25 * duration_score) + (0.25 * occupancy_score) + (0.15 * shape_score)))
        )
        if confidence < 0.35:
            continue

        mask[onset:offset] = True
        events.append(
            [
                float(onset),
                float(offset),
                duration_ms,
                plateau_height,
                1.0 if internal_spikes else 0.0,
                confidence,
            ]
        )
        next_available = offset + return_len

    if not events:
        return mask, np.zeros((0, 6), dtype=float)
    return mask, np.asarray(events, dtype=float)


def _merge_plateau_masks(long_mask: np.ndarray, short_mask: np.ndarray, fs: float) -> np.ndarray:
    n = max(int(np.asarray(long_mask).size), int(np.asarray(short_mask).size))
    combined = np.zeros(n, dtype=bool)
    if long_mask is not None:
        lm = np.asarray(long_mask, dtype=bool)
        combined[: lm.size] |= lm
    if short_mask is not None:
        sm = np.asarray(short_mask, dtype=bool)
        combined[: sm.size] |= sm
    if not np.any(combined):
        return combined
    merge_gap = _to_samples(5.0, fs, minimum=1)
    return _mask_from_regions(_merge_close_regions(_contiguous_regions(combined), merge_gap), n)


def _suppress_short_impulse_regions(
    raw_signal: np.ndarray,
    plateau_mask: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Suppress short, impulse-like regions that are simple spikes, not plateaus."""
    raw = np.asarray(raw_signal, dtype=float)
    mask = np.asarray(plateau_mask, dtype=bool)
    n = min(raw.size, mask.size)
    if n <= 0:
        return mask[:0]

    raw = raw[:n]
    mask = mask[:n].copy()
    if not np.any(mask):
        return mask

    short_len = _to_samples(80.0, fs, minimum=2)

    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size == 0:
        return mask
    med = float(np.median(finite_raw))
    global_noise = float(np.median(np.abs(finite_raw - med)))
    global_noise = max(1e-9, global_noise)
    span_min = 4.0 * global_noise

    kept: List[Tuple[int, int]] = []
    for start, end in _contiguous_regions(mask):
        if (end - start) <= short_len:
            seg = raw[start:end]
            seg = seg[np.isfinite(seg)]
            if seg.size >= 8:
                p5, p95 = np.percentile(seg, [5.0, 95.0])
                span = float(p95 - p5)
            elif seg.size > 0:
                span = float(np.max(seg) - np.min(seg))
            else:
                span = 0.0
            # A short high-span region is spike-shaped. A brief but genuinely
            # flat-topped event remains eligible as a plateau.
            if span >= span_min:
                continue
        kept.append((int(start), int(end)))

    return _mask_from_regions(kept, n)


def _merge_split_long_plateaus(mask: np.ndarray, fs: float) -> np.ndarray:
    """Merge adjacent long plateau regions separated by a short interruption."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not np.any(m):
        return m

    regions = _contiguous_regions(m)
    if len(regions) < 2:
        return m

    min_long = _to_samples(140.0, fs, minimum=2)
    max_join_gap = _to_samples(120.0, fs, minimum=1)

    merged: List[Tuple[int, int]] = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        prev_len = int(prev_end - prev_start)
        cur_len = int(end - start)
        gap = int(start - prev_end)

        if prev_len >= min_long and cur_len >= min_long and gap <= max_join_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return _mask_from_regions(merged, m.size)


def _rescue_overmasked_plateau(raw_signal: np.ndarray, plateau_mask: np.ndarray) -> np.ndarray:
    """If plateau coverage is implausibly large, keep only high-contrast regions."""
    raw = np.asarray(raw_signal, dtype=float)
    mask = np.asarray(plateau_mask, dtype=bool)
    n = min(raw.size, mask.size)
    if n <= 0:
        return mask[:0]

    raw = raw[:n]
    mask = mask[:n].copy()
    if not np.any(mask):
        return mask

    # Trigger only for pathological over-labeling; normal traces are untouched.
    cover = float(np.mean(mask)) if n > 0 else 0.0
    if cover < 0.45:
        return mask

    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return mask
    g_med = float(np.median(finite))
    g_noise = float(np.median(np.abs(finite - g_med)))
    g_noise = max(1e-9, g_noise)

    candidates: List[Tuple[Tuple[int, int], float]] = []
    for start, end in _contiguous_regions(mask):
        seg = raw[start:end]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        lift = float(np.median(seg) - g_med)
        if seg.size >= 8:
            p5, p95 = np.percentile(seg, [5.0, 95.0])
            span = float(p95 - p5)
        else:
            span = float(np.max(seg) - np.min(seg))

        if lift >= (1.8 * g_noise) or span >= (7.0 * g_noise):
            score = lift * math.sqrt(max(1.0, float(end - start)))
            candidates.append(((int(start), int(end)), float(score)))

    if not candidates:
        return np.zeros(n, dtype=bool)

    # Keep a few strongest regions to avoid re-expanding into broad low-contrast bands.
    candidates.sort(key=lambda item: item[1], reverse=True)
    kept_regions = [region for region, _ in candidates[:3]]
    return _mask_from_regions(kept_regions, n)


def _detect_startup_transient_mask(signal: np.ndarray, fs: float, startup_exclusion_ms: float = 50.0) -> np.ndarray:
    """Mask a startup window (configurable, default 50 ms) for every trace."""
    y = np.asarray(signal, dtype=float)
    n = y.size
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    end = min(n, _to_samples(startup_exclusion_ms, fs, minimum=1))
    if end > 0:
        out[:end] = True
    return out


def _bridge_short_artifact_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or max_gap_samples <= 0:
        return m
    merged = _merge_close_regions(_contiguous_regions(m), max_gap_samples=int(max_gap_samples))
    return _mask_from_regions(merged, m.size)


def _expand_artifact_block_mask(mask: np.ndarray, fs: float) -> np.ndarray:
    # Artifact masks often mark individual dips; bridge short inter-dip gaps so
    # peaks in the same artifact epoch cannot slip through.
    blocked = np.asarray(mask, dtype=bool)
    if blocked.size == 0 or not np.any(blocked):
        return blocked

    bridge_primary = _to_samples(180.0, fs, minimum=1)
    bridge_secondary = _to_samples(90.0, fs, minimum=1)
    edge_pad = _to_samples(45.0, fs, minimum=1)

    blocked = _bridge_short_artifact_gaps(blocked, max_gap_samples=bridge_primary)
    blocked = _dilate_mask(blocked, radius=edge_pad)
    blocked = _bridge_short_artifact_gaps(blocked, max_gap_samples=bridge_secondary)
    return blocked
def _butterworth_lowpass(signal: np.ndarray, fs: float, cutoff_hz: float, order: int = 5) -> np.ndarray:
    y = np.asarray(signal, dtype=float)
    if y.size < 8:
        return y.copy()

    finite_idx = np.flatnonzero(np.isfinite(y))
    if finite_idx.size < y.size:
        if finite_idx.size < 2:
            return np.nan_to_num(y, nan=0.0)
        all_idx = np.arange(y.size)
        y = y.copy()
        missing = ~np.isfinite(y)
        y[missing] = np.interp(all_idx[missing], all_idx[finite_idx], y[finite_idx])

    # Reflective edge padding reduces startup/end transients in filtfilt output.
    edge_pad = min(max(1, _to_samples(1500.0, fs, minimum=1)), max(0, y.size - 2))
    if edge_pad > 0:
        y_work = np.pad(y, (edge_pad, edge_pad), mode="reflect")
    else:
        y_work = y

    nyquist = 0.5 * max(1e-6, float(fs))
    cutoff = min(max(1e-6, float(cutoff_hz)), 0.95 * nyquist)
    wn = cutoff / nyquist
    if not (0.0 < wn < 1.0):
        return y.copy()
    sos = butter(int(max(1, order)), wn, btype="low", analog=False, output="sos")
    max_pad = max(0, y_work.size - 1)
    padlen = min(max_pad, 3 * (2 * sos.shape[0] + 1))
    if padlen <= 0:
        return y.copy()
    try:
        filtered = sosfiltfilt(sos, y_work, padlen=padlen)
        if edge_pad > 0:
            filtered = filtered[edge_pad:-edge_pad]

        # Guard against rare numerical blow-ups in very low-cutoff regimes.
        in_scale = float(np.nanpercentile(np.abs(y), 99)) if y.size else 0.0
        out_scale = float(np.nanpercentile(np.abs(filtered), 99)) if filtered.size else 0.0
        in_scale = max(1e-9, in_scale)
        if (not np.all(np.isfinite(filtered))) or (out_scale > 50.0 * in_scale):
            sigma = max(1.0, _to_samples(600.0, fs, minimum=1) / 6.0)
            return gaussian_filter1d(y, sigma=float(sigma), mode="nearest", truncate=2.0)

        return filtered
    except Exception:
        return y.copy()


def _butterworth_highpass(signal: np.ndarray, fs: float, cutoff_hz: float, order: int = 3) -> np.ndarray:
    y = np.asarray(signal, dtype=float)
    if y.size < 8:
        return y.copy()

    finite_idx = np.flatnonzero(np.isfinite(y))
    if finite_idx.size < y.size:
        if finite_idx.size < 2:
            return np.nan_to_num(y, nan=0.0)
        all_idx = np.arange(y.size)
        y = y.copy()
        missing = ~np.isfinite(y)
        y[missing] = np.interp(all_idx[missing], all_idx[finite_idx], y[finite_idx])

    edge_pad = min(max(1, _to_samples(1500.0, fs, minimum=1)), max(0, y.size - 2))
    if edge_pad > 0:
        y_work = np.pad(y, (edge_pad, edge_pad), mode="reflect")
    else:
        y_work = y

    nyquist = 0.5 * max(1e-6, float(fs))
    cutoff = min(max(1e-6, float(cutoff_hz)), 0.95 * nyquist)
    wn = cutoff / nyquist
    if not (0.0 < wn < 1.0):
        return y.copy()
    sos = butter(int(max(1, order)), wn, btype="high", analog=False, output="sos")
    max_pad = max(0, y_work.size - 1)
    padlen = min(max_pad, 3 * (2 * sos.shape[0] + 1))
    if padlen <= 0:
        return y.copy()
    try:
        filtered = sosfiltfilt(sos, y_work, padlen=padlen)
        if edge_pad > 0:
            filtered = filtered[edge_pad:-edge_pad]
        return filtered
    except Exception:
        return y.copy()


def _compute_dff(smoothed: np.ndarray, baseline: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    f0_window = max(5, _to_samples(2000.0, fs, minimum=5))
    f0 = _rolling_median(baseline, window=f0_window)
    denom = np.maximum(1e-9, np.abs(f0))
    dff = (smoothed - f0) / denom
    return dff, f0


def _crossing_index(y: np.ndarray, level: float, i1: int, i2: int) -> float:
    """Linearly interpolate a threshold crossing between adjacent samples."""
    y1 = float(y[i1])
    y2 = float(y[i2])
    if not np.isfinite(y1) or not np.isfinite(y2) or y2 == y1:
        return float("nan")
    return float(i1 + ((level - y1) / (y2 - y1)))


def _measure_spike_fwhm_ms(
    signal: np.ndarray,
    peak_index: int,
    fs: float,
    half_window_ms: float = 5.0,
) -> float:
    """Measure local spike FWHM without treating its plateau as baseline."""
    y = np.asarray(signal, dtype=float)
    peak = int(peak_index)
    if y.size == 0 or peak <= 0 or peak >= y.size - 1:
        return float("nan")

    half_window = _to_samples(half_window_ms, fs, minimum=2)
    start = max(0, peak - half_window)
    stop = min(y.size, peak + half_window + 1)
    left = y[start : peak + 1]
    right = y[peak:stop]
    if left.size < 2 or right.size < 2 or not np.all(np.isfinite(y[start:stop])):
        return float("nan")

    # The higher neighboring valley is the local shoulder of a spike riding on
    # a plateau. Measuring from the global trace baseline would overestimate it.
    shoulder = max(float(np.min(left)), float(np.min(right)))
    peak_value = float(y[peak])
    amplitude = peak_value - shoulder
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return float("nan")
    level = shoulder + (0.5 * amplitude)

    left_cross = float("nan")
    for idx in range(peak - 1, start - 1, -1):
        if y[idx] <= level <= y[idx + 1]:
            left_cross = _crossing_index(y, level, idx, idx + 1)
            break

    right_cross = float("nan")
    for idx in range(peak, stop - 1):
        if y[idx] >= level >= y[idx + 1]:
            right_cross = _crossing_index(y, level, idx, idx + 1)
            break

    if not np.isfinite(left_cross) or not np.isfinite(right_cross):
        return float("nan")
    return float((right_cross - left_cross) * 1000.0 / float(fs))


def _finite_noise_scale(values: np.ndarray, mode: str) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1e-9
    if mode == "mad":
        center = float(np.median(finite))
        lower = finite[finite < center]
        source = lower if lower.size else finite
        noise = float(np.median(np.abs(source - np.median(source))) / 0.6745)
    else:
        center = float(np.mean(finite))
        lower = finite[finite < center]
        source = lower if lower.size else finite
        noise = float(np.std(source))
    return center, max(noise, 1e-9)


def _spike_snr_trace(signal: np.ndarray, fs: float, window_ms: float = 12.5) -> np.ndarray:
    """Lagged SNR transform for the SNR spike-threshold mode."""
    trace = np.asarray(signal, dtype=float)
    lag = _to_samples(window_ms, fs, minimum=1)
    noise = max(float(np.mean(np.abs(np.diff(trace)))) if trace.size > 1 else 0.0, 1e-9)
    transformed = np.zeros_like(trace)
    if lag < trace.size:
        transformed[lag:] = (trace[lag:] - trace[:-lag]) / noise
    return transformed


def _detect_spikes_threshold(
    raw_signal: np.ndarray,
    time: np.ndarray,
    baseline: np.ndarray,
    blocked: np.ndarray,
    fs: float,
    mode: str,
    threshold_k: float,
    prominence_k: float,
    min_separation_ms: float,
    fwhm_half_window_ms: float,
    trace_name: str,
) -> Tuple[List[SpikeRecord], np.ndarray, np.ndarray]:
    """Run MAD, STD, and SNR spike threshold modes."""
    raw = np.asarray(raw_signal, dtype=float)
    t = np.asarray(time, dtype=float)
    base = np.asarray(baseline, dtype=float)
    excluded = np.asarray(blocked, dtype=bool)
    analysis = _spike_snr_trace(raw, fs) if mode == "snr" else raw.copy()
    analysis[excluded] = -np.inf
    refractory = _to_samples(min_separation_ms, fs, minimum=1)

    if mode == "snr":
        threshold = float(threshold_k)
        prominence = None
    else:
        center, noise = _finite_noise_scale(raw[~excluded], mode=mode)
        threshold = center + (float(threshold_k) * noise)
        prominence = float(prominence_k) * noise

    peak_kwargs: Dict[str, object] = {"height": threshold, "distance": refractory}
    if prominence is not None:
        peak_kwargs["prominence"] = prominence
    peaks, properties = find_peaks(analysis, **peak_kwargs)
    threshold_line = np.full(raw.shape, threshold, dtype=float)
    noise_for_records = max(float(np.mean(np.abs(np.diff(raw[~excluded])))) if np.count_nonzero(~excluded) > 1 else 0.0, 1e-9)
    spikes: List[SpikeRecord] = []
    widths = np.asarray(properties.get("widths", np.ones(peaks.shape)), dtype=float)
    prominences = np.asarray(properties.get("prominences", np.zeros(peaks.shape)), dtype=float)
    for idx, peak in enumerate(peaks):
        peak = int(peak)
        baseline_value = float(base[peak])
        amplitude = float(raw[peak] - baseline_value)
        spikes.append(
            SpikeRecord(
                trace_name=trace_name,
                candidate_index=peak,
                spike_index=peak,
                spike_time=float(t[peak]),
                spike_amplitude_raw_or_corrected=float(raw[peak]),
                baseline_at_spike=baseline_value,
                amplitude_above_baseline=amplitude,
                prominence=float(prominences[idx]) if idx < prominences.size else 0.0,
                width=float(widths[idx]) if idx < widths.size else 1.0,
                local_noise_estimate=noise_for_records,
                snr=float(amplitude / noise_for_records),
                detection_threshold_used=threshold,
                fwhm_ms=_measure_spike_fwhm_ms(raw, peak, fs, half_window_ms=fwhm_half_window_ms),
                status="pending",
                source="auto",
                notes=f"threshold-{mode}",
            )
        )
    return spikes, threshold_line, analysis


def detect_spikes(
    trace_name: str,
    time: np.ndarray,
    corrected: np.ndarray,
    params: DetectionParams,
    artifact_mask: np.ndarray | None = None,
    baseline_override: np.ndarray | None = None,
    plateau_mask_override: np.ndarray | None = None,
    sampling_rate_hz: float | None = None,
    short_plateau_trace_override: np.ndarray | None = None,
) -> DetectionResult:
    signal = _interpolate_invalid(time, corrected)
    fs = float(sampling_rate_hz) if sampling_rate_hz is not None and np.isfinite(sampling_rate_hz) and sampling_rate_hz > 0 else _estimate_sampling_rate(time)

    # Per user request: do not apply any filters. Keep only interpolation of invalid samples.
    processed_signal = signal
    # No smoothing/filtering: use the raw interpolated signal as the smoothed channel.
    smoothed = processed_signal

    blocked = np.zeros_like(smoothed, dtype=bool)
    if artifact_mask is not None:
        mask_arr = np.asarray(artifact_mask, dtype=bool)
        n = min(mask_arr.size, blocked.size)
        if n > 0:
            blocked[:n] = mask_arr[:n]

    startup_block = _detect_startup_transient_mask(smoothed, fs=fs, startup_exclusion_ms=params.startup_exclusion_ms)
    if startup_block.size > 0:
        n_start = min(startup_block.size, blocked.size)
        if n_start > 0:
            blocked[:n_start] |= startup_block[:n_start]

    if np.any(blocked):
        blocked = _expand_artifact_block_mask(blocked, fs=fs)

    # Global raw baseline/noise for spike thresholding on plateau-restricted windows.
    raw_finite = smoothed[np.isfinite(smoothed) & (~blocked)]
    if raw_finite.size == 0:
        raw_finite = smoothed[np.isfinite(smoothed)]
    if raw_finite.size > 0:
        raw_baseline = float(np.median(raw_finite))
        raw_noise = float(np.median(np.abs(raw_finite - raw_baseline)))
        raw_noise = max(raw_noise, 1e-9)
    else:
        raw_baseline = 0.0
        raw_noise = 1e-9

    # Keep plateau detection, but do not apply baseline correction to the signal.
    (
        baseline,
        plateau_mask,
        plateau_debug_signal,
        plateau_debug_threshold_line,
        local_noise_estimate,
        long_plateau_debug,
    ) = _estimate_baseline_simple(
        smoothed,
        fs=fs,
        excluded_mask=blocked,
        plateau_median_window_ms=params.plateau_median_window_ms,
        baseline_rolling_window_ms=params.baseline_rolling_window_ms,
        baseline_rolling_percentile=params.baseline_rolling_percentile,
        long_plateau_drift_window_ms=params.long_plateau_drift_window_ms,
        long_plateau_drift_percentile=params.long_plateau_drift_percentile,
        long_plateau_entry_noise_k=params.long_plateau_entry_noise_k,
        long_plateau_exit_noise_k=params.long_plateau_exit_noise_k,
        long_plateau_exit_fraction=params.long_plateau_exit_fraction,
        long_plateau_enter_sustain_ms=params.long_plateau_enter_sustain_ms,
        long_plateau_exit_sustain_ms=params.long_plateau_exit_sustain_ms,
    )
    if baseline_override is not None:
        override = np.asarray(baseline_override, dtype=float)
        if override.shape != baseline.shape:
            raise ValueError("baseline_override must have the same shape as corrected")
        baseline = override
    if plateau_mask_override is not None:
        override = np.asarray(plateau_mask_override, dtype=bool)
        if override.shape != plateau_mask.shape:
            raise ValueError("plateau_mask_override must have the same shape as corrected")
        plateau_mask = override & (~blocked)

    # The long detector is used here as the baseline-exclusion/protection mask.
    long_plateau_mask = np.asarray(plateau_mask, dtype=bool).copy()
    run_short_plateau_detector = baseline_override is not None
    if run_short_plateau_detector:
        short_plateau_signal = processed_signal
        if short_plateau_trace_override is not None:
            override = np.asarray(short_plateau_trace_override, dtype=float)
            if override.shape != processed_signal.shape:
                raise ValueError("short_plateau_trace_override must have the same shape as corrected")
            short_plateau_signal = _interpolate_invalid(time, override)
        baseline_corrected_signal = short_plateau_signal - baseline
        short_plateau_mask, short_plateau_events = _detect_short_plateaus_baseline_corrected(
            baseline_corrected_signal,
            fs=fs,
            excluded_mask=blocked,
            long_plateau_mask=long_plateau_mask,
        )
        short_plateau_mask &= ~long_plateau_mask
    else:
        short_plateau_mask = np.zeros_like(long_plateau_mask, dtype=bool)
        short_plateau_events = np.zeros((0, 6), dtype=float)

    plateau_mask = _merge_plateau_masks(long_plateau_mask, short_plateau_mask, fs=fs)

    # Compute a single noise estimate from truly quiet regions.
    quiet_mask = (~plateau_mask) & (~blocked)
    quiet_samples = processed_signal[quiet_mask & np.isfinite(processed_signal)]
    if quiet_samples.size > 0:
        quiet_baseline = float(np.median(quiet_samples))
        quiet_noise = float(np.median(np.abs(quiet_samples - quiet_baseline)))
        quiet_noise = max(quiet_noise, 1e-9)
    else:
        quiet_baseline = float(raw_baseline)
        quiet_noise = float(raw_noise)

    candidate_windows = _contiguous_regions(plateau_mask & (~blocked))

    merge_gap = _to_samples(float(params.min_separation_ms) * 0.5, fs, minimum=1)
    candidate_windows = _merge_close_regions(candidate_windows, merge_gap)

    mode = str(params.detection_mode).strip().lower()
    old_prefix = "neuro" + "box" + "_"
    if mode.startswith(old_prefix):
        mode = mode.removeprefix(old_prefix)
    if mode not in {"mad", "std", "snr"}:
        mode = "mad"
    spikes, threshold, detector_signal = _detect_spikes_threshold(
        raw_signal=processed_signal,
        time=time,
        baseline=baseline,
        blocked=blocked,
        fs=fs,
        mode=mode,
        threshold_k=float(params.spike_noise_k),
        prominence_k=float(params.spike_prominence_k),
        min_separation_ms=float(params.min_separation_ms),
        fwhm_half_window_ms=float(params.spike_zoom_half_window_ms),
        trace_name=trace_name,
    )

    qc_plot_data: Dict[str, np.ndarray] = {
        "corrected_trace": processed_signal,
        "input_trace": signal,
        "highpass_signal": processed_signal,
        "confirmation_threshold_line": threshold,
        "dff": np.zeros_like(smoothed, dtype=float),
        "residual": detector_signal,
        "baseline_seed": baseline,
        "baseline_mask": plateau_mask,
        "long_plateau_mask": long_plateau_mask,
        "short_plateau_mask": short_plateau_mask,
        # Explicit aliases documenting the different roles without changing UI/export naming.
        "baseline_exclusion_mask": long_plateau_mask,
        "detected_plateau_mask": short_plateau_mask,
        "short_plateau_event_table": short_plateau_events,
        "local_noise_estimate": local_noise_estimate,
        "plateau_debug_signal": plateau_debug_signal,
        "plateau_debug_threshold_line": plateau_debug_threshold_line,
    }
    qc_plot_data.update(long_plateau_debug)

    return DetectionResult(
        spikes=spikes,
        processed_signal=processed_signal,
        smoothed=smoothed,
        baseline=baseline,
        threshold=threshold,
        candidate_windows=candidate_windows,
        qc_plot_data=qc_plot_data,
    )
