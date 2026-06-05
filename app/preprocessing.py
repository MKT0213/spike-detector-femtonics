from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_PARAMS = {
    "absolute_low": 120.0,
    "min_depth": 8.0,
    "drop_start_abs": 6.0,
    "recover_frac": 0.9,
    "recover_margin": 2.0,
    "max_recover_len": 120,
    "baseline_window": 80,
    "join_gap": 1,
    "pad_left": 1,
    "pad_right": 2,
    "min_region_len": 2,
    "max_region_len": 250,
}


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _regions_to_mask(regions: list[tuple[int, int]], n: int, pad_left: int = 0, pad_right: int = 0) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in regions:
        start2 = max(0, start - pad_left)
        end2 = min(n, end + pad_right)
        mask[start2:end2] = True
    return mask


def _merge_regions(regions: list[tuple[int, int]], join_gap: int = 0) -> list[tuple[int, int]]:
    if not regions:
        return []
    regions = sorted(regions, key=lambda region: region[0])
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + join_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_artifact_mask(signal: np.ndarray, params: dict[str, float] | None = None) -> np.ndarray:
    config = dict(DEFAULT_ARTIFACT_PARAMS)
    if params:
        config.update(params)

    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < 3 or np.all(np.isnan(x)):
        return np.zeros(n, dtype=bool)

    baseline_window = int(config.get("baseline_window", 80))
    baseline = pd.Series(x).rolling(window=baseline_window, min_periods=1).median().to_numpy()

    depth = baseline - x
    dx = np.diff(x, prepend=x[0])

    abs_low = float(config.get("absolute_low", 120.0))
    min_depth = float(config.get("min_depth", 8.0))
    drop_start_abs = float(config.get("drop_start_abs", 6.0))

    low_and_deep = (x <= abs_low) & (depth >= min_depth)
    falling_now = dx <= -drop_start_abs
    falling_prev = np.r_[False, falling_now[:-1]]

    seed = low_and_deep & (falling_now | falling_prev)
    regions = _contiguous_regions(seed)

    grown: list[tuple[int, int]] = []
    recover_frac = float(config.get("recover_frac", 0.9))
    recover_margin = float(config.get("recover_margin", 2.0))
    max_recover_len = int(config.get("max_recover_len", 120))

    for start, end in regions:
        b0 = max(0, start - baseline_window)
        local_baseline = np.nanmedian(x[b0:start]) if start > b0 else baseline[start]
        trough = np.nanmin(x[start:end])

        if np.isnan(local_baseline) or np.isnan(trough):
            grown.append((start, end))
            continue

        target = max(abs_low + recover_margin, trough + recover_frac * (local_baseline - trough))

        left = start
        while left > 0 and (x[left - 1] > x[left]) and ((x[left - 1] - x[left]) >= 0.5):
            left -= 1

        right = end
        max_end = min(n, start + max_recover_len)
        while right < max_end and x[right] < target:
            right += 1

        grown.append((left, right))

    regions = _merge_regions(grown, join_gap=int(config.get("join_gap", 0)))

    min_len = int(config.get("min_region_len", 1))
    max_len = int(config.get("max_region_len", 10**9))
    regions = [
        (start, end)
        for (start, end) in regions
        if min_len <= (end - start) <= max_len and np.nanmin(x[start:end]) <= abs_low
    ]

    return _regions_to_mask(
        regions,
        n,
        pad_left=int(config.get("pad_left", 0)),
        pad_right=int(config.get("pad_right", 0)),
    )


def interpolate_artifacts(time_values: np.ndarray, signal: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(signal, dtype=float)
    mask = np.asarray(artifact_mask, dtype=bool)

    n = min(t.size, y.size, mask.size)
    corrected = y[:n].copy()
    if n == 0:
        return corrected

    t = t[:n]
    y = y[:n]
    mask = mask[:n]

    finite_time = np.isfinite(t)
    finite_signal = np.isfinite(y)
    keep = (~mask) & finite_signal & finite_time
    fill = mask & finite_time
    if int(np.count_nonzero(keep)) < 2 or not np.any(fill):
        return corrected

    keep_idx = np.flatnonzero(keep)
    order = np.argsort(t[keep_idx], kind="mergesort")
    keep_idx = keep_idx[order]
    anchor_t = t[keep_idx]
    anchor_y = y[keep_idx]

    unique_t, unique_start = np.unique(anchor_t, return_index=True)
    if unique_t.size < 2:
        return corrected
    anchor_y = anchor_y[unique_start]

    corrected[fill] = np.interp(t[fill], unique_t, anchor_y)
    return corrected


def apply_artifact_interpolation(
    time_values: np.ndarray,
    signal: np.ndarray,
    params: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    artifact_mask = detect_artifact_mask(signal, params=params)
    corrected = interpolate_artifacts(time_values, signal, artifact_mask)
    return corrected, artifact_mask[: corrected.size]


def interpolate_downward_artifacts(signal: np.ndarray, z_threshold: float = 4.0, pad: int = 2) -> np.ndarray:
    x = np.asarray(signal, dtype=float).copy()
    if x.size < 5:
        return x

    series = pd.Series(x)
    med = series.rolling(window=25, center=True, min_periods=1).median().to_numpy()
    resid = x - med
    mad = pd.Series(np.abs(resid)).rolling(window=51, center=True, min_periods=1).median().to_numpy()
    sigma = 1.4826 * np.maximum(mad, 1e-9)

    mask = resid < (-z_threshold * sigma)

    if pad > 0 and np.any(mask):
        expanded = mask.copy()
        indices = np.where(mask)[0]
        for idx in indices:
            start = max(0, idx - pad)
            end = min(len(mask), idx + pad + 1)
            expanded[start:end] = True
        mask = expanded

    if np.any(mask):
        valid = ~mask
        if valid.sum() >= 2:
            x[mask] = np.interp(np.where(mask)[0], np.where(valid)[0], x[valid])

    return x
