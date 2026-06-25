"""Display/export normalization helpers for trace rendering.

The functions in this module return derived display arrays only. They never
write normalized values back into ``TraceCuration``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .models import TraceCuration
from .trace_identity import split_trace_identity


NORMALIZATION_NATIVE = "native"
NORMALIZATION_PERCENT_DFF = "percent_dff"
PERCENT_DFF_UNIT = "% ΔF/F0"
PERCENT_DFF_SCALE_LABEL = "% ΔF/F"


def normalization_unit_label(mode: str) -> str:
    return PERCENT_DFF_UNIT if str(mode) == NORMALIZATION_PERCENT_DFF else "native"


def finite_median(values: object) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size <= 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return float("nan")
    return float(np.median(finite))


def f0_values_for_trace(state: TraceCuration, size: int | None = None) -> Tuple[np.ndarray, str, float]:
    """Return F0 values, source label, and finite median for a trace."""
    corrected = np.asarray(state.corrected, dtype=float)
    n = int(size if size is not None else corrected.size)
    n = max(0, n)

    baseline = getattr(state, "correction_baseline", None)
    if baseline is not None:
        baseline_arr = np.asarray(baseline, dtype=float)
        if baseline_arr.ndim == 1 and baseline_arr.size == corrected.size and n <= baseline_arr.size:
            values = baseline_arr[:n].astype(float, copy=True)
            finite = values[np.isfinite(values) & (np.abs(values) > 1.0e-12)]
            if finite.size > 0:
                return values, "correction_baseline", float(np.median(finite))

    for source_name in ("raw", "corrected_source"):
        median = finite_median(getattr(state, source_name, None))
        if np.isfinite(median) and abs(float(median)) > 1.0e-12:
            return np.full(n, float(median), dtype=float), source_name, float(median)

    return np.ones(n, dtype=float), "unit_fallback", 1.0


def normalize_trace_values(values: object, state: TraceCuration, mode: str) -> np.ndarray:
    """Normalize a display array using the trace F0 definition."""
    arr = np.asarray(values, dtype=float).copy()
    if str(mode) != NORMALIZATION_PERCENT_DFF:
        return arr
    f0, _source, _median = f0_values_for_trace(state, arr.size)
    return normalize_values_with_f0(arr, f0)


def normalize_values_with_f0(values: object, f0_values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    f0 = np.asarray(f0_values, dtype=float)
    if f0.ndim == 0:
        f0 = np.full(arr.size, float(f0), dtype=float)
    if f0.size != arr.size:
        out = np.ones(arr.size, dtype=float)
        n = min(out.size, f0.size)
        if n > 0:
            out[:n] = f0[:n]
        f0 = out
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = 100.0 * arr / f0
    normalized[~np.isfinite(normalized)] = np.nan
    return normalized


def trace_normalization_row(trace_name: str, state: TraceCuration, mode: str = NORMALIZATION_PERCENT_DFF) -> Dict[str, object]:
    recording_id, roi_id = split_trace_identity(trace_name)
    _f0, f0_source, f0_median = f0_values_for_trace(state)
    return {
        "trace_name": str(trace_name),
        "recording_id": recording_id,
        "roi_id": roi_id,
        "mode": str(mode),
        "f0_source": f0_source,
        "f0_median": f0_median,
        "unit": normalization_unit_label(mode),
    }


def render_normalization_rows(
    trace_names: Iterable[str],
    traces: Dict[str, TraceCuration],
    mode: str = NORMALIZATION_PERCENT_DFF,
) -> List[Dict[str, object]]:
    return [
        trace_normalization_row(str(name), traces[str(name)], mode)
        for name in trace_names
        if str(name) in traces
    ]


def build_render_normalization_payload(
    trace_names: Iterable[str],
    traces: Dict[str, TraceCuration],
    mode: str = NORMALIZATION_PERCENT_DFF,
) -> Dict[str, object]:
    rows = render_normalization_rows(trace_names, traces, mode)
    return {
        "version": 1,
        "mode": str(mode),
        "unit": normalization_unit_label(mode),
        "traces": rows,
    }
