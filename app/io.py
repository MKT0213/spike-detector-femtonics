from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TIME_CANDIDATES = [
    "time",
    "time_ms",
    "time (ms)",
    "time(ms)",
    "timestamp",
    "t",
    "t_ms",
    "t (ms)",
    "t(ms)",
    "ms",
]


def _pick_time_column(columns: List[object]) -> object:
    lowered = {str(col).lower().strip(): col for col in columns}
    for candidate in TIME_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for key, col in lowered.items():
        if "time" in key or key.startswith("t(") or key.startswith("t_"):
            return col
    return columns[0]


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _time_column_unit_scale_to_seconds(column: object) -> float | None:
    name = str(column).lower().replace(" ", "")
    if "(ms)" in name or name.endswith("_ms") or name == "ms" or "millisecond" in name:
        return 0.001
    if "(s)" in name or name.endswith("_s") or name in {"s", "sec", "secs", "second", "seconds"}:
        return 1.0
    return None


def estimate_sampling_rate_hz(time: np.ndarray, time_column: object | None = None) -> float:
    values = np.asarray(time, dtype=float)
    if values.size < 2:
        return 1.0
    dt = np.diff(values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    median_dt = float(np.median(dt))
    unit_scale = _time_column_unit_scale_to_seconds(time_column) if time_column is not None else None
    if unit_scale is not None:
        return float(1.0 / (median_dt * unit_scale))
    direct_hz = 1.0 / median_dt
    return float(1000.0 / median_dt) if direct_hz < 5.0 and median_dt >= 0.001 else float(direct_hz)


def _load_single_sheet(
    xl: pd.ExcelFile,
    sheet: str,
) -> Tuple[np.ndarray, object, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = xl.parse(sheet)
    if df.empty:
        raise ValueError(f"Sheet '{sheet}' is empty.")

    time_col = _pick_time_column(df.columns.tolist())
    time = _to_numeric_array(df[time_col])
    valid_time = np.isfinite(time)
    if int(np.count_nonzero(valid_time)) < 2:
        raise ValueError(f"Sheet '{sheet}' does not contain a numeric time column.")
    positive_steps = np.diff(time[valid_time])
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    if positive_steps.size == 0:
        raise ValueError(f"Sheet '{sheet}' does not contain an increasing numeric time column.")
    if not np.all(valid_time):
        df = df.loc[valid_time].reset_index(drop=True)
        time = time[valid_time]

    work = df.drop(columns=[time_col]).copy()
    if work.empty:
        raise ValueError(f"No trace columns found in sheet '{sheet}' after removing time column.")

    for column in work.columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    raw_map: Dict[str, np.ndarray] = {}
    corrected_map: Dict[str, np.ndarray] = {}
    corrected_suffixes = ("_corrected", " corrected", "-corrected")

    for col in list(work.columns):
        col_clean = str(col).strip()
        lower = col_clean.lower()

        matched = None
        for suffix in corrected_suffixes:
            if lower.endswith(suffix):
                matched = suffix
                break

        values = _to_numeric_array(work[col])
        if matched:
            base = col_clean[: -len(matched)].strip()
            corrected_map[base or col_clean] = values
        else:
            raw_map[col_clean] = values

    if not raw_map:
        raw_map = {name: values.copy() for name, values in corrected_map.items()}

    for name in list(raw_map.keys()):
        if name not in corrected_map:
            corrected_map[name] = raw_map[name].copy()

    common = sorted(set(raw_map).intersection(corrected_map))
    if not common:
        raise ValueError(f"Could not align raw and corrected trace columns in sheet '{sheet}'.")

    n = min(
        len(time),
        *(len(v) for v in raw_map.values()),
        *(len(v) for v in corrected_map.values()),
    )
    time = time[:n]

    return (
        time,
        time_col,
        {k: v[:n] for k, v in raw_map.items() if k in common},
        {k: v[:n] for k, v in corrected_map.items() if k in common},
    )


def load_excel_traces(
    excel_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    with pd.ExcelFile(path) as xl:
        sheets = xl.sheet_names
        if not sheets:
            raise ValueError("No sheets found in Excel file.")

        all_time: Dict[str, np.ndarray] = {}
        all_raw: Dict[str, np.ndarray] = {}
        all_corrected: Dict[str, np.ndarray] = {}
        all_sampling_rate_hz: Dict[str, float] = {}

        for sheet in sheets:
            try:
                time_sheet, time_col, raw_sheet, corrected_sheet = _load_single_sheet(xl, sheet)
            except ValueError:
                continue

            for name in raw_sheet:
                trace_name = f"{sheet} | {name}"
                all_time[trace_name] = time_sheet
                all_raw[trace_name] = raw_sheet[name]
                all_corrected[trace_name] = corrected_sheet[name]
                all_sampling_rate_hz[trace_name] = estimate_sampling_rate_hz(time_sheet, time_col)

    if not all_raw:
        raise ValueError("No valid trace data found in any sheet.")

    # Truncate each trace to its own shortest aligned length. Do not pad or fill:
    # raw import must remain a faithful copy of the Excel samples and time axis.
    for name in list(all_raw.keys()):
        n = min(len(all_time[name]), len(all_raw[name]), len(all_corrected[name]))
        all_raw[name] = all_raw[name][:n]
        all_corrected[name] = all_corrected[name][:n]
        all_time[name] = all_time[name][:n]

    return all_time, all_raw, all_corrected, all_sampling_rate_hz
