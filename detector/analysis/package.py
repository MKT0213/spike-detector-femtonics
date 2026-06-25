"""Analysis package save/load support for the review application and exporter."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from detector.core.models import ArtifactParams, DetectionParams, SpikeRecord, TraceCuration
from detector.core.render_normalization import (
    NORMALIZATION_PERCENT_DFF,
    build_render_normalization_payload,
    render_normalization_rows,
)
from detector.core.session_io import _spike_from_dict
from detector.core.trace_identity import recording_groups, split_trace_identity, trace_identity_columns


ANALYSIS_MANIFEST_NAME = "analysis_manifest.json"
ANALYSIS_PACKAGE_VERSION = 1


@dataclass
class AnalysisPackage:
    root: Path
    manifest: Dict[str, Any]
    traces: Dict[str, TraceCuration]
    trace_names: List[str]
    detection_params: DetectionParams
    artifact_params: ArtifactParams
    source_excel: str = ""


def safe_trace_id(trace_name: str, used: Optional[set[str]] = None) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(trace_name)).strip("._")
    if not value:
        value = "trace"
    if used is None:
        return value
    base = value
    idx = 2
    while value in used:
        value = f"{base}_{idx}"
        idx += 1
    used.add(value)
    return value


def default_analysis_package_dir(result_root: Path, source_excel: str = "") -> Path:
    stem = Path(source_excel).stem if source_excel else "analysis"
    safe_stem = safe_trace_id(stem)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(result_root) / f"analysis_{safe_stem}_{timestamp}"


def save_analysis_package(
    out_dir: str | Path,
    *,
    source_excel: str,
    detection_params: DetectionParams,
    artifact_params: ArtifactParams,
    trace_names: Iterable[str],
    traces: Dict[str, TraceCuration],
    pipeline_config: Dict[str, object] | None = None,
    render_normalization_mode: str = NORMALIZATION_PERCENT_DFF,
) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    tables_dir = root / "tables"
    reload_dir = root / "reload"
    traces_dir = root / "traces"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reload_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    active_names = [str(name) for name in trace_names if str(name) in traces]
    _write_tables_and_reload(tables_dir, reload_dir, active_names, traces)
    normalization_mode = str(render_normalization_mode)
    normalization_rows = render_normalization_rows(active_names, traces, normalization_mode)
    pd.DataFrame(normalization_rows).to_csv(tables_dir / "recording_roi_render_metadata.csv", index=False)

    used_ids: set[str] = set()
    trace_rows: List[Dict[str, object]] = []
    for trace_name in active_names:
        state = traces[trace_name]
        trace_id = safe_trace_id(trace_name, used_ids)
        trace_dir = traces_dir / trace_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        _write_trace_payload(trace_dir, state)
        trace_rows.append(
            {
                "trace_id": trace_id,
                "trace_name": trace_name,
                **trace_identity_columns(trace_name),
                "sample_count": int(np.asarray(state.time).size),
                "sampling_rate_hz": state.sampling_rate_hz,
                "analysis_status": state.analysis_status,
            }
        )

    pd.DataFrame(trace_rows).to_csv(traces_dir / "trace_index.csv", index=False)
    grouped_recordings = recording_groups(active_names)

    manifest = {
        "analysis_package_version": ANALYSIS_PACKAGE_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_excel": str(source_excel or ""),
        "trace_count": len(active_names),
        "traces": active_names,
        "recordings": [
            {
                "recording_id": recording_id,
                "roi_count": len(names),
                "roi_ids": [split_trace_identity(name)[1] for name in names],
                "traces": [str(name) for name in names],
            }
            for recording_id, names in grouped_recordings.items()
        ],
        "detection_params": detection_params.to_dict(),
        "artifact_params": artifact_params.to_dict(),
        "pipeline_config": pipeline_config or {},
        "display_normalization_mode": normalization_mode,
        "render_normalization": build_render_normalization_payload(
            active_names,
            traces,
            normalization_mode,
        ),
        "folders": {
            "tables": "tables",
            "reload": "reload",
            "traces": "traces",
        },
    }
    (root / ANALYSIS_MANIFEST_NAME).write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return root


def load_analysis_package(folder: str | Path) -> AnalysisPackage:
    root = Path(folder)
    manifest_path = root / ANALYSIS_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"{ANALYSIS_MANIFEST_NAME} not found in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = int(manifest.get("analysis_package_version", 0))
    if version != ANALYSIS_PACKAGE_VERSION:
        raise ValueError(f"Unsupported analysis package version: {version}")

    detection_params = DetectionParams.from_dict(manifest.get("detection_params", {}))
    artifact_params = ArtifactParams.from_dict(manifest.get("artifact_params", {}))

    trace_index_path = root / "traces" / "trace_index.csv"
    if not trace_index_path.exists():
        raise FileNotFoundError(f"trace_index.csv not found in {root / 'traces'}")
    trace_index = pd.read_csv(trace_index_path)

    traces: Dict[str, TraceCuration] = {}
    trace_names: List[str] = []
    for _, row in trace_index.iterrows():
        trace_id = str(row.get("trace_id", ""))
        trace_name = str(row.get("trace_name", trace_id))
        if not trace_id:
            continue
        state = _read_trace_payload(root / "traces" / trace_id, trace_name)
        traces[trace_name] = state
        trace_names.append(trace_name)

    return AnalysisPackage(
        root=root,
        manifest=manifest,
        traces=traces,
        trace_names=trace_names,
        detection_params=detection_params,
        artifact_params=artifact_params,
        source_excel=str(manifest.get("source_excel", "")),
    )


def _write_tables_and_reload(
    tables_dir: Path,
    reload_dir: Path,
    trace_names: List[str],
    traces: Dict[str, TraceCuration],
) -> None:
    from detector.core import gui_main

    export_tables = gui_main._build_export_dataframes(trace_names, traces)
    cfg = gui_main.TRACE_SUMMARY_CONFIG
    table_paths = {
        "trace_summary": tables_dir / str(cfg["csv_name"]),
        "trace_qc": tables_dir / str(cfg["qc_csv_name"]),
        "plateau_events": tables_dir / str(cfg["events_csv_name"]),
        "spike_events": tables_dir / str(cfg["spike_events_csv_name"]),
        "spike_summary": tables_dir / str(cfg["spike_summary_csv_name"]),
    }
    for key, path in table_paths.items():
        export_tables[key].to_csv(path, index=False)
    export_tables["corrected_traces"].to_csv(reload_dir / "corrected_traces.csv", index=False)
    export_tables["spike_events"].to_csv(reload_dir / str(cfg["spike_events_csv_name"]), index=False)


def _write_trace_payload(trace_dir: Path, state: TraceCuration) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for name in [
        "time",
        "raw",
        "corrected_source",
        "corrected",
        "baseline",
        "correction_baseline",
        "smoothed",
        "highpass_trace",
        "lowpass_trace",
        "hybrid_denoised",
        "artifact_mask",
        "plateau_mask",
        "long_plateau_mask",
        "short_plateau_mask",
        "short_plateau_events",
        "burst_plateau_mask",
        "burst_plateau_events",
        "debug_threshold",
    ]:
        value = getattr(state, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    np.savez_compressed(trace_dir / "arrays.npz", **arrays)

    metadata = {
        "trace_name": state.trace_name,
        "sampling_rate_hz": state.sampling_rate_hz,
        "analysis_status": state.analysis_status,
        "analysis_error": state.analysis_error,
        "analysis_generation": state.analysis_generation,
        "candidate_windows": state.candidate_windows,
        "hybrid_denoised_cache_key": state.hybrid_denoised_cache_key,
        "gonzalez_cwt_debug_cache_key": state.gonzalez_cwt_debug_cache_key,
    }
    (trace_dir / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")

    _spikes_to_dataframe(state.spikes).to_csv(trace_dir / "spikes.csv", index=False)
    _spikes_to_dataframe(state.deleted_spikes).to_csv(trace_dir / "deleted_spikes.csv", index=False)
    _write_plateau_rows(trace_dir / "plateaus.csv", state)
    if state.short_plateau_events is not None:
        pd.DataFrame(np.asarray(state.short_plateau_events)).to_csv(
            trace_dir / "short_plateau_events.csv",
            index=False,
        )
    if state.burst_plateau_events is not None:
        pd.DataFrame(np.asarray(state.burst_plateau_events)).to_csv(
            trace_dir / "burst_plateau_events.csv",
            index=False,
        )
    _write_plateau_signal_classification(trace_dir, state.plateau_signal_classification)
    _write_debug_payload(trace_dir, "long_plateau_debug", state.long_plateau_debug)
    _write_debug_payload(trace_dir, "burst_plateau_debug", state.burst_plateau_debug)
    _write_debug_payload(trace_dir, "gonzalez_cwt_debug", state.gonzalez_cwt_debug)


def _read_trace_payload(trace_dir: Path, trace_name: str) -> TraceCuration:
    arrays_path = trace_dir / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"arrays.npz not found in {trace_dir}")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}

    metadata_path = trace_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    time = np.asarray(arrays.get("time", np.array([], dtype=float)), dtype=float)
    corrected = np.asarray(arrays.get("corrected", np.array([], dtype=float)), dtype=float)
    raw = np.asarray(arrays.get("raw", corrected.copy()), dtype=float)
    corrected_source = np.asarray(arrays.get("corrected_source", corrected.copy()), dtype=float)

    state = TraceCuration(
        trace_name=str(metadata.get("trace_name", trace_name)),
        raw=raw,
        corrected_source=corrected_source,
        corrected=corrected,
        time=time,
        sampling_rate_hz=_optional_float(metadata.get("sampling_rate_hz")),
        original_time=time.copy(),
        original_raw=raw.copy(),
        original_corrected_source=corrected_source.copy(),
        original_sampling_rate_hz=_optional_float(metadata.get("sampling_rate_hz")),
        baseline=_optional_array(arrays, "baseline", float),
        correction_baseline=_optional_array(arrays, "correction_baseline", float),
        smoothed=_optional_array(arrays, "smoothed", float),
        highpass_trace=_optional_array(arrays, "highpass_trace", float),
        lowpass_trace=_optional_array(arrays, "lowpass_trace", float),
        hybrid_denoised=_optional_array(arrays, "hybrid_denoised", float),
        artifact_mask=_optional_array(arrays, "artifact_mask", bool),
        plateau_mask=_optional_array(arrays, "plateau_mask", bool),
        long_plateau_mask=_optional_array(arrays, "long_plateau_mask", bool),
        long_plateau_debug=_read_debug_payload(trace_dir, "long_plateau_debug"),
        short_plateau_mask=_optional_array(arrays, "short_plateau_mask", bool),
        short_plateau_events=_optional_array(arrays, "short_plateau_events", float),
        burst_plateau_mask=_optional_array(arrays, "burst_plateau_mask", bool),
        burst_plateau_events=_optional_array(arrays, "burst_plateau_events", float),
        burst_plateau_debug=_read_debug_payload(trace_dir, "burst_plateau_debug"),
        plateau_signal_classification=_read_plateau_signal_classification(trace_dir),
        debug_threshold=_optional_array(arrays, "debug_threshold", float),
        candidate_windows=[tuple(item) for item in metadata.get("candidate_windows", [])],
        spikes=_read_spikes(trace_dir / "spikes.csv"),
        deleted_spikes=_read_spikes(trace_dir / "deleted_spikes.csv"),
        hybrid_denoised_cache_key=_optional_tuple(metadata.get("hybrid_denoised_cache_key")),
        gonzalez_cwt_debug_cache_key=_optional_tuple(metadata.get("gonzalez_cwt_debug_cache_key")),
        gonzalez_cwt_debug=_read_debug_payload(trace_dir, "gonzalez_cwt_debug"),
        analysis_status=str(metadata.get("analysis_status", "done")),
        analysis_error=str(metadata.get("analysis_error", "")),
        analysis_generation=int(metadata.get("analysis_generation", 0)),
    )
    state.sort_spikes()
    return state


def _optional_array(arrays: Dict[str, np.ndarray], key: str, dtype: object) -> Optional[np.ndarray]:
    if key not in arrays:
        return None
    return np.asarray(arrays[key], dtype=dtype)


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None


def _optional_tuple(value: object) -> Optional[tuple[object, ...]]:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _spikes_to_dataframe(spikes: List[SpikeRecord]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spike in spikes:
        row = spike.to_dict()
        row["status"] = spike.status
        row["spike_type"] = spike.spike_type
        rows.append(row)
    return pd.DataFrame(rows)


def _read_spikes(path: Path) -> List[SpikeRecord]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    return [_spike_from_dict(row.dropna().to_dict()) for _, row in df.iterrows()]


def _write_plateau_rows(path: Path, state: TraceCuration) -> None:
    from detector.core import gui_main

    if state.plateau_mask is None:
        pd.DataFrame().to_csv(path, index=False)
        return
    rows = gui_main._plateau_rows(
        trace_name=state.trace_name,
        time=np.asarray(state.time, dtype=float),
        plateau_mask=np.asarray(state.plateau_mask, dtype=bool),
        long_plateau_mask=state.long_plateau_mask,
        short_plateau_mask=state.short_plateau_mask,
        burst_plateau_mask=state.burst_plateau_mask,
        burst_plateau_events=state.burst_plateau_events,
        plateau_signal_classification=state.plateau_signal_classification,
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_plateau_signal_classification(
    trace_dir: Path,
    rows: Optional[List[Dict[str, Any]]],
) -> None:
    clean_rows = [dict(row) for row in rows] if isinstance(rows, list) else []
    (trace_dir / "plateau_signal_classification.json").write_text(
        json.dumps(_json_safe(clean_rows), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(clean_rows).to_csv(trace_dir / "plateau_signal_classification.csv", index=False)


def _read_plateau_signal_classification(trace_dir: Path) -> Optional[List[Dict[str, Any]]]:
    json_path = trace_dir / "plateau_signal_classification.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
        return []
    csv_path = trace_dir / "plateau_signal_classification.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return []
    return df.to_dict("records")


def _write_debug_payload(trace_dir: Path, name: str, payload: Optional[Dict[str, Any]]) -> None:
    if not isinstance(payload, dict):
        (trace_dir / f"{name}_metadata.json").write_text("{}", encoding="utf-8")
        return
    arrays_dir = trace_dir / f"{name}_arrays"
    arrays_dir.mkdir(exist_ok=True)
    array_counter = {"value": 0}
    safe_payload = _debug_json_safe(payload, arrays_dir, array_counter)
    (trace_dir / f"{name}_metadata.json").write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")


def _read_debug_payload(trace_dir: Path, name: str) -> Optional[Dict[str, Any]]:
    path = trace_dir / f"{name}_metadata.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        return None
    return _debug_json_restore(payload, trace_dir)


def _debug_json_safe(value: Any, arrays_dir: Path, counter: Dict[str, int]) -> Any:
    if isinstance(value, np.ndarray):
        counter["value"] += 1
        filename = f"array_{counter['value']:04d}.npy"
        np.save(arrays_dir / filename, value, allow_pickle=False)
        return {
            "__ndarray_file__": str(arrays_dir.name + "/" + filename),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, dict):
        return {str(k): _debug_json_safe(v, arrays_dir, counter) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_json_safe(v, arrays_dir, counter) for v in value]
    return _json_safe(value)


def _debug_json_restore(value: Any, trace_dir: Path) -> Any:
    if isinstance(value, dict) and "__ndarray_file__" in value:
        return np.load(trace_dir / str(value["__ndarray_file__"]), allow_pickle=False)
    if isinstance(value, dict):
        return {k: _debug_json_restore(v, trace_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_debug_json_restore(v, trace_dir) for v in value]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
