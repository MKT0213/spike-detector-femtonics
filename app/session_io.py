from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import AppSession, ArtifactParams, BurstRecord, DetectionParams, SpikeRecord, TraceCuration


def _spike_from_dict(data: Dict[str, object]) -> SpikeRecord:
    return SpikeRecord(
        trace_name=str(data.get("trace_name", "")),
        candidate_index=int(data.get("candidate_index", 0)),
        spike_index=int(data.get("spike_index", 0)),
        spike_time=float(data.get("spike_time", 0.0)),
        spike_amplitude_raw_or_corrected=float(data.get("spike_amplitude_raw_or_corrected", 0.0)),
        baseline_at_spike=float(data.get("baseline_at_spike", 0.0)),
        amplitude_above_baseline=float(data.get("amplitude_above_baseline", 0.0)),
        prominence=float(data.get("prominence", 0.0)),
        width=float(data.get("width", 0.0)),
        local_noise_estimate=float(data.get("local_noise_estimate", 0.0)),
        snr=float(data.get("snr", 0.0)),
        detection_threshold_used=float(data.get("detection_threshold_used", 0.0)),
        fwhm_ms=float(data.get("fwhm_ms", 0.0)),
        delta_f_over_f0=float(data.get("delta_f_over_f0", float("nan"))),
        rise_time_ms=float(data.get("rise_time_ms", float("nan"))),
        return_to_baseline_time_ms=float(data.get("return_to_baseline_time_ms", float("nan"))),
        post_spike_level=float(data.get("post_spike_level", float("nan"))),
        status=str(data.get("status", "pending")),
        source=str(data.get("source", "auto")),
        spike_type=str(data.get("spike_type", "single")),
        notes=str(data.get("notes", "")),
    )


def _burst_from_dict(data: Dict[str, object]) -> BurstRecord:
    return BurstRecord(
        trace_name=str(data.get("trace_name", "")),
        burst_index=int(data.get("burst_index", 0)),
        start_index=int(data.get("start_index", 0)),
        end_index=int(data.get("end_index", 0)),
        start_time=float(data.get("start_time", 0.0)),
        end_time=float(data.get("end_time", 0.0)),
        duration_ms=float(data.get("duration_ms", 0.0)),
        peak_index=int(data.get("peak_index", 0)),
        peak_time=float(data.get("peak_time", 0.0)),
        peak_amplitude=float(data.get("peak_amplitude", 0.0)),
        baseline_at_peak=float(data.get("baseline_at_peak", 0.0)),
        amplitude_above_baseline=float(data.get("amplitude_above_baseline", 0.0)),
        mean_amplitude=float(data.get("mean_amplitude", 0.0)),
        spike_count=int(data.get("spike_count", 0)),
        mean_firing_rate_hz=float(data.get("mean_firing_rate_hz", 0.0)),
        mean_snr=float(data.get("mean_snr", 0.0)),
        mean_prominence=float(data.get("mean_prominence", 0.0)),
        local_noise_estimate=float(data.get("local_noise_estimate", 0.0)),
        status=str(data.get("status", "pending")),
        source=str(data.get("source", "auto")),
        notes=str(data.get("notes", "")),
    )


def _normalize_spike_payload(data: object) -> List[Dict[str, object]]:
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def save_session(
    path: str,
    excel_path: str,
    detection_params: DetectionParams,
    artifact_params: ArtifactParams,
    traces: Dict[str, TraceCuration],
) -> None:
    session = AppSession(excel_path=excel_path, detection_params=detection_params, artifact_params=artifact_params)

    for trace_name, trace_state in traces.items():
        session.trace_states[trace_name] = {
            "spikes": [s.to_dict() for s in trace_state.spikes],
            "bursts": [b.to_dict() for b in trace_state.bursts],
            "deleted_bursts": [b.to_dict() for b in trace_state.deleted_bursts],
        }

    payload = {
        "excel_path": session.excel_path,
        "detection_params": session.detection_params.to_dict(),
        "artifact_params": session.artifact_params.to_dict(),
        "trace_states": session.trace_states,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_session(path: str) -> AppSession:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    app_session = AppSession(
        excel_path=str(payload.get("excel_path", "")),
        detection_params=DetectionParams.from_dict(payload.get("detection_params", {})),
        artifact_params=ArtifactParams.from_dict(payload.get("artifact_params", {})),
    )

    raw_states = payload.get("trace_states", {})
    if isinstance(raw_states, dict):
        for trace_name, state in raw_states.items():
            spikes_data: List[Dict[str, object]] = []
            if isinstance(state, dict):
                spikes_data = _normalize_spike_payload(state.get("spikes", []))
            app_session.trace_states[trace_name] = {
                "spikes": spikes_data,
            }

    return app_session


def apply_session_to_traces(app_session: AppSession, traces: Dict[str, TraceCuration]) -> None:
    for trace_name, trace_state in traces.items():
        state = app_session.trace_states.get(trace_name)
        if not state:
            continue
        spikes_data = _normalize_spike_payload(state.get("spikes", []))
        trace_state.spikes = [_spike_from_dict(item) for item in spikes_data]
        trace_state.deleted_spikes = []
        trace_state.sort_spikes()
