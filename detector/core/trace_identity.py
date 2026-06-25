from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


TRACE_NAME_SEPARATOR = " | "
DEFAULT_RECORDING_ID = "Ungrouped"


def split_trace_identity(trace_name: object) -> Tuple[str, str]:
    """Return (recording_id, roi_id) from the analyser trace name."""
    text = str(trace_name)
    if TRACE_NAME_SEPARATOR in text:
        recording, roi = text.split(TRACE_NAME_SEPARATOR, 1)
        recording = recording.strip()
        roi = roi.strip()
        return recording or DEFAULT_RECORDING_ID, roi or text
    return DEFAULT_RECORDING_ID, text


def trace_identity_columns(trace_name: object) -> Dict[str, str]:
    recording_id, roi_id = split_trace_identity(trace_name)
    return {"recording_id": recording_id, "roi_id": roi_id}


def recording_groups(trace_names: Iterable[object]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for trace_name in trace_names:
        name = str(trace_name)
        recording_id, _roi_id = split_trace_identity(name)
        groups.setdefault(recording_id, []).append(name)
    return groups
