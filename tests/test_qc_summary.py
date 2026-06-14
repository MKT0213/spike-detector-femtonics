from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gui_main import (  # noqa: E402
    _add_event_qc_flags,
    _add_trace_qc_flags,
    _build_all_trace_summary_row,
    _build_export_dataframes,
    _build_plateau_event_rows,
    _build_spike_event_rows,
    _build_trace_spike_summary_rows,
    _build_trace_summary_rows,
    _find_export_file,
    _save_spike_statistics_summary,
    _spike_record_from_event_stats_row,
)
from app.models import SpikeRecord, TraceCuration  # noqa: E402


def _state(name: str, mask: np.ndarray, values: np.ndarray | None = None) -> TraceCuration:
    time = np.arange(mask.size, dtype=float)
    corrected = np.zeros(mask.size, dtype=float) if values is None else np.asarray(values, dtype=float)
    baseline = np.zeros(mask.size, dtype=float)
    return TraceCuration(
        trace_name=name,
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=time,
        sampling_rate_hz=1000.0,
        baseline=baseline,
        plateau_mask=np.asarray(mask, dtype=bool),
        long_plateau_mask=np.asarray(mask, dtype=bool),
    )


def _spike(index: int, time: float, amplitude: float, status: str = "pending") -> SpikeRecord:
    return SpikeRecord(
        trace_name="trace",
        candidate_index=index,
        spike_index=index,
        spike_time=time,
        spike_amplitude_raw_or_corrected=100.0 + amplitude,
        baseline_at_spike=100.0,
        amplitude_above_baseline=amplitude,
        prominence=amplitude,
        width=1.0,
        local_noise_estimate=2.0,
        snr=amplitude / 2.0,
        detection_threshold_used=0.0,
        fwhm_ms=0.0,
        status=status,  # type: ignore[arg-type]
        source="auto",
    )


def test_spike_event_stats_include_all_active_spikes_and_exclude_deleted() -> None:
    corrected = np.full(50, 100.0, dtype=float)
    baseline = np.full(50, 100.0, dtype=float)
    state = TraceCuration(
        trace_name="trace",
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=np.arange(50, dtype=float),
        sampling_rate_hz=1000.0,
        baseline=baseline,
        spikes=[
            _spike(10, 10.0, 5.0, "pending"),
            _spike(20, 20.0, 5.0, "accepted"),
            _spike(30, 30.0, 5.0, "rejected"),
            _spike(40, 40.0, 5.0, "manual"),
        ],
        deleted_spikes=[_spike(45, 45.0, 5.0, "deleted")],
    )

    rows = _build_spike_event_rows(["trace"], {"trace": state})

    assert len(rows) == 4
    assert {row["peak_index"] for row in rows} == {10, 20, 30, 40}
    assert "status" not in rows[0]


def test_spike_waveform_metrics_dff_isi_and_rate() -> None:
    amplitude = np.zeros(40, dtype=float)
    amplitude[5:16] = [0, 2, 4, 6, 8, 10, 8, 5, 1, 0, 0]
    amplitude[25:36] = [0, 2, 4, 6, 8, 10, 8, 5, 1, 0, 0]
    baseline = np.full(40, 100.0, dtype=float)
    corrected = baseline + amplitude
    state = TraceCuration(
        trace_name="trace",
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=np.arange(40, dtype=float),
        sampling_rate_hz=1000.0,
        baseline=baseline,
        spikes=[_spike(10, 10.0, 10.0), _spike(30, 30.0, 10.0)],
    )

    rows = _build_spike_event_rows(["trace"], {"trace": state})
    first = rows[0]
    second = rows[1]

    assert first["amplitude"] == 10.0
    assert first["delta_f_over_f0"] == 0.1
    assert first["snr"] == 5.0
    assert first["rise_time_ms"] == 4.0
    assert first["fwhm_ms"] == 4.5
    assert first["return_to_baseline_time_ms"] == 3.0
    assert first["post_spike_level"] == 0.0
    assert math.isnan(float(first["isi_ms"]))
    assert math.isnan(float(first["instantaneous_firing_rate_hz"]))
    assert second["isi_ms"] == 20.0
    assert second["instantaneous_firing_rate_hz"] == 50.0


def test_spike_isi_and_rate_use_sampling_rate_for_seconds_axis() -> None:
    amplitude = np.zeros(40, dtype=float)
    amplitude[10] = 10.0
    amplitude[30] = 10.0
    baseline = np.full(40, 100.0, dtype=float)
    corrected = baseline + amplitude
    state = TraceCuration(
        trace_name="trace",
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=np.arange(40, dtype=float) * 0.001,
        sampling_rate_hz=1000.0,
        baseline=baseline,
        spikes=[_spike(10, 0.010, 10.0), _spike(30, 0.030, 10.0)],
    )

    rows = _build_spike_event_rows(["trace"], {"trace": state})

    assert rows[1]["isi_ms"] == 20.0
    assert rows[1]["instantaneous_firing_rate_hz"] == 50.0


def test_spike_waveform_metrics_handle_spike_on_plateau() -> None:
    amplitude = np.zeros(40, dtype=float)
    amplitude[5:16] = 8.0
    amplitude[10] = 10.0
    baseline = np.full(40, 100.0, dtype=float)
    corrected = baseline + amplitude
    spike = _spike(10, 10.0, 10.0)
    spike.fwhm_ms = 1.25
    state = TraceCuration(
        trace_name="trace",
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=np.arange(40, dtype=float),
        sampling_rate_hz=1000.0,
        baseline=baseline,
        spikes=[spike],
    )

    row = _build_spike_event_rows(["trace"], {"trace": state})[0]

    assert row["fwhm_ms"] == 1.25
    assert math.isfinite(float(row["rise_time_ms"]))
    assert math.isfinite(float(row["return_to_baseline_time_ms"]))
    assert row["post_spike_level"] == 8.0


def test_reloaded_spike_event_metrics_are_preserved_on_reexport() -> None:
    row = pd.Series(
        {
            "trace_id": "trace",
            "peak_index": 10,
            "peak_time": 10.0,
            "amplitude": 12.0,
            "delta_f_over_f0": 0.24,
            "snr": 6.0,
            "rise_time_ms": 0.8,
            "fwhm_ms": 1.4,
            "return_to_baseline_time_ms": 2.2,
            "post_spike_level": 0.3,
            "source": "auto",
        }
    )
    spike = _spike_record_from_event_stats_row(row, "trace")
    corrected = np.zeros(30, dtype=float)
    state = TraceCuration(
        trace_name="trace",
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=np.arange(30, dtype=float),
        sampling_rate_hz=1000.0,
        baseline=None,
        spikes=[spike],
    )

    exported = _build_spike_event_rows(["trace"], {"trace": state})[0]

    assert exported["delta_f_over_f0"] == 0.24
    assert exported["rise_time_ms"] == 0.8
    assert exported["fwhm_ms"] == 1.4
    assert exported["return_to_baseline_time_ms"] == 2.2
    assert exported["post_spike_level"] == 0.3


def test_trace_spike_summary_handles_zero_and_multiple_spikes() -> None:
    events = [
        {"trace_id": "trace", "amplitude": 4.0, "delta_f_over_f0": 0.04, "snr": 2.0, "rise_time_ms": 1.0, "fwhm_ms": 2.0, "return_to_baseline_time_ms": 3.0, "post_spike_level": 0.5, "isi_ms": float("nan"), "instantaneous_firing_rate_hz": float("nan")},
        {"trace_id": "trace", "amplitude": 8.0, "delta_f_over_f0": 0.08, "snr": 4.0, "rise_time_ms": 3.0, "fwhm_ms": 4.0, "return_to_baseline_time_ms": 5.0, "post_spike_level": 1.5, "isi_ms": 20.0, "instantaneous_firing_rate_hz": 50.0},
    ]

    rows = _build_trace_spike_summary_rows(["empty", "trace"], events)
    by_trace = {row["trace_id"]: row for row in rows}

    assert by_trace["empty"]["spike_count"] == 0
    assert math.isnan(float(by_trace["empty"]["mean_amplitude"]))
    assert by_trace["trace"]["spike_count"] == 2
    assert by_trace["trace"]["mean_amplitude"] == 6.0
    assert by_trace["trace"]["median_isi_ms"] == 20.0
    assert by_trace["trace"]["mean_instantaneous_firing_rate_hz"] == 50.0


def test_plateau_sd_uses_baseline_subtracted_amplitude() -> None:
    mask = np.zeros(20, dtype=bool)
    mask[5:15] = True
    baseline = np.linspace(100.0, 200.0, 20)
    amplitude = np.zeros(20, dtype=float)
    amplitude[5:15] = 10.0
    corrected = baseline + amplitude
    state = TraceCuration(
        trace_name="flat_amp",
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=np.arange(20, dtype=float),
        sampling_rate_hz=1000.0,
        baseline=baseline,
        plateau_mask=mask,
        long_plateau_mask=mask,
    )

    event = _build_plateau_event_rows(["flat_amp"], {"flat_amp": state})[0]

    assert event["mean_amp_above_baseline"] == 10.0
    assert event["plateau_sd"] == 0.0


def test_trace_with_no_plateaus_is_kept_and_flagged() -> None:
    mask = np.zeros(100, dtype=bool)
    traces = {"empty": _state("empty", mask)}

    rows = _build_trace_summary_rows(["empty"], traces, event_rows=[])
    qc = _add_trace_qc_flags(pd.DataFrame(rows))

    assert rows[0]["plateau_count"] == 0
    assert rows[0]["total_plateau_time_ms"] == 0.0
    assert qc.loc[0, "qc_status"] == "review"
    assert "no_plateaus" in qc.loc[0, "qc_notes"]


def test_known_plateau_durations_and_intervals() -> None:
    mask = np.zeros(80, dtype=bool)
    mask[10:20] = True
    mask[35:50] = True
    values = np.zeros(80, dtype=float)
    values[10:20] = np.linspace(1.0, 10.0, 10)
    values[35:50] = np.linspace(2.0, 16.0, 15)
    traces = {"trace": _state("trace", mask, values)}

    events = _build_plateau_event_rows(["trace"], traces)
    summary = _build_trace_summary_rows(["trace"], traces, event_rows=events)[0]

    assert [event["duration_ms"] for event in events] == [10.0, 15.0]
    assert summary["plateau_count"] == 2
    assert summary["first_onset_ms"] == 10.0
    assert summary["mean_duration_ms"] == 12.5
    assert summary["median_duration_ms"] == 12.5
    assert summary["mean_inter_plateau_interval_ms"] == 25.0
    assert summary["median_inter_plateau_interval_ms"] == 25.0


def test_all_row_pools_within_trace_intervals_only() -> None:
    events = pd.DataFrame(
        [
            {"trace_id": "a", "onset_time_ms": 10.0, "duration_ms": 5.0},
            {"trace_id": "a", "onset_time_ms": 30.0, "duration_ms": 5.0},
            {"trace_id": "b", "onset_time_ms": 10.0, "duration_ms": 5.0},
            {"trace_id": "b", "onset_time_ms": 60.0, "duration_ms": 5.0},
        ]
    )

    row = _build_all_trace_summary_row(events)

    assert row["mean_inter_plateau_interval_ms"] == 35.0
    assert row["median_inter_plateau_interval_ms"] == 35.0
    assert math.isnan(float(row["first_onset_ms"]))


def test_trace_and_event_qc_flags_high_outliers() -> None:
    trace_df = pd.DataFrame(
        {
            "trace_id": [f"t{i}" for i in range(6)],
            "plateau_count": [2, 2, 3, 2, 2, 50],
            "median_plateau_sd": [1, 1, 1, 1, 1, 20],
            "median_duration_ms": [10, 11, 10, 10, 12, 200],
            "median_fwhm_ms": [3, 3, 3, 4, 3, 80],
            "plateau_time_fraction": [0.05, 0.05, 0.06, 0.05, 0.04, 0.8],
        }
    )
    qc_trace = _add_trace_qc_flags(trace_df)
    assert qc_trace.loc[5, "qc_status"] == "review"
    assert "high_plateau_count" in qc_trace.loc[5, "qc_notes"]

    events = pd.DataFrame(
        {
            "duration_ms": [10, 11, 12, 10, 11, 300],
            "fwhm_ms": [3, 3, 4, 3, 3, 50],
            "peak_amp_above_baseline": [5, 6, 5, 5, 6, 100],
            "plateau_sd": [1, 1, 1, 1, 1, 20],
        }
    )
    qc_events = _add_event_qc_flags(events)
    assert qc_events.loc[5, "duration_qc"] == "high_outlier"
    assert qc_events.loc[5, "fwhm_qc"] == "high_outlier"
    assert qc_events.loc[5, "amplitude_qc"] == "high_outlier"
    assert qc_events.loc[5, "noise_qc"] == "high_outlier"


def test_export_dataframes_include_tables_reload_data_and_stable_columns() -> None:
    mask = np.zeros(50, dtype=bool)
    mask[10:20] = True
    values = np.full(50, 100.0, dtype=float)
    values[10:20] += 5.0
    state = _state("trace", mask, values)
    state.baseline = np.full(50, 100.0, dtype=float)
    state.spikes = [_spike(15, 15.0, 5.0)]

    export_tables = _build_export_dataframes(["trace"], {"trace": state})

    assert set(export_tables) == {
        "plateau_events",
        "trace_summary",
        "trace_qc",
        "spike_events",
        "spike_summary",
        "corrected_traces",
    }
    assert list(export_tables["corrected_traces"].columns) == ["time", "trace"]
    assert "plateau_id" in export_tables["plateau_events"].columns
    assert "qc_status" in export_tables["trace_qc"].columns
    assert export_tables["spike_events"].loc[0, "spike_id"] == "S0001"
    assert export_tables["trace_summary"].iloc[-1]["trace_id"] == "ALL"


def test_spike_statistics_summary_image_is_written() -> None:
    spike_events = pd.DataFrame(
        [
            {"trace_id": "a", "amplitude": 10.0, "delta_f_over_f0": 0.10, "snr": 5.0, "fwhm_ms": 1.2, "isi_ms": float("nan"), "instantaneous_firing_rate_hz": float("nan"), "rise_time_ms": 0.4, "return_to_baseline_time_ms": 1.5},
            {"trace_id": "a", "amplitude": 12.0, "delta_f_over_f0": 0.12, "snr": 6.0, "fwhm_ms": 1.4, "isi_ms": 20.0, "instantaneous_firing_rate_hz": 50.0, "rise_time_ms": 0.5, "return_to_baseline_time_ms": 1.8},
            {"trace_id": "b", "amplitude": 8.0, "delta_f_over_f0": 0.08, "snr": 4.0, "fwhm_ms": 1.1, "isi_ms": float("nan"), "instantaneous_firing_rate_hz": float("nan"), "rise_time_ms": 0.3, "return_to_baseline_time_ms": 1.2},
        ]
    )
    spike_summary = pd.DataFrame(_build_trace_spike_summary_rows(["a", "b"], spike_events.to_dict("records")))

    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "spike_statistics_summary.png"
        _save_spike_statistics_summary(spike_summary, spike_events, png_path)

        assert png_path.exists()
        assert png_path.stat().st_size > 0


def test_find_export_file_supports_organized_and_legacy_layouts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reload_dir = root / "reload"
        reload_dir.mkdir()
        organized = reload_dir / "corrected_traces.csv"
        organized.write_text("time,trace\n0,1\n", encoding="utf-8")

        assert _find_export_file(root, "corrected_traces.csv") == organized
        assert _find_export_file(reload_dir, "corrected_traces.csv") == organized

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        legacy = root / "corrected_traces.csv"
        legacy.write_text("time,trace\n0,1\n", encoding="utf-8")

        assert _find_export_file(root, "corrected_traces.csv") == legacy
