from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.core.models import ArtifactParams, BurstRecord, DetectionParams, SpikeRecord, TraceCuration  # noqa: E402
from detector.core.pipeline import PipelineConfig  # noqa: E402
from detector.core.session_io import apply_session_to_traces, load_session, save_session  # noqa: E402


def _spike(index: int, status: str, notes: str = "") -> SpikeRecord:
    return SpikeRecord(
        trace_name="trace",
        candidate_index=index,
        spike_index=index,
        spike_time=float(index),
        spike_amplitude_raw_or_corrected=10.0,
        baseline_at_spike=1.0,
        amplitude_above_baseline=9.0,
        prominence=8.0,
        width=1.0,
        local_noise_estimate=0.5,
        snr=18.0,
        detection_threshold_used=2.0,
        status=status,  # type: ignore[arg-type]
        notes=notes,
    )


def _burst(index: int, status: str, notes: str = "") -> BurstRecord:
    return BurstRecord(
        trace_name="trace",
        burst_index=index,
        start_index=10,
        end_index=20,
        start_time=10.0,
        end_time=20.0,
        duration_ms=10.0,
        peak_index=15,
        peak_time=15.0,
        peak_amplitude=20.0,
        baseline_at_peak=1.0,
        amplitude_above_baseline=19.0,
        mean_amplitude=12.0,
        spike_count=3,
        mean_firing_rate_hz=300.0,
        mean_snr=8.0,
        mean_prominence=7.0,
        local_noise_estimate=0.5,
        status=status,  # type: ignore[arg-type]
        notes=notes,
    )


def _trace_state() -> TraceCuration:
    values = np.zeros(30, dtype=float)
    return TraceCuration(
        trace_name="trace",
        raw=values.copy(),
        corrected_source=values.copy(),
        corrected=values.copy(),
        time=np.arange(values.size, dtype=float),
        spikes=[_spike(12, "accepted", "keep")],
        deleted_spikes=[_spike(18, "deleted", "bad")],
        bursts=[_burst(1, "manual", "burst keep")],
        deleted_bursts=[_burst(2, "deleted", "burst bad")],
    )


def test_session_round_trip_preserves_curation_records() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "session.json"
        source = _trace_state()

        save_session(
            str(path),
            "source.xlsx",
            DetectionParams(spike_noise_k=7.0, rolling_downsample_factor=4),
            ArtifactParams(min_depth=3.0),
            {"trace": source},
            pipeline_config=PipelineConfig.from_dict(
                {
                    "active_preset": "custom",
                    "ordered_step_ids": [
                        "baseline_correction",
                        "rolling_average_downsample",
                        "plateau_detection",
                        "denoising",
                        "spike_detection",
                    ],
                    "enabled_steps": {"rolling_average_downsample": True},
                }
            ).to_dict(),
        )
        session = load_session(str(path))
        target = TraceCuration(
            trace_name="trace",
            raw=source.raw.copy(),
            corrected_source=source.corrected_source.copy(),
            corrected=source.corrected.copy(),
            time=source.time.copy(),
        )

        apply_session_to_traces(session, {"trace": target})

        assert session.excel_path == "source.xlsx"
        assert session.detection_params.spike_noise_k == 7.0
        assert session.detection_params.rolling_downsample_factor == 4
        assert session.artifact_params.min_depth == 3.0
        assert "rolling_average_downsample" in session.pipeline_config["ordered_step_ids"]
        assert session.pipeline_config["enabled_steps"]["rolling_average_downsample"] is True
        assert target.spikes[0].status == "accepted"
        assert target.spikes[0].notes == "keep"
        assert target.deleted_spikes[0].status == "deleted"
        assert target.deleted_spikes[0].notes == "bad"
        assert target.bursts[0].status == "manual"
        assert target.bursts[0].notes == "burst keep"
        assert target.deleted_bursts[0].status == "deleted"
        assert target.deleted_bursts[0].notes == "burst bad"
