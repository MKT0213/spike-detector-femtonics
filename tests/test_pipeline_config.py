from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.core.pipeline import ADDABLE_PIPELINE_STEP_IDS, PIPELINE_STEP_DEFINITIONS, PipelineConfig  # noqa: E402
from detector.analysis.package import ANALYSIS_MANIFEST_NAME, save_analysis_package  # noqa: E402
from detector.core.models import ArtifactParams, DetectionParams, SpikeRecord, TraceCuration  # noqa: E402


def _state(name: str) -> TraceCuration:
    time = np.arange(20, dtype=float)
    corrected = np.zeros(20, dtype=float)
    corrected[8] = 5.0
    baseline = np.zeros(20, dtype=float)
    spike = SpikeRecord(
        trace_name=name,
        candidate_index=8,
        spike_index=8,
        spike_time=8.0,
        spike_amplitude_raw_or_corrected=5.0,
        baseline_at_spike=0.0,
        amplitude_above_baseline=5.0,
        prominence=4.0,
        width=1.0,
        local_noise_estimate=0.5,
        snr=10.0,
        detection_threshold_used=2.0,
    )
    return TraceCuration(
        trace_name=name,
        raw=corrected.copy(),
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=time,
        sampling_rate_hz=1000.0,
        baseline=baseline,
        plateau_mask=np.zeros(20, dtype=bool),
        spikes=[spike],
    )


def test_pipeline_config_groups_parameters_by_analysis_step() -> None:
    config = PipelineConfig.from_preset("current")
    data = config.to_dict()

    assert data["active_preset"] == "current"
    assert "not interchangeable" in data["order_summary"]
    assert "Denoising Input Normalization" in data["order_summary"]
    assert data["parameter_groups"]["artifact_cleanup"] == [
        "absolute_low",
        "min_depth",
        "drop_start_abs",
    ]
    assert "baseline_mode" in data["parameter_groups"]["baseline_correction"]
    assert "long_plateau_entry_noise_k" in data["parameter_groups"]["plateau_detection"]
    assert "long_plateau_exit_sustain_ms" in data["parameter_groups"]["plateau_detection"]
    assert "short_plateau_min_confidence" in data["parameter_groups"]["plateau_detection"]
    plateau_subgroups = data["parameter_subgroups"]["plateau_detection"]
    assert plateau_subgroups["Shared / setup"] == [
        "startup_exclusion_ms",
        "plateau_median_window_ms",
    ]
    assert "long_plateau_entry_noise_k" in plateau_subgroups["Long baseline-protection plateaus"]
    assert "long_plateau_exit_sustain_ms" in plateau_subgroups["Long baseline-protection plateaus"]
    assert "short_plateau_min_confidence" in plateau_subgroups["Short and burst raised-floor plateaus"]
    assert "short_plateau_min_duration_ms" in plateau_subgroups["Short and burst raised-floor plateaus"]
    assert "burst-associated" in PIPELINE_STEP_DEFINITIONS["plateau_detection"]["description"]
    assert ADDABLE_PIPELINE_STEP_IDS == list(PIPELINE_STEP_DEFINITIONS)
    assert "artifact_cleanup" not in config.ordered_step_ids
    assert data["enabled_steps"]["artifact_cleanup"] is False
    assert "highpass_filter" not in config.ordered_step_ids
    assert "lowpass_filter" not in config.ordered_step_ids
    assert "rolling_average_downsample" not in config.ordered_step_ids
    assert "highpass_cutoff_hz" in data["parameter_groups"]["highpass_filter"]
    assert "lowpass_cutoff_hz" in data["parameter_groups"]["lowpass_filter"]
    assert data["enabled_steps"]["rolling_average_downsample"] is False
    assert data["parameter_groups"]["rolling_average_downsample"] == ["rolling_downsample_factor"]
    assert "denoising_method" in data["parameter_groups"]["denoising"]
    assert "denoising_input_normalization" in data["parameter_groups"]["denoising"]
    assert "spike_noise_k" in data["parameter_groups"]["spike_detection"]
    assert "template_matching_enable_autopick" in data["parameter_groups"]["spike_detection"]
    assert "template_matching_template_indices" in data["parameter_groups"]["spike_detection"]
    assert "template_matching_false_positive_rate" in data["parameter_groups"]["spike_detection"]
    spike_subgroups = data["parameter_subgroups"]["spike_detection"]
    assert spike_subgroups["Shared detection setup"] == [
        "detection_mode",
        "min_separation_ms",
        "spike_zoom_half_window_ms",
    ]
    assert spike_subgroups["Threshold methods"] == [
        "spike_noise_k",
        "spike_prominence_k",
    ]
    assert spike_subgroups["Template matching"] == [
        "template_matching_template_indices",
        "template_matching_enable_autopick",
        "template_matching_pre_ms",
        "template_matching_post_ms",
        "template_matching_filter_cutoff_hz",
        "template_matching_false_positive_rate",
    ]
    assert "display_units" in data["parameter_groups"]["signal_display"]
    assert "selected_overlay" in data["parameter_groups"]["signal_display"]
    assert "show_denoise_residual" in data["parameter_groups"]["signal_display"]
    assert data["order_constraints"]["baseline_correction"]["policy"] == "fixed"
    assert "Optional" in data["order_constraints"]["artifact_cleanup"]["note"]
    assert "Not interchangeable" in data["order_constraints"]["plateau_detection"]["note"]
    assert "input normalization is fixed inside Denoising" in data["order_constraints"]["denoising"]["note"]
    assert data["order_constraints"]["highpass_filter"]["policy"] == "movable_filter"
    assert data["order_constraints"]["rolling_average_downsample"]["policy"] == "movable_preprocessing"
    assert data["order_constraints"]["signal_display"]["policy"] == "output"


def test_pipeline_config_preserves_added_filter_steps() -> None:
    config = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "artifact_cleanup",
                "baseline_correction",
                "highpass_filter",
                "lowpass_filter",
                "plateau_detection",
                "denoising",
                "spike_detection",
                "review_save",
                "signal_display",
            ],
            "enabled_steps": {
                "highpass_filter": True,
                "lowpass_filter": True,
                "denoising": False,
            },
        }
    )
    data = config.to_dict()

    assert data["ordered_step_ids"][2:4] == ["highpass_filter", "lowpass_filter"]
    assert data["enabled_steps"]["highpass_filter"] is True
    assert data["enabled_steps"]["lowpass_filter"] is True
    assert data["enabled_steps"]["denoising"] is False


def test_pipeline_config_preserves_removed_default_steps() -> None:
    config = PipelineConfig.from_dict(
        {
            "active_preset": "custom",
            "ordered_step_ids": [
                "artifact_cleanup",
                "baseline_correction",
                "spike_detection",
            ],
            "enabled_steps": {
                "plateau_detection": False,
                "denoising": False,
                "review_save": False,
                "signal_display": False,
            },
        }
    )
    data = config.to_dict()

    assert data["ordered_step_ids"] == [
        "artifact_cleanup",
        "baseline_correction",
        "spike_detection",
    ]
    assert data["enabled_steps"]["plateau_detection"] is False
    assert data["enabled_steps"]["review_save"] is False


def test_pipeline_config_tolerates_malformed_saved_values() -> None:
    config = PipelineConfig.from_dict(
        {
            "active_preset": "no_denoising",
            "ordered_step_ids": "not-a-list",
            "enabled_steps": "not-a-dict",
            "trace_order": "trace_a",
        }
    )
    data = config.to_dict()

    assert data["active_preset"] == "no_denoising"
    assert data["ordered_step_ids"] == PipelineConfig.from_preset("no_denoising").ordered_step_ids
    assert data["enabled_steps"]["denoising"] is False
    assert data["trace_order"] == []

    empty_order = PipelineConfig.from_dict(
        {
            "active_preset": "current",
            "ordered_step_ids": ["missing_step"],
        }
    )
    assert empty_order.ordered_step_ids == PipelineConfig.from_preset("current").ordered_step_ids


def test_analysis_package_manifest_stores_pipeline_config(tmp_path: Path) -> None:
    traces = {"trace_a": _state("trace_a")}
    pipeline_config = PipelineConfig.from_preset("no_denoising")
    pipeline_config.trace_order = ["trace_a"]

    package_dir = save_analysis_package(
        tmp_path / "analysis_trace_a",
        source_excel="source.xlsx",
        detection_params=DetectionParams(),
        artifact_params=ArtifactParams(),
        trace_names=["trace_a"],
        traces=traces,
        pipeline_config=pipeline_config.to_dict(),
    )

    manifest = json.loads((package_dir / ANALYSIS_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["pipeline_config"]["active_preset"] == "no_denoising"
    assert manifest["pipeline_config"]["enabled_steps"]["denoising"] is False
    assert manifest["pipeline_config"]["trace_order"] == ["trace_a"]
