from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List


PIPELINE_ORDER_SUMMARY = (
    "Core detector stages are not interchangeable: Baseline Correction -> Plateau Detection -> "
    "Denoising Input Normalization -> Gonzalez Denoising -> Spike Detection. "
    "Artifact Cleanup is optional and runs before Baseline Correction only when explicitly enabled. "
    "Denoising Input Normalization is a fixed substage inside Denoising, not a draggable step. "
    "High-Pass, Low-Pass, and Rolling-Average Downsampling have execution placement; "
    "Review/Save and Signal Display are output stages."
)

PIPELINE_ORDER_POLICY_LABELS: Dict[str, str] = {
    "fixed": "fixed stage",
    "movable_filter": "movable filter",
    "movable_preprocessing": "movable preprocessing",
    "output": "output stage",
}


PIPELINE_STEP_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "artifact_cleanup": {
        "label": "Artifact Cleanup (Optional)",
        "description": "Optionally interpolates trace dropouts before detection.",
        "order_policy": "fixed",
        "order_note": "Optional; when enabled, artifact interpolation runs before baseline, plateau, denoising, and spike detection.",
    },
    "baseline_correction": {
        "label": "Baseline Correction",
        "description": "Estimates baseline and corrected trace units.",
        "order_policy": "fixed",
        "order_note": "Not interchangeable; baseline correction feeds plateau detection and corrected-trace measurements.",
    },
    "highpass_filter": {
        "label": "High-Pass Filter",
        "description": "Removes slow drift before downstream detection steps.",
        "order_policy": "movable_filter",
        "order_note": "Movable optional filter; placement before Baseline, Denoising, or Spike Detection changes which working trace is filtered.",
    },
    "lowpass_filter": {
        "label": "Low-Pass Filter",
        "description": "Suppresses fast noise before downstream detection steps.",
        "order_policy": "movable_filter",
        "order_note": "Movable optional filter; placement before Baseline, Denoising, or Spike Detection changes which working trace is filtered.",
    },
    "rolling_average_downsample": {
        "label": "Rolling-Average Downsampling",
        "description": "Optionally reduces the trace sample rate by averaging non-overlapping sample blocks.",
        "order_policy": "movable_preprocessing",
        "order_note": "Movable optional step; placement controls which downstream stages use the lower sampling rate.",
    },
    "plateau_detection": {
        "label": "Plateau Detection",
        "description": "Long baseline-protection pass, clean short events, and burst-associated raised-floor plateaus.",
        "order_policy": "fixed",
        "order_note": "Not interchangeable; plateau masks are computed after baseline correction and guide denoising and spike detection.",
    },
    "denoising": {
        "label": "Denoising",
        "description": "Normalizes the denoiser input and builds the denoised detection aid.",
        "order_policy": "fixed",
        "order_note": "Not interchangeable when enabled; input normalization is fixed inside Denoising and the denoised aid runs before spike detection.",
    },
    "spike_detection": {
        "label": "Spike Detection",
        "description": "Detects spike candidates and event metrics.",
        "order_policy": "fixed",
        "order_note": "Not interchangeable; spike detection consumes the baseline-corrected and optional denoised working trace.",
    },
    "review_save": {
        "label": "Review / Save Package",
        "description": "Reviews traces and writes the reusable package.",
        "order_policy": "output",
        "order_note": "Output stage; review and saving happen after analysis and do not change detector execution order.",
    },
    "signal_display": {
        "label": "Signal Display / Overlay",
        "description": "Controls plot signal overlays only.",
        "order_policy": "output",
        "order_note": "Output stage; display toggles affect plotting only and do not change detector execution order.",
    },
}

DEFAULT_PIPELINE_STEP_IDS: List[str] = [
    "baseline_correction",
    "plateau_detection",
    "denoising",
    "spike_detection",
    "review_save",
    "signal_display",
]

OPTIONAL_PIPELINE_STEP_IDS: List[str] = [
    "artifact_cleanup",
    "highpass_filter",
    "lowpass_filter",
    "rolling_average_downsample",
]

ADDABLE_PIPELINE_STEP_IDS: List[str] = list(PIPELINE_STEP_DEFINITIONS)


def pipeline_order_constraints() -> Dict[str, Dict[str, str]]:
    return {
        str(step_id): {
            "policy": str(definition.get("order_policy", "")),
            "policy_label": PIPELINE_ORDER_POLICY_LABELS.get(
                str(definition.get("order_policy", "")),
                str(definition.get("order_policy", "")),
            ),
            "note": str(definition.get("order_note", "")),
        }
        for step_id, definition in PIPELINE_STEP_DEFINITIONS.items()
    }

PIPELINE_PARAMETER_GROUPS: Dict[str, List[str]] = {
    "artifact_cleanup": [
        "absolute_low",
        "min_depth",
        "drop_start_abs",
    ],
    "baseline_correction": [
        "baseline_mode",
        "baseline_rolling_window_ms",
        "baseline_rolling_percentile",
        "baseline_savgol_window_ms",
        "baseline_savgol_polyorder",
    ],
    "highpass_filter": [
        "use_highpass_filter",
        "highpass_cutoff_hz",
        "highpass_order",
    ],
    "lowpass_filter": [
        "use_lowpass_filter",
        "lowpass_cutoff_hz",
        "lowpass_order",
    ],
    "rolling_average_downsample": [
        "rolling_downsample_factor",
    ],
    "plateau_detection": [
        "startup_exclusion_ms",
        "plateau_median_window_ms",
        "long_plateau_drift_window_ms",
        "long_plateau_drift_percentile",
        "long_plateau_entry_noise_k",
        "long_plateau_exit_noise_k",
        "long_plateau_exit_fraction",
        "long_plateau_enter_sustain_ms",
        "long_plateau_exit_sustain_ms",
        "short_plateau_step_noise_k",
        "short_plateau_elevated_noise_k",
        "short_plateau_survival_noise_k",
        "short_plateau_loss_noise_k",
        "short_plateau_median_noise_k",
        "short_plateau_pre_quiet_noise_k",
        "short_plateau_min_support_fraction",
        "short_plateau_min_confidence",
        "short_plateau_early_dwell_ms",
        "short_plateau_min_support_ms",
        "short_plateau_min_duration_ms",
    ],
    "denoising": [
        "denoising_method",
        "denoising_input_normalization",
        "gonzalez_threshold_sd",
        "gonzalez_max_clusters",
        "gonzalez_attenuation_min",
        "gonzalez_attenuation_max",
        "gonzalez_enable_noise_cluster_rejection",
        "gonzalez_enable_event_template_rejection",
    ],
    "spike_detection": [
        "detection_mode",
        "spike_noise_k",
        "spike_prominence_k",
        "template_matching_enable_autopick",
        "template_matching_template_indices",
        "template_matching_pre_ms",
        "template_matching_post_ms",
        "template_matching_filter_cutoff_hz",
        "template_matching_false_positive_rate",
        "min_separation_ms",
        "spike_zoom_half_window_ms",
    ],
    "signal_display": [
        "display_units",
        "selected_overlay",
        "show_raw",
        "show_corrected",
        "show_baseline",
        "show_highpass",
        "show_denoised",
        "show_normalized_denoised",
        "show_denoise_residual",
    ],
}

PIPELINE_PARAMETER_SUBGROUPS: Dict[str, Dict[str, List[str]]] = {
    "plateau_detection": {
        "Shared / setup": [
            "startup_exclusion_ms",
            "plateau_median_window_ms",
        ],
        "Long baseline-protection plateaus": [
            "long_plateau_drift_window_ms",
            "long_plateau_drift_percentile",
            "long_plateau_entry_noise_k",
            "long_plateau_exit_noise_k",
            "long_plateau_exit_fraction",
            "long_plateau_enter_sustain_ms",
            "long_plateau_exit_sustain_ms",
        ],
        "Short and burst raised-floor plateaus": [
            "short_plateau_step_noise_k",
            "short_plateau_elevated_noise_k",
            "short_plateau_survival_noise_k",
            "short_plateau_loss_noise_k",
            "short_plateau_median_noise_k",
            "short_plateau_pre_quiet_noise_k",
            "short_plateau_min_support_fraction",
            "short_plateau_min_confidence",
            "short_plateau_early_dwell_ms",
            "short_plateau_min_support_ms",
            "short_plateau_min_duration_ms",
        ],
    },
    "spike_detection": {
        "Shared detection setup": [
            "detection_mode",
            "min_separation_ms",
            "spike_zoom_half_window_ms",
        ],
        "Threshold methods": [
            "spike_noise_k",
            "spike_prominence_k",
        ],
        "Template matching": [
            "template_matching_template_indices",
            "template_matching_enable_autopick",
            "template_matching_pre_ms",
            "template_matching_post_ms",
            "template_matching_filter_cutoff_hz",
            "template_matching_false_positive_rate",
        ],
    },
}

PIPELINE_PRESETS: Dict[str, Dict[str, Any]] = {
    "current": {
        "label": "Current Algorithm",
        "locked": True,
        "order": DEFAULT_PIPELINE_STEP_IDS.copy(),
        "enabled": {step_id: True for step_id in DEFAULT_PIPELINE_STEP_IDS},
        "warning": "Default preset: artifact interpolation is off unless explicitly added.",
    },
    "no_denoising": {
        "label": "No Denoising Aid",
        "locked": False,
        "order": DEFAULT_PIPELINE_STEP_IDS.copy(),
        "enabled": {
            **{step_id: True for step_id in DEFAULT_PIPELINE_STEP_IDS},
            "denoising": False,
        },
        "warning": "Uses denoising_method = none.",
    },
    "denoise_first": {
        "label": "Denoising First",
        "locked": False,
        "order": [
            "denoising",
            "baseline_correction",
            "plateau_detection",
            "spike_detection",
            "review_save",
            "signal_display",
        ],
        "enabled": {step_id: True for step_id in DEFAULT_PIPELINE_STEP_IDS},
        "warning": "Experimental order: saved in package metadata; default detector path is unchanged.",
    },
    "baseline_after_denoising": {
        "label": "Baseline After Denoising",
        "locked": False,
        "order": [
            "denoising",
            "baseline_correction",
            "plateau_detection",
            "spike_detection",
            "review_save",
            "signal_display",
        ],
        "enabled": {step_id: True for step_id in DEFAULT_PIPELINE_STEP_IDS},
        "warning": "Experimental order: saved in package metadata; default detector path is unchanged.",
    },
    "no_baseline": {
        "label": "No Baseline Correction",
        "locked": False,
        "order": [
            "plateau_detection",
            "denoising",
            "spike_detection",
            "review_save",
            "signal_display",
        ],
        "enabled": {
            **{step_id: True for step_id in DEFAULT_PIPELINE_STEP_IDS},
            "baseline_correction": False,
        },
        "warning": "Experimental preset: records intent only; current detector baseline remains protected.",
    },
    "custom": {
        "label": "Custom Editable Pipeline",
        "locked": False,
        "order": DEFAULT_PIPELINE_STEP_IDS.copy(),
        "enabled": {step_id: True for step_id in DEFAULT_PIPELINE_STEP_IDS},
        "warning": "Custom order changes workflow metadata and should be compared against Current Algorithm.",
    },
}


@dataclass
class PipelineConfig:
    active_preset: str = "current"
    ordered_step_ids: List[str] = field(default_factory=lambda: DEFAULT_PIPELINE_STEP_IDS.copy())
    enabled_steps: Dict[str, bool] = field(
        default_factory=lambda: {step_id: True for step_id in DEFAULT_PIPELINE_STEP_IDS}
    )
    trace_order: List[str] = field(default_factory=list)

    @classmethod
    def from_preset(cls, preset_id: str) -> "PipelineConfig":
        preset = PIPELINE_PRESETS.get(preset_id, PIPELINE_PRESETS["current"])
        ordered = [str(step_id) for step_id in preset["order"] if step_id in PIPELINE_STEP_DEFINITIONS]
        enabled_raw = dict(preset["enabled"])
        return cls(
            active_preset=str(preset_id if preset_id in PIPELINE_PRESETS else "current"),
            ordered_step_ids=ordered,
            enabled_steps={
                step_id: bool(enabled_raw.get(step_id, step_id in DEFAULT_PIPELINE_STEP_IDS))
                for step_id in PIPELINE_STEP_DEFINITIONS
            },
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        base = cls.from_preset(str(data.get("active_preset", "current")))
        ordered_source = data.get("ordered_step_ids", base.ordered_step_ids)
        if not isinstance(ordered_source, (list, tuple)):
            ordered_source = base.ordered_step_ids
        ordered = [str(step_id) for step_id in ordered_source if str(step_id) in PIPELINE_STEP_DEFINITIONS]
        if not ordered:
            ordered = list(base.ordered_step_ids)
        enabled_source = data.get("enabled_steps", base.enabled_steps)
        enabled_raw = dict(enabled_source) if isinstance(enabled_source, Mapping) else {}
        trace_order_source = data.get("trace_order", [])
        if not isinstance(trace_order_source, (list, tuple)):
            trace_order_source = []
        trace_order = [str(name) for name in trace_order_source]
        return cls(
            active_preset=base.active_preset,
            ordered_step_ids=ordered,
            enabled_steps={
                step_id: bool(
                    enabled_raw.get(
                        step_id,
                        base.enabled_steps.get(step_id, step_id in DEFAULT_PIPELINE_STEP_IDS),
                    )
                )
                for step_id in PIPELINE_STEP_DEFINITIONS
            },
            trace_order=trace_order,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_preset": self.active_preset,
            "preset_label": str(PIPELINE_PRESETS.get(self.active_preset, {}).get("label", self.active_preset)),
            "ordered_step_ids": list(self.ordered_step_ids),
            "enabled_steps": {str(key): bool(value) for key, value in self.enabled_steps.items()},
            "trace_order": list(self.trace_order),
            "parameter_groups": {
                str(step_id): list(fields)
                for step_id, fields in PIPELINE_PARAMETER_GROUPS.items()
            },
            "parameter_subgroups": {
                str(step_id): {
                    str(subgroup): list(fields)
                    for subgroup, fields in subgroups.items()
                }
                for step_id, subgroups in PIPELINE_PARAMETER_SUBGROUPS.items()
            },
            "order_summary": PIPELINE_ORDER_SUMMARY,
            "order_constraints": pipeline_order_constraints(),
        }

    def is_locked(self) -> bool:
        return bool(PIPELINE_PRESETS.get(self.active_preset, PIPELINE_PRESETS["current"]).get("locked", False))

    def warning(self) -> str:
        return str(PIPELINE_PRESETS.get(self.active_preset, PIPELINE_PRESETS["current"]).get("warning", ""))
