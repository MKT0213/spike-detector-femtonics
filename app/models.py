from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

SpikeStatus = Literal["pending", "accepted", "rejected", "manual", "deleted"]
BurstStatus = Literal["pending", "accepted", "rejected", "manual", "deleted"]
AnalysisStatus = Literal["pending", "running", "done", "error"]


@dataclass
class DetectionParams:
    # Spike detector strategy selected in the UI.
    detection_mode: str = "mad"
    # Startup exclusion window (ms), excluding first N ms from all analysis and viewer.
    startup_exclusion_ms: float = 50.0
    # Plateau median filter window (ms), used for centered rolling median plateau detection.
    plateau_median_window_ms: float = 25.0
    # Long-plateau detector controls. Drift correction is detector-only; the
    # final correction baseline still uses rolling percentile + plateau interpolation.
    long_plateau_drift_window_ms: float = 2000.0
    long_plateau_drift_percentile: float = 20.0
    long_plateau_entry_noise_k: float = 5.0
    long_plateau_exit_noise_k: float = 2.0
    long_plateau_exit_fraction: float = 0.5
    long_plateau_enter_sustain_ms: float = 10.0
    long_plateau_exit_sustain_ms: float = 25.0
    # Flat horizontal threshold value (in signal units, same as centered signal).
    absolute_threshold: float = 250.0
    # When enabled, threshold is computed automatically from the no-event period as:
    #   percentile(no_event, 99.5) + auto_threshold_mad_k * MAD(no_event) + auto_threshold_offset
    # This is robust against single outlier samples that would inflate a simple max.
    use_auto_threshold_from_no_event: bool = True
    # MAD multiplier for the auto threshold margin (scales with trace noise level).
    auto_threshold_mad_k: float = 2.0
    # Optional fixed offset added on top of the auto-threshold.
    auto_threshold_offset: float = 0.0
    # Legacy spike-threshold multiplier kept for backward compatibility.
    spike_noise_k: float = 5.0
    # find_peaks prominence multiplier for MAD/STD spike threshold modes.
    spike_prominence_k: float = 3.0
    # Retained smoothing setting for older saved configurations.
    gaussian_sigma_frames: float = 1.0
    min_separation_ms: float = 25.0
    # Half-width of zoom window used for spike-focused plots (UI/export), in ms.
    spike_zoom_half_window_ms: float = 5.0
    max_peaks_per_window: int = 3
    peak_noise_window_ms: float = 50.0
    high_activity_min_separation_ms: float = 10.0
    high_activity_window_ms: float = 120.0
    high_activity_peak_count: int = 3

    # State-guided denoising: Gonzalez adaptive wavelet on bridged low-state
    # samples, TV for plateau cores, raw corrected samples around boundaries.
    low_state_denoiser: str = "gonzalez_adaptive_wavelet"
    gonzalez_threshold_sd: float = 2.0
    gonzalez_max_clusters: int = 20
    gonzalez_attenuation_min: float = 0.5
    gonzalez_attenuation_max: float = 1.0
    gonzalez_enable_noise_cluster_rejection: bool = False
    gonzalez_enable_event_template_rejection: bool = False
    plateau_tv_weight: float = 2.0
    low_state_boundary_protect_ms: float = 20.0
    enable_low_state_safety_gate: bool = True
    min_event_peak_preservation: float = 0.80
    min_local_max_ratio: float = 0.80
    min_reliable_low_state_fraction: float = 0.10
    min_reliable_low_state_samples: int = 20
    plateau_tv_min_duration_ms: float = 5.0
    plateau_tv_max_weight_factor: float = 0.25

    # Optional preprocessing before baseline/plateau/spike computation.
    use_highpass_filter: bool = False
    highpass_cutoff_hz: float = 1.0
    highpass_order: int = 3

    # Local baseline/noise estimation windows remain editable for tuning.
    baseline_mode: str = "percentile"
    baseline_rolling_window_ms: float = 300.0
    baseline_rolling_percentile: float = 10.0
    baseline_savgol_window_ms: float = 300.0
    baseline_savgol_polyorder: int = 3
    # Legacy field kept for backward compatibility.
    plateau_sd_k: float = 2.0
    baseline_second_pass_percentile: float = 50.0  # legacy no-op; kept for backward compatibility
    baseline_window: int = 81
    noise_window: int = 121

    # Legacy fields kept for backward compatibility with older sessions.
    smoothing_window: int = 9
    savgol_polyorder: int = 2
    mad_k_amp: float = 2.2
    mad_k_prom: float = 1.3
    min_amp_above_baseline: float = 4.0
    min_prominence_abs: float = 0.8
    min_distance: int = 4
    min_width: float = 1.0
    max_width: float = 220.0
    plateau_size: int = 1
    event_merge_distance: int = 10
    refine_window: int = 4
    use_absolute_height_threshold: bool = False
    absolute_height_threshold: float = 250.0

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DetectionParams":
        base = cls()
        legacy_prefix = "neuro" + "box"
        interim_prefix = "hy" + "brid"
        for key, value in data.items():
            normalized_key = str(key)
            if normalized_key == f"{legacy_prefix}_prominence_k":
                normalized_key = "spike_prominence_k"
            elif normalized_key == f"{interim_prefix}_prominence_k":
                normalized_key = "spike_prominence_k"
            elif normalized_key == f"{interim_prefix}_threshold_sd":
                normalized_key = "gonzalez_threshold_sd"
            elif normalized_key == f"{interim_prefix}_plateau_protect_ms":
                normalized_key = "low_state_boundary_protect_ms"
            if normalized_key == "detection_mode" and isinstance(value, str):
                for prefix in (legacy_prefix, interim_prefix):
                    value = value.replace(f"{prefix}_", "")
            if normalized_key == "low_state_denoiser" and isinstance(value, str):
                mode = value.strip().lower()
                if mode in {"hybrid", "legacy", "legacy_hybrid"}:
                    value = "legacy_hybrid"
                elif mode in {"none", "off", "disabled"}:
                    value = "none"
                else:
                    value = "gonzalez_adaptive_wavelet"
            if normalized_key in {
                "gonzalez_enable_event_template_rejection",
                "gonzalez_enable_noise_cluster_rejection",
            }:
                # Safety migration: older saved configs may have enabled
                # Gonzalez cleanup steps that can suppress real spikes. Load
                # them off by default; users can re-enable in the current UI.
                value = False
            if normalized_key == "baseline_mode" and isinstance(value, str):
                mode = value.strip().lower()
                if mode in {"savgol", "savitzky-golay", "savitzky_golay"}:
                    value = "savgol"
                elif mode != "percentile":
                    value = "percentile"
            if hasattr(base, normalized_key):
                setattr(base, normalized_key, value)
        return base


@dataclass
class ArtifactParams:
    absolute_low: float = 120.0
    min_depth: float = 8.0
    drop_start_abs: float = 6.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ArtifactParams":
        base = cls()
        for key, value in data.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base


@dataclass
class SpikeRecord:
    trace_name: str
    candidate_index: int
    spike_index: int
    spike_time: float
    spike_amplitude_raw_or_corrected: float
    baseline_at_spike: float
    amplitude_above_baseline: float
    prominence: float
    width: float
    local_noise_estimate: float
    snr: float
    detection_threshold_used: float
    fwhm_ms: float = 0.0  # Full Width at Half Maximum in milliseconds
    delta_f_over_f0: float = float("nan")
    rise_time_ms: float = float("nan")
    return_to_baseline_time_ms: float = float("nan")
    post_spike_level: float = float("nan")
    status: SpikeStatus = "pending"
    source: Literal["auto", "manual"] = "auto"
    spike_type: Literal["single", "burst"] = "single"
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "trace_name": self.trace_name,
            "candidate_index": int(self.candidate_index),
            "spike_index": int(self.spike_index),
            "spike_time": float(self.spike_time),
            "spike_amplitude_raw_or_corrected": float(self.spike_amplitude_raw_or_corrected),
            "baseline_at_spike": float(self.baseline_at_spike),
            "amplitude_above_baseline": float(self.amplitude_above_baseline),
            "prominence": float(self.prominence),
            "width": float(self.width),
            "local_noise_estimate": float(self.local_noise_estimate),
            "snr": float(self.snr),
            "detection_threshold_used": float(self.detection_threshold_used),
            "fwhm_ms": float(self.fwhm_ms),
            "delta_f_over_f0": float(self.delta_f_over_f0),
            "rise_time_ms": float(self.rise_time_ms),
            "return_to_baseline_time_ms": float(self.return_to_baseline_time_ms),
            "post_spike_level": float(self.post_spike_level),
            "source": self.source,
            "spike_type": self.spike_type,
            "notes": self.notes,
        }


@dataclass
class BurstRecord:
    trace_name: str
    burst_index: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration_ms: float
    peak_index: int
    peak_time: float
    peak_amplitude: float
    baseline_at_peak: float
    amplitude_above_baseline: float
    mean_amplitude: float
    spike_count: int
    mean_firing_rate_hz: float
    mean_snr: float
    mean_prominence: float
    local_noise_estimate: float
    status: BurstStatus = "pending"
    source: Literal["auto", "manual"] = "auto"
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "trace_name": self.trace_name,
            "burst_index": int(self.burst_index),
            "start_index": int(self.start_index),
            "end_index": int(self.end_index),
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "duration_ms": float(self.duration_ms),
            "peak_index": int(self.peak_index),
            "peak_time": float(self.peak_time),
            "peak_amplitude": float(self.peak_amplitude),
            "baseline_at_peak": float(self.baseline_at_peak),
            "amplitude_above_baseline": float(self.amplitude_above_baseline),
            "mean_amplitude": float(self.mean_amplitude),
            "spike_count": int(self.spike_count),
            "mean_firing_rate_hz": float(self.mean_firing_rate_hz),
            "mean_snr": float(self.mean_snr),
            "mean_prominence": float(self.mean_prominence),
            "local_noise_estimate": float(self.local_noise_estimate),
            "status": self.status,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass
class TraceCuration:
    trace_name: str
    raw: np.ndarray
    corrected_source: np.ndarray
    corrected: np.ndarray
    time: np.ndarray
    sampling_rate_hz: Optional[float] = None
    baseline: Optional[np.ndarray] = None
    # Fluorescence-unit baseline subtracted before displaying corrected channels.
    correction_baseline: Optional[np.ndarray] = None
    smoothed: Optional[np.ndarray] = None
    highpass_trace: Optional[np.ndarray] = None
    # Lazily computed state-guided denoising overlay. Field names are retained
    # for compatibility with older sessions and UI code paths.
    hybrid_denoised: Optional[np.ndarray] = None
    hybrid_denoised_cache_key: Optional[Tuple[object, ...]] = None
    hybrid_denoised_pending_key: Optional[Tuple[object, ...]] = None
    # Artifact mask from downward-artifact interpolation (True = interpolated region).
    artifact_mask: Optional[np.ndarray] = None
    # Combined plateau mask from long and short detectors (True = plateau region).
    plateau_mask: Optional[np.ndarray] = None
    # Existing long-plateau detector mask.
    long_plateau_mask: Optional[np.ndarray] = None
    # Complementary short-plateau detector mask.
    short_plateau_mask: Optional[np.ndarray] = None
    # Short detector event table:
    # onset, offset_exclusive, duration_ms, plateau_height, internal_spikes, confidence.
    short_plateau_events: Optional[np.ndarray] = None
    debug_threshold: Optional[np.ndarray] = None
    candidate_windows: List[Tuple[int, int]] = field(default_factory=list)
    spikes: List[SpikeRecord] = field(default_factory=list)
    deleted_spikes: List[SpikeRecord] = field(default_factory=list)
    bursts: List[BurstRecord] = field(default_factory=list)
    deleted_bursts: List[BurstRecord] = field(default_factory=list)
    analysis_status: AnalysisStatus = "done"
    analysis_error: str = ""
    analysis_generation: int = 0

    def sort_spikes(self) -> None:
        self.spikes.sort(key=lambda spike: spike.spike_index)

    def to_spike_dataframe(self) -> pd.DataFrame:
        if not self.spikes:
            return pd.DataFrame(
                columns=[
                    "trace_name",
                    "candidate_index",
                    "spike_index",
                    "spike_time",
                    "spike_amplitude_raw_or_corrected",
                    "baseline_at_spike",
                    "amplitude_above_baseline",
                    "prominence",
                    "width",
                    "local_noise_estimate",
                    "snr",
                    "detection_threshold_used",
                    "fwhm_ms",
                    "delta_f_over_f0",
                    "rise_time_ms",
                    "return_to_baseline_time_ms",
                    "post_spike_level",
                    "source",
                    "notes",
                ]
            )
        return pd.DataFrame([spike.to_dict() for spike in self.spikes])


@dataclass
class AppSession:
    excel_path: str = ""
    detection_params: DetectionParams = field(default_factory=DetectionParams)
    artifact_params: ArtifactParams = field(default_factory=ArtifactParams)
    trace_states: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass
class DetectionResult:
    spikes: List[SpikeRecord]
    processed_signal: np.ndarray
    smoothed: np.ndarray
    baseline: np.ndarray
    threshold: np.ndarray
    candidate_windows: List[Tuple[int, int]]
    qc_plot_data: Dict[str, np.ndarray]
