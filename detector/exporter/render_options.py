from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from detector.core.render_normalization import NORMALIZATION_PERCENT_DFF


ChannelId = str

CHANNEL_DEFINITIONS: Tuple[Tuple[ChannelId, str], ...] = (
    ("raw", "Raw"),
    ("corrected_source", "Corrected source"),
    ("corrected", "Corrected"),
    ("baseline", "Baseline"),
    ("correction_baseline", "Correction baseline"),
    ("highpass", "High-pass"),
    ("lowpass", "Low-pass"),
    ("denoised", "Denoised"),
    ("denoise_residual", "Denoise residual"),
)
CHANNEL_LABELS: Dict[ChannelId, str] = dict(CHANNEL_DEFINITIONS)
DEFAULT_SELECTED_CHANNELS: Tuple[ChannelId, ...] = ("raw", "corrected", "denoised")
DEFAULT_ALIGNED_PANEL_CHANNEL: ChannelId = "denoised"
DEFAULT_ALIGNED_PANEL_Y_MIN: float = -15.0
DEFAULT_ALIGNED_PANEL_Y_MAX: float = 30.0
DEFAULT_ALIGNED_PANEL_Y_LOW_PERCENTILE: float = 0.1
DEFAULT_ALIGNED_PANEL_Y_HIGH_PERCENTILE: float = 99.9
DEFAULT_ALIGNED_PANEL_Y_PADDING: float = 0.15
DEFAULT_ALIGNED_PANEL_MIN_Y_SPAN: float = DEFAULT_ALIGNED_PANEL_Y_MAX - DEFAULT_ALIGNED_PANEL_Y_MIN
MARKER_STYLE_DOT: str = "dot"
MARKER_STYLE_BAR: str = "bar"
MARKER_STYLE_DEFINITIONS: Tuple[Tuple[str, str], ...] = (
    (MARKER_STYLE_DOT, "Debug dots + shading"),
    (MARKER_STYLE_BAR, "Bars only"),
)


@dataclass
class TraceWindow:
    enabled: bool = False
    start_time: float | None = None
    end_time: float | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_values(cls, enabled: bool, start_time: float, end_time: float) -> "TraceWindow":
        start = float(start_time)
        end = float(end_time)
        if end < start:
            start, end = end, start
        return cls(enabled=bool(enabled), start_time=start, end_time=end)


@dataclass
class RenderOptions:
    shade_plateaus: bool = True
    show_spike_markers: bool = True
    label_spikes: bool = False
    label_plateaus: bool = False
    marker_style: str = MARKER_STYLE_DOT
    trace_window: TraceWindow = field(default_factory=TraceWindow)
    spike_half_window_ms: float | None = None
    plateau_pre_ms: float = 100.0
    plateau_post_ms: float = 500.0
    selected_channels: Tuple[ChannelId, ...] = DEFAULT_SELECTED_CHANNELS
    normalization_mode: str = NORMALIZATION_PERCENT_DFF
    aligned_panel_channel: ChannelId = DEFAULT_ALIGNED_PANEL_CHANNEL
    aligned_panel_y_min: float = DEFAULT_ALIGNED_PANEL_Y_MIN
    aligned_panel_y_max: float = DEFAULT_ALIGNED_PANEL_Y_MAX
    aligned_panel_auto_y: bool = True
    aligned_panel_share_y: bool = False
    aligned_panel_negative_indicator: bool = False
    aligned_panel_y_low_percentile: float = DEFAULT_ALIGNED_PANEL_Y_LOW_PERCENTILE
    aligned_panel_y_high_percentile: float = DEFAULT_ALIGNED_PANEL_Y_HIGH_PERCENTILE
    aligned_panel_y_padding: float = DEFAULT_ALIGNED_PANEL_Y_PADDING
    aligned_panel_min_y_span: float = DEFAULT_ALIGNED_PANEL_MIN_Y_SPAN

    def normalized_channels(self) -> Tuple[ChannelId, ...]:
        valid = {channel for channel, _label in CHANNEL_DEFINITIONS}
        channels = tuple(str(channel) for channel in self.selected_channels if str(channel) in valid)
        return channels or ("corrected",)

    def effective_spike_half_window_ms(self, default_value: float) -> float:
        try:
            value = float(self.spike_half_window_ms)
        except (TypeError, ValueError):
            return float(default_value)
        if value <= 0.0:
            return float(default_value)
        return value

    def normalized_aligned_panel_channel(self) -> ChannelId:
        valid = {channel for channel, _label in CHANNEL_DEFINITIONS}
        channel = str(self.aligned_panel_channel or DEFAULT_ALIGNED_PANEL_CHANNEL)
        return channel if channel in valid else DEFAULT_ALIGNED_PANEL_CHANNEL

    def normalized_marker_style(self) -> str:
        valid = {style for style, _label in MARKER_STYLE_DEFINITIONS}
        style = str(self.marker_style or MARKER_STYLE_DOT)
        return style if style in valid else MARKER_STYLE_DOT

    def aligned_panel_y_range(self) -> Tuple[float, float]:
        try:
            y_min = float(self.aligned_panel_y_min)
            y_max = float(self.aligned_panel_y_max)
        except (TypeError, ValueError):
            return DEFAULT_ALIGNED_PANEL_Y_MIN, DEFAULT_ALIGNED_PANEL_Y_MAX
        if not y_min < y_max:
            return DEFAULT_ALIGNED_PANEL_Y_MIN, DEFAULT_ALIGNED_PANEL_Y_MAX
        return y_min, y_max

    def aligned_panel_y_percentiles(self) -> Tuple[float, float]:
        try:
            low = float(self.aligned_panel_y_low_percentile)
            high = float(self.aligned_panel_y_high_percentile)
        except (TypeError, ValueError):
            return DEFAULT_ALIGNED_PANEL_Y_LOW_PERCENTILE, DEFAULT_ALIGNED_PANEL_Y_HIGH_PERCENTILE
        low = min(max(low, 0.0), 99.0)
        high = min(max(high, low + 0.1), 100.0)
        return low, high

    def aligned_panel_y_padding_fraction(self) -> float:
        try:
            padding = float(self.aligned_panel_y_padding)
        except (TypeError, ValueError):
            return DEFAULT_ALIGNED_PANEL_Y_PADDING
        return min(max(padding, 0.0), 1.0)

    def aligned_panel_min_y_span_value(self) -> float:
        try:
            span = float(self.aligned_panel_min_y_span)
        except (TypeError, ValueError):
            return DEFAULT_ALIGNED_PANEL_MIN_Y_SPAN
        return max(0.0, span)

    def to_dict(self) -> Dict[str, object]:
        return {
            "shade_plateaus": bool(self.shade_plateaus),
            "show_spike_markers": bool(self.show_spike_markers),
            "label_spikes": bool(self.label_spikes),
            "label_plateaus": bool(self.label_plateaus),
            "marker_style": self.normalized_marker_style(),
            "trace_window": self.trace_window.to_dict(),
            "spike_half_window_ms": self.spike_half_window_ms,
            "plateau_pre_ms": float(self.plateau_pre_ms),
            "plateau_post_ms": float(self.plateau_post_ms),
            "selected_channels": list(self.normalized_channels()),
            "normalization_mode": str(self.normalization_mode),
            "aligned_panel_channel": self.normalized_aligned_panel_channel(),
            "aligned_panel_y_min": self.aligned_panel_y_range()[0],
            "aligned_panel_y_max": self.aligned_panel_y_range()[1],
            "aligned_panel_auto_y": bool(self.aligned_panel_auto_y),
            "aligned_panel_share_y": bool(self.aligned_panel_share_y),
            "aligned_panel_negative_indicator": bool(self.aligned_panel_negative_indicator),
            "aligned_panel_y_low_percentile": self.aligned_panel_y_percentiles()[0],
            "aligned_panel_y_high_percentile": self.aligned_panel_y_percentiles()[1],
            "aligned_panel_y_padding": self.aligned_panel_y_padding_fraction(),
            "aligned_panel_min_y_span": self.aligned_panel_min_y_span_value(),
        }


def selected_channels_from_iterable(values: Iterable[object]) -> Tuple[ChannelId, ...]:
    valid = {channel for channel, _label in CHANNEL_DEFINITIONS}
    channels = tuple(str(value) for value in values if str(value) in valid)
    return channels or ("corrected",)
