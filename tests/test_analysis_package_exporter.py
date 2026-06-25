from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.analysis.package import load_analysis_package, save_analysis_package  # noqa: E402
from detector.core import gui_main  # noqa: E402
from detector.exporter.service import ExportOptions, export_analysis_package  # noqa: E402
from detector.exporter.render_options import MARKER_STYLE_BAR, RenderOptions, TraceWindow  # noqa: E402
from detector.exporter import rendering as exporter_rendering  # noqa: E402
from detector.exporter.rendering import available_channel_ids  # noqa: E402
from detector.exporter.window import ExporterWindow  # noqa: E402
from detector.core.models import ArtifactParams, DetectionParams, SpikeRecord, TraceCuration  # noqa: E402


def _state(name: str, offset: float = 0.0) -> TraceCuration:
    time = np.arange(20, dtype=float)
    corrected = np.zeros(20, dtype=float) + offset
    corrected[8] = 5.0 + offset
    baseline = np.zeros(20, dtype=float) + offset
    plateau_mask = np.zeros(20, dtype=bool)
    plateau_mask[5:10] = True
    spike = SpikeRecord(
        trace_name=name,
        candidate_index=8,
        spike_index=8,
        spike_time=8.0,
        spike_amplitude_raw_or_corrected=float(corrected[8]),
        baseline_at_spike=float(baseline[8]),
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
        plateau_mask=plateau_mask,
        long_plateau_mask=plateau_mask.copy(),
        short_plateau_mask=np.zeros_like(plateau_mask),
        spikes=[spike],
        long_plateau_debug={"input_signal": corrected.copy(), "long_plateau_region_debug_rows": []},
    )


def test_startup_trimmed_export_uses_absolute_time_axis() -> None:
    state = _state("trace_offset")
    state.time = 250.0 + np.arange(20, dtype=float)
    state.spikes[0].spike_time = 258.0

    export_view = gui_main._startup_trimmed_trace_export_data(state, startup_exclusion_ms=5.0)

    assert export_view["startup_idx"] == 5
    np.testing.assert_allclose(export_view["time"][:3], np.array([255.0, 256.0, 257.0]))
    assert export_view["spikes"][0].spike_index == 3
    assert export_view["spikes"][0].spike_time == 258.0

    plateau_rows = gui_main._plateau_rows(
        state.trace_name,
        export_view["time"],
        np.asarray(export_view["plateau_mask"], dtype=bool),
        np.asarray(export_view["long_plateau_mask"], dtype=bool),
        np.asarray(export_view["short_plateau_mask"], dtype=bool),
        np.asarray(export_view["burst_plateau_mask"], dtype=bool),
        export_view["burst_plateau_events"],
        export_view["plateau_signal_classification"],
        sampling_rate_hz=state.sampling_rate_hz,
    )

    assert plateau_rows[0]["start_index"] == 0
    assert plateau_rows[0]["start_time"] == 255.0
    assert plateau_rows[0]["onset_time_ms"] == 255.0


def _state_with_two_events(name: str) -> TraceCuration:
    time = np.arange(40, dtype=float)
    corrected = np.zeros(40, dtype=float)
    corrected[8] = 5.0
    corrected[28] = 4.0
    baseline = np.zeros(40, dtype=float)
    plateau_mask = np.zeros(40, dtype=bool)
    plateau_mask[5:11] = True
    plateau_mask[24:31] = True
    spikes = [
        SpikeRecord(
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
        ),
        SpikeRecord(
            trace_name=name,
            candidate_index=28,
            spike_index=28,
            spike_time=28.0,
            spike_amplitude_raw_or_corrected=4.0,
            baseline_at_spike=0.0,
            amplitude_above_baseline=4.0,
            prominence=3.0,
            width=1.0,
            local_noise_estimate=0.5,
            snr=8.0,
            detection_threshold_used=2.0,
        ),
    ]
    return TraceCuration(
        trace_name=name,
        raw=corrected + 100.0,
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=time,
        sampling_rate_hz=1000.0,
        baseline=baseline,
        highpass_trace=corrected * 0.5,
        hybrid_denoised=corrected * 0.9,
        plateau_mask=plateau_mask,
        long_plateau_mask=plateau_mask.copy(),
        short_plateau_mask=np.zeros_like(plateau_mask),
        spikes=spikes,
        long_plateau_debug={"input_signal": corrected.copy(), "long_plateau_region_debug_rows": []},
    )


def _state_with_burst_plateau(name: str) -> TraceCuration:
    time = np.arange(40, dtype=float)
    corrected = np.zeros(40, dtype=float)
    corrected[10:18] = 12.0
    corrected[[11, 13, 15]] = 32.0
    baseline = np.zeros(40, dtype=float)
    plateau_mask = np.zeros(40, dtype=bool)
    plateau_mask[10:18] = True
    burst_events = np.asarray([[10.0, 18.0, 8.0, 12.0, 1.0, 9.5, 7.5, 0.88]], dtype=float)
    return TraceCuration(
        trace_name=name,
        raw=corrected + 100.0,
        corrected_source=corrected.copy(),
        corrected=corrected,
        time=time,
        sampling_rate_hz=1000.0,
        baseline=baseline,
        plateau_mask=plateau_mask,
        hybrid_denoised=np.where(plateau_mask, 12.0, corrected),
        long_plateau_mask=np.zeros_like(plateau_mask),
        short_plateau_mask=np.zeros_like(plateau_mask),
        burst_plateau_mask=plateau_mask.copy(),
        burst_plateau_events=burst_events,
        burst_plateau_debug={
            "burst_plateau_candidate_rows": [
                {
                    "source": "long_near_miss",
                    "accepted": True,
                    "start_index": 10,
                    "end_index_exclusive": 18,
                    "confidence": 0.88,
                    "rejection_reasons": "",
                }
            ]
        },
        plateau_signal_classification=[
            {
                "start_index": 10,
                "end_index_exclusive": 18,
                "duration_ms": 8.0,
                "detector_source": "burst",
                "signal_type": "mixed",
                "signal_confidence": 0.72,
                "raw_highfreq_rms": 7.5,
                "denoised_highfreq_rms": 1.0,
                "residual_rms": 6.5,
                "residual_fraction": 0.86,
                "noise_reduction_fraction": 0.75,
                "raw_peak_density_hz": 375.0,
                "denoised_peak_density_hz": 125.0,
                "preserved_peak_fraction": 0.33,
            }
        ],
        spikes=[],
        long_plateau_debug={"input_signal": corrected.copy(), "long_plateau_region_debug_rows": []},
    )


def test_analysis_package_round_trip_restores_trace_state(tmp_path: Path) -> None:
    traces = {"trace_a": _state("trace_a")}
    traces["trace_a"].hybrid_denoised_cache_key = ("gonzalez_full_trace", 1000.0, 2.0)
    traces["trace_a"].gonzalez_cwt_debug_cache_key = ("debug", 1)
    package_dir = save_analysis_package(
        tmp_path / "analysis_trace_a",
        source_excel="source.xlsx",
        detection_params=DetectionParams(),
        artifact_params=ArtifactParams(),
        trace_names=["trace_a"],
        traces=traces,
    )

    package = load_analysis_package(package_dir)

    assert package.source_excel == "source.xlsx"
    assert package.trace_names == ["trace_a"]
    loaded = package.traces["trace_a"]
    np.testing.assert_allclose(loaded.corrected, traces["trace_a"].corrected)
    np.testing.assert_array_equal(loaded.plateau_mask, traces["trace_a"].plateau_mask)
    assert len(loaded.spikes) == 1
    assert loaded.spikes[0].spike_index == 8
    assert isinstance(loaded.long_plateau_debug, dict)
    assert loaded.hybrid_denoised_cache_key == traces["trace_a"].hybrid_denoised_cache_key
    assert loaded.gonzalez_cwt_debug_cache_key == traces["trace_a"].gonzalez_cwt_debug_cache_key


def test_analysis_package_and_exporter_preserve_burst_plateau_source(tmp_path: Path) -> None:
    traces = {"trace_burst": _state_with_burst_plateau("trace_burst")}
    package_dir = save_analysis_package(
        tmp_path / "analysis_trace_burst",
        source_excel="source.xlsx",
        detection_params=DetectionParams(startup_exclusion_ms=0.0),
        artifact_params=ArtifactParams(),
        trace_names=["trace_burst"],
        traces=traces,
    )
    package = load_analysis_package(package_dir)
    loaded = package.traces["trace_burst"]

    np.testing.assert_array_equal(loaded.burst_plateau_mask, traces["trace_burst"].burst_plateau_mask)
    np.testing.assert_allclose(loaded.burst_plateau_events, traces["trace_burst"].burst_plateau_events)
    assert isinstance(loaded.burst_plateau_debug, dict)
    assert loaded.plateau_signal_classification is not None
    assert loaded.plateau_signal_classification[0]["signal_type"] == "mixed"
    assert "denoise_residual" in available_channel_ids(loaded)

    out_dir = tmp_path / "export_burst"
    export_analysis_package(
        package,
        out_dir,
        ["trace_burst"],
        ExportOptions(
            tables=False,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            trace_images=False,
            spike_images=False,
            plateau_images=True,
        ),
    )

    plateau_csv = out_dir / "images" / "plateaus" / "trace_burst_plateaus.csv"
    signal_csv = out_dir / "images" / "plateaus" / "trace_burst_plateau_signal_classification.csv"
    burst_csv = out_dir / "images" / "plateaus" / "trace_burst_burst_plateau_detector_events.csv"
    debug_csv = out_dir / "images" / "plateaus" / "trace_burst_burst_plateau_candidate_debug.csv"
    assert plateau_csv.exists()
    assert signal_csv.exists()
    assert burst_csv.exists()
    assert debug_csv.exists()
    rows = pd.read_csv(plateau_csv)
    assert rows.loc[0, "detector_source"] == "burst"
    assert rows.loc[0, "signal_type"] == "mixed"
    assert rows.loc[0, "signal_confidence"] == 0.72
    assert rows.loc[0, "burst_peak_dominance_noise"] == 9.5
    assert (out_dir / "images" / "plateaus" / "trace_burst_plateau_zoom_001.png").exists()


def test_exporter_writes_selected_trace_tables_without_detection(tmp_path: Path) -> None:
    traces = {"trace_a": _state("trace_a"), "trace_b": _state("trace_b", offset=10.0)}
    package_dir = save_analysis_package(
        tmp_path / "analysis_two_traces",
        source_excel="source.xlsx",
        detection_params=DetectionParams(),
        artifact_params=ArtifactParams(),
        trace_names=["trace_a", "trace_b"],
        traces=traces,
    )
    package = load_analysis_package(package_dir)

    out_dir = tmp_path / "export_selected"
    written = export_analysis_package(
        package,
        out_dir,
        ["trace_a"],
        ExportOptions(
            tables=True,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            trace_images=False,
            spike_images=False,
            plateau_images=False,
            denoising_debug=False,
            long_plateau_debug=False,
        ),
    )

    assert (out_dir / "tables" / "trace_summary_stats.csv").exists()
    assert not (out_dir / "reload").exists()
    assert (out_dir / "export_manifest.json") in written
    trace_summary = pd.read_csv(out_dir / "tables" / "trace_summary_stats.csv")
    assert set(trace_summary["trace_id"]) == {"trace_a", "ALL"}


def test_export_manifest_records_render_options_defaults(tmp_path: Path) -> None:
    trace_name = "Recording 01 | ROI A"
    traces = {trace_name: _state(trace_name)}
    package_dir = save_analysis_package(
        tmp_path / "analysis_trace_a",
        source_excel="source.xlsx",
        detection_params=DetectionParams(),
        artifact_params=ArtifactParams(),
        trace_names=[trace_name],
        traces=traces,
    )
    package = load_analysis_package(package_dir)
    out_dir = tmp_path / "export_defaults"

    export_analysis_package(
        package,
        out_dir,
        [trace_name],
        ExportOptions(tables=True, reload_files=False, qc_report=False, spike_statistics=False),
    )

    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recordings"] == [
        {
            "recording_id": "Recording 01",
            "roi_count": 1,
            "roi_ids": ["ROI A"],
            "traces": [trace_name],
        }
    ]
    render_options = manifest["options"]["render_options"]
    assert render_options["shade_plateaus"] is True
    assert render_options["show_spike_markers"] is True
    assert render_options["label_spikes"] is False
    assert render_options["label_plateaus"] is False
    assert render_options["marker_style"] == "dot"
    assert render_options["trace_window"]["enabled"] is False
    assert render_options["selected_channels"] == ["raw", "corrected", "denoised"]


def test_custom_render_window_filters_image_events_and_writes_labels_option(tmp_path: Path) -> None:
    traces = {"trace_a": _state_with_two_events("trace_a")}
    package_dir = save_analysis_package(
        tmp_path / "analysis_trace_a",
        source_excel="source.xlsx",
        detection_params=DetectionParams(startup_exclusion_ms=0.0),
        artifact_params=ArtifactParams(),
        trace_names=["trace_a"],
        traces=traces,
    )
    package = load_analysis_package(package_dir)
    out_dir = tmp_path / "export_custom_images"

    export_analysis_package(
        package,
        out_dir,
        ["trace_a"],
        ExportOptions(
            tables=False,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            trace_images=True,
            spike_images=True,
            plateau_images=True,
            render_options=RenderOptions(
                shade_plateaus=False,
                show_spike_markers=False,
                label_spikes=True,
                label_plateaus=True,
                marker_style=MARKER_STYLE_BAR,
                trace_window=TraceWindow(enabled=True, start_time=0.0, end_time=14.0),
                spike_half_window_ms=6.0,
                plateau_pre_ms=25.0,
                plateau_post_ms=120.0,
                selected_channels=("corrected", "denoised", "highpass"),
            ),
        ),
    )

    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["options"]["render_options"]["shade_plateaus"] is False
    assert manifest["options"]["render_options"]["show_spike_markers"] is False
    assert manifest["options"]["render_options"]["label_spikes"] is True
    assert manifest["options"]["render_options"]["label_plateaus"] is True
    assert manifest["options"]["render_options"]["marker_style"] == MARKER_STYLE_BAR
    assert manifest["options"]["render_options"]["trace_window"]["enabled"] is True
    assert (out_dir / "images" / "traces" / "trace_a_selected_trace.png").exists()
    assert (out_dir / "images" / "spikes" / "trace_a_spike_event_0001.png").exists()
    assert not (out_dir / "images" / "spikes" / "trace_a_spike_event_0002.png").exists()
    assert (out_dir / "images" / "plateaus" / "trace_a_plateau_zoom_001.png").exists()
    assert not (out_dir / "images" / "plateaus" / "trace_a_plateau_zoom_001_no_shading.png").exists()
    assert not (out_dir / "images" / "plateaus" / "trace_a_plateau_zoom_002.png").exists()


def test_spike_export_renderers_do_not_draw_marker_dots(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    state = _state("trace_a")
    options = RenderOptions(show_spike_markers=False, label_spikes=True)
    data = exporter_rendering.prepare_trace_for_render(
        state,
        DetectionParams(startup_exclusion_ms=0.0),
        options,
        apply_trace_window=False,
    )

    def fail_scatter_item(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spike marker dots should not be drawn")

    monkeypatch.setattr(exporter_rendering.pg, "ScatterPlotItem", fail_scatter_item)

    trace_pix = exporter_rendering.render_trace_channels(data, options)
    spike_pix = exporter_rendering.render_spike_event(
        data,
        state.spikes[0],
        options,
        spike_label="S0001",
        default_half_window_ms=5.0,
    )

    assert not trace_pix.isNull()
    assert not spike_pix.isNull()
    app.processEvents()


def test_exporter_black_bar_marker_style_does_not_draw_marker_dots(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    state = _state("trace_a")
    options = RenderOptions(
        show_spike_markers=True,
        label_plateaus=False,
        marker_style=MARKER_STYLE_BAR,
    )
    data = exporter_rendering.prepare_trace_for_render(
        state,
        DetectionParams(startup_exclusion_ms=0.0),
        options,
        apply_trace_window=False,
    )

    def fail_scatter_item(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("black bar marker style should not draw marker dots")

    monkeypatch.setattr(exporter_rendering.pg, "ScatterPlotItem", fail_scatter_item)

    assert exporter_rendering.PLATEAU_MARKER_COLORS["long_plateau"] != "#111111"
    assert exporter_rendering.PLATEAU_MARKER_COLORS["short_plateau"] != "#111111"
    assert exporter_rendering.PLATEAU_MARKER_COLORS["burst_plateau"] != "#111111"
    assert exporter_rendering._show_bar_plateau_bars(options) is True
    trace_pix = exporter_rendering.render_trace_channels(data, options)
    spike_pix = exporter_rendering.render_spike_event(
        data,
        state.spikes[0],
        options,
        spike_label="S0001",
        default_half_window_ms=5.0,
    )
    plateau_pix = exporter_rendering.render_plateau_zoom(
        data,
        (5, 10),
        options,
        plateau_label="P0001",
    )
    panel_pix, _metadata = exporter_rendering.render_recording_aligned_trace_panels(
        "recording",
        [state],
        DetectionParams(startup_exclusion_ms=0.0),
        options,
    )

    assert not trace_pix.isNull()
    assert not spike_pix.isNull()
    assert not plateau_pix.isNull()
    assert not panel_pix.isNull()
    assert panel_pix.height() >= 300
    app.processEvents()


def test_exporter_bar_event_mode_uses_bars_without_plateau_shading(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    state = _state("trace_a")
    options = RenderOptions(
        shade_plateaus=True,
        show_spike_markers=True,
        marker_style=MARKER_STYLE_BAR,
    )
    data = exporter_rendering.prepare_trace_for_render(
        state,
        DetectionParams(startup_exclusion_ms=0.0),
        options,
        apply_trace_window=False,
    )

    def fail_shading(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bar event mode should not draw shaded plateau regions")

    monkeypatch.setattr(exporter_rendering, "_add_shaded_regions", fail_shading)

    assert exporter_rendering._uses_plateau_shading(options) is False
    assert exporter_rendering._show_bar_plateau_bars(options) is True
    assert exporter_rendering._show_bar_plateau_bars(RenderOptions(shade_plateaus=False, marker_style=MARKER_STYLE_BAR)) is True
    trace_pix = exporter_rendering.render_trace_channels(data, options)
    plateau_pix = exporter_rendering.render_plateau_zoom(
        data,
        (5, 10),
        options,
        plateau_label="P0001",
    )

    assert not trace_pix.isNull()
    assert not plateau_pix.isNull()
    app.processEvents()


def test_recording_event_raster_renders_spikes_and_colored_plateaus() -> None:
    app = QApplication.instance() or QApplication([])
    state_a = _state("RecA | ROI1")
    state_b = _state("RecA | ROI2", offset=2.0)
    state_b.short_plateau_mask = state_b.plateau_mask.copy()
    state_b.long_plateau_mask = np.zeros_like(state_b.plateau_mask)

    pix, metadata = exporter_rendering.render_recording_event_raster(
        "RecA",
        [state_a, state_b],
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(),
    )

    assert not pix.isNull()
    assert metadata["recording_id"] == "RecA"
    assert metadata["trace_count"] == 2
    assert metadata["spike_count"] == 2
    assert metadata["plateau_interval_count"] == 2
    assert set(metadata["plateau_sources"]) == {"long_plateau", "short_plateau"}
    app.processEvents()


def test_exporter_writes_recording_event_raster_plot(tmp_path: Path) -> None:
    trace_names = ["RecA | ROI1", "RecA | ROI2"]
    traces = {
        "RecA | ROI1": _state("RecA | ROI1"),
        "RecA | ROI2": _state("RecA | ROI2", offset=2.0),
    }
    package_dir = save_analysis_package(
        tmp_path / "analysis_raster",
        source_excel="source.xlsx",
        detection_params=DetectionParams(startup_exclusion_ms=0.0),
        artifact_params=ArtifactParams(),
        trace_names=trace_names,
        traces=traces,
    )
    package = load_analysis_package(package_dir)
    out_dir = tmp_path / "export_raster"

    export_analysis_package(
        package,
        out_dir,
        trace_names,
        ExportOptions(
            tables=False,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            recording_event_raster=True,
        ),
    )

    raster_path = out_dir / "images" / "recordings" / "RecA_event_raster.png"
    assert raster_path.exists()
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["options"]["recording_event_raster"] is True
    assert manifest["recording_event_rasters"][0]["image_path"] == "images/recordings/RecA_event_raster.png"
    assert manifest["recording_event_rasters"][0]["spike_count"] == 2


def test_exporter_black_bar_marker_height_is_capped() -> None:
    levels = exporter_rendering._bar_marker_levels(np.array([0.0, 1000.0], dtype=float))

    assert levels is not None
    assert np.isclose(
        levels["spike_top"] - levels["spike_bottom"],
        exporter_rendering.BAR_MARKER_MAX_SIGNAL_HEIGHT,
    )
    assert np.isclose(
        levels["plateau_spike_top"] - levels["plateau_spike_bottom"],
        exporter_rendering.BAR_MARKER_MAX_SIGNAL_HEIGHT,
    )


def test_aligned_panel_bar_height_scales_with_panel_y_span() -> None:
    small_span = 300.0
    large_span = 450.0
    small_levels = exporter_rendering._aligned_panel_marker_levels(450.0, small_span)
    large_levels = exporter_rendering._aligned_panel_marker_levels(600.0, large_span)

    small_height = small_levels["spike_top"] - small_levels["spike_bottom"]
    large_height = large_levels["spike_top"] - large_levels["spike_bottom"]
    small_visual_ratio = small_height / (small_span + exporter_rendering._aligned_panel_marker_lane(small_span))
    large_visual_ratio = large_height / (large_span + exporter_rendering._aligned_panel_marker_lane(large_span))

    assert large_height > small_height
    assert np.isclose(small_visual_ratio, large_visual_ratio)


def test_main_export_renderers_do_not_draw_marker_dots(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    state = _state("trace_a")

    def fail_scatter_item(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("spike marker dots should not be drawn")

    monkeypatch.setattr(gui_main.pg, "ScatterPlotItem", fail_scatter_item)

    trace_pix = gui_main._render_whole_trace_png(
        state.trace_name,
        state.time,
        state.corrected,
        "corrected trace",
        spikes=state.spikes,
        show_spike_markers=False,
    )
    spike_pix = gui_main._render_spike_png(
        state.time,
        state.corrected,
        state.corrected * 0.9,
        state.baseline,
        None,
        state.spikes[0],
        show_spike_markers=False,
    )
    aligned_pix = gui_main._render_aligned_spikes_png(
        state.time,
        state.corrected,
        state.baseline,
        None,
        state.spikes,
        show_spike_markers=False,
    )

    assert not trace_pix.isNull()
    assert not spike_pix.isNull()
    assert not aligned_pix.isNull()
    app.processEvents()


def test_exporter_window_refreshes_preview_for_loaded_package(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    traces = {"trace_a": _state_with_two_events("trace_a")}
    package_dir = save_analysis_package(
        tmp_path / "analysis_trace_a",
        source_excel="source.xlsx",
        detection_params=DetectionParams(startup_exclusion_ms=0.0),
        artifact_params=ArtifactParams(),
        trace_names=["trace_a"],
        traces=traces,
    )

    window = ExporterWindow()
    window.load_package(package_dir)
    window.chk_manual_window.setChecked(True)
    window.chk_shade_plateaus.setChecked(False)
    window.chk_spike_markers.setChecked(False)
    window.chk_label_spikes.setChecked(False)
    window.chk_label_plateaus.setChecked(False)
    window.window_start.setValue(0.0)
    window.window_end.setValue(14.0)
    options = window._render_options_from_ui()
    assert options.shade_plateaus is False
    assert options.show_spike_markers is False
    assert options.label_spikes is False
    assert options.label_plateaus is False
    assert options.marker_style == "dot"
    window.marker_style_combo.setCurrentIndex(window.marker_style_combo.findData(MARKER_STYLE_BAR))
    assert window._render_options_from_ui().marker_style == MARKER_STYLE_BAR
    window.preview_type_combo.setCurrentIndex(window.preview_type_combo.findData("trace_view"))
    window.refresh_preview()

    pixmap = window.preview_image.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    app.processEvents()
