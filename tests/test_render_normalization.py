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
from detector.core.models import ArtifactParams, DetectionParams, TraceCuration  # noqa: E402
from detector.core.render_normalization import (  # noqa: E402
    NORMALIZATION_NATIVE,
    NORMALIZATION_PERCENT_DFF,
    normalize_trace_values,
    trace_normalization_row,
)
from detector.exporter.render_options import RenderOptions, TraceWindow  # noqa: E402
from detector.exporter.rendering import (  # noqa: E402
    render_recording_aligned_trace_panels,
    render_recording_aligned_traces,
    robust_percent_scale,
)
from detector.exporter.service import ExportOptions, export_analysis_package  # noqa: E402


def _trace(name: str, corrected: np.ndarray, *, f0: float = 100.0) -> TraceCuration:
    corrected = np.asarray(corrected, dtype=float)
    time = np.linspace(0.0, 5.0, corrected.size)
    baseline = np.full(corrected.size, float(f0), dtype=float)
    return TraceCuration(
        trace_name=name,
        raw=baseline + corrected,
        corrected_source=corrected.copy(),
        corrected=corrected.copy(),
        time=time,
        sampling_rate_hz=100.0,
        correction_baseline=baseline,
        baseline=np.zeros_like(corrected),
        spikes=[],
    )


def test_percent_dff_uses_correction_baseline_and_does_not_mutate() -> None:
    state = TraceCuration(
        trace_name="RecA | ROI1",
        raw=np.array([101.0, 104.0, 190.0]),
        corrected_source=np.array([1.0, 4.0, -10.0]),
        corrected=np.array([1.0, 4.0, -10.0]),
        time=np.arange(3, dtype=float),
        correction_baseline=np.array([100.0, 200.0, 50.0]),
    )
    before = state.corrected.copy()

    normalized = normalize_trace_values(state.corrected, state, NORMALIZATION_PERCENT_DFF)

    np.testing.assert_allclose(normalized, np.array([1.0, 2.0, -20.0]))
    np.testing.assert_allclose(state.corrected, before)
    row = trace_normalization_row(state.trace_name, state)
    assert row["recording_id"] == "RecA"
    assert row["roi_id"] == "ROI1"
    assert row["f0_source"] == "correction_baseline"
    assert row["unit"] == "% ΔF/F0"


def test_percent_dff_falls_back_to_raw_median_for_older_traces() -> None:
    state = TraceCuration(
        trace_name="legacy_trace",
        raw=np.array([10.0, 20.0, 30.0]),
        corrected_source=np.array([1.0, 2.0, 3.0]),
        corrected=np.array([1.0, 2.0, 3.0]),
        time=np.arange(3, dtype=float),
        correction_baseline=np.array([0.0, 0.0]),
    )

    normalized = normalize_trace_values(state.corrected, state, NORMALIZATION_PERCENT_DFF)

    np.testing.assert_allclose(normalized, np.array([5.0, 10.0, 15.0]))
    assert trace_normalization_row(state.trace_name, state)["f0_source"] == "raw"


def test_saved_package_includes_render_normalization_metadata(tmp_path: Path) -> None:
    trace_name = "Recording 01 | ROI A"
    traces = {trace_name: _trace(trace_name, np.array([0.0, 1.0, -1.0]))}

    package_dir = save_analysis_package(
        tmp_path / "analysis",
        source_excel="source.xlsx",
        detection_params=DetectionParams(),
        artifact_params=ArtifactParams(),
        trace_names=[trace_name],
        traces=traces,
    )

    manifest = json.loads((package_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_normalization"]["version"] == 1
    assert manifest["render_normalization"]["mode"] == "percent_dff"
    assert manifest["render_normalization"]["traces"][0]["recording_id"] == "Recording 01"
    metadata = pd.read_csv(package_dir / "tables" / "recording_roi_render_metadata.csv")
    assert set(["recording_id", "roi_id", "f0_source", "f0_median", "unit"]).issubset(metadata.columns)


def test_saved_package_honors_requested_render_normalization_mode(tmp_path: Path) -> None:
    trace_name = "Recording 01 | ROI A"
    traces = {trace_name: _trace(trace_name, np.array([0.0, 1.0, -1.0]))}

    package_dir = save_analysis_package(
        tmp_path / "analysis_native",
        source_excel="source.xlsx",
        detection_params=DetectionParams(),
        artifact_params=ArtifactParams(),
        trace_names=[trace_name],
        traces=traces,
        render_normalization_mode=NORMALIZATION_NATIVE,
    )

    manifest = json.loads((package_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["display_normalization_mode"] == NORMALIZATION_NATIVE
    assert manifest["render_normalization"]["mode"] == NORMALIZATION_NATIVE
    assert manifest["render_normalization"]["unit"] == "native"
    metadata = pd.read_csv(package_dir / "tables" / "recording_roi_render_metadata.csv")
    assert metadata.loc[0, "mode"] == NORMALIZATION_NATIVE
    assert metadata.loc[0, "unit"] == "native"


def test_robust_scale_ignores_outlier_and_labels_negative_polarity() -> None:
    scale = robust_percent_scale(np.concatenate([np.linspace(-80.0, 10.0, 1000), np.array([10000.0])]))

    assert scale["value"] < 100.0
    assert str(scale["label"]).startswith("−")
    assert scale["polarity"] == "negative"


def test_recording_aligned_trace_renderer_respects_manual_window() -> None:
    app = QApplication.instance() or QApplication([])
    corrected = np.sin(np.linspace(0.0, np.pi * 2.0, 501)) * 10.0
    states = [
        _trace("RecA | ROI1", corrected),
        _trace("RecA | ROI2", corrected * -3.0),
    ]
    full_pix, full_meta = render_recording_aligned_traces(
        "RecA",
        states,
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(),
    )
    window_pix, window_meta = render_recording_aligned_traces(
        "RecA",
        states,
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(trace_window=TraceWindow(enabled=True, start_time=0.0, end_time=2.0)),
    )

    assert not full_pix.isNull()
    assert not window_pix.isNull()
    assert full_meta["horizontal_label"] != window_meta["horizontal_label"]
    assert window_meta["horizontal_label"] == "500 ms"
    assert str(window_meta["vertical_label"]).startswith("−")
    app.processEvents()


def test_recording_aligned_native_mode_uses_native_scale_label() -> None:
    app = QApplication.instance() or QApplication([])
    states = [_trace("RecA | ROI1", np.array([0.0, 5.0, 10.0, 5.0, 0.0]))]

    pix, metadata = render_recording_aligned_traces(
        "RecA",
        states,
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(normalization_mode=NORMALIZATION_NATIVE),
    )

    assert not pix.isNull()
    assert metadata["unit"] == "native"
    assert str(metadata["vertical_label"]).endswith(" native")
    assert metadata["vertical_scale_value"] == metadata["vertical_scale_percent"]
    app.processEvents()


def test_recording_aligned_trace_panels_keep_true_denoised_percent_traces() -> None:
    app = QApplication.instance() or QApplication([])
    state_a = _trace("RecA | ROI1", np.array([0.0, 10.0, 45.0, -20.0, 0.0]))
    state_b = _trace("RecA | ROI2", np.array([0.0, -8.0, 5.0, 32.0, 0.0]))
    state_a.hybrid_denoised = state_a.corrected * 0.5
    state_b.hybrid_denoised = state_b.corrected * 0.25

    pix, metadata = render_recording_aligned_trace_panels(
        "RecA",
        [state_a, state_b],
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(aligned_panel_channel="denoised"),
    )

    assert not pix.isNull()
    assert metadata["source_channel"] == "denoised"
    assert metadata["unit"] == "dF/F0 (%)"
    assert metadata["share_y"] is False
    assert metadata["y_min"] < 0.0
    assert metadata["y_max"] > 0.0
    assert metadata["trace_count"] == 2
    assert not any(str(key).startswith("panel_scaling") for key in metadata)
    for row in metadata["trace_normalization"]:
        assert not any(str(key).startswith("panel_scaling") for key in row)
    app.processEvents()


def test_recording_aligned_trace_panels_can_render_pure_raw_native_signal() -> None:
    app = QApplication.instance() or QApplication([])
    state = _trace("RecA | ROI1", np.array([0.0, 10.0, 45.0, -20.0, 0.0]))
    state.hybrid_denoised = state.corrected * 0.5

    pix, metadata = render_recording_aligned_trace_panels(
        "RecA",
        [state],
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(normalization_mode=NORMALIZATION_NATIVE, aligned_panel_channel="raw"),
    )

    assert not pix.isNull()
    assert metadata["normalization_mode"] == NORMALIZATION_NATIVE
    assert metadata["source_channel"] == "raw"
    assert metadata["unit"] == "native"
    assert metadata["plot_y_label"] == "Raw signal"
    assert metadata["y_min"] > 0.0
    assert metadata["trace_normalization"][0]["source_label"] == "Raw"
    assert metadata["trace_normalization"][0]["panel_y_min"] > 0.0
    app.processEvents()


def test_recording_aligned_trace_panels_default_to_independent_y_without_data_scaling() -> None:
    app = QApplication.instance() or QApplication([])
    state_a = _trace("RecA | ROI1", np.array([0.0, 2.0, 4.0, 2.0, 0.0]))
    state_b = _trace("RecA | ROI2", np.array([0.0, 20.0, 80.0, 20.0, 0.0]))
    state_a.hybrid_denoised = state_a.corrected.copy()
    state_b.hybrid_denoised = state_b.corrected.copy()

    _pix, metadata = render_recording_aligned_trace_panels(
        "RecA",
        [state_a, state_b],
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(aligned_panel_channel="denoised"),
    )

    rows = {row["roi_id"]: row for row in metadata["trace_normalization"]}
    assert metadata["unit"] == "dF/F0 (%)"
    assert metadata["share_y"] is False
    assert metadata["y_limit_min_span"] >= 45.0
    assert not any(str(key).startswith("panel_scaling") for key in metadata)
    for row in rows.values():
        assert not any(str(key).startswith("panel_scaling") for key in row)
    assert rows["ROI1"]["panel_y_max"] - rows["ROI1"]["panel_y_min"] >= 45.0
    assert rows["ROI2"]["panel_y_max"] > rows["ROI1"]["panel_y_max"]
    app.processEvents()


def test_recording_aligned_trace_panels_preserve_event_y_limits_without_data_scaling() -> None:
    app = QApplication.instance() or QApplication([])
    baseline_activity = np.sin(np.linspace(0.0, np.pi * 8.0, 1000)) * 3.0
    plateau_trace = baseline_activity.copy()
    plateau_trace[350:520] += 85.0
    state_a = _trace("RecA | ROI1", baseline_activity)
    state_b = _trace("RecA | ROI2", plateau_trace)
    state_b.plateau_mask = np.zeros(plateau_trace.size, dtype=bool)
    state_b.plateau_mask[350:520] = True
    state_a.hybrid_denoised = state_a.corrected.copy()
    state_b.hybrid_denoised = state_b.corrected.copy()

    _pix, metadata = render_recording_aligned_trace_panels(
        "RecA",
        [state_a, state_b],
        DetectionParams(startup_exclusion_ms=0.0),
        RenderOptions(aligned_panel_channel="denoised"),
    )

    rows = {row["roi_id"]: row for row in metadata["trace_normalization"]}
    assert rows["ROI2"]["panel_y_max"] >= 85.0
    assert rows["ROI2"]["panel_y_max"] > rows["ROI1"]["panel_y_max"]
    for row in rows.values():
        assert not any(str(key).startswith("panel_scaling") for key in row)
    app.processEvents()


def test_exporter_writes_one_recording_plot_per_selected_recording(tmp_path: Path) -> None:
    trace_names = ["RecA | ROI1", "RecA | ROI2", "RecB | ROI1"]
    traces = {
        "RecA | ROI1": _trace("RecA | ROI1", np.array([0.0, -10.0, -20.0, -10.0, 0.0])),
        "RecA | ROI2": _trace("RecA | ROI2", np.array([0.0, -5.0, -12.0, -5.0, 0.0])),
        "RecB | ROI1": _trace("RecB | ROI1", np.array([0.0, 5.0, 10.0, 5.0, 0.0])),
    }
    package_dir = save_analysis_package(
        tmp_path / "analysis",
        source_excel="source.xlsx",
        detection_params=DetectionParams(startup_exclusion_ms=0.0),
        artifact_params=ArtifactParams(),
        trace_names=trace_names,
        traces=traces,
    )
    package = load_analysis_package(package_dir)

    out_dir = tmp_path / "export"
    export_analysis_package(
        package,
        out_dir,
        trace_names,
        ExportOptions(
            tables=False,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            recording_aligned_traces=True,
        ),
    )

    assert (out_dir / "images" / "recordings" / "RecA_aligned_traces_percent_dff.png").exists()
    assert (out_dir / "images" / "recordings" / "RecB_aligned_traces_percent_dff.png").exists()
    scales = pd.read_csv(out_dir / "tables" / "recording_aligned_trace_scales.csv")
    assert set(scales["recording_id"]) == {"RecA", "RecB"}
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["recording_aligned_trace_scales"]) == 2

    out_dir_single = tmp_path / "export_single_roi"
    export_analysis_package(
        package,
        out_dir_single,
        ["RecA | ROI2"],
        ExportOptions(
            tables=False,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            recording_aligned_traces=True,
        ),
    )
    assert (out_dir_single / "images" / "recordings" / "RecA_aligned_traces_percent_dff.png").exists()
    single_manifest = json.loads((out_dir_single / "export_manifest.json").read_text(encoding="utf-8"))
    assert single_manifest["recording_aligned_trace_scales"][0]["trace_count"] == 1


def test_exporter_writes_recording_aligned_trace_panel_plot_and_csv(tmp_path: Path) -> None:
    trace_names = ["RecA | ROI1", "RecA | ROI2"]
    traces = {
        "RecA | ROI1": _trace("RecA | ROI1", np.array([0.0, 10.0, 45.0, -20.0, 0.0])),
        "RecA | ROI2": _trace("RecA | ROI2", np.array([0.0, -8.0, 5.0, 32.0, 0.0])),
    }
    for state in traces.values():
        state.hybrid_denoised = np.asarray(state.corrected, dtype=float) * 0.5
    package_dir = save_analysis_package(
        tmp_path / "analysis_panels",
        source_excel="source.xlsx",
        detection_params=DetectionParams(startup_exclusion_ms=0.0),
        artifact_params=ArtifactParams(),
        trace_names=trace_names,
        traces=traces,
    )
    package = load_analysis_package(package_dir)

    out_dir = tmp_path / "export_panels"
    export_analysis_package(
        package,
        out_dir,
        trace_names,
        ExportOptions(
            tables=False,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            recording_aligned_trace_panels=True,
            render_options=RenderOptions(aligned_panel_channel="denoised"),
        ),
    )

    assert (out_dir / "images" / "recordings" / "RecA_aligned_trace_panels_percent_dff.png").exists()
    panel_csv = out_dir / "tables" / "RecA_aligned_trace_panels_percent_dff.csv"
    assert panel_csv.exists()
    panel_table = pd.read_csv(panel_csv)
    assert list(panel_table.columns) == ["time", "ROI1", "ROI2"]
    np.testing.assert_allclose(panel_table["ROI1"].to_numpy(), np.array([0.0, 5.0, 22.5, -10.0, 0.0]))
    scales = pd.read_csv(out_dir / "tables" / "recording_aligned_trace_panel_scales.csv")
    assert scales.loc[0, "source_channel"] == "denoised"
    assert scales.loc[0, "unit"] == "dF/F0 (%)"
    assert "panel_scaling_mode" not in scales.columns
    assert str(scales.loc[0, "share_y"]).lower() == "false"
    assert scales.loc[0, "y_min"] < 0.0
    assert scales.loc[0, "y_max"] > 0.0
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recording_aligned_trace_panel_scales"][0]["trace_count"] == 2
    assert "panel_scaling_mode" not in manifest["recording_aligned_trace_panel_scales"][0]
    assert manifest["recording_aligned_trace_panel_scales"][0]["share_y"] is False
    for row in manifest["recording_aligned_trace_panel_scales"][0]["trace_normalization"]:
        assert not any(str(key).startswith("panel_scaling") for key in row)


def test_exporter_writes_pure_raw_recording_aligned_trace_panel_plot_and_csv(tmp_path: Path) -> None:
    trace_names = ["RecA | ROI1", "RecA | ROI2"]
    traces = {
        "RecA | ROI1": _trace("RecA | ROI1", np.array([0.0, 10.0, 45.0, -20.0, 0.0])),
        "RecA | ROI2": _trace("RecA | ROI2", np.array([0.0, -8.0, 5.0, 32.0, 0.0])),
    }
    for state in traces.values():
        state.hybrid_denoised = np.asarray(state.corrected, dtype=float) * 0.5
    package_dir = save_analysis_package(
        tmp_path / "analysis_raw_panels",
        source_excel="source.xlsx",
        detection_params=DetectionParams(startup_exclusion_ms=0.0),
        artifact_params=ArtifactParams(),
        trace_names=trace_names,
        traces=traces,
    )
    package = load_analysis_package(package_dir)

    out_dir = tmp_path / "export_raw_panels"
    export_analysis_package(
        package,
        out_dir,
        trace_names,
        ExportOptions(
            tables=False,
            reload_files=False,
            qc_report=False,
            spike_statistics=False,
            recording_aligned_trace_panels_raw=True,
            render_options=RenderOptions(aligned_panel_channel="denoised"),
        ),
    )

    assert (out_dir / "images" / "recordings" / "RecA_aligned_trace_panels_raw.png").exists()
    panel_csv = out_dir / "tables" / "RecA_aligned_trace_panels_raw.csv"
    assert panel_csv.exists()
    panel_table = pd.read_csv(panel_csv)
    assert list(panel_table.columns) == ["time", "ROI1", "ROI2"]
    np.testing.assert_allclose(panel_table["ROI1"].to_numpy(), np.array([100.0, 110.0, 145.0, 80.0, 100.0]))

    scales = pd.read_csv(out_dir / "tables" / "recording_aligned_trace_panel_scales.csv")
    assert scales.loc[0, "normalization_mode"] == NORMALIZATION_NATIVE
    assert scales.loc[0, "source_channel"] == "raw"
    assert scales.loc[0, "unit"] == "native"
    assert scales.loc[0, "plot_y_label"] == "Raw signal"
    assert scales.loc[0, "y_min"] > 0.0
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["options"]["recording_aligned_trace_panels_raw"] is True
    assert manifest["recording_aligned_trace_panel_scales"][0]["source_channel"] == "raw"
