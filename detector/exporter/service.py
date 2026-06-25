from __future__ import annotations

from dataclasses import dataclass, field, replace
import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pyqtgraph as pg

from detector.analysis.package import AnalysisPackage
from detector.core.models import TraceCuration
from detector.core.render_normalization import NORMALIZATION_NATIVE
from detector.core.trace_identity import recording_groups, split_trace_identity

from .render_options import RenderOptions
from .rendering import (
    _recording_aligned_panel_entries,
    _uses_plateau_shading,
    filtered_spikes,
    plateau_regions,
    prepare_trace_for_render,
    render_full_trace_plateaus,
    render_plateau_zoom,
    render_recording_event_raster,
    render_recording_aligned_trace_panels,
    render_recording_aligned_traces,
    render_raw_processed_comparison,
    render_spike_event,
    render_superimposed_plateaus,
    render_superimposed_spikes,
    render_trace_channels,
)


@dataclass
class ExportOptions:
    tables: bool = True
    reload_files: bool = True
    qc_report: bool = True
    spike_statistics: bool = True
    trace_images: bool = False
    recording_event_raster: bool = False
    recording_aligned_traces: bool = False
    recording_aligned_trace_panels: bool = False
    recording_aligned_trace_panels_raw: bool = False
    spike_images: bool = False
    plateau_images: bool = False
    denoising_debug: bool = False
    long_plateau_debug: bool = False
    render_options: RenderOptions = field(default_factory=RenderOptions)

    def has_any_output(self) -> bool:
        return any(
            [
                self.tables,
                self.reload_files,
                self.qc_report,
                self.spike_statistics,
                self.trace_images,
                self.recording_event_raster,
                self.recording_aligned_traces,
                self.recording_aligned_trace_panels,
                self.recording_aligned_trace_panels_raw,
                self.spike_images,
                self.plateau_images,
                self.denoising_debug,
                self.long_plateau_debug,
            ]
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "tables": bool(self.tables),
            "reload_files": bool(self.reload_files),
            "qc_report": bool(self.qc_report),
            "spike_statistics": bool(self.spike_statistics),
            "trace_images": bool(self.trace_images),
            "recording_event_raster": bool(self.recording_event_raster),
            "recording_aligned_traces": bool(self.recording_aligned_traces),
            "recording_aligned_trace_panels": bool(self.recording_aligned_trace_panels),
            "recording_aligned_trace_panels_raw": bool(self.recording_aligned_trace_panels_raw),
            "spike_images": bool(self.spike_images),
            "plateau_images": bool(self.plateau_images),
            "denoising_debug": bool(self.denoising_debug),
            "long_plateau_debug": bool(self.long_plateau_debug),
            "render_options": self.render_options.to_dict(),
        }


def default_export_dir(result_root: Path, package: AnalysisPackage) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(result_root) / f"export_from_{package.root.name}_{timestamp}"


def export_analysis_package(
    package: AnalysisPackage,
    out_dir: str | Path,
    trace_names: Iterable[str],
    options: ExportOptions,
) -> List[Path]:
    from detector.core import gui_main

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected_names = [str(name) for name in trace_names if str(name) in package.traces]
    if not selected_names:
        raise ValueError("No selected traces are available in the analysis package")

    written: List[Path] = []
    export_tables: Dict[str, pd.DataFrame] | None = None
    if options.tables or options.reload_files or options.qc_report or options.spike_statistics:
        export_tables = gui_main._build_export_dataframes(selected_names, package.traces)

    if options.tables and export_tables is not None:
        written.extend(_write_tables(root / "tables", export_tables))
    if options.reload_files and export_tables is not None:
        written.extend(_write_reload_files(root / "reload", export_tables))
    if options.qc_report and export_tables is not None:
        written.extend(_write_qc_report(root / "reports", export_tables))
    if options.spike_statistics and export_tables is not None:
        written.extend(_write_spike_statistics(root / "reports", export_tables))

    recording_scale_metadata: List[Dict[str, object]] = []
    recording_panel_scale_metadata: List[Dict[str, object]] = []
    recording_raster_metadata: List[Dict[str, object]] = []

    if (
        options.trace_images
        or options.recording_event_raster
        or options.recording_aligned_traces
        or options.recording_aligned_trace_panels
        or options.recording_aligned_trace_panels_raw
        or options.spike_images
        or options.plateau_images
    ):
        pg.setConfigOptions(antialias=True)

    if options.trace_images:
        traces_dir = root / "images" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        for trace_name in selected_names:
            written.extend(
                _write_trace_images(
                    traces_dir,
                    package.traces[trace_name],
                    package.detection_params,
                    options.render_options,
                )
            )

    if options.recording_event_raster:
        recordings_dir = root / "images" / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        for recording_id, names in recording_groups(selected_names).items():
            states = [package.traces[name] for name in names if name in package.traces]
            pix, metadata = render_recording_event_raster(
                recording_id,
                states,
                package.detection_params,
                options.render_options,
            )
            if pix.isNull():
                continue
            path = recordings_dir / f"{_safe_name(recording_id)}_event_raster.png"
            if pix.save(str(path)):
                written.append(path)
                row = dict(metadata)
                row["image_path"] = str(path.relative_to(root))
                recording_raster_metadata.append(row)

    if options.recording_aligned_traces:
        recordings_dir = root / "images" / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        for recording_id, names in recording_groups(selected_names).items():
            states = [package.traces[name] for name in names if name in package.traces]
            pix, metadata = render_recording_aligned_traces(
                recording_id,
                states,
                package.detection_params,
                options.render_options,
            )
            if pix.isNull():
                continue
            path = recordings_dir / f"{_safe_name(recording_id)}_aligned_traces_percent_dff.png"
            if pix.save(str(path)):
                written.append(path)
                row = dict(metadata)
                row["image_path"] = str(path.relative_to(root))
                recording_scale_metadata.append(row)
        if recording_scale_metadata:
            scales_dir = root / "tables"
            scales_dir.mkdir(parents=True, exist_ok=True)
            scale_path = scales_dir / "recording_aligned_trace_scales.csv"
            pd.DataFrame([_recording_scale_csv_row(row) for row in recording_scale_metadata]).to_csv(
                scale_path,
                index=False,
            )
            written.append(scale_path)

    panel_variants: List[tuple[str, RenderOptions]] = []
    if options.recording_aligned_trace_panels:
        panel_variants.append(("percent_dff", options.render_options))
    if options.recording_aligned_trace_panels_raw:
        panel_variants.append(("raw", _raw_aligned_panel_render_options(options.render_options)))

    if panel_variants:
        recordings_dir = root / "images" / "recordings"
        tables_dir = root / "tables"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        for recording_id, names in recording_groups(selected_names).items():
            states = [package.traces[name] for name in names if name in package.traces]
            safe_recording = _safe_name(recording_id)
            for suffix, panel_options in panel_variants:
                pix, metadata = render_recording_aligned_trace_panels(
                    recording_id,
                    states,
                    package.detection_params,
                    panel_options,
                )
                if pix.isNull():
                    continue
                path = recordings_dir / f"{safe_recording}_aligned_trace_panels_{suffix}.png"
                if pix.save(str(path)):
                    written.append(path)
                    row = dict(metadata)
                    row["image_path"] = str(path.relative_to(root))
                    recording_panel_scale_metadata.append(row)

                table = _recording_aligned_panel_dataframe(
                    states,
                    package.detection_params,
                    panel_options,
                )
                if table is not None and not table.empty:
                    table_path = tables_dir / f"{safe_recording}_aligned_trace_panels_{suffix}.csv"
                    table.to_csv(table_path, index=False)
                    written.append(table_path)

        if recording_panel_scale_metadata:
            scale_path = tables_dir / "recording_aligned_trace_panel_scales.csv"
            pd.DataFrame([_recording_panel_scale_csv_row(row) for row in recording_panel_scale_metadata]).to_csv(
                scale_path,
                index=False,
            )
            written.append(scale_path)

    if options.spike_images:
        spikes_dir = root / "images" / "spikes"
        spikes_dir.mkdir(parents=True, exist_ok=True)
        for trace_name in selected_names:
            written.extend(
                _write_spike_images(
                    spikes_dir,
                    package.traces[trace_name],
                    package.detection_params,
                    options.render_options,
                )
            )

    if options.plateau_images:
        plateaus_dir = root / "images" / "plateaus"
        plateaus_dir.mkdir(parents=True, exist_ok=True)
        for trace_name in selected_names:
            written.extend(
                _write_plateau_images(
                    plateaus_dir,
                    package.traces[trace_name],
                    package.detection_params,
                    options.render_options,
                )
            )

    if options.denoising_debug:
        debug_root = root / "images" / "denoising_debug"
        debug_root.mkdir(parents=True, exist_ok=True)
        for trace_name in selected_names:
            state = package.traces[trace_name]
            if not isinstance(state.gonzalez_cwt_debug, dict):
                continue
            safe_name = _safe_name(trace_name)
            written.extend(
                gui_main._write_gonzalez_cwt_debug_export(
                    trace_name=trace_name,
                    time=state.time,
                    corrected=state.corrected,
                    debug=dict(state.gonzalez_cwt_debug),
                    out_dir=debug_root / safe_name,
                    startup_exclusion_ms=float(package.detection_params.startup_exclusion_ms),
                )
            )

    if options.long_plateau_debug:
        debug_root = root / "images" / "plateau_debug" / "long"
        debug_root.mkdir(parents=True, exist_ok=True)
        for trace_name in selected_names:
            state = package.traces[trace_name]
            if not isinstance(state.long_plateau_debug, dict):
                continue
            safe_name = _safe_name(trace_name)
            written.extend(
                gui_main._write_long_plateau_debug_export(
                    trace_name=trace_name,
                    time=state.time,
                    corrected=state.corrected,
                    debug=dict(state.long_plateau_debug),
                    out_dir=debug_root / safe_name,
                    startup_exclusion_ms=float(package.detection_params.startup_exclusion_ms),
                )
            )

    manifest_path = _write_export_manifest(
        root,
        package,
        selected_names,
        options,
        written,
        recording_scale_metadata,
        recording_panel_scale_metadata,
        recording_raster_metadata,
    )
    written.append(manifest_path)
    return written


def _raw_aligned_panel_render_options(render_options: RenderOptions) -> RenderOptions:
    return replace(render_options, normalization_mode=NORMALIZATION_NATIVE, aligned_panel_channel="raw")


def _write_tables(out_dir: Path, export_tables: Dict[str, pd.DataFrame]) -> List[Path]:
    from detector.core import gui_main

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = gui_main.TRACE_SUMMARY_CONFIG
    table_paths = {
        "trace_summary": out_dir / str(cfg["csv_name"]),
        "trace_qc": out_dir / str(cfg["qc_csv_name"]),
        "plateau_events": out_dir / str(cfg["events_csv_name"]),
        "spike_events": out_dir / str(cfg["spike_events_csv_name"]),
        "spike_summary": out_dir / str(cfg["spike_summary_csv_name"]),
    }
    for key, path in table_paths.items():
        export_tables[key].to_csv(path, index=False)
    return list(table_paths.values())


def _write_reload_files(out_dir: Path, export_tables: Dict[str, pd.DataFrame]) -> List[Path]:
    from detector.core import gui_main

    out_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = out_dir / "corrected_traces.csv"
    spike_path = out_dir / str(gui_main.TRACE_SUMMARY_CONFIG["spike_events_csv_name"])
    export_tables["corrected_traces"].to_csv(corrected_path, index=False)
    export_tables["spike_events"].to_csv(spike_path, index=False)
    return [corrected_path, spike_path]


def _write_qc_report(out_dir: Path, export_tables: Dict[str, pd.DataFrame]) -> List[Path]:
    from detector.core import gui_main

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = gui_main.TRACE_SUMMARY_CONFIG
    png_path = out_dir / str(cfg["png_name"])
    pdf_path = out_dir / str(cfg["pdf_name"]) if bool(cfg.get("save_pdf", False)) else None
    gui_main._save_qc_summary_report(
        trace_df=export_tables["trace_qc"],
        events_df=export_tables["plateau_events"],
        png_path=png_path,
        pdf_path=pdf_path,
    )
    return [path for path in [png_path, pdf_path] if path is not None]


def _write_spike_statistics(out_dir: Path, export_tables: Dict[str, pd.DataFrame]) -> List[Path]:
    from detector.core import gui_main

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / str(gui_main.TRACE_SUMMARY_CONFIG["spike_png_name"])
    gui_main._save_spike_statistics_summary(
        spike_summary_df=export_tables["spike_summary"],
        spike_events_df=export_tables["spike_events"],
        png_path=path,
    )
    return [path]


def _write_trace_images(
    out_dir: Path,
    state: TraceCuration,
    detection_params: object,
    render_options: RenderOptions,
) -> List[Path]:
    safe_name = _safe_name(state.trace_name)
    data = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
    written: List[Path] = []

    selected_path = out_dir / f"{safe_name}_selected_trace.png"
    pix = render_trace_channels(data, render_options, title_suffix="selected trace")
    if not pix.isNull() and pix.save(str(selected_path)):
        written.append(selected_path)

    channel_filenames = {
        "raw": "raw_trace",
        "corrected_source": "corrected_source_trace",
        "corrected": "corrected_trace_plateaus",
        "baseline": "baseline_trace",
        "correction_baseline": "correction_baseline_trace",
        "highpass": "highpass_trace",
        "lowpass": "lowpass_trace",
        "denoised": "denoised_trace_spikes",
    }
    selected_channels = render_options.normalized_channels()
    for channel in selected_channels:
        if data.array(channel) is None:
            continue
        path = out_dir / f"{safe_name}_{channel_filenames.get(channel, channel + '_trace')}.png"
        pix = render_trace_channels(
            data,
            render_options,
            title_suffix=channel_filenames.get(channel, channel).replace("_", " "),
            channels=(channel,),
        )
        if not pix.isNull() and pix.save(str(path)):
            written.append(path)

    for processed_channel in ("corrected", "denoised"):
        if data.array("raw") is None or data.array(processed_channel) is None:
            continue
        if "raw" not in selected_channels or processed_channel not in selected_channels:
            continue
        path = out_dir / f"{safe_name}_raw_vs_{processed_channel}.png"
        pix = render_raw_processed_comparison(data, processed_channel)
        if not pix.isNull() and pix.save(str(path)):
            written.append(path)
    return written


def _write_spike_images(
    out_dir: Path,
    state: TraceCuration,
    detection_params: object,
    render_options: RenderOptions,
) -> List[Path]:
    from detector.core import gui_main

    safe_name = _safe_name(state.trace_name)
    data = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=False)
    spikes = filtered_spikes(data, render_options)
    half_window_ms = render_options.effective_spike_half_window_ms(float(detection_params.spike_zoom_half_window_ms))
    single_w, single_h, aligned_row_h, _ = gui_main._spike_export_dimensions(half_window_ms)
    pixmaps = []
    written: List[Path] = []

    for idx, spike in enumerate(spikes, 1):
        pix = render_spike_event(
            data,
            spike,
            render_options,
            spike_label=f"S{idx:04d}",
            default_half_window_ms=float(detection_params.spike_zoom_half_window_ms),
            width=single_w,
            height=single_h * 2 if data.array("denoised") is not None else single_h,
        )
        if pix.isNull():
            continue
        path = out_dir / f"{safe_name}_spike_event_{idx:04d}.png"
        if pix.save(str(path)):
            pixmaps.append(pix)
            written.append(path)

    superimposed = render_superimposed_spikes(
        data,
        spikes,
        render_options,
        default_half_window_ms=float(detection_params.spike_zoom_half_window_ms),
        width=single_w,
        height=single_h,
    )
    if not superimposed.isNull():
        path = out_dir / f"{safe_name}_superimposed.png"
        if superimposed.save(str(path)):
            written.append(path)

    aligned = gui_main._compose_stacked_spike_png(
        pixmaps,
        title=f"{state.trace_name} | aligned spikes (count={len(pixmaps)})",
        row_height=aligned_row_h,
    )
    if not aligned.isNull():
        path = out_dir / f"{safe_name}_aligned.png"
        if aligned.save(str(path)):
            written.append(path)
    return written


def _write_plateau_images(
    out_dir: Path,
    state: TraceCuration,
    detection_params: object,
    render_options: RenderOptions,
) -> List[Path]:
    from detector.core import gui_main

    safe_name = _safe_name(state.trace_name)
    cropped = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=True)
    full = prepare_trace_for_render(state, detection_params, render_options, apply_trace_window=False)
    plateau_mask = cropped.mask("plateau")
    long_mask = cropped.mask("long_plateau")
    short_mask = cropped.mask("short_plateau")
    burst_mask = cropped.mask("burst_plateau")
    selected_regions = plateau_regions(full, render_options)
    written: List[Path] = []

    if plateau_mask is not None and np.any(np.asarray(plateau_mask, dtype=bool)):
        pix = render_full_trace_plateaus(cropped, render_options)
        if not pix.isNull():
            path = out_dir / f"{safe_name}_full_trace_plateaus.png"
            if pix.save(str(path)):
                written.append(path)

        cropped_burst_events = gui_main._adjust_event_table_for_offset(
            state.burst_plateau_events,
            int(cropped.startup_idx + cropped.window_offset),
            int(cropped.time.size),
            8,
        )
        cropped_signal_rows = gui_main._adjust_plateau_signal_rows_for_offset(
            state.plateau_signal_classification,
            int(cropped.startup_idx + cropped.window_offset),
            int(cropped.time.size),
        )
        rows = gui_main._plateau_rows(
            state.trace_name,
            cropped.time,
            np.asarray(plateau_mask, dtype=bool),
            long_mask,
            short_mask,
            burst_mask,
            cropped_burst_events,
            cropped_signal_rows,
            sampling_rate_hz=gui_main._trace_sampling_rate_hz(state),
        )
        if rows:
            path = out_dir / f"{safe_name}_plateaus.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            written.append(path)
        if cropped_signal_rows:
            signal_path = out_dir / f"{safe_name}_plateau_signal_classification.csv"
            pd.DataFrame(cropped_signal_rows).to_csv(signal_path, index=False)
            written.append(signal_path)
        burst_events = np.asarray(
            cropped_burst_events,
            dtype=float,
        )
        if burst_events.ndim == 2 and burst_events.shape[0] > 0 and burst_events.shape[1] == 8:
            burst_path = out_dir / f"{safe_name}_burst_plateau_detector_events.csv"
            burst_df = pd.DataFrame(
                burst_events,
                columns=[
                    "onset_index",
                    "offset_index_exclusive",
                    "duration_ms",
                    "median_level",
                    "floor_support_fraction",
                    "peak_dominance_noise",
                    "iqr_over_noise",
                    "confidence",
                ],
            )
            burst_df.insert(0, "trace_name", state.trace_name)
            burst_df["onset_index"] = burst_df["onset_index"].astype(int)
            burst_df["offset_index_exclusive"] = burst_df["offset_index_exclusive"].astype(int)
            burst_df.to_csv(burst_path, index=False)
            written.append(burst_path)
        burst_debug_rows = []
        if isinstance(state.burst_plateau_debug, dict):
            raw_rows = state.burst_plateau_debug.get("burst_plateau_candidate_rows", [])
            if isinstance(raw_rows, list):
                burst_debug_rows = [row for row in raw_rows if isinstance(row, dict)]
        if burst_debug_rows:
            debug_path = out_dir / f"{safe_name}_burst_plateau_candidate_debug.csv"
            pd.DataFrame(burst_debug_rows).to_csv(debug_path, index=False)
            written.append(debug_path)

    if selected_regions:
        pix = render_superimposed_plateaus(full, selected_regions, render_options)
        if not pix.isNull():
            path = out_dir / f"{safe_name}_plateaus_superimposed_onset0.png"
            if pix.save(str(path)):
                written.append(path)

    full_long = full.mask("long_plateau")
    full_short = full.mask("short_plateau")
    full_burst = full.mask("burst_plateau")
    for idx, region in enumerate(selected_regions, 1):
        source = gui_main._plateau_source_for_region(region[0], region[1], full_long, full_short, full_burst)
        signal_type = str(
            gui_main._signal_metrics_for_region(
                state.plateau_signal_classification,
                int(region[0] + full.startup_idx + full.window_offset),
                int(region[1] + full.startup_idx + full.window_offset),
            ).get("signal_type", "unknown")
        )
        pix = render_plateau_zoom(
            full,
            region,
            render_options,
            plateau_label=f"P{idx:04d}",
            plateau_source=source,
            signal_type=signal_type,
        )
        if not pix.isNull():
            path = out_dir / f"{safe_name}_plateau_zoom_{idx:03d}.png"
            if pix.save(str(path)):
                written.append(path)
        if _uses_plateau_shading(render_options):
            unshaded_options = replace(render_options, shade_plateaus=False)
            pix = render_plateau_zoom(
                full,
                region,
                unshaded_options,
                plateau_label=f"P{idx:04d}",
                plateau_source=source,
                signal_type=signal_type,
            )
            if not pix.isNull():
                path = out_dir / f"{safe_name}_plateau_zoom_{idx:03d}_no_shading.png"
                if pix.save(str(path)):
                    written.append(path)

    written.extend(_write_rejected_plateau_candidates(out_dir, state, detection_params, render_options, full))
    return written


def _write_rejected_plateau_candidates(
    out_dir: Path,
    state: TraceCuration,
    detection_params: object,
    render_options: RenderOptions,
    full_data: object,
) -> List[Path]:
    from detector.core import gui_main

    safe_name = _safe_name(state.trace_name)
    startup_idx = gui_main._startup_exclusion_index(np.asarray(state.time, dtype=float), float(detection_params.startup_exclusion_ms))
    rejected_rows = gui_main._rejected_long_plateau_rows_for_export(state.long_plateau_debug, startup_idx, int(len(full_data.time)))
    if render_options.trace_window.enabled and full_data.time.size:
        start_time = (
            float(render_options.trace_window.start_time)
            if render_options.trace_window.start_time is not None
            else float(full_data.time[0])
        )
        end_time = (
            float(render_options.trace_window.end_time)
            if render_options.trace_window.end_time is not None
            else float(full_data.time[-1])
        )
        if end_time < start_time:
            start_time, end_time = end_time, start_time
        rejected_rows = [
            row
            for row in rejected_rows
            if 0 <= int(row.get("display_start_index", row.get("start_index", 0))) < full_data.time.size
            and start_time <= float(full_data.time[int(row.get("display_start_index", row.get("start_index", 0)))]) <= end_time
        ]

    written: List[Path] = []
    for row in rejected_rows[:50]:
        region_id = int(row.get("region_id", 0))
        start = int(row.get("display_start_index", row.get("start_index", 0)))
        end = int(row.get("display_end_index_exclusive", start + 1))
        pix = render_plateau_zoom(
            full_data,
            (start, end),
            render_options,
            plateau_label=f"R{region_id:03d}",
            plateau_source="rejected",
        )
        if not pix.isNull():
            path = out_dir / f"{safe_name}_failed_candidate_zoom_{region_id:03d}.png"
            if pix.save(str(path)):
                written.append(path)
    return written


def _recording_aligned_panel_dataframe(
    states: List[TraceCuration],
    detection_params: object,
    render_options: RenderOptions,
) -> pd.DataFrame | None:
    rows: Dict[str, np.ndarray] = {}
    time_values: np.ndarray | None = None
    entries, _metadata = _recording_aligned_panel_entries(states, detection_params, render_options)
    for state, data, values, _row_metadata in entries:
        n = min(data.time.size, values.size)
        if n <= 0:
            continue
        if time_values is None:
            time_values = np.asarray(data.time[:n], dtype=float)
        n = min(n, time_values.size)
        _recording_id, roi_id = split_trace_identity(state.trace_name)
        rows[str(roi_id)] = np.asarray(values[:n], dtype=float)
    if time_values is None or not rows:
        return None
    count = min([time_values.size, *(values.size for values in rows.values())])
    table = pd.DataFrame({"time": time_values[:count]})
    for roi_id, values in rows.items():
        table[roi_id] = values[:count]
    return table


def _recording_scale_csv_row(metadata: Dict[str, object]) -> Dict[str, object]:
    traces = metadata.get("traces", [])
    if isinstance(traces, list):
        trace_text = ";".join(str(value) for value in traces)
    else:
        trace_text = str(traces)
    return {
        "recording_id": metadata.get("recording_id", ""),
        "trace_count": metadata.get("trace_count", 0),
        "traces": trace_text,
        "image_path": metadata.get("image_path", ""),
        "normalization_mode": metadata.get("normalization_mode", ""),
        "unit": metadata.get("unit", ""),
        "time_start": metadata.get("time_start", float("nan")),
        "time_end": metadata.get("time_end", float("nan")),
        "visible_duration": metadata.get("visible_duration", float("nan")),
        "time_axis_is_ms": metadata.get("time_axis_is_ms", False),
        "horizontal_scale_seconds": metadata.get("horizontal_scale_seconds", float("nan")),
        "horizontal_scale_native_units": metadata.get("horizontal_scale_native_units", float("nan")),
        "horizontal_label": metadata.get("horizontal_label", ""),
        "vertical_scale_percent": metadata.get("vertical_scale_percent", float("nan")),
        "vertical_scale_value": metadata.get("vertical_scale_value", metadata.get("vertical_scale_percent", float("nan"))),
        "vertical_label": metadata.get("vertical_label", ""),
        "vertical_polarity": metadata.get("vertical_polarity", ""),
        "robust_p05": metadata.get("robust_p05", float("nan")),
        "robust_p95": metadata.get("robust_p95", float("nan")),
    }


def _recording_panel_scale_csv_row(metadata: Dict[str, object]) -> Dict[str, object]:
    traces = metadata.get("traces", [])
    if isinstance(traces, list):
        trace_text = ";".join(str(value) for value in traces)
    else:
        trace_text = str(traces)
    return {
        "recording_id": metadata.get("recording_id", ""),
        "trace_count": metadata.get("trace_count", 0),
        "traces": trace_text,
        "image_path": metadata.get("image_path", ""),
        "normalization_mode": metadata.get("normalization_mode", ""),
        "unit": metadata.get("unit", ""),
        "source_channel": metadata.get("source_channel", ""),
        "time_start": metadata.get("time_start", float("nan")),
        "time_end": metadata.get("time_end", float("nan")),
        "visible_duration": metadata.get("visible_duration", float("nan")),
        "time_axis_is_ms": metadata.get("time_axis_is_ms", False),
        "horizontal_scale_seconds": metadata.get("horizontal_scale_seconds", float("nan")),
        "horizontal_scale_native_units": metadata.get("horizontal_scale_native_units", float("nan")),
        "horizontal_label": metadata.get("horizontal_label", ""),
        "share_y": metadata.get("share_y", False),
        "plot_negative_indicator": metadata.get("plot_negative_indicator", False),
        "plot_y_label": metadata.get("plot_y_label", ""),
        "y_limit_low_percentile": metadata.get("y_limit_low_percentile", float("nan")),
        "y_limit_high_percentile": metadata.get("y_limit_high_percentile", float("nan")),
        "y_limit_padding": metadata.get("y_limit_padding", float("nan")),
        "y_limit_min_span": metadata.get("y_limit_min_span", float("nan")),
        "y_min": metadata.get("y_min", float("nan")),
        "y_max": metadata.get("y_max", float("nan")),
        "robust_p05": metadata.get("robust_p05", float("nan")),
        "robust_p95": metadata.get("robust_p95", float("nan")),
        "show_peak_markers": metadata.get("show_peak_markers", False),
        "peak_marker_count": metadata.get("peak_marker_count", 0),
    }


def _write_export_manifest(
    root: Path,
    package: AnalysisPackage,
    trace_names: List[str],
    options: ExportOptions,
    files: List[Path],
    recording_scale_metadata: List[Dict[str, object]] | None = None,
    recording_panel_scale_metadata: List[Dict[str, object]] | None = None,
    recording_raster_metadata: List[Dict[str, object]] | None = None,
) -> Path:
    grouped_recordings = recording_groups(trace_names)
    manifest = {
        "export_version": 3,
        "export_type": "analysis_package_selection",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_analysis_package": str(package.root),
        "source_excel": package.source_excel,
        "trace_count": len(trace_names),
        "traces": trace_names,
        "recordings": [
            {
                "recording_id": recording_id,
                "roi_count": len(names),
                "roi_ids": [split_trace_identity(name)[1] for name in names],
                "traces": [str(name) for name in names],
            }
            for recording_id, names in grouped_recordings.items()
        ],
        "recording_aligned_trace_scales": recording_scale_metadata or [],
        "recording_aligned_trace_panel_scales": recording_panel_scale_metadata or [],
        "recording_event_rasters": recording_raster_metadata or [],
        "options": options.to_dict(),
        "files": [str(path.relative_to(root)) for path in files if path.exists()],
    }
    path = root / "export_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _safe_name(value: object) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value))
