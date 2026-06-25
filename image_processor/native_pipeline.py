#!/usr/bin/env python3
"""Single-command backend for the native image processor workflow."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from .app import default_average_png_path, save_tiff_average_png
    from .comparison_signal_export import write_signal_comparison_workbook
    from .mask_signal_export import export_mask_signal_documents, extract_mask_signals, write_mask_signals_csv
    from .pipeline_hooks import load_hook, parse_hook_params, run_hook
    from .qc_images import save_motion_comparison_png, save_raw_average_png, save_roi_label_map_png
    from .roi_core import is_tiff_path, iter_tiff_inputs, metadata_path_for_tiff, split_tiff
    from .signal_export import default_export_output_dir, export_signal_documents, extract_roi_signals, write_signals_csv
except ImportError:  # pragma: no cover - supports direct script execution.
    from app import default_average_png_path, save_tiff_average_png
    from comparison_signal_export import write_signal_comparison_workbook
    from mask_signal_export import export_mask_signal_documents, extract_mask_signals, write_mask_signals_csv
    from pipeline_hooks import load_hook, parse_hook_params, run_hook
    from qc_images import save_motion_comparison_png, save_raw_average_png, save_roi_label_map_png
    from roi_core import is_tiff_path, iter_tiff_inputs, metadata_path_for_tiff, split_tiff
    from signal_export import default_export_output_dir, export_signal_documents, extract_roi_signals, write_signals_csv


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the native-viewer analysis pipeline for Femtonics TIFF stacks."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Input TIFF file(s), metadata file(s), or folder(s) containing TIFF/metadata pairs.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output root. Per-TIFF outputs are written under <output>/<tiff_stem>/.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Override metadata sampling rate in Hz for signal export.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip signal CSV export.")
    parser.add_argument("--no-average", action="store_true", help="Skip average-projection PNG export.")
    parser.add_argument("--no-signals", action="store_true", help="Skip signal workbook export.")
    parser.add_argument("--no-split", action="store_true", help="Skip ROI TIFF splitting.")
    parser.add_argument(
        "--mask-background",
        default="none",
        choices=["none", "parent_tile", "global"],
        help="Background subtraction for mask signal export. Default: none.",
    )
    parser.add_argument(
        "--motion-hook",
        default=None,
        help="Optional module:function or file.py:function hook run before exports for motion correction.",
    )
    parser.add_argument(
        "--motion-hook-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra keyword argument for the motion hook. Repeat for multiple values. "
            "VALUE is parsed as JSON when possible, otherwise as a string."
        ),
    )
    parser.add_argument(
        "--roi-hook",
        default=None,
        help="Optional module:function or file.py:function hook run before exports for ROI detection.",
    )
    parser.add_argument(
        "--roi-hook-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra keyword argument for the ROI hook. Repeat for multiple values. "
            "VALUE is parsed as JSON when possible, otherwise as a string."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON path.",
    )
    return parser.parse_args(argv)


def per_tiff_output_dir(tiff_path: Path, output_root: Path | None) -> Path | None:
    if output_root is None:
        return None
    return output_root / tiff_path.stem


def split_output_root(output_root: Path | None) -> Path | None:
    if output_root is None:
        return None
    return output_root / "split_rois"


def default_manifest_path(input_paths: list[Path], output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root / "native_pipeline_manifest.json"
    if len(input_paths) != 1:
        return Path.cwd() / "exports" / "native_pipeline_manifest.json"
    input_path = input_paths[0]
    if input_path.is_file():
        return input_path.parent / "exports" / f"{input_path.stem}_native_pipeline_manifest.json"
    return input_path / "exports" / "native_pipeline_manifest.json"


def path_text(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def existing_path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve()) if path.exists() else str(path)


class CsvSignalTable:
    """Small adapter so an existing CSV can be inserted into comparison workbooks."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            self.headers = next(reader)
            self.frame_count = sum(1 for _row in reader)
        self.roi_count = max(0, len(self.headers) - 2)

    def iter_signal_rows(self) -> Iterable[list[float | int | str]]:
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                converted: list[float | int | str] = []
                for index, value in enumerate(row):
                    if index == 0:
                        try:
                            converted.append(int(value))
                            continue
                        except ValueError:
                            pass
                    try:
                        converted.append(float(value))
                    except ValueError:
                        converted.append(value)
                yield converted


def first_path_value(values: dict[str, Any], keys: list[str]) -> Path | None:
    for key in keys:
        value = values.get(key)
        if value:
            return Path(value)
    return None


def resolve_returned_path(path: Path, *, fallback_base: Path) -> Path:
    resolved = path.expanduser()
    if resolved.is_absolute():
        return resolved
    if resolved.exists():
        return resolved.resolve()
    return (fallback_base / resolved).resolve()


def resolve_existing_returned_path(path: Path, fallback_bases: list[Path]) -> Path:
    resolved = path.expanduser()
    if resolved.is_absolute():
        return resolved
    if resolved.exists():
        return resolved.resolve()
    for base in fallback_bases:
        candidate = (base / resolved).resolve()
        if candidate.exists():
            return candidate
    return (fallback_bases[0] / resolved).resolve()


def normalize_hook_result_paths(hook_result: dict[str, Any], fallback_bases: list[Path]) -> None:
    path_keys = [
        "tiff_path",
        "output_tiff",
        "output_tiff_path",
        "corrected_tiff",
        "corrected_tiff_path",
        "downstream_tiff",
        "metadata_path",
        "output_metadata",
        "roi_metadata_path",
        "mask_path",
        "masks_path",
        "roi_mask_path",
        "output_masks",
        "segmentation_mask_path",
        "label_tiff",
        "mask_manifest_path",
        "manifest_path",
        "segmentation_manifest_path",
        "roi_manifest_path",
        "summary_mean_png",
        "summary_std_png",
        "summary_max_png",
        "summary_manifest_path",
        "qc_overlay_png",
        "segmentation_overlay_png",
        "qc_json",
        "shifts_csv",
        "tracked_trace_csv",
        "tracked_trace_xlsx",
        "tracked_manifest_path",
        "tracked_tracks_csv",
        "tracked_debug_tiffs",
        "tracked_debug_raw_tiffs",
        "tracked_debug_mask_tiffs",
        "initial_mask_tiff",
        "initial_mask_motion_qc_json",
        "initial_mask_motion_shifts_csv",
    ]

    def normalize_value(value: Any) -> Any:
        if isinstance(value, str) and value:
            resolved = resolve_existing_returned_path(Path(value), fallback_bases)
            return str(resolved) if resolved.exists() else value
        if isinstance(value, list):
            return [normalize_value(item) for item in value]
        return value

    for key in path_keys:
        if key in hook_result:
            hook_result[key] = normalize_value(hook_result[key])
    if "summary_paths" in hook_result:
        hook_result["summary_paths"] = normalize_value(hook_result["summary_paths"])


def resolve_hook_file(path: Path, *, base_dir: Path, description: str) -> Path:
    resolved = resolve_returned_path(path, fallback_base=base_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    return resolved


def copy_metadata_if_needed(source_metadata: Path, target_tiff: Path) -> Path | None:
    target_metadata = metadata_path_for_tiff(target_tiff)
    if target_metadata.exists():
        return target_metadata
    if source_metadata.exists():
        target_metadata.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_metadata, target_metadata)
        return target_metadata
    return None


def comparison_signal_output_dir(tiff_path: Path, export_dir: Path | None) -> Path:
    return default_export_output_dir(tiff_path, export_dir)


def comparison_csv_path(tiff_path: Path, export_dir: Path | None, variant: str) -> Path:
    return comparison_signal_output_dir(tiff_path, export_dir) / f"{tiff_path.stem}_{variant}_signals.csv"


def export_rectangular_comparison_csv(
    *,
    variant: str,
    source_tiff: Path,
    original_tiff: Path,
    export_dir: Path | None,
    sampling_rate_hz: float | None,
) -> dict[str, Any]:
    def progress(done: int, total: int) -> None:
        print(f"{source_tiff.name}: {variant} signal frames {done}/{total}")

    signal_table = extract_roi_signals(source_tiff, sampling_rate_hz=sampling_rate_hz, progress=progress)
    csv_path = comparison_csv_path(original_tiff, export_dir, variant)
    write_signals_csv(signal_table, csv_path)
    return {
        "ok": True,
        "skipped": False,
        "signal_source": "rectangular_roi",
        "csv": existing_path_text(csv_path),
        "source_tiff": str(source_tiff),
        "sampling_rate_hz": signal_table.sampling_rate_hz,
        "roi_count": len(signal_table.roi_labels),
        "frame_count": signal_table.tiff_info.frame_count,
    }


def export_mask_comparison_csv(
    *,
    variant: str,
    source_tiff: Path,
    original_tiff: Path,
    mask_path: Path,
    export_dir: Path | None,
    sampling_rate_hz: float | None,
    mask_manifest_path: Path | None,
    background_mode: str,
) -> dict[str, Any]:
    def progress(done: int, total: int) -> None:
        print(f"{source_tiff.name}: {variant} signal frames {done}/{total}")

    signal_table = extract_mask_signals(
        source_tiff,
        mask_path,
        mask_manifest_path=mask_manifest_path,
        sampling_rate_hz=sampling_rate_hz,
        background_mode=background_mode,
        progress=progress,
    )
    csv_path = comparison_csv_path(original_tiff, export_dir, variant)
    write_mask_signals_csv(signal_table, csv_path)
    result = {
        "ok": True,
        "skipped": False,
        "signal_source": "mask",
        "csv": existing_path_text(csv_path),
        "source_tiff": str(source_tiff),
        "mask_path": str(mask_path),
        "sampling_rate_hz": signal_table.sampling_rate_hz,
        "roi_count": len(signal_table.roi_labels),
        "frame_count": signal_table.tiff_info.frame_count,
        "background_mode": signal_table.background_mode,
    }
    if mask_manifest_path is not None:
        result["mask_manifest_path"] = str(mask_manifest_path)
    return result


def skipped_comparison_variant(reason: str) -> dict[str, Any]:
    return {"ok": True, "skipped": True, "reason": reason}


def export_signal_comparison_csvs(
    *,
    original_tiff: Path,
    motion_corrected_tiff: Path | None,
    raw_mask_path: Path | None,
    raw_mask_manifest_path: Path | None,
    motion_mask_path: Path | None,
    motion_mask_manifest_path: Path | None,
    tracked_trace_csv: Path | None,
    tracked_trace_xlsx: Path | None,
    tracked_manifest_path: Path | None,
    tracked_tracks_csv: Path | None,
    export_dir: Path | None,
    sampling_rate_hz: float | None,
    background_mode: str,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    signal_tables: dict[str, Any] = {}

    def rectangular_variant(name: str, source_tiff: Path) -> dict[str, Any]:
        def progress(done: int, total: int) -> None:
            print(f"{source_tiff.name}: {name} signal frames {done}/{total}")

        signal_table = extract_roi_signals(source_tiff, sampling_rate_hz=sampling_rate_hz, progress=progress)
        csv_path = comparison_csv_path(original_tiff, export_dir, name)
        write_signals_csv(signal_table, csv_path)
        signal_tables[name] = signal_table
        return {
            "ok": True,
            "skipped": False,
            "signal_source": "rectangular_roi",
            "csv": existing_path_text(csv_path),
            "source_tiff": str(source_tiff),
            "sampling_rate_hz": signal_table.sampling_rate_hz,
            "roi_count": len(signal_table.roi_labels),
            "frame_count": signal_table.tiff_info.frame_count,
        }

    def mask_variant(
        name: str,
        source_tiff: Path,
        mask_path: Path,
        mask_manifest_path: Path | None,
        mask_source_tiff: Path | None = None,
    ) -> dict[str, Any]:
        def progress(done: int, total: int) -> None:
            print(f"{source_tiff.name}: {name} signal frames {done}/{total}")

        signal_table = extract_mask_signals(
            source_tiff,
            mask_path,
            mask_manifest_path=mask_manifest_path,
            sampling_rate_hz=sampling_rate_hz,
            background_mode=background_mode,
            progress=progress,
        )
        csv_path = comparison_csv_path(original_tiff, export_dir, name)
        write_mask_signals_csv(signal_table, csv_path)
        signal_tables[name] = signal_table
        result = {
            "ok": True,
            "skipped": False,
            "signal_source": "mask",
            "csv": existing_path_text(csv_path),
            "source_tiff": str(source_tiff),
            "mask_source_tiff": str(mask_source_tiff or source_tiff),
            "mask_path": str(mask_path),
            "sampling_rate_hz": signal_table.sampling_rate_hz,
            "roi_count": len(signal_table.roi_labels),
            "frame_count": signal_table.tiff_info.frame_count,
            "background_mode": signal_table.background_mode,
        }
        if mask_manifest_path is not None:
            result["mask_manifest_path"] = str(mask_manifest_path)
        return result

    try:
        variants["raw"] = rectangular_variant("raw", original_tiff)
    except Exception as exc:
        variants["raw"] = {"ok": False, "skipped": False, "error": str(exc)}

    tracked_mode = tracked_trace_csv is not None
    if tracked_mode:
        try:
            if not tracked_trace_csv.exists():
                raise FileNotFoundError(f"Tracked segmentation CSV does not exist: {tracked_trace_csv}")
            signal_tables["tracked_segmentation"] = CsvSignalTable(tracked_trace_csv)
            variants["tracked_segmentation"] = {
                "ok": True,
                "skipped": False,
                "signal_source": "tracked_segmentation",
                "csv": existing_path_text(tracked_trace_csv),
                "xlsx": existing_path_text(tracked_trace_xlsx),
                "source_tiff": str(original_tiff),
                "roi_count": signal_tables["tracked_segmentation"].roi_count,
                "frame_count": signal_tables["tracked_segmentation"].frame_count,
                "tracked_manifest_path": existing_path_text(tracked_manifest_path),
                "tracked_tracks_csv": existing_path_text(tracked_tracks_csv),
            }
        except Exception as exc:
            variants["tracked_segmentation"] = {"ok": False, "skipped": False, "error": str(exc)}
        variants["segmentation"] = skipped_comparison_variant(
            "Dynamic tracked segmentation writes tracked_segmentation instead of static segmentation comparison."
        )
        variants["motion_corrected"] = skipped_comparison_variant(
            "Dynamic tracked segmentation does not run motion correction."
        )
        variants["motion_corrected_segmentation"] = skipped_comparison_variant(
            "Dynamic tracked segmentation does not run motion-corrected segmentation."
        )
        variants["motion_segmentation_on_raw"] = skipped_comparison_variant(
            "Dynamic tracked segmentation does not run motion-corrected static segmentation."
        )
    else:
        variants["tracked_segmentation"] = skipped_comparison_variant(
            "No tracked segmentation trace is available."
        )

    if not tracked_mode and motion_corrected_tiff is not None:
        try:
            variants["motion_corrected"] = rectangular_variant("motion_corrected", motion_corrected_tiff)
        except Exception as exc:
            variants["motion_corrected"] = {"ok": False, "skipped": False, "error": str(exc)}
    elif not tracked_mode:
        variants["motion_corrected"] = skipped_comparison_variant("No motion-corrected TIFF is available.")

    if not tracked_mode and raw_mask_path is not None:
        try:
            variants["segmentation"] = mask_variant(
                "segmentation",
                original_tiff,
                raw_mask_path,
                raw_mask_manifest_path,
                mask_source_tiff=original_tiff,
            )
        except Exception as exc:
            variants["segmentation"] = {"ok": False, "skipped": False, "error": str(exc)}
    elif not tracked_mode:
        variants["segmentation"] = skipped_comparison_variant("No raw segmentation mask is available.")

    if not tracked_mode and motion_mask_path is not None:
        try:
            variants["motion_segmentation_on_raw"] = mask_variant(
                "motion_segmentation_on_raw",
                original_tiff,
                motion_mask_path,
                motion_mask_manifest_path,
                mask_source_tiff=motion_corrected_tiff,
            )
        except Exception as exc:
            variants["motion_segmentation_on_raw"] = {"ok": False, "skipped": False, "error": str(exc)}
    elif not tracked_mode and motion_corrected_tiff is None:
        variants["motion_segmentation_on_raw"] = skipped_comparison_variant(
            "No motion-corrected TIFF is available."
        )
    elif not tracked_mode:
        variants["motion_segmentation_on_raw"] = skipped_comparison_variant(
            "No motion-corrected segmentation mask is available."
        )

    if not tracked_mode and motion_corrected_tiff is not None and motion_mask_path is not None:
        try:
            variants["motion_corrected_segmentation"] = mask_variant(
                "motion_corrected_segmentation",
                motion_corrected_tiff,
                motion_mask_path,
                motion_mask_manifest_path,
                mask_source_tiff=motion_corrected_tiff,
            )
        except Exception as exc:
            variants["motion_corrected_segmentation"] = {"ok": False, "skipped": False, "error": str(exc)}
    elif not tracked_mode and motion_corrected_tiff is None:
        variants["motion_corrected_segmentation"] = skipped_comparison_variant(
            "No motion-corrected TIFF is available."
        )
    elif not tracked_mode:
        variants["motion_corrected_segmentation"] = skipped_comparison_variant(
            "No motion-corrected segmentation mask is available."
        )

    csvs = [
        variant["csv"]
        for variant in variants.values()
        if isinstance(variant, dict) and variant.get("ok") and not variant.get("skipped") and variant.get("csv")
    ]
    workbook_path = comparison_signal_output_dir(original_tiff, export_dir) / f"{original_tiff.stem}_signal_comparison.xlsx"
    workbook_error = None
    try:
        write_signal_comparison_workbook(variants, signal_tables, workbook_path)
    except Exception as exc:
        workbook_error = str(exc)

    failed = {
        name: variant
        for name, variant in variants.items()
        if isinstance(variant, dict) and not variant.get("ok", False)
    }
    if workbook_error is not None:
        failed["signal_comparison_workbook"] = {"ok": False, "error": workbook_error}

    workbook_result = {"ok": workbook_error is None, "path": existing_path_text(workbook_path)}
    if workbook_error is not None:
        workbook_result["error"] = workbook_error

    return {
        "ok": not failed,
        "variants": variants,
        "csvs": csvs,
        "workbook": workbook_result,
        "xlsx": existing_path_text(workbook_path) if workbook_error is None else None,
        "failed_variants": sorted(failed),
    }


def export_pipeline_qc_images(
    *,
    original_tiff: Path,
    motion_corrected_tiff: Path | None,
    export_dir: Path | None,
    record: dict[str, Any],
) -> dict[str, Any]:
    qc_dir = default_export_output_dir(original_tiff, export_dir)
    result: dict[str, Any] = {"ok": True}

    raw_average = qc_dir / f"{original_tiff.stem}_raw_average.png"
    save_raw_average_png(original_tiff, raw_average)
    result["raw_average_png"] = existing_path_text(raw_average)

    if motion_corrected_tiff is not None:
        motion_qc = qc_dir / f"{original_tiff.stem}_motion_vs_raw_qc.png"
        save_motion_comparison_png(original_tiff, motion_corrected_tiff, motion_qc)
        result["motion_vs_raw_qc_png"] = existing_path_text(motion_qc)
    else:
        result["motion_vs_raw_qc_png"] = None
        result["motion_vs_raw_qc_skipped"] = "No motion-corrected TIFF is available."

    roi_action = record.get("actions", {}).get("roi_detection", {})
    if isinstance(roi_action, dict):
        overlay = roi_action.get("qc_overlay_png") or roi_action.get("segmentation_overlay_png")
        if overlay:
            result["segmentation_overlay_png"] = overlay

    raw_roi_action = record.get("actions", {}).get("roi_detection_raw", {})
    if isinstance(raw_roi_action, dict):
        raw_overlay = raw_roi_action.get("qc_overlay_png") or raw_roi_action.get("segmentation_overlay_png")
        if raw_overlay:
            result["raw_segmentation_overlay_png"] = raw_overlay

    return result


def same_resolved_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def apply_hook_outputs(
    hook_result: dict[str, Any],
    *,
    current_tiff: Path,
    current_metadata: Path,
) -> tuple[Path, Path]:
    next_tiff = current_tiff
    hook_tiff = first_path_value(
        hook_result,
        [
            "tiff_path",
            "output_tiff",
            "output_tiff_path",
            "corrected_tiff",
            "corrected_tiff_path",
            "downstream_tiff",
        ],
    )
    if hook_tiff is not None:
        hook_tiff = resolve_returned_path(hook_tiff, fallback_base=current_tiff.parent)
        if not hook_tiff.exists():
            raise FileNotFoundError(f"Hook returned TIFF path that does not exist: {hook_tiff}")
        if not is_tiff_path(hook_tiff):
            raise ValueError(f"Hook returned non-TIFF path for downstream TIFF: {hook_tiff}")
        next_tiff = hook_tiff

    next_metadata = metadata_path_for_tiff(next_tiff)
    hook_metadata = first_path_value(hook_result, ["metadata_path", "output_metadata", "roi_metadata_path"])
    if hook_metadata is not None:
        hook_metadata = resolve_returned_path(hook_metadata, fallback_base=next_tiff.parent)
        if not hook_metadata.exists():
            raise FileNotFoundError(f"Hook returned metadata path that does not exist: {hook_metadata}")
        if not same_resolved_path(hook_metadata, next_metadata):
            next_metadata.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hook_metadata, next_metadata)
    elif next_tiff != current_tiff:
        copy_metadata_if_needed(current_metadata, next_tiff)

    return next_tiff, metadata_path_for_tiff(next_tiff)


def resolve_hook_mask_outputs(
    hook_result: dict[str, Any],
    *,
    hook_output_base: Path,
    stage: str,
) -> tuple[Path | None, Path | None]:
    mask_path: Path | None = None
    mask_manifest_path: Path | None = None
    hook_mask = first_path_value(
        hook_result,
        ["mask_path", "masks_path", "roi_mask_path", "output_masks", "segmentation_mask_path"],
    )
    if hook_mask is not None:
        mask_path = resolve_hook_file(
            hook_mask,
            base_dir=hook_output_base,
            description=f"{stage} hook returned mask path",
        )
    hook_mask_manifest = first_path_value(
        hook_result,
        ["mask_manifest_path", "manifest_path", "segmentation_manifest_path", "roi_manifest_path"],
    )
    if hook_mask_manifest is not None:
        mask_manifest_path = resolve_hook_file(
            hook_mask_manifest,
            base_dir=hook_output_base,
            description=f"{stage} hook returned mask manifest path",
        )
    return mask_path, mask_manifest_path


def run_raw_roi_debug_hook(
    *,
    tiff_path: Path,
    export_dir: Path | None,
    record: dict[str, Any],
    roi_hook: str,
    roi_hook_kwargs: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    action_name = "roi_detection_raw"
    try:
        hook_result = run_hook(
            "roi_detection",
            roi_hook,
            tiff_path=tiff_path,
            output_dir=export_dir,
            record=record,
            hook_kwargs=roi_hook_kwargs,
        )
        if roi_hook_kwargs and not hook_result.get("skipped"):
            hook_result["hook_parameters"] = roi_hook_kwargs
        record["actions"][action_name] = hook_result
        if hook_result.get("skipped"):
            print(f"[SKIP] {tiff_path.name}: raw ROI debug: {hook_result.get('reason')}")
            return None, None
        print(f"[OK] {tiff_path.name}: raw ROI debug hook {roi_hook}")
        if not hook_result.get("ok", True):
            return None, None
        hook_output_base = export_dir if export_dir is not None else tiff_path.parent
        mask_path, mask_manifest_path = resolve_hook_mask_outputs(
            hook_result,
            hook_output_base=hook_output_base,
            stage="raw ROI debug",
        )
        normalize_hook_result_paths(
            hook_result,
            [
                hook_output_base,
                tiff_path.parent,
                Path.cwd(),
            ],
        )
        return mask_path, mask_manifest_path
    except Exception as exc:
        record["actions"][action_name] = {
            "ok": False,
            "skipped": False,
            "hook": roi_hook,
            "error": str(exc),
        }
        print(f"[ERROR] {tiff_path.name}: raw ROI debug: {exc}", file=sys.stderr)
        return None, None


def run_tiff(
    tiff_path: Path,
    output_root: Path | None,
    sampling_rate_hz: float | None,
    write_csv: bool,
    do_average: bool,
    do_signals: bool,
    do_split: bool,
    motion_hook: str | None,
    roi_hook: str | None,
    motion_hook_kwargs: dict[str, Any] | None = None,
    roi_hook_kwargs: dict[str, Any] | None = None,
    mask_background_mode: str = "none",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "tiff": str(tiff_path),
        "ok": True,
        "actions": {},
        "errors": [],
    }
    export_dir = per_tiff_output_dir(tiff_path, output_root)
    working_tiff_path = tiff_path
    working_metadata_path = metadata_path_for_tiff(tiff_path)
    working_mask_path: Path | None = None
    working_mask_manifest_path: Path | None = None
    motion_corrected_tiff_path: Path | None = None
    raw_mask_path: Path | None = None
    raw_mask_manifest_path: Path | None = None
    motion_mask_path: Path | None = None
    motion_mask_manifest_path: Path | None = None
    tracked_trace_csv_path: Path | None = None
    tracked_trace_xlsx_path: Path | None = None
    tracked_manifest_path: Path | None = None
    tracked_tracks_csv_path: Path | None = None
    tracked_debug_tiff_paths: list[Path] = []
    tracked_debug_raw_tiff_paths: list[Path] = []
    tracked_debug_mask_tiff_paths: list[Path] = []
    effective_roi_hook_kwargs = dict(roi_hook_kwargs or {})
    if sampling_rate_hz is not None:
        effective_roi_hook_kwargs.setdefault("sampling_rate_hz", sampling_rate_hz)

    for stage, spec, action_name, hook_kwargs in [
        ("motion_correction", motion_hook, "motion_correction", motion_hook_kwargs or {}),
        ("roi_detection", roi_hook, "roi_detection", effective_roi_hook_kwargs),
    ]:
        try:
            stage_input_tiff_path = working_tiff_path
            hook_result = run_hook(
                stage,
                spec,
                tiff_path=working_tiff_path,
                output_dir=export_dir,
                record=record,
                hook_kwargs=hook_kwargs,
            )
            if hook_kwargs and not hook_result.get("skipped"):
                hook_result["hook_parameters"] = hook_kwargs
            record["actions"][action_name] = hook_result
            if hook_result.get("skipped"):
                print(f"[SKIP] {tiff_path.name}: {stage}: {hook_result.get('reason')}")
            else:
                print(f"[OK] {tiff_path.name}: {stage} hook {spec}")
            if not hook_result.get("ok", True):
                record["ok"] = False
                record["errors"].append(f"{stage}: hook returned ok=false")
            else:
                working_tiff_path, working_metadata_path = apply_hook_outputs(
                    hook_result,
                    current_tiff=working_tiff_path,
                    current_metadata=working_metadata_path,
                )
                if action_name == "motion_correction" and not same_resolved_path(working_tiff_path, tiff_path):
                    motion_corrected_tiff_path = working_tiff_path
                hook_output_base = export_dir if export_dir is not None else working_tiff_path.parent
                hook_mask_path, hook_mask_manifest_path = resolve_hook_mask_outputs(
                    hook_result,
                    hook_output_base=hook_output_base,
                    stage=stage,
                )
                if hook_mask_path is not None:
                    working_mask_path = hook_mask_path
                if hook_mask_manifest_path is not None:
                    working_mask_manifest_path = hook_mask_manifest_path
                if action_name == "roi_detection" and hook_mask_path is not None:
                    if same_resolved_path(stage_input_tiff_path, tiff_path):
                        raw_mask_path = hook_mask_path
                        raw_mask_manifest_path = hook_mask_manifest_path
                    elif (
                        motion_corrected_tiff_path is not None
                        and same_resolved_path(stage_input_tiff_path, motion_corrected_tiff_path)
                    ):
                        motion_mask_path = hook_mask_path
                        motion_mask_manifest_path = hook_mask_manifest_path
                normalize_hook_result_paths(
                    hook_result,
                    [
                        hook_output_base,
                        working_tiff_path.parent,
                        stage_input_tiff_path.parent,
                        Path.cwd(),
                    ],
                )
                if action_name == "roi_detection":
                    tracked_trace_csv_path = first_path_value(hook_result, ["tracked_trace_csv"])
                    tracked_trace_xlsx_path = first_path_value(hook_result, ["tracked_trace_xlsx"])
                    tracked_manifest_path = first_path_value(hook_result, ["tracked_manifest_path"])
                    tracked_tracks_csv_path = first_path_value(hook_result, ["tracked_tracks_csv"])
                    tracked_debug_value = hook_result.get("tracked_debug_tiffs")
                    if isinstance(tracked_debug_value, list):
                        tracked_debug_tiff_paths = [Path(path) for path in tracked_debug_value if path]
                    tracked_raw_value = hook_result.get("tracked_debug_raw_tiffs")
                    if isinstance(tracked_raw_value, list):
                        tracked_debug_raw_tiff_paths = [Path(path) for path in tracked_raw_value if path]
                    tracked_mask_value = hook_result.get("tracked_debug_mask_tiffs")
                    if isinstance(tracked_mask_value, list):
                        tracked_debug_mask_tiff_paths = [Path(path) for path in tracked_mask_value if path]
        except Exception as exc:
            record["ok"] = False
            record["actions"][action_name] = {
                "ok": False,
                "skipped": False,
                "hook": spec,
                "error": str(exc),
            }
            record["errors"].append(f"{stage}: {exc}")
            print(f"[ERROR] {tiff_path.name}: {stage}: {exc}", file=sys.stderr)

    if (
        record["ok"]
        and do_signals
        and write_csv
        and roi_hook
        and motion_corrected_tiff_path is not None
        and raw_mask_path is None
    ):
        raw_mask_path, raw_mask_manifest_path = run_raw_roi_debug_hook(
            tiff_path=tiff_path,
            export_dir=export_dir,
            record=record,
            roi_hook=roi_hook,
            roi_hook_kwargs=roi_hook_kwargs or {},
        )

    record["working_tiff"] = str(working_tiff_path)
    record["working_metadata"] = str(working_metadata_path)
    if working_mask_path is not None:
        record["working_mask"] = str(working_mask_path)
    if working_mask_manifest_path is not None:
        record["working_mask_manifest"] = str(working_mask_manifest_path)

    if not record["ok"]:
        print(f"[ERROR] {tiff_path.name}: downstream outputs skipped because a hook stage failed.", file=sys.stderr)
        return record

    try:
        qc_result = export_pipeline_qc_images(
            original_tiff=tiff_path,
            motion_corrected_tiff=motion_corrected_tiff_path,
            export_dir=export_dir,
            record=record,
        )
        record["actions"]["qc_images"] = qc_result
        for key in [
            "raw_average_png",
            "motion_vs_raw_qc_png",
            "segmentation_overlay_png",
            "raw_segmentation_overlay_png",
        ]:
            path = qc_result.get(key)
            if path:
                print(f"[OK] {tiff_path.name}: QC image {path}")
    except Exception as exc:
        record["ok"] = False
        record["actions"]["qc_images"] = {"ok": False, "error": str(exc)}
        record["errors"].append(f"qc_images: {exc}")
        print(f"[ERROR] {tiff_path.name}: QC images: {exc}", file=sys.stderr)

    if do_average:
        try:
            def average_progress(done: int, total: int, source: Path = working_tiff_path) -> None:
                print(f"{source.name}: average frames {done}/{total}")

            average_path = default_average_png_path(working_tiff_path, export_dir)
            save_tiff_average_png(working_tiff_path, average_path, progress=average_progress)
            print(f"[OK] {working_tiff_path.name}: average PNG {average_path}")
            record["actions"]["average_png"] = {
                "ok": True,
                "path": existing_path_text(average_path),
                "source_tiff": str(working_tiff_path),
            }
        except Exception as exc:
            record["ok"] = False
            record["actions"]["average_png"] = {"ok": False, "error": str(exc)}
            record["errors"].append(f"average_png: {exc}")
            print(f"[ERROR] {tiff_path.name}: average PNG: {exc}", file=sys.stderr)

    if do_signals:
        try:
            def signal_progress(done: int, total: int, source: Path = working_tiff_path) -> None:
                print(f"{source.name}: signal frames {done}/{total}")

            if tracked_trace_csv_path is not None and tracked_trace_xlsx_path is not None:
                if not tracked_trace_csv_path.exists():
                    raise FileNotFoundError(f"Tracked signal CSV is missing: {tracked_trace_csv_path}")
                if not tracked_trace_xlsx_path.exists():
                    raise FileNotFoundError(f"Tracked signal workbook is missing: {tracked_trace_xlsx_path}")
                csv_table = CsvSignalTable(tracked_trace_csv_path)
                signal_source = "tracked_segmentation"
                print(f"[OK] {working_tiff_path.name}: tracked segmentation signals {tracked_trace_xlsx_path}")
                record["actions"]["signals"] = {
                    "ok": True,
                    "signal_source": signal_source,
                    "xlsx": existing_path_text(tracked_trace_xlsx_path),
                    "csv": existing_path_text(tracked_trace_csv_path),
                    "source_tiff": str(working_tiff_path),
                    "roi_count": csv_table.roi_count,
                    "frame_count": csv_table.frame_count,
                    "tracked_manifest_path": existing_path_text(tracked_manifest_path),
                    "tracked_tracks_csv": existing_path_text(tracked_tracks_csv_path),
                    "tracked_debug_tiffs": [existing_path_text(path) for path in tracked_debug_tiff_paths],
                    "tracked_debug_raw_tiffs": [
                        existing_path_text(path) for path in tracked_debug_raw_tiff_paths
                    ],
                    "tracked_debug_mask_tiffs": [
                        existing_path_text(path) for path in tracked_debug_mask_tiff_paths
                    ],
                }
            elif working_mask_path is not None:
                signal_result = export_mask_signal_documents(
                    working_tiff_path,
                    working_mask_path,
                    output_dir=export_dir,
                    sampling_rate_hz=sampling_rate_hz,
                    write_csv=write_csv,
                    progress=signal_progress,
                    mask_manifest_path=working_mask_manifest_path,
                    background_mode=mask_background_mode,
                )
                signal_source = "mask"
                print(f"[OK] {working_tiff_path.name}: mask signals {signal_result.xlsx_path}")
            else:
                signal_result = export_signal_documents(
                    working_tiff_path,
                    output_dir=export_dir,
                    sampling_rate_hz=sampling_rate_hz,
                    write_csv=write_csv,
                    progress=signal_progress,
                )
                signal_source = "rectangular_roi"
                print(f"[OK] {working_tiff_path.name}: signals {signal_result.xlsx_path}")

            if tracked_trace_csv_path is None or tracked_trace_xlsx_path is None:
                record["actions"]["signals"] = {
                    "ok": True,
                    "signal_source": signal_source,
                    "xlsx": existing_path_text(signal_result.xlsx_path),
                    "csv": existing_path_text(signal_result.csv_path),
                    "source_tiff": str(working_tiff_path),
                    "sampling_rate_hz": signal_result.signal_table.sampling_rate_hz,
                    "roi_count": len(signal_result.signal_table.roi_labels),
                    "frame_count": signal_result.signal_table.tiff_info.frame_count,
                }
                if signal_source == "mask":
                    record["actions"]["signals"]["background_mode"] = signal_result.signal_table.background_mode
                if working_mask_path is not None:
                    record["actions"]["signals"]["mask_path"] = str(working_mask_path)
                if working_mask_manifest_path is not None:
                    record["actions"]["signals"]["mask_manifest_path"] = str(working_mask_manifest_path)
        except Exception as exc:
            record["ok"] = False
            record["actions"]["signals"] = {"ok": False, "error": str(exc)}
            record["errors"].append(f"signals: {exc}")
            print(f"[ERROR] {tiff_path.name}: signals: {exc}", file=sys.stderr)

        if write_csv:
            comparison = export_signal_comparison_csvs(
                original_tiff=tiff_path,
                motion_corrected_tiff=motion_corrected_tiff_path,
                raw_mask_path=raw_mask_path,
                raw_mask_manifest_path=raw_mask_manifest_path,
                motion_mask_path=motion_mask_path,
                motion_mask_manifest_path=motion_mask_manifest_path,
                tracked_trace_csv=tracked_trace_csv_path,
                tracked_trace_xlsx=tracked_trace_xlsx_path,
                tracked_manifest_path=tracked_manifest_path,
                tracked_tracks_csv=tracked_tracks_csv_path,
                export_dir=export_dir,
                sampling_rate_hz=sampling_rate_hz,
                background_mode=mask_background_mode,
            )
            record["actions"]["signal_comparison_csvs"] = comparison
            for csv_path in comparison.get("csvs", []):
                print(f"[OK] {tiff_path.name}: comparison CSV {csv_path}")
            if not comparison.get("ok", False):
                record["ok"] = False
                failed = ", ".join(comparison.get("failed_variants", []))
                record["errors"].append(f"signal_comparison_csvs: failed variants: {failed}")
                print(
                    f"[ERROR] {tiff_path.name}: signal comparison CSVs failed for {failed}",
                    file=sys.stderr,
                )
        else:
            record["actions"]["signal_comparison_csvs"] = {
                "ok": True,
                "skipped": True,
                "reason": "CSV export is disabled.",
            }

    if do_split:
        try:
            split_result = split_tiff(
                working_tiff_path,
                debug=True,
                output_root=split_output_root(output_root),
                write_manifest=True,
            )
            print(split_result.message)
            record["actions"]["split"] = {
                "ok": split_result.ok,
                "message": split_result.message,
                "output_dir": existing_path_text(split_result.output_dir),
                "source_tiff": str(working_tiff_path),
                "error": split_result.error,
            }
            if split_result.ok:
                label_map_path = split_result.output_dir / f"{working_tiff_path.stem}_split_roi_labels.png"
                save_roi_label_map_png(working_tiff_path, label_map_path)
                record["actions"]["split"]["roi_label_png"] = existing_path_text(label_map_path)
                print(f"[OK] {working_tiff_path.name}: split ROI labels {label_map_path}")
            if not split_result.ok:
                record["ok"] = False
                record["errors"].append(f"split: {split_result.error or split_result.message}")
        except Exception as exc:
            record["ok"] = False
            record["actions"]["split"] = {"ok": False, "error": str(exc)}
            record["errors"].append(f"split: {exc}")
            print(f"[ERROR] {tiff_path.name}: split: {exc}", file=sys.stderr)

    return record


def collect_tiff_paths(input_paths: list[Path], recursive: bool) -> list[Path]:
    tiff_paths: list[Path] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        for tiff_path in iter_tiff_inputs(input_path, recursive):
            resolved = tiff_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            tiff_paths.append(tiff_path)
    return tiff_paths


def preflight_hook_specs(motion_hook: str | None, roi_hook: str | None) -> bool:
    ok = True
    for label, spec in [
        ("motion_correction", motion_hook),
        ("roi_detection", roi_hook),
    ]:
        if not spec:
            continue
        try:
            hook = load_hook(spec)
        except Exception as exc:
            print(f"[ERROR] {label} hook is not valid: {exc}", file=sys.stderr)
            ok = False
            continue
        print(f"[OK] {label} hook preflight: {spec} -> {hook.__module__}.{hook.__name__}")
    return ok


def run(
    input_paths: Path | list[Path],
    recursive: bool,
    output_root: Path | None,
    sampling_rate_hz: float | None,
    write_csv: bool,
    do_average: bool,
    do_signals: bool,
    do_split: bool,
    motion_hook: str | None,
    roi_hook: str | None,
    manifest_path: Path | None,
    motion_hook_kwargs: dict[str, Any] | None = None,
    roi_hook_kwargs: dict[str, Any] | None = None,
    mask_background_mode: str = "none",
) -> int:
    if isinstance(input_paths, Path):
        input_paths = [input_paths]
    try:
        tiff_paths = collect_tiff_paths(input_paths, recursive)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not tiff_paths:
        input_text = ", ".join(str(path) for path in input_paths)
        print(f"[ERROR] No TIFF files found under {input_text}.", file=sys.stderr)
        return 2

    if not preflight_hook_specs(motion_hook, roi_hook):
        print("[ERROR] Hook preflight failed; pipeline was not started.", file=sys.stderr)
        return 2

    motion_hook_kwargs = dict(motion_hook_kwargs or {})
    roi_hook_kwargs = dict(roi_hook_kwargs or {})

    manifest_path = manifest_path or default_manifest_path(input_paths, output_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_paths[0]) if len(input_paths) == 1 else [str(path) for path in input_paths],
        "inputs": [str(path) for path in input_paths],
        "output_root": path_text(output_root),
        "settings": {
            "recursive": recursive,
            "sampling_rate_hz": sampling_rate_hz,
            "write_csv": write_csv,
            "average_png": do_average,
            "signals": do_signals,
            "split": do_split,
            "motion_hook": motion_hook,
            "motion_hook_params": motion_hook_kwargs,
            "roi_hook": roi_hook,
            "roi_hook_params": roi_hook_kwargs,
            "mask_background": mask_background_mode,
        },
        "tiffs": [],
    }

    for tiff_path in tiff_paths:
        manifest["tiffs"].append(
            run_tiff(
                tiff_path,
                output_root=output_root,
                sampling_rate_hz=sampling_rate_hz,
                write_csv=write_csv,
                do_average=do_average,
                do_signals=do_signals,
                do_split=do_split,
                motion_hook=motion_hook,
                roi_hook=roi_hook,
                motion_hook_kwargs=motion_hook_kwargs,
                roi_hook_kwargs=roi_hook_kwargs,
                mask_background_mode=mask_background_mode,
            )
        )

    successes = sum(1 for record in manifest["tiffs"] if record["ok"])
    failures = len(manifest["tiffs"]) - successes
    manifest["ok"] = failures == 0
    manifest["summary"] = {
        "tiff_count": len(manifest["tiffs"]),
        "successes": successes,
        "failures": failures,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] manifest: {manifest_path}")
    print(f"Processed {len(manifest['tiffs'])} TIFF file(s): {successes} succeeded, {failures} failed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        motion_hook_kwargs = parse_hook_params(args.motion_hook_param, label="motion")
        roi_hook_kwargs = parse_hook_params(args.roi_hook_param, label="ROI")
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return run(
        args.input,
        recursive=args.recursive,
        output_root=args.output,
        sampling_rate_hz=args.sampling_rate,
        write_csv=not args.no_csv,
        do_average=not args.no_average,
        do_signals=not args.no_signals,
        do_split=not args.no_split,
        motion_hook=args.motion_hook,
        roi_hook=args.roi_hook,
        manifest_path=args.manifest,
        motion_hook_kwargs=motion_hook_kwargs,
        roi_hook_kwargs=roi_hook_kwargs,
        mask_background_mode=args.mask_background,
    )


if __name__ == "__main__":
    raise SystemExit(main())
