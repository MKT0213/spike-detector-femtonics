#!/usr/bin/env python3
"""CLI wrapper for exporting Femtonics ROI signal documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .roi_core import iter_tiff_inputs
    from .signal_export import export_signal_documents
except ImportError:  # pragma: no cover - supports direct script execution.
    from roi_core import iter_tiff_inputs
    from signal_export import export_signal_documents


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ROI mean-intensity signals from Femtonics TIFF stacks."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input TIFF file, metadata file, or folder containing TIFF/metadata pairs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search folders recursively for .tif/.tiff files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to an exports folder beside each TIFF.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Override metadata sampling rate in Hz.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write only the Excel workbook.",
    )
    return parser.parse_args(argv)


def run(
    input_path: Path,
    recursive: bool,
    output: Path | None,
    sampling_rate_hz: float | None,
    write_csv: bool,
) -> int:
    try:
        tiff_paths = iter_tiff_inputs(input_path, recursive)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not tiff_paths:
        print(f"[ERROR] No TIFF files found under {input_path}.", file=sys.stderr)
        return 2

    successes = 0
    failures = 0
    for tiff_path in tiff_paths:
        try:
            def progress(done: int, total: int, source: Path = tiff_path) -> None:
                print(f"{source.name}: signal frames {done}/{total}")

            result = export_signal_documents(
                tiff_path,
                output_dir=output,
                sampling_rate_hz=sampling_rate_hz,
                write_csv=write_csv,
                progress=progress,
            )
            print(f"[OK] {tiff_path.name}: {result.xlsx_path}")
            if result.csv_path is not None:
                print(f"[OK] {tiff_path.name}: {result.csv_path}")
            successes += 1
        except Exception as exc:
            print(f"[ERROR] {tiff_path.name}: {exc}", file=sys.stderr)
            failures += 1

    print(f"Processed {len(tiff_paths)} TIFF file(s): {successes} succeeded, {failures} failed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run(
        args.input,
        recursive=args.recursive,
        output=args.output,
        sampling_rate_hz=args.sampling_rate,
        write_csv=not args.no_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
