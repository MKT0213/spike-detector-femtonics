#!/usr/bin/env python3
"""CLI wrapper for splitting Femtonics multi-ROI TIFF stacks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .roi_core import iter_tiff_inputs, split_tiff
except ImportError:  # pragma: no cover - supports direct script execution.
    from roi_core import iter_tiff_inputs, split_tiff


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split MESc MultiROIChessBoard TIFF stacks into one multi-page TIFF "
            "per ROI using row-major ROI order."
        )
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
        "--debug",
        action="store_true",
        help="Print detailed mapping information and write debug_manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output root. ROI stacks are written to <output>/<tiff_stem>/.",
    )
    return parser.parse_args(argv)


def run(input_path: Path, recursive: bool, debug: bool, output: Path | None = None) -> int:
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
        result = split_tiff(tiff_path, debug=debug, output_root=output)
        print(result.message)
        if result.ok:
            successes += 1
        else:
            failures += 1

    print(f"Processed {len(tiff_paths)} TIFF file(s): {successes} succeeded, {failures} failed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run(args.input, args.recursive, args.debug, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
