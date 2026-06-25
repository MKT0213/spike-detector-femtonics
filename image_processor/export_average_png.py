#!/usr/bin/env python3
"""CLI wrapper for exporting average-projection PNGs from TIFF stacks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .app import default_average_png_path, save_tiff_average_png
    from .roi_core import iter_tiff_inputs
except ImportError:  # pragma: no cover - supports direct script execution.
    from app import default_average_png_path, save_tiff_average_png
    from roi_core import iter_tiff_inputs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export average-projection PNGs from TIFF stacks.")
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
        help="Optional output root. Each TIFF writes into <output>/<tiff_stem>/.",
    )
    return parser.parse_args(argv)


def output_dir_for_tiff(tiff_path: Path, output_root: Path | None) -> Path | None:
    if output_root is None:
        return None
    return output_root / tiff_path.stem


def run(input_path: Path, recursive: bool, output_root: Path | None) -> int:
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
                print(f"{source.name}: average frames {done}/{total}")

            output_dir = output_dir_for_tiff(tiff_path, output_root)
            output_path = default_average_png_path(tiff_path, output_dir)
            save_tiff_average_png(tiff_path, output_path, progress=progress)
            print(f"[OK] {tiff_path.name}: {output_path}")
            successes += 1
        except Exception as exc:
            print(f"[ERROR] {tiff_path.name}: {exc}", file=sys.stderr)
            failures += 1

    print(f"Processed {len(tiff_paths)} TIFF file(s): {successes} succeeded, {failures} failed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run(args.input, recursive=args.recursive, output_root=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
