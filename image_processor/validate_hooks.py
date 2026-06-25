#!/usr/bin/env python3
"""Validate native pipeline hook specs without running a TIFF pipeline."""

from __future__ import annotations

import argparse
import importlib
import sys

try:
    from .pipeline_hooks import load_hook, parse_hook_params
except ImportError:  # pragma: no cover - supports direct script execution.
    from pipeline_hooks import load_hook, parse_hook_params


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate motion/ROI hook specs.")
    parser.add_argument("--motion-hook", default=None, help="module:function or file.py:function")
    parser.add_argument(
        "--motion-hook-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Validate an extra keyword argument for the motion hook.",
    )
    parser.add_argument("--roi-hook", default=None, help="module:function or file.py:function")
    parser.add_argument(
        "--roi-hook-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Validate an extra keyword argument for the ROI hook.",
    )
    return parser.parse_args(argv)


def validate_hook(label: str, spec: str | None) -> bool:
    if not spec:
        print(f"[SKIP] {label}: no hook configured")
        return True
    try:
        hook = load_hook(spec)
    except Exception as exc:
        print(f"[ERROR] {label}: {exc}", file=sys.stderr)
        return False
    print(f"[OK] {label}: {spec} -> {hook.__module__}.{hook.__name__}")
    return True


def validate_backend_imports() -> bool:
    print(f"[OK] python: {sys.executable}")
    print(f"[OK] python version: {sys.version.split()[0]}")

    required_modules = [
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("openpyxl", "openpyxl"),
    ]
    optional_modules = [
        ("tifffile fast TIFF reader", "tifffile"),
        ("zarr persistent preview cache", "zarr"),
    ]

    ok = True
    for label, module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            print(f"[ERROR] backend import {label}: {exc}", file=sys.stderr)
            ok = False
        else:
            print(f"[OK] backend import {label}")

    for label, module_name in optional_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            print(f"[WARN] optional backend import {label}: {exc}")
        else:
            print(f"[OK] optional backend import {label}")

    return ok


def validate_hook_params(label: str, entries: list[str]) -> bool:
    try:
        params = parse_hook_params(entries, label=label)
    except ValueError as exc:
        print(f"[ERROR] {label} params: {exc}", file=sys.stderr)
        return False
    if params:
        keys = ", ".join(sorted(params))
        print(f"[OK] {label} params: {keys}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    backend_ok = validate_backend_imports()
    ok = True
    ok = validate_hook_params("motion", args.motion_hook_param) and ok
    ok = validate_hook_params("roi", args.roi_hook_param) and ok
    ok = validate_hook("motion", args.motion_hook) and ok
    ok = validate_hook("roi", args.roi_hook) and ok
    if backend_ok and ok:
        print("Backend and hook validation succeeded.")
        return 0
    print("Backend or hook validation failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
