"""Optional extension hooks for the native image-processing pipeline."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


HookFunction = Callable[..., Any]
RESERVED_HOOK_KWARGS = {"stage", "tiff_path", "output_dir", "record"}


def parse_hook_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if value == "":
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_hook_params(entries: list[str], *, label: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"{label} hook parameter must use KEY=VALUE format: {entry!r}")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} hook parameter has an empty key: {entry!r}")
        if not key.isidentifier():
            raise ValueError(f"{label} hook parameter key must be a valid Python identifier: {key!r}")
        if key in RESERVED_HOOK_KWARGS:
            raise ValueError(f"{label} hook parameter cannot override reserved argument {key!r}.")
        params[key] = parse_hook_value(raw_value)
    return params


def load_hook(spec: str) -> HookFunction:
    if ":" not in spec:
        raise ValueError(f"Hook spec must use module:function or file.py:function format, got {spec!r}.")
    module_name, function_name = spec.rsplit(":", 1)
    module_name = module_name.strip().strip('"')
    function_name = function_name.strip()
    if not module_name or not function_name:
        raise ValueError(f"Hook spec must use module:function or file.py:function format, got {spec!r}.")

    module_path = Path(module_name).expanduser()
    if module_path.suffix.lower() == ".py" or module_path.exists():
        module_path = module_path.resolve()
        if not module_path.is_file():
            raise FileNotFoundError(f"Hook file does not exist: {module_path}")
        digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
        loaded_name = f"_spike_pipeline_hook_{module_path.stem}_{digest}"
        spec_obj = importlib.util.spec_from_file_location(loaded_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load hook file: {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[loaded_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    hook = getattr(module, function_name)
    if not callable(hook):
        raise TypeError(f"Hook target is not callable: {spec}")
    return hook


def _call_hook(
    hook: HookFunction,
    *,
    stage: str,
    tiff_path: Path,
    output_dir: Path | None,
    record: dict[str, Any],
    hook_kwargs: dict[str, Any] | None = None,
) -> Any:
    signature = inspect.signature(hook)
    kwargs = {
        "stage": stage,
        "tiff_path": tiff_path,
        "output_dir": output_dir,
        "record": record,
    }
    if hook_kwargs:
        reserved = sorted(set(hook_kwargs).intersection(RESERVED_HOOK_KWARGS))
        if reserved:
            raise ValueError(f"Hook keyword arguments cannot override reserved values: {', '.join(reserved)}")
        kwargs.update(hook_kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return hook(**kwargs)

    selected_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return hook(**selected_kwargs)


def normalize_hook_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {"ok": True}
    if isinstance(result, dict):
        normalized = dict(result)
        normalized.setdefault("ok", True)
        return normalized
    return {"ok": True, "result": result}


def run_hook(
    stage: str,
    spec: str | None,
    *,
    tiff_path: Path,
    output_dir: Path | None,
    record: dict[str, Any],
    hook_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not spec:
        return {"ok": True, "skipped": True, "reason": "No hook configured."}

    hook = load_hook(spec)
    result = _call_hook(
        hook,
        stage=stage,
        tiff_path=tiff_path,
        output_dir=output_dir,
        record=record,
        hook_kwargs=hook_kwargs,
    )
    normalized = normalize_hook_result(result)
    normalized["skipped"] = False
    normalized["hook"] = spec
    return normalized
