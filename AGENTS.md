# Agent Instructions

This repository contains the spike detector Python app plus a native C++/Qt
image processor. Treat it as an active research/tooling workspace: some folders
are source code, while others are generated outputs from experiments or builds.

## Preserve Source Code

Do not delete, overwrite, or hide these project-code paths during cleanup:

- `detector/`
- `image_processor/`
- `image_processor/native_viewer/src/`
- `image_processor/native_viewer/CMakeLists.txt`
- `image_processor/native_viewer/README.md`
- `tools/`
- `tests/`
- `docs/`
- `main.py`
- `launch_image_processor.bat`
- `requirements.txt`
- `detection_params.json`

The old `app/` files may appear as deleted in git status because the code has
been reorganized under `detector/`. Do not restore or remove those changes
unless the user asks.

## Generated Or Disposable Paths

These paths are generated artifacts and can be ignored by git:

- `image_processor/native_viewer/build/` - CMake build tree, executable, Qt DLL deployment, and
  probe outputs. It is rebuildable, but deleting it means the native UI must be
  rebuilt before it can run.
- `__pycache__/` and nested `__pycache__/` folders - Python bytecode cache.
- `*.log`, `*.obj` - local build/install byproducts.
- `tmp/` - scratch probes and temporary research artifacts.

Do not delete `analysis_outputs/` without explicit user confirmation. It is
generated output, but it may contain experiment results the user wants to keep.

## Cleanup Rules

Before recursive deletion, resolve absolute paths and verify every target is
inside the repository root. Use PowerShell `Remove-Item -LiteralPath` for
Windows cleanup. Do not compose delete commands by passing path strings between
different shells.

When in doubt, prefer updating `.gitignore` and summarizing candidates over
deleting data. Keep cleanup scoped to generated artifacts, never source code.

## Native Viewer Notes

`launch_image_processor.bat` and `image_processor.app` expect the native UI at:

```text
image_processor/native_viewer/build/femtonics_image_processor.exe
```

If that executable is missing, the launcher attempts to rebuild it with CMake,
Ninja, Qt, and vcpkg using `tools/native_dev_env.bat`. Keep the native source
tree even when deleting `image_processor/native_viewer/build/`.
