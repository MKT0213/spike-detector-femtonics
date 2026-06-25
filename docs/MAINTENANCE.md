# Maintenance Notes

This file records the repository cleanup and rebuild conventions so generated
files do not get mixed up with source code.

## Project Map

- `detector/` - Python spike detection and curation package.
- `image_processor/` - Python image-processing UI/backend helpers plus the native C++/Qt UI source.
- `image_processor/native_viewer/src/` - C++/Qt native image processor source.
- `image_processor/native_viewer/build/` - generated native build output; safe to recreate.
- `analysis_outputs/` - generated analysis runs; keep unless intentionally
  archiving or deleting experiment output.
- `docs/` - repository structure, maintenance, and planning notes.
- `tmp/` - scratch probes and temporary files.
- `tests/` and `image_processor/tests/` - automated tests.
- `tools/` - local development environment helpers.

## Safe Cleanup

Safe cleanup may remove:

- Python `__pycache__/` folders.
- `*.log` and `*.obj` byproducts.
- `tmp/` scratch output.
- `image_processor/native_viewer/build/` when disk space matters and rebuilding the native UI
  is acceptable.

Safe cleanup must not remove:

- Source folders listed in the project map.
- `analysis_outputs/` unless the user explicitly confirms that generated
  experiment outputs can be deleted.
- Any untracked source file, test, script, or Markdown planning document without
  checking its purpose.

## Rebuild Native Viewer

From the repository root:

```bat
call tools\native_dev_env.bat
cmake -S image_processor\native_viewer -B image_processor\native_viewer\build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH="%QT_ROOT%" ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build image_processor\native_viewer\build
```

The batch launcher also attempts this rebuild automatically if
`image_processor\native_viewer\build\femtonics_image_processor.exe` is missing.

## Useful Checks

```powershell
git status --short --ignored
Get-ChildItem -Force
```

To inspect top-level disk usage in PowerShell:

```powershell
Get-ChildItem -Force | ForEach-Object {
    $item = $_
    if ($item.PSIsContainer) {
        $stats = Get-ChildItem -LiteralPath $item.FullName -Recurse -Force -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum
        $bytes = [int64]$stats.Sum
        $count = $stats.Count
    } else {
        $bytes = [int64]$item.Length
        $count = 1
    }
    [pscustomobject]@{
        Name = $item.Name
        Kind = if ($item.PSIsContainer) { "dir" } else { "file" }
        Files = $count
        MB = [math]::Round($bytes / 1MB, 2)
    }
} | Sort-Object MB -Descending | Format-Table -AutoSize
```
