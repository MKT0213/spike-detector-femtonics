# Spike Detector / Curation App

A desktop Python application for loading fluorescence traces, detecting spike-like events, reviewing detections, and exporting curated spike results.

The app is built with PySide6 and pyqtgraph, with numerical processing handled by NumPy, SciPy, pandas, PyWavelets, scikit-learn, and related scientific Python packages.

## Features

- Load Excel trace data
- Detect spikes with configurable detector parameters
- Review and curate detected events
- Save and reload curation sessions
- Export CSV summaries and PNG/PDF visualizations
- Reload detector defaults from `detection_params.json`

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Analysis/review program:

```powershell
python detector/analysis/main.py
```

Standalone exporter program:

```powershell
python detector/exporter/main.py
```

`python main.py` is kept as a compatibility launcher for the analysis/review program.

For a plain folder map, see `docs/PROJECT_STRUCTURE.md`.

## Femtonics TIFF Image Processor

For smooth TIFF stack loading, switching, and scrubbing, use the native C++/Qt
image processor UI. The Python analysis modules still run behind the scenes for
exports, motion-correction hooks, ROI-detection hooks, current-TIFF pipeline
runs, and full queue pipeline runs.
The native window has a Python backend field next to the motion/ROI hook fields;
leave it blank to auto-detect Python, or point it at the environment that
contains your analysis packages and custom hook code.
Python backend output is parsed immediately for progress/results, but visible
Process Log updates are batched so noisy hooks do not force thousands of widget
updates in one burst. Partial stdout/stderr chunks are buffered until a newline
or process exit so progress/result parsing sees complete backend lines.
If a pipeline produces a corrected/downstream TIFF, auto-open only changes the
viewer when the user is still looking at the source TIFF; if they browse away
during analysis, the result stays available through the Working TIFF button.

Folder scanning and large queue population are asynchronous/batched, and TIFF
frame reads/rendering use cancellable worker tasks so rapid switching and
scrubbing settle on the latest requested frame instead of blocking the UI.
Opening a TIFF uses a two-stage path: the first frame is previewed after reading
only the first directory, then the full IFD index updates the frame controls in
the background. Preview metadata and full-stack indexing use separate worker
queues so a cancelled or slow index from a previous TIFF does not delay the next
file's first-frame preview.
The native reader supports grayscale uint8, uint16, int16, and float32 TIFF
stacks in strip or tiled layout, including classic TIFF or BigTIFF containers
and deflate-compressed files handled by libtiff.
After loading TIFFs, the Benchmark Queue button in the native UI measures
native metadata, frame-read, and render timings for the current queue and writes
the slowest file/frame summary to the Process Log.
Last Results can open TIFFs, images, text outputs, and folders; folder result
inspection is bounded so selecting a large output folder does not recursively
walk the whole tree on the UI thread.

Useful native responsiveness probes:

```powershell
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-ui path\to\tiff_folder --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-ui path\to\tiff_folder --require-first-display --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-ui path\to\tiff_folder --direct-paths --require-first-display --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-switch-ui --folder path\to\tiff_folder --limit 25 --rounds 4 --interval-ms 1 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-open-latency path\to\movie.tiff --timeout-ms 20000 --max-first-render-ms 1000
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-prefetch-ui path\to\movie.tiff --target-frame 2 --wait-ms 500 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-prefetch-ui path\to\movie.tiff --prime-frame 4 --pivot-frame 3 --target-frame 2 --wait-ms 500 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-scrub-ui path\to\movie.tiff --events 80 --interval-ms 1 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-compat-folder path\to\tiff_folder --limit 100 --full-index
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-perf-folder path\to\tiff_folder --limit 20 --frames-per-stack 3 --max-read-ms 250 --max-render-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-info-cache path\to\movie.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-frame-cache --frames 8 --max-frames 8 --width 2048 --height 2048 --bits 16 --budget-mb 32
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-mainwindow-benchmark path\to\tiff_folder --timeout-ms 30000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-validate-backend-ui --timeout-ms 30000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-current-pipeline-ui path\to\movie.tiff --output path\to\probe_output --timeout-ms 60000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-pipeline-ui path\to\first.tiff path\to\second.tiff --output path\to\queue_probe_output --no-signals --no-split --timeout-ms 60000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-python-average-ui path\to\movie.tiff --timeout-ms 60000 --max-gap-ms 250
```

The batch launcher and Python image-processor entry point will try to build the
native UI automatically if the executable is missing. You can also build it
manually:

```powershell
cmd /v:on /c "call tools\native_dev_env.bat && cmake -S image_processor\native_viewer -B image_processor\native_viewer\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=!QT_ROOT! -DCMAKE_TOOLCHAIN_FILE=!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows && cmake --build image_processor\native_viewer\build"
```

Launch the single native UI:

```powershell
.\launch_image_processor.bat
```

You can pass a TIFF path directly:

```powershell
.\launch_image_processor.bat path\to\movie.tiff
```

The Python image-processor entry point also routes to the native UI when the
environment supports module execution:

```powershell
python -m image_processor path\to\movie.tiff
```

On Windows, prefer `launch_image_processor.bat`. If you specifically need to
launch through Python and a virtual-environment launcher reports an access or
process-start error with `-m`, use the direct runner form instead:

```powershell
python -c "from image_processor.app import main; raise SystemExit(main())" path\to\movie.tiff
```

The older PySide image-processor viewer is still available only for debugging
through `image_processor.app.run_legacy_python_viewer`.

## Program Folders

- `detector/analysis/` contains the analyzer/reviewer entrypoint and the analysis-package save/load format.
- `detector/exporter/` contains the independent exporter program for opening an analysis package and exporting selected outputs.
- `detector/core/` contains shared GUI, detection, plotting, and data-model code used by both detector programs.
- `image_processor/` contains the Femtonics TIFF image processor Python backend and native C++/Qt UI source.
- `image_processor/native_viewer/` contains the native image processor UI.
- `docs/` contains maintenance notes, project-structure notes, and longer planning/research notes.

The usual workflow is to save an analysis package from the analysis/review program, then open that package in the standalone exporter.

## Configuration

Default detector settings live in:

```text
detection_params.json
```

You can adjust these values before launching the app, or use the app's reload button after editing the file.

## Notes For Data

This repository should contain the application source code only. Keep raw experiment data, exported CSV/PNG/PDF files, and saved session files outside git unless you intentionally want to publish example data.
