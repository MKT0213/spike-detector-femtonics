# Native Image Processor UI

This is the native C++/Qt user interface for the Femtonics image processor.
The intent is one visible application window: C++ handles smooth TIFF display,
file switching, scrubbing, and UI responsiveness, while existing Python modules
run headlessly in the background for analysis.

## Build

```bat
call tools\native_dev_env.bat
cmake -S image_processor\native_viewer -B image_processor\native_viewer\build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH="%QT_ROOT%" ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build image_processor\native_viewer\build
```

## Run

```bat
image_processor\native_viewer\build\femtonics_image_processor.exe path\to\movie.tiff
```

The build produces one user-facing executable: `femtonics_image_processor.exe`.
Obsolete compatibility copies are removed during the native build so launch
scripts do not drift back to the earlier prototype shape.

The native window can run the existing Python analysis backend for the current
TIFF through `QProcess`. It probes candidate interpreters before starting a job:
the Python backend path selected in the UI, `SPIKE_PYTHON`, the sibling project
venv, the venv base interpreter declared in `pyvenv.cfg`, a repository-local
venv and its base interpreter, a bundled development fallback when present,
then `python` on `PATH`. When a venv base interpreter is used, the venv
`Lib\site-packages` folder is added to `PYTHONPATH` for that backend process.
The selected interpreter and any failed candidates are written to the Process
Log so backend launch problems are visible without opening a second UI. The
`--probe-python-average-ui` smoke check also accepts `--python` to verify a
specific interpreter path when diagnosing Windows launcher or environment
issues.
The bridge invokes Python through a small `-c` module runner, which avoids
Windows launcher failures seen with `python -m` in some virtual environments.
Python is run unbuffered (`-u` / `PYTHONUNBUFFERED=1`) so analysis logs and
frame-progress updates reach the native UI as they are printed.
Backend output is parsed immediately for progress and result paths, while
visible Process Log rendering is batched into small timed updates so noisy
hooks do not force thousands of text-widget appends in one burst. Partial
stdout/stderr chunks are buffered until a newline or process exit so progress
and result parsing sees complete backend lines.
The analysis panel supports a shared output folder, optional manual sampling
rate override for signal export, CSV on/off, ROI splitting, signal export,
average-projection PNG export, optional hooks for motion correction and ROI
detection, stage toggles for motion correction and segmentation, selectable
pipeline steps for average PNG, signals, and split ROI outputs,
mask-background subtraction mode for hook-produced segmentation masks,
cancellation for long-running analysis, and queue pipeline runs across every
loaded TIFF. The Python backend field is saved with the rest of the UI settings
so custom environments do not need a separate launcher script. Hook fields
accept either `module:function` or
`C:\path\to\method.py:function`; the adjacent picker buttons fill a selected
Python file as `path.py:run`, and the function suffix can be edited in place.
The Built-in buttons fill the bundled first-pass hooks:
`image_processor.motion_correction:motion_hook` and
`image_processor.segmentation_hooks:roi_hook`.
The Motion params and ROI params fields pass extra hook keyword arguments as
semicolon-separated `key=value` entries. Values are parsed as JSON when
possible, so numeric tuning such as `max_shift_px=12`, `min_area=20`,
`foreground_percentile=75`, `threshold_method="brightest"`, `percentile=92.5`,
or `std_factor=2.0` reaches the Python hook as numbers or strings.
Use Validate Backend / Hooks to check the selected Python interpreter, required
backend packages, configured hook paths/functions, and hook parameter syntax
before starting a pipeline run. Pipeline runs also perform the hook preflight
automatically and stop before writing outputs if a configured hook is invalid.
When a motion hook returns a TIFF path using `tiff_path`, `output_tiff`,
`corrected_tiff`, or `downstream_tiff`, downstream average/signal/split steps
use that TIFF. A hook can also return `metadata_path` to pair ROI metadata with
the working TIFF. If a hook returns `ok: false` or fails at runtime, downstream
outputs are skipped for that TIFF.
The bundled motion hook defaults to the adapted shared-template residual method:
`method=shared_template_residual; max_shift_px=2; subpixel=false;
residual_max_shift_px=1; residual_min_peak_ratio=1.10`. It performs a shared
global correction first, then accepts conservative per-rectangle residual shifts
only when the residual match is confident enough.
For a single-frame 2D TIFF, the bundled motion hook intentionally returns
`motion_skipped` plus `*_motion_qc.json` instead of creating a fake corrected
movie; segmentation can still run on that frame.
When an ROI hook returns `mask_path`, the signal step can export mask-based
traces. The Mask background control passes `--mask-background` to the Python
pipeline: `None` keeps raw mask means, `Parent tile` subtracts non-mask pixels
inside the same Femtonics acquisition rectangle, and `Global frame` subtracts
non-mask pixels across the full frame.
The bundled ROI hook defaults to tile-local foreground segmentation inside each
Femtonics acquisition rectangle, using the upper foreground of the average image
(`foreground_percentile=60` by default) instead of only the brightest specks or
the whole tile foreground. The Otsu and legacy high-percentile modes are still
available with `threshold_method="otsu"` or `threshold_method="brightest"`. The
hook also writes a segmentation overlay PNG focused on the detected mask pixels
and boundaries. Parent Femtonics acquisition-tile locations are kept in the
segmentation manifest instead of being drawn over the overlay.
After a run, the Results button opens the expected output folder; for pipeline
runs it follows the emitted manifest path when available. The Processed button
opens the preferred processed display from the manifest: segmentation overlay
first, then segmentation labels, corrected/downstream TIFF, and finally any
different working TIFF. The Last Results list is filled from the same manifest
so generated PNG, workbook, CSV, split folder, corrected TIFF, hook QC files,
segmentation masks/labels, summary images, segmentation overlays, and manifest
outputs are visible in the native window.
When Auto-open processed output is enabled, a successful pipeline run loads the
segmentation overlay if ROI detection ran, or the corrected/downstream TIFF if
motion correction ran without ROI detection.
The visible export controls are consolidated around the current/queue pipeline
buttons. Pipeline runs write a signal-comparison workbook with stable sheets for
raw rectangular ROI traces, raw segmentation-mask traces, motion-corrected
rectangular ROI traces, and motion-corrected segmentation-mask traces. When CSV
signal export is enabled, the same variants are also written as separate CSV
files. These paths are recorded under `signal_comparison_csvs` in the manifest
and appear in Last Results.
Pipeline runs also write QC images: raw average projection, a motion-corrected
average versus raw average panel with a signed heatmap, segmentation overlays
showing mask locations, and a split-ROI label map PNG next to the per-ROI TIFF
stacks.
If the user switches to another TIFF while the Python pipeline is still
running, the working TIFF is left available through the Working TIFF button
instead of pulling the viewer away from the user's current file.
Selecting a TIFF, image, CSV, JSON, or TXT result in Last Results opens it
inside the native image processor UI; folders and binary documents still open through the
operating system. Split result folders that contain TIFF stacks are added to
the native TIFF queue instead of opening externally. Result-folder inspection is
bounded so selecting a large output folder does not recursively walk the whole
tree on the UI thread.
Long-running average, signal, and split operations update the native progress
bar by parsing Python backend progress messages.
One-off Average PNG, Export Signals, and Split TIFF actions also populate Last
Results by parsing their `[OK] ... path` log lines.
Double-clicking a Last Results row, pressing Enter, or using the action button
opens it; the button label changes to the selected result type.
Benchmark Queue runs a native background benchmark over the current TIFF queue,
sampling up to three frames per TIFF and writing metadata, read, render, and
slowest-file/frame timings into the Process Log. It uses the same cancellation
button as the Python analysis actions.

The queue panel supports loading individual TIFFs or folders, removing the
selected TIFF, and clearing the queue without leaving a stale image displayed.
Folder scans run in the background and older scans are cancelled when another
folder is selected, so large recursive experiment folders do not block the UI.
When a TIFF opens, the native reader records its image-file-directory offsets
and reuses them for frame reads and prefetching, which avoids repeated
directory walking during large-stack scrubbing. TIFF opening is two-stage: a
preview metadata read makes frame 1 visible after only the first directory is
read, then full indexing updates the frame slider/count and prefetching in the
background. Preview metadata and full-stack indexing use separate worker queues
so a cancelled or slow index from a previous TIFF does not delay the next
file's first-frame preview. The native reader supports
grayscale uint8, uint16, int16, and float32 TIFF stacks in strip or tiled
layout, including classic TIFF or BigTIFF containers and deflate-compressed
files handled by libtiff. The viewer status and Benchmark Queue report show whether a stack is
classic TIFF or BigTIFF and whether it uses strips or tiles. If the user
switches TIFFs while a previous metadata scan is still running, the old scan is
cancelled between directories and queued stale opens are dropped. Stale
foreground frame loads and prefetches are also cancelled between scanlines or
tiles so rapid scrubbing can move on to the latest requested frame sooner.
Full TIFF metadata is cached in memory by absolute path, file size, and
modification time, so switching back to a recently opened stack can reuse its
IFD offsets without a second full directory walk.
Frame-to-display grayscale
conversion runs on a dedicated render worker, so large-frame scaling does not
run on the UI thread during normal frame display. Integer grayscale frames use
per-render lookup tables for contrast mapping, which keeps 1024/2048 px frame
rendering cheap while preserving adjustable black/white levels. Cached frame
sample buffers are moved from the TIFF worker into shared immutable storage,
then reused by the cache, current display state, and render worker to avoid
duplicating large pixel arrays during handoff. Frame status shows separate read
and render timings to help diagnose real-file latency. The frame cache is
bounded by both frame count and byte budget, so very large TIFF frames evict
older cached frames instead of allowing memory use to grow with a fixed number
of frames.

The native UI remembers session preferences such as output folder, hook specs,
sampling/CSV settings, pipeline step toggles, recursive folder scanning, and
window geometry. TIFF queue contents are intentionally not restored on startup,
so each processing batch is selected deliberately.

For automated smoke checks:

```bat
image_processor\native_viewer\build\femtonics_image_processor.exe --probe path\to\movie.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-sequence path\to\first.tiff path\to\second.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-metadata path\to\movie.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-folder-scan path\to\folder
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-perf-folder path\to\folder --limit 20 --frames-per-stack 3 --max-read-ms 250 --max-render-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-info-cache path\to\movie.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-frame-cache --frames 8 --max-frames 8 --width 2048 --height 2048 --bits 16 --budget-mb 32
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-ui path\to\folder --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-ui path\to\folder --require-first-display --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-ui path\to\folder --direct-paths --require-first-display --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-switch-ui --folder path\to\folder --limit 25 --rounds 4 --interval-ms 1 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-open-latency path\to\movie.tiff --timeout-ms 20000 --max-first-render-ms 1000
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-prefetch-ui path\to\movie.tiff --target-frame 2 --wait-ms 500 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-prefetch-ui path\to\movie.tiff --prime-frame 4 --pivot-frame 3 --target-frame 2 --wait-ms 500 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-viewer-scrub-ui path\to\movie.tiff --events 80 --interval-ms 1 --timeout-ms 20000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-compat-folder path\to\folder --limit 100 --full-index
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-mainwindow-benchmark path\to\folder --timeout-ms 30000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-validate-backend-ui --timeout-ms 30000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-validate-backend-ui --motion-hook path\to\motion.py:run --roi-hook package.module:detect
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-current-pipeline-ui path\to\movie.tiff --output path\to\probe_output --timeout-ms 60000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-current-pipeline-ui path\to\movie.tiff --roi-hook image_processor.segmentation_hooks:roi_hook --mask-background parent_tile --timeout-ms 60000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-current-pipeline-ui path\to\movie.tiff --no-signals --no-split --motion-hook path\to\motion.py:run --expect-working-tiff path\to\corrected.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-queue-pipeline-ui path\to\first.tiff path\to\second.tiff --output path\to\queue_probe_output --no-signals --no-split --timeout-ms 60000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-python-average-ui path\to\movie.tiff --timeout-ms 60000 --max-gap-ms 250
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-scrub path\to\movie.tiff 0 5 1 6
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-frame-access path\to\movie.tiff 999
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-render-frame path\to\movie.tiff 999
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-info-cancel path\to\movie.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-frame-cancel path\to\movie.tiff 999
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-manifest-working path\to\native_pipeline_manifest.json path\to\source.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-manifest-display path\to\native_pipeline_manifest.json path\to\source.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-manifest-results path\to\native_pipeline_manifest.json
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-progress-line "movie.tiff: signal frames 25/100"
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-result-open-mode path\to\result.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-result-action-label path\to\result.tiff
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-image path\to\average_projection.png
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-text path\to\native_pipeline_manifest.json
image_processor\native_viewer\build\femtonics_image_processor.exe --probe-result-log-line "[OK] movie.tiff: path\to\movie_average_projection.png"
```
