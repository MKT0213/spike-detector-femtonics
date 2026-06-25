# Image Processor Motion Correction and Segmentation Plan

Created: 2026-06-20

Source memory: `VOLPY_PIPELINE_REVIEW_MEMORY.md`

## Objective

Add a practical movie-level preprocessing layer to the existing Femtonics TIFF
image processor so a 2D TIFF stack can be motion corrected, summarized,
segmented into masks, converted into traces, and then analyzed by the existing
trace detector.

The first implementation should be local, small, and testable. VolPy/CaImAn
should guide the architecture, but the project should not start by copying or
installing the full CaImAn stack.

## Core Adaptation Need

The main problem is not simply missing motion correction. The current Femtonics
workflow captures and exports activity through rectangular ROIs, but the desired
workflow is neuron-shaped segmentation.

That means the adaptation must change the data model:

- keep rectangular Femtonics ROI processing as a legacy path;
- keep each Femtonics rectangle as an acquisition tile / coordinate container;
- add arbitrary soma/dendrite masks as first-class ROI objects;
- extract traces from mask pixels inside those acquisition tiles, not from the
  whole rectangular tile mean;
- allow weighted masks for processes or uneven fluorescence;
- preserve the corrected TIFF movie so segmentation and trace extraction happen
  before pixels are averaged into one-dimensional traces.

This is why a simple ROI splitter improvement is not enough. The implementation
needs a mask-aware branch beside the existing rectangular branch.

## Femtonics Acquisition Geometry

The metadata file is important and should remain central. The current image
processor reconstructs the packed rectangular ROI mosaic from metadata fields:

- `Width` and `Height`: full exported TIFF mosaic size.
- `RoiCount`: number of rectangular acquisition ROIs.
- `RoiSize`: width/height of each acquisition rectangle.
- `RoiIndex`: acquisition ROI labels/order.
- `Channel`, `DataType`, `Comment`, and sampling-rate hints.

In `image_processor/roi_core.py`, `compute_roi_layout()` assumes equal-sized
ROIs that evenly tile the exported TIFF dimensions, then creates row-major
`RoiBox` objects with `left`, `upper`, `right`, and `lower` crop coordinates.
The native viewer mirrors this in `image_processor/native_viewer/src/NativeMetadata.cpp`.

For our adaptation, these rectangles are not the final biological ROIs. They
are the containers that tell us where each acquired patch lives inside the TIFF
mosaic. Segmentation should happen within each container, or on a mask mapped
back to a container, so every neuron-shaped mask carries:

- parent Femtonics `roi_index`;
- parent tile crop box in mosaic coordinates;
- local mask coordinates inside the tile;
- optional global mosaic mask coordinates for overlays/QC.

This also affects motion correction. A packed multi-ROI mosaic may not be a
continuous physical field of view, so a single global rigid shift over the whole
mosaic can be misleading. The safer first design is tile-aware:

- parse metadata into `RoiLayout`;
- crop each rectangular acquisition tile as a local stack;
- estimate/apply motion correction per tile, or estimate a shared shift only
  when QC shows the tiles move coherently;
- generate per-tile summary images;
- segment neurons/processes inside each tile;
- extract mask traces from the corrected tile pixels.

## Scope Rule: 2D Frame vs 2D Stack

The user target was described as a "2D TIFF image." This must be split into two
cases before implementation:

- Single 2D TIFF frame: segmentation/ROI marking is meaningful; motion
  correction, trace extraction, and spike extraction are not meaningful.
- 2D TIFF stack/movie, shaped `T x H x W`: motion correction, segmentation,
  mask-aware trace extraction, background correction, and detector handoff are
  meaningful.

The implementation should detect `frame_count == 1` and either skip motion
correction with a clear manifest note or run segmentation-only behavior.

## Current Codebase Facts

The image processor now has two internal engines:

- Image processing engine: `image_processor/`
  - Python backend for TIFF metadata parsing, splitting, average PNG export,
    signal export, pipeline orchestration, motion hooks, and ROI hooks.
  - Called from CLI modules such as `image_processor.native_pipeline`,
    `image_processor.export_signals`, and `image_processor.split_multi_roi_tiff`.
- Image viewer engine: `image_processor/native_viewer/`
  - Native C++/Qt UI built as `femtonics_image_processor.exe`.
  - Owns fast TIFF loading, frame caching, queue display, scrubbing, native
    metadata overlays, result opening, and the visible pipeline controls.
  - Runs the Python processing engine through `QProcess`.

Implementation rule: motion correction, segmentation, and mask-aware trace
extraction belong first in the Python image processing engine. The native image
viewer should expose controls, previews, overlays, and results for those backend
artifacts, but it should not own the scientific algorithm logic.

The best integration point already exists:

- `image_processor/native_pipeline.py` has `--motion-hook` and `--roi-hook`.
- `image_processor/pipeline_hooks.py` supports `module:function` and
  `file.py:function` hooks.
- A motion hook can return `corrected_tiff`, `tiff_path`, `output_tiff`, or
  `downstream_tiff`; downstream average PNG, signal export, and split export
  then use that corrected TIFF.
- `image_processor/native_viewer/src/MainWindow.cpp` exposes hook fields, hook validation,
  current-TIFF pipeline runs, queue pipeline runs, and auto-open of the
  hook-produced working TIFF.

Current hook wiring contract:

- Launch app through `launch_image_processor.bat`.
- In the native UI, wire motion correction through the Motion hook field as
  `path\to\hook_file.py:function_name`.
- Wire segmentation / ROI detection through the ROI hook field as
  `path\to\hook_file.py:function_name`.
- Built-in first-pass motion hook spec:
  `image_processor.motion_correction:motion_hook`.
- Built-in first-pass ROI/segmentation hook spec:
  `image_processor.segmentation_hooks:roi_hook`.
- Hook functions may accept any subset of `tiff_path`, `output_dir`, `stage`,
  `record`, and `**kwargs`.
- Hook params can be supplied from the native Motion params / ROI params fields
  as semicolon-separated `key=value` entries, or from the backend CLI as
  repeatable `--motion-hook-param KEY=VALUE` and `--roi-hook-param KEY=VALUE`.
  Values are JSON-parsed when possible and stored in the pipeline manifest.
- A motion hook should return at least
  `{"ok": True, "corrected_tiff": "path/to/corrected.tiff"}`.
- A segmentation / ROI hook should return at least
  `{"ok": True, "metadata_path": "path/to/roi_metadata.txt",
  "detected_roi_count": N}`.
- `native_pipeline.py` also accepts TIFF keys `tiff_path`, `output_tiff`,
  `output_tiff_path`, `corrected_tiff_path`, and `downstream_tiff`, and metadata
  keys `output_metadata` and `roi_metadata_path`.

Important constraint: the current downstream signal/split exporters consume
metadata through the existing Femtonics-style metadata parser. If segmentation
outputs neuron-shaped masks rather than rectangular metadata, the backend must
either generate a compatible metadata artifact for the current path or extend
`native_pipeline.py` with a mask-aware export branch.

The current limitation is also clear:

- `image_processor/signal_export.py` computes mean raw intensity inside each
  rectangular ROI.
- `image_processor/roi_core.py` assumes Femtonics metadata is an evenly tiled
  rectangular grid.
- `image_processor/native_viewer/src/NativeMetadata.cpp` and `TiffViewerWidget.cpp` render that
  same rectangular metadata overlay.
- Arbitrary soma or dendrite segmentation masks do not fit the current metadata
  model.

Dependency reality:

- `image_processor/requirements.txt` currently contains only `numpy`,
  `Pillow`, `tifffile`, `zarr`, `openpyxl`, and `PySide6`.
- First-pass code should not assume SciPy, OpenCV, scikit-image, PyTorch,
  Cellpose, Suite2p, EXTRACT, or CaImAn.

## How the 8 Papers Should Be Used

| Paper | Use in this implementation |
| --- | --- |
| Cai et al. 2021, VolPy | Primary architecture reference: motion correction, summary images, ROI masks, movie-aware trace/spike extraction. Use SpikePursuit later after masks and corrected movies exist. |
| Xie et al. 2021, SGPMD-NMF | Design pressure for background subtraction, bleaching handling, and QC before trusting traces. Do not implement full SGPMD-NMF first. |
| Gonzalez et al. 2026, 3D-RTMC | QC warning: 2D offline registration cannot fix z drift/focus artifacts and can leave motion-induced false voltage events. |
| Platisa et al. 2023, DeepVID | Later denoising reference for low-light voltage movies. First pass should register before any denoising and keep denoising optional. |
| Evans et al. 2023, ASAP4 | Practical baseline: rigid registration, manual/background ROIs, bleach correction, and trace analysis are acceptable first workflow pieces. |
| Lee/Lin ASAP6c manuscript | Modern hybrid workflow reference: NoRMCorre plus EXTRACT/Cellpose/manual masks plus VolPy/SpikePursuit, depending on data domain. |
| Russell et al. 2022 | Scaling reference: repeated TIFF IO can bottleneck large jobs; keep chunked iteration now and consider zarr/memmap intermediates later. |
| Carolan et al. 2025 | Dendrite/process mask reference: support weighted masks, running-window dF/F, and two-pass baseline/event cleanup later. |

## Proposed Pipeline

1. Load TIFF as either one frame or a `T x H x W` movie.
2. Motion-correct the movie before reducing pixels to traces.
3. Build summary images from the corrected movie.
4. Segment or import masks on corrected summary images.
5. Extract mask-aware traces from the corrected movie.
6. Apply local background and optional baseline correction.
7. Export a detector-compatible workbook/CSV.
8. Feed traces into the existing detector GUI or analysis package workflow.

## Phase 1: Local Rigid Motion Correction

Add `image_processor/motion_correction.py`.

First-pass algorithm:

- Load frames through `iter_tiff_frames()`.
- Build a reference image from a robust mean of selected frames.
- Estimate per-frame integer shifts by phase correlation using `numpy.fft`.
- Limit accepted shifts with `max_shift_px`.
- Apply same-size integer shifts with an explicit border fill policy.
- Write the corrected stack with `save_tiff_array_stack()`.
- Copy the source `.metadata.txt` beside the corrected TIFF so current
  downstream exports still work.

First-pass hook:

- Use the implemented module function
  `image_processor.motion_correction:motion_hook`.
- For multi-frame stacks, return `{"ok": true, "corrected_tiff": "...",
  "qc_json": "...", "shifts_csv": "..."}` so the existing native pipeline uses
  the corrected TIFF immediately.
- For a single still 2D TIFF, return `{"ok": true, "motion_skipped": true,
  "qc_json": "..."}` without `corrected_tiff`; segmentation can still run on
  the original frame.

QC outputs:

- `*_motion_shifts.csv`: frame, dx, dy, peak score for multi-frame stacks.
- `*_motion_qc.json`: frame count, layout source, max shift, tile count, and
  skipped reason for single-frame TIFFs.
- Optional `*_motion_reference.png` and `*_motion_average_before_after.png`.

Later improvements:

- Subpixel shifting with optional SciPy/OpenCV.
- Iterative template refinement.
- Piecewise-rigid correction if residual deformation remains high.
- Optional high-pass preprocessing for weak landmarks or one-photon background.

## Phase 2: Summary Images

Implemented as `image_processor/summary_images.py`.

First summaries:

- Mean projection.
- Standard deviation projection.
- Max projection.
- Optional local correlation image when memory allows.

Outputs:

- `*_summary_mean.png`.
- `*_summary_std.png`.
- `*_summary_max.png`.
- `*_summary_manifest.json`.

These images become the user-facing segmentation surface and the QC surface for
motion correction.

## Phase 3: Mask Artifact Contract

Do not encode arbitrary masks as fake rectangular Femtonics metadata.

Add a separate mask format:

- `*_masks.npz`
  - `masks`: shape `N x H x W`, dtype `bool` or float weights.
  - `labels`: ROI names.
  - optional `centroids`, `boxes`, `source_tiff`, `source_summary`.
- `*_mask_manifest.json`
  - mask source, creation method, image shape, ROI count, labels, parameters,
    and any manual curation notes.

First segmentation sources:

- Imported masks from `.npz`.
- Simple threshold plus connected components on summary images, implemented
  locally as `image_processor.segmentation_hooks:roi_hook`.
- Manual masks later through native viewer or an external editor workflow.

Current first-pass segmentation output:

- `*_segmentation_masks.npz` with `masks`, `label_image`, and `labels`.
- `*_segmentation_labels.tiff` for overlay/QC.
- `*_segmentation_overlay.png` for fast visual QC: colored detected mask pixels
  and boundaries over the mean projection. Femtonics acquisition-tile locations
  stay in the manifest instead of being drawn over this image.
- `*_segmentation_manifest.json` with parent Femtonics tile, local/global
  bounding boxes, centroid, area, threshold metadata, and per-tile detection
  counts.
- Summary artifacts are returned beside segmentation artifacts:
  `*_summary_mean.png`, `*_summary_std.png`, `*_summary_max.png`, and
  `*_summary_manifest.json`.
- It returns the existing rectangular metadata path for compatibility. This
  keeps rectangular downstream tools available while the pipeline can use the
  returned `mask_path` for mask-aware traces.

Deferred segmentation sources:

- Cellpose for soma-like cell bodies.
- EXTRACT for dense one-photon voltage fields.
- Suite2p-style registration/ROI workflows when calcium-style assumptions fit.
- VolPy Mask R-CNN only if data/domain and model weights match well enough.

## Phase 4: Mask-Aware Trace Export

Implemented as `image_processor/mask_signal_export.py`.

Core behavior:

- Load corrected TIFF stack through `iter_tiff_frames()`.
- Load masks from `*_masks.npz` or `label_image`.
- For each ROI, compute either binary mean or weighted mean intensity.
- Optionally subtract background through `--mask-background`.
  - `none`: raw mask mean, default.
  - `parent_tile`: mean of non-mask pixels inside the parent Femtonics
    acquisition rectangle.
  - `global`: mean of non-mask pixels across the frame.
- Export workbook and CSV in a detector-compatible shape:
  `Frame`, `Time_ms`, `ROI01`, `ROI02`, ...

Workbook differences from current rectangular export:

- Metadata row should say mask-weighted or binary-mask extraction.
- Add `Mask_Map` sheet instead of, or in addition to, `ROI_Map`.
- Add a `Background` sheet when background subtraction is enabled.
- Record mask file, source corrected TIFF, segmentation method, background
  method, and polarity/baseline settings.

Important: keep `signal_export.py` stable for existing rectangular Femtonics
metadata workflows. Add the mask-aware path beside it rather than replacing it.

## Phase 5: Native Pipeline Integration

Implemented for the hook-produced mask path.

Current behavior:

- Accept `mask_path` from ROI hook results.
- Store mask paths in the manifest.
- If a mask path exists and signal export is enabled, run mask-aware export
  instead of rectangular `export_signal_documents()`.
- Keep rectangular export as the fallback when no mask path exists.

Later CLI options if useful:

  - `--mask-path`
  - `--mask-signal-mode`

Implemented CLI option:

- `--mask-background {none,parent_tile,global}`

This preserves the current hook strategy while making segmentation actually
usable downstream.

## Phase 6: UI Work

Backend first, then UI.

Implemented native UI additions:

- A "Use masks for signals" result when ROI hook returns a mask file.
- Built-in hook fill buttons for
  `image_processor.motion_correction:motion_hook` and
  `image_processor.segmentation_hooks:roi_hook`.
- A Mask background selector that passes
  `--mask-background {none,parent_tile,global}` to the Python backend for
  current-TIFF and queue pipeline runs.
- Motion params and ROI params fields for tuning hook kwargs such as
  `max_shift_px`, `min_area`, `percentile`, `std_factor`, and
  `max_area_fraction`.
- Last Results entries for `*_masks.npz`, mask manifest, summary images,
  segmentation overlay, and motion QC.
- Manifest result discovery normalizes hook artifact paths so relative CLI runs
  and native UI runs can both surface generated files reliably.

Remaining native UI additions:

- Native in-canvas mask overlay rendering beyond the generated QC PNG.
- Manual mask drawing/editing.
- Summary-image view mode for segmentation.
- Per-mask trace preview and accept/reject curation.

## Phase 7: VolPy SpikePursuit Later

Only consider VolPy `spikepursuit.py` after these are working:

- Corrected TIFF movie.
- ROI masks.
- Mask-aware traces.
- Background/QC metadata.
- Confirmed indicator polarity.

SpikePursuit is useful for movie-aware spike timing, spatial weights, local
background modeling, and matched filtering. It is not useful for a single still
image and should not be the first dependency introduced.

## Test Plan

Motion correction tests:

- Synthetic shifted stack with known integer shifts.
- No-motion stack remains unchanged.
- Single-frame TIFF returns `skipped` or equivalent QC without failure.
- Motion hook returned corrected TIFF is used downstream by native pipeline.
- Metadata is copied beside the corrected TIFF.

Summary image tests:

- Mean/std/max projections match known synthetic stacks.
- Mixed-shape frame errors are clear.

Mask export tests:

- Binary mask extraction returns known frame means.
- Weighted mask extraction returns known weighted means.
- Empty masks and shape mismatches fail clearly.
- Workbook metadata identifies mask-based extraction, not rectangular ROI
  extraction.

Pipeline tests:

- ROI hook returns `mask_path`; pipeline records it in manifest.
- Mask path triggers mask-aware signal export.
- No mask path keeps current rectangular export behavior.
- Hook failure still skips downstream outputs.

Native viewer checks:

- Hook validation still works.
- Working TIFF auto-open still works after motion correction.
- Last Results includes corrected TIFF, QC JSON/CSV, summary images, masks, and
  exported signals.

## First Implementation Recommendation

Start with a Python-only backend slice:

1. `image_processor/motion_correction.py`
2. tests for synthetic rigid shifts
3. hook integration through existing `--motion-hook`
4. summary images from corrected TIFFs
5. mask artifact loader plus mask-aware trace export

This gives immediate value for TIFF stacks while keeping the design honest:
VolPy guides the pipeline, but current project constraints decide the first
implementation.
