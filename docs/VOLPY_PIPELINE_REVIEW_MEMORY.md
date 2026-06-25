# VolPy Pipeline Review Memory

Created: 2026-06-20

Purpose: durable project memory for future motion correction and segmentation work on 2D TIFF image stacks in this repository.

## Current Decision

VolPy is useful as a reference pipeline, but the full CaImAn/VolPy stack should not be copied into this project as the first implementation step.

The core adaptation need is to move beyond Femtonics rectangular ROI capture.
The current image processor can split and average rectangular ROI tiles, but the
desired workflow is neuron-shaped segmentation: soma or dendrite masks should
be treated as the real ROIs, and traces should be extracted from those masks
after motion correction. Rectangular ROI support should remain as a legacy path,
not define the new segmentation model.

Important nuance: the Femtonics rectangles should not be discarded. The metadata
file describes how the rectangular acquisition ROIs are packed/aligned into the
exported TIFF mosaic. In this project, `RoiSize`, `RoiCount`, `RoiIndex`,
`Width`, and `Height` are parsed into a row-major `RoiLayout`. For the new
workflow, each rectangle is the parent acquisition tile/coordinate container;
the biological ROI should be a mask inside that tile. A mask record should keep
its parent `roi_index`, parent crop box, local tile coordinates, and optional
global mosaic coordinates.

This also means motion correction should be tile-aware first. A packed
multi-ROI TIFF mosaic may not be one continuous physical field of view, so a
single global rigid shift across the whole mosaic can be misleading. Safer
first-pass processing is: parse metadata, crop each rectangular acquisition
tile, motion-correct/segment/extract inside each tile, then export mask traces.

For our target, the most valuable transfer is the pipeline order:

1. Load TIFF stack as a movie, shaped `T x H x W`.
2. Motion-correct frames while spatial information is still present.
3. Build summary images from the corrected movie.
4. Segment ROIs or load/manual-correct masks.
5. Extract traces from masks with local background handling.
6. Feed extracted traces into the existing detector.

If the target is a single 2D TIFF frame, motion correction is not meaningful. Segmentation can still be done on the image, but VolPy spike extraction and most voltage analysis require a time series.

## Local Project Context

The current repo is primarily a trace-level detector. It already has a TIFF processing path, but that path reduces image stacks to rectangular ROI mean-intensity traces.

The image processor has two internal engines:

- `image_processor/`: Python processing backend for metadata parsing, ROI
  splitting, average/signal exports, native pipeline orchestration, and motion
  or ROI hooks.
- `image_processor/native_viewer/`: native C++/Qt image viewer built as
  `femtonics_image_processor.exe`; it owns fast TIFF loading/scrubbing, queue
  display, native metadata overlays, result opening, and UI controls. It calls
  the Python backend through `QProcess`.

Implementation implication: scientific adaptations such as tile-aware motion
correction, segmentation, and mask-aware trace extraction should land first in
`image_processor/`. `image_processor/native_viewer/` should expose controls and visual feedback
for those artifacts, not duplicate the algorithm logic.

Current hook wiring contract:

- Launch app with `launch_image_processor.bat`.
- Motion hook field: `path\to\hook_file.py:function_name`.
- ROI hook field: `path\to\hook_file.py:function_name`.
- Motion/ROI params fields pass semicolon-separated `key=value` entries as
  typed hook kwargs; values are JSON-parsed when possible. Useful first-pass
  tuning examples: `max_shift_px=12`, `min_area=20`, `percentile=92.5`,
  `std_factor=2.0`, and `max_area_fraction=0.4`.
- Built-in first-pass motion hook: `image_processor.motion_correction:motion_hook`.
- Built-in first-pass ROI/segmentation hook:
  `image_processor.segmentation_hooks:roi_hook`.
- Hook functions may accept `tiff_path`, `output_dir`, `stage`, `record`, and
  `**kwargs`.
- Motion hook expected return:
  `{"ok": True, "corrected_tiff": "path/to/corrected.tiff"}`.
- ROI/segmentation hook expected return:
  `{"ok": True, "metadata_path": "path/to/roi_metadata.txt",
  "detected_roi_count": N}`.
- `native_pipeline.py` copies returned metadata beside the working TIFF before
  downstream average/signal/split exports.

Implemented first-pass hook behavior:

- `image_processor.motion_correction` performs metadata-aware tile-wise integer
  phase-correlation correction, writes `*_motion_corrected.tiff`,
  `*_motion_qc.json`, and `*_motion_shifts.csv`, and copies source metadata
  beside the corrected TIFF for multi-frame stacks. For a single still 2D TIFF
  it writes QC JSON with `motion_skipped: true` and does not return a
  `corrected_tiff`, because temporal motion correction is not meaningful.
- `image_processor.summary_images` streams TIFF stacks into explicit mean,
  standard-deviation, and max-projection PNGs plus `*_summary_manifest.json`.
- `image_processor.segmentation_hooks` uses the summary mean image, segments
  connected bright components inside each metadata tile, writes
  `*_segmentation_masks.npz`, `*_segmentation_labels.tiff`,
  `*_segmentation_overlay.png`, and `*_segmentation_manifest.json`, returns
  summary artifact paths, and returns the original rectangular metadata path so
  legacy downstream exporters remain stable. The overlay focuses on the
  neuron-like mask pixels and boundaries; Femtonics acquisition rectangle
  locations remain available in the segmentation manifest.
- `image_processor.mask_signal_export` reads `N x H x W` masks or a
  `label_image`, exports detector-compatible mask traces, and writes a
  `Mask_Map` workbook sheet. It supports raw mask means by default plus
  optional `parent_tile`/`global` background subtraction with a `Background`
  workbook sheet.
- `native_pipeline.py` now records returned `mask_path` and
  `mask_manifest_path`; when a mask exists and signal export is enabled, it
  writes `*_mask_signals.xlsx`/CSV from mask pixels instead of rectangular ROI
  means.
- `image_processor/native_viewer/src/MainWindow.cpp` exposes a Mask background selector and
  passes `--mask-background` for both current-TIFF and queue pipeline runs.
- The same native UI exposes Built-in buttons that fill
  `image_processor.motion_correction:motion_hook` and
  `image_processor.segmentation_hooks:roi_hook`.
- The native UI and backend CLI now pass validated hook parameters through
  `--motion-hook-param KEY=VALUE` and `--roi-hook-param KEY=VALUE`, and the
  pipeline manifest records those params for reproducibility.
- Native manifest result discovery now includes hook artifacts such as motion
  QC JSON/CSV, corrected TIFFs, segmentation mask `.npz`, label TIFFs,
  segmentation manifests, overlay PNGs, and summary PNG/JSON files in Last
  Results.

Key local files:

- `image_processor/native_pipeline.py`
  - Already has `motion_hook` and `roi_hook` stages.
  - Best integration point for motion correction and segmentation before exports.
  - Uses mask-aware signal export when a ROI hook returns `mask_path`.
- `image_processor/signal_export.py`
  - Currently calculates "Mean raw pixel intensity inside each rectangular ROI".
  - It uses metadata-driven rectangular grid ROIs.
  - It remains the compatibility path when no mask is available.
- `image_processor/mask_signal_export.py`
  - Calculates mean raw intensity inside binary or weighted segmentation masks.
  - Produces detector-compatible `Frame`, `Time_ms`, ROI-label columns plus
    `Metadata` and `Mask_Map` workbook sheets.
- `image_processor/summary_images.py`
  - Produces `*_summary_mean.png`, `*_summary_std.png`,
    `*_summary_max.png`, and `*_summary_manifest.json` for segmentation/QC.
- `image_processor/roi_core.py`
  - `RoiMetadata`, `RoiBox`, and `compute_roi_layout()` assume evenly tiled rectangular ROIs.
  - Arbitrary segmentation masks are carried by separate mask artifacts, not by
    fake rectangular metadata.
- `detector/core/gui_main.py`
  - Runs the trace analysis pipeline after signals are loaded.
- `detector/core/detection.py`
  - Detects plateau and spike events from one-dimensional traces.
- `detector/core/gonzalez_denoising.py`
  - Trace-level CWT/PCA/Ward style denoising aid.
  - This is not movie-level motion correction or segmentation.

Important architectural implication: motion correction and segmentation must happen upstream of `signal_export.py`, before traces are averaged out of the movie.

## Reference Folder

Reference materials were found outside the repo:

`C:\Users\makat\Desktop\Spike detector\Volpy reference`

Reviewed PDFs:

- Cai et al. 2021, *VolPy: Automated and scalable analysis pipelines for voltage imaging datasets*.
- Xie et al. 2021, *High-fidelity estimates of spikes and subthreshold waveforms from 1-photon voltage imaging in vivo*.
- Russell et al. 2022, *All-optical interrogation of neural circuits in behaving mice*.
- Platisa et al. 2023, *High-speed low-light in vivo two-photon voltage imaging of large neuronal populations*.
- Evans et al. 2023, *A positively tuned voltage indicator for extended electrical recordings in the brain*.
- Lee/Lin ASAP6c manuscript, *Large-scale long-read voltage imaging of spatial activity patterns in the brain*, stored as `LeeLin_Nature_inconfidence.pdf`.
- Carolan et al. 2025, *All-optical voltage interrogation for probing synaptic plasticity in vivo*.
- Gonzalez et al. 2026, *Movement-stabilized three-dimensional optical recordings of membrane potential changes and calcium dynamics in hippocampal CA1 dendrites*.

Extracted paper text files were generated under:

`tmp/volpy_paper_texts/`

## Accuracy Audit - 2026-06-20

I rechecked the reference folder, the extracted-text index, and the local image
processor code. The memory is accurate for implementation planning, with one
important scope rule: statements about VolPy/CaImAn are being used as design
guidance, not as a promise that we should vendor or install the whole CaImAn
stack first.

Extraction audit:

| Paper | Extracted? | Pages | Characters |
| --- | --- | ---: | ---: |
| Cai et al. 2021, VolPy | Yes | 27 | 93,394 |
| Xie et al. 2021, SGPMD-NMF | Yes | 15 | 67,837 |
| Russell et al. 2022 | Yes | 70 | 202,271 |
| Platisa et al. 2023, DeepVID | Yes | 28 | 94,777 |
| Evans et al. 2023, ASAP4 | Yes | 33 | 130,428 |
| Lee/Lin ASAP6c manuscript | Yes | 46 | 143,058 |
| Carolan et al. 2025 | Yes | 12 | 74,091 |
| Gonzalez et al. 2026, 3D-RTMC | Yes | 25 | 135,927 |

Code audit:

- `image_processor/native_pipeline.py` already supports `--motion-hook` and
  `--roi-hook`. A motion hook can return a corrected TIFF through keys such as
  `corrected_tiff`, `tiff_path`, or `downstream_tiff`, and downstream average,
  signal, and split exports will use that working TIFF.
- `image_processor/pipeline_hooks.py` loads hooks from either
  `module:function` or `file.py:function`, then calls them with `stage`,
  `tiff_path`, `output_dir`, and `record` when accepted by the hook signature.
- `image_processor/signal_export.py` currently exports mean raw intensity from
  rectangular ROIs only.
- `image_processor/roi_core.py` parses Femtonics metadata into an evenly tiled
  rectangular grid. It does not represent arbitrary soma/dendrite masks.
- `image_processor/native_viewer/src/MainWindow.cpp` and `image_processor/native_viewer/README.md` already expose
  hook fields and working-TIFF handling in the native UI.
- `image_processor/requirements.txt` is intentionally small:
  `numpy`, `Pillow`, `tifffile`, `zarr`, `openpyxl`, `PySide6`. First-pass
  motion correction and mask extraction should avoid assuming SciPy, OpenCV,
  scikit-image, PyTorch, or CaImAn are installed.

Resulting correction to keep in mind:

- Motion correction is meaningful for a 2D TIFF stack/movie, shaped
  `T x H x W`. For a single still 2D TIFF frame, only segmentation/ROI marking
  is meaningful.
- Segmentation should not be forced into the existing rectangular Femtonics
  metadata format. The current implementation uses separate mask artifacts and
  `image_processor.mask_signal_export` for mask-aware trace export.

## Eight-Paper Use Checklist

All eight papers were used, but not in the same way. The algorithmic papers should guide implementation; the indicator/acquisition papers mainly constrain preprocessing choices and QC.

| Paper | Main use for this project | Should drive first implementation? |
| --- | --- | --- |
| Cai et al. 2021, VolPy | Defines the voltage-imaging pipeline order: motion correction, summary images, masks, movie-aware trace/spike extraction. | Yes, as architecture and later SpikePursuit reference. |
| Xie et al. 2021, SGPMD-NMF | Explains why local background, bleaching, motion, and blood/structured artifacts must be modeled before trusting traces. | Yes, for background/QC design; full SGPMD-NMF later. |
| Gonzalez et al. 2026, 3D-RTMC | Shows the limits of 2D offline registration when z-motion/focus changes create false voltage events. | Yes, for QC warnings and residual-motion interpretation. |
| Platisa et al. 2023, DeepVID | Shows low-light voltage denoising strategy and a workflow of rigid correction, denoising, manual ROIs, and traces. | Not first; useful if shot noise dominates. |
| Evans et al. 2023, ASAP4 | Documents practical TIFF/ROI workflows: NoRMCorre or pyStackReg, manual ROIs, background subtraction, bleach correction. | Yes, as pragmatic baseline validation. |
| Lee/Lin ASAP6c manuscript | Shows modern hybrid processing: NormCorre, EXTRACT/Cellpose/manual ROIs, then VolPy/SpikePursuit for spikes. | Yes, for tool-selection strategy. |
| Russell et al. 2022 | Warns that TIFF IO can bottleneck large pipelines; supports memmap/binary/zarr intermediate design. | Not algorithmically, but yes for scaling decisions. |
| Carolan et al. 2025 | Gives dendritic mask and event extraction ideas: Suite2p registration, weighted masks, running-window dF/F, two-pass baseline. | Yes if target structures are dendrites/processes. |

Implementation priority from the eight papers:

1. Cai + Xie define the movie-level preprocessing architecture.
2. Evans + Lee/Lin validate practical ROI and hybrid-tool workflows.
3. Gonzalez defines what 2D motion correction cannot guarantee.
4. Carolan guides dendrite/process masks and event cleanup.
5. Platisa and Russell guide later denoising/scaling choices.

## Paper-by-Paper Memory

### Cai et al. 2021 - VolPy

Most important paper for the pipeline shape.

Pipeline:

1. CaImAn/NoRMCorre motion correction.
2. Memory mapping.
3. Summary images: mean image plus local correlation image.
4. Segmentation with Mask R-CNN, GUI/manual masks, or imported masks.
5. Spike extraction with a SpikePursuit-derived algorithm.

Motion correction:

- Uses CaImAn implementation of NoRMCorre.
- Supports rigid and piecewise-rigid correction.
- The paper reports rigid correction was sufficient for their tested datasets, likely because fields of view were small.
- `gSig_filt` spatial high-pass filtering can help one-photon data when landmarks are weak or background is high.

Segmentation:

- Mask R-CNN takes a 3-channel summary image: mean, mean, local correlation.
- It works well on similar-looking training data but is not universal.
- The paper reports strong performance on L1 data and weaker performance on more difficult HPC/TEG data.
- The authors explicitly recommend GUI/manual correction or retraining on different datasets.

Spike extraction:

- Requires ROI masks and the full movie, not just pre-extracted traces.
- For each ROI, it crops a local context window.
- It flips signal if needed for negative-going indicators.
- It high-pass filters to remove photobleaching.
- It estimates local background from pixels outside the ROI using SVD/PCA.
- It subtracts background by ridge regression.
- It detects candidate spikes, builds a spike template, then applies a whitened matched filter.
- It refines spatial weights and spike times iteratively.
- It outputs traces, spike times, reconstructed signal, subthreshold signal, SNR, spatial weights, F0, and dF/F.

Use for us:

- Very useful as a design reference.
- `MotionCorrect` and summary-image logic are directly relevant.
- `spikepursuit.py` is useful later if spike timing from movie data becomes central.
- Mask R-CNN is too heavy and dataset-sensitive for the first segmentation implementation unless we already have compatible trained weights and a similar data domain.

### Xie et al. 2021 - SGPMD-NMF

Most important paper for noisy one-photon voltage movies and background contamination.

Pipeline:

1. In-plane motion correction with NoRMCorre.
2. Pixel-wise photobleach correction using spline detrending.
3. PMD denoising to suppress temporally uncorrelated shot noise while preserving spatially contiguous signal.
4. Motion decorrelation using NoRMCorre x/y shifts and polynomial regressors.
5. Optional heartbeat/blood vessel artifact handling.
6. localNMF demixing for cell footprints.
7. Joint demixing of cells and smooth background components.

Key insight:

- Simple ROI averaging can preserve background contamination.
- Background can be correlated with cell signals, so ICA-style independence assumptions can fail.
- Subthreshold waveform accuracy needs explicit background modeling.

Use for us:

- Strong argument to add local background subtraction early.
- Good design reference if we need subthreshold fidelity, not only spike/event timing.
- Full SGPMD-NMF is probably too complex for the first implementation.

### Gonzalez et al. 2026 - 3D Real-Time Motion Correction

Most important warning about limits of offline 2D registration.

Key points:

- Their system uses 3D real-time motion correction with reference cells and orthogonal AO drifts.
- It corrects x, y, and z movement during acquisition.
- They show motion can mimic rhythmic or burst-like voltage activity, especially in small dendritic structures.
- Current 2D motion correction cannot compensate out-of-plane z motion because it lacks z-axis fluorescence information.

Use for us:

- Offline 2D correction can align frames in x-y.
- It cannot fully fix focus changes, z drift, or intensity modulation caused by axial motion.
- QC should report residual motion and warn that corrected-looking images can still contain motion-induced trace artifacts.

### Platisa et al. 2023 - DeepVID

Most relevant for low-light two-photon voltage imaging and denoising.

Key points:

- Voltage imaging needs high frame rates, which pushes data near shot-noise limits.
- DeepVID is a self-supervised deep learning denoiser based on DeepInterpolation/Noise2Void ideas.
- It predicts a central frame from nearby frames while hiding blind pixels.
- It improved SNR without sacrificing voltage spike temporal dynamics in their setting.
- Their in vivo workflow: 2D rigid FFT motion correction, DeepVID denoising, manual ROI segmentation, trace extraction.

Use for us:

- Useful later if low SNR becomes the dominant problem.
- Not a first-pass requirement because it needs a substantial model/training/inference setup.
- Simpler denoising and background subtraction should come first.
- Do not use DeepVID as a substitute for motion correction or segmentation; it sits after registration and before/around trace extraction.

### Evans et al. 2023 - ASAP4 Indicator Paper

Mostly a biology/indicator paper, but useful for practical processing norms. It should not be treated as a segmentation algorithm paper.

Observed workflows:

- TIFF stacks analyzed in Fiji with manual cellular ROIs and background ROIs.
- In vivo stacks first registered with NoRMCorre.
- ROIs drawn by hand in ImageJ/Fiji.
- Mean background ROI subtracted at each timepoint.
- Some TIFF conversion workflows used pyStackReg-based rigid registration.

Use for us:

- Validates a pragmatic baseline: rigid motion correction, manual or semi-automatic ROIs, background subtraction, trace analysis.
- Does not provide a new segmentation algorithm.

### Lee/Lin ASAP6c Manuscript - Large-Scale Long-Read Voltage Imaging

Important modern hybrid workflow.

Key points:

- Uses NoRMCorre/NormCorre for lateral motion correction.
- Uses EXTRACT for automated cell extraction in large dense one-photon imaging.
- Uses Cellpose plus manual correction in some two-photon contexts.
- Imports EXTRACT pixel weights into VolPy processing in the large-scale MPOP workflow.
- Uses VolPy/SpikePursuit for spike locations/SNR in several analyses.
- Some workflows used background subtraction and normalization without extra denoising/bleach correction, so preprocessing should stay modular rather than forced globally.

Use for us:

- Strong evidence that modern workflows mix tools:
  - NoRMCorre for motion correction.
  - EXTRACT/Cellpose/manual correction for segmentation.
  - VolPy/SpikePursuit for voltage spike extraction.
- This supports not treating VolPy Mask R-CNN as the only segmentation path.

### Russell et al. 2022 - All-Optical Protocol

Useful for practical high-throughput imaging workflow.

Key points:

- Avoids slow TIFF read/write overhead during online workflows by streaming raw binary data.
- Uses real-time motion-corrected binary files.
- Uses Suite2p or stimulus-triggered average images for ROI identification.
- Memory-mapping speeds ROI segmentation and trace extraction.
- Manual ROI curation remains part of the workflow.

Use for us:

- For large TIFF stacks, repeated full-stack TIFF IO will become a bottleneck.
- A future optimized pipeline should consider zarr, memmap, or binary intermediate storage.
- For now, `tifffile` plus chunked iteration is acceptable for a first implementation.

### Carolan et al. 2025 - All-Optical Synaptic Plasticity

Useful for dendritic mask handling and event extraction.

Key points:

- Raw images registered with Suite2p.
- Dendrites manually identified from mean registered images.
- 2D masks smoothed with a Gaussian filter, normalized, then cubed to suppress low-intensity pixels.
- Dendritic segments clustered by trace correlation.
- Slow photobleaching handled with running-window dF/F.
- Spike/event detection used baseline filtering and a two-pass procedure where detected spikes were removed before recalculating baseline.

Use for us:

- Weighted masks can improve trace SNR compared with hard binary masks.
- A two-pass baseline/spike procedure is useful when events contaminate baseline estimation.

## CaImAn/VolPy Code Map

Reference code folder:

`C:\Users\makat\Desktop\Spike detector\Volpy reference\CaImAn-main`

Relevant files:

- `demos/general/demo_pipeline_voltage_imaging.py`
  - Main practical workflow.
  - Shows motion correction, memmap, summary image generation, segmentation options, and `VOLPY.fit()`.
- `caiman/motion_correction.py`
  - `MotionCorrect`
  - `motion_correct_oneP_rigid`
  - `motion_correct_oneP_nonrigid`
  - `register_translation`
  - `tile_and_correct`
- `caiman/source_extraction/volpy/volparams.py`
  - VolPy data, motion, and spike parameter groups.
- `caiman/source_extraction/volpy/volpy.py`
  - `VOLPY.fit()` dispatches each ROI to `spikepursuit.volspike`.
- `caiman/source_extraction/volpy/spikepursuit.py`
  - Main trace extraction/spike detection implementation.
  - Per-ROI context crop, background SVD, ridge regression, adaptive thresholding, whitened matched filter, spatial weight refinement.
- `caiman/source_extraction/volpy/mrcnn/`
  - Current local source uses PyTorch Mask R-CNN.
  - Heavy dependency path.

## Recommended First Implementation Path

Do not start by installing all of CaImAn.

Start with a smaller, local implementation that fits this repo:

1. Add a motion correction hook for TIFF stacks.
   - Input: TIFF stack.
   - Output: corrected TIFF stack plus shift/QC JSON or CSV.
   - First method: rigid phase-correlation translation.
   - Later method: piecewise-rigid if residual deformation remains high.
2. Add summary image generation.
   - Mean image.
   - Standard deviation image.
   - Max/mean local correlation image if feasible.
3. Add a segmentation path.
   - First pass: manual or simple semi-automatic segmentation on corrected summary images.
   - Consider Cellpose/EXTRACT later if installation and data domain justify it.
   - Avoid starting with Mask R-CNN unless we confirm compatible weights and data similarity.
4. Add mask-aware trace extraction.
   - Accept binary or weighted masks.
   - Extract mean or weighted average trace.
   - Add local annulus/context background subtraction.
   - Preserve ROI labels and mask metadata.
5. Feed mask-extracted traces into the existing detector.
6. Only then consider VolPy SpikePursuit integration for spike timing and subthreshold extraction.

Current first-pass status: items 1, 2, 3, and the basic binary/weighted part of
4 are implemented in `image_processor/`. A first local background mode
(`parent_tile`, using non-mask pixels inside the parent Femtonics rectangle) is
also implemented. Annulus/context masks, SVD/ridge background regression,
subpixel/piecewise motion correction, and UI mask curation remain later work.

## First-Pass Motion Correction Design

Use rigid 2D registration for the first pass:

- Build a reference image from a robust average or median of selected frames.
- Estimate per-frame shifts against reference by phase correlation.
- Apply subpixel shifts where available.
- Save corrected TIFF.
- Save shift traces and QC:
  - x/y shift over time.
  - max absolute shift.
  - frame-to-reference correlation before/after.
  - border/crop policy.

Important parameters:

- `max_shift_px`: expected maximum movement.
- `reference_mode`: mean, median, or iterative template.
- `high_pass_sigma`: optional spatial high-pass for weak landmarks/high background.
- `border_mode`: copy/reflect/crop/nan.

## First-Pass Segmentation Design

Because current local metadata assumes rectangular grids, mask segmentation needs its own format.

Recommended artifact format:

- `*_summary_mean.tif` or png.
- `*_summary_corr.tif` or png.
- `*_masks.npz` containing:
  - `masks`: shape `N x H x W`, boolean or float weights.
  - `labels`: ROI names.
  - optional `centroids`, `boxes`, `source_tiff`, `created_at`.
- `*_mask_manifest.json` with human-readable metadata.

Trace extraction should support:

- binary masks.
- weighted masks.
- local background ring/context mask.
- optional neuropil/background coefficient.

## Risks and Open Questions

- Is the user target a single 2D TIFF frame or a TIFF stack/movie?
  - Single frame: segmentation only.
  - Stack: motion correction and trace extraction.
- What indicator polarity is used?
  - VolPy uses `flip_signal=True` for Voltron-like negative-going indicators.
  - Positive-going indicators should not be flipped.
- Does the data contain rectangular Femtonics multi-ROI tiles or a full FOV movie?
  - Current repo expects rectangular ROI tiling.
  - VolPy-style segmentation expects full FOV frame geometry.
- Are there z-motion/focus artifacts?
  - 2D registration cannot solve these fully.
- Is segmentation goal soma/cell bodies or dendrites/processes?
  - Dendritic masks may need weighted/segmented line masks instead of cell-body masks.
- Are there enough frames and spikes for VolPy/SpikePursuit?
  - VolPy spike extraction needs enough events to build a template.

## Practical Answer to "Is VolPy Helpful Here?"

Yes, but selectively.

For motion correction:

- Use CaImAn/NoRMCorre ideas.
- A small local rigid phase-correlation hook is the right first step.
- Full CaImAn can come later if local correction is insufficient.

For segmentation:

- VolPy Mask R-CNN is conceptually useful but risky as the first solution because it is trained-domain dependent and dependency-heavy.
- Start with summary images plus manual/semi-automatic masks.
- Consider Cellpose, EXTRACT, Suite2p, or retrained Mask R-CNN later.

For spike extraction:

- VolPy `spikepursuit.py` is very helpful if we want movie-aware spike timing with local background subtraction and matched filtering.
- It requires a corrected TIFF movie and ROI masks.
- It is not useful for a single still image.

Bottom line: build a local movie-level preprocessing layer first, then decide whether to integrate VolPy spike extraction.
