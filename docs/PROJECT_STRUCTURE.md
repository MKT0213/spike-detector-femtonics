# Project Structure

The repository has three user-facing programs:

- Analyzer/reviewer: `python detector/analysis/main.py` or compatibility launcher `python main.py`.
- Exporter: `python detector/exporter/main.py`.
- Image processor: `launch_image_processor.bat` or `python -m image_processor`.

## Root

- `README.md` - quick start and high-level usage.
- `main.py` - compatibility launcher for the analyzer/reviewer.
- `launch_image_processor.bat` - Windows launcher for the native image processor.
- `detection_params.json` - default detector parameters loaded by the analyzer.
- `requirements.txt` - Python dependencies for the detector/analyzer/exporter.
- `AGENTS.md` - repository safety rules for agents.
- `.gitignore`, `.gitattributes` - repository metadata.

## Source Folders

- `detector/` - spike detection, review, and export code.
- `detector/core/` - shared detection logic, data models, preprocessing, plotting, and session I/O.
- `detector/analysis/` - analyzer/reviewer entrypoint and analysis-package save/load support.
- `detector/exporter/` - standalone exporter window and export rendering logic.
- `image_processor/` - Femtonics TIFF image-processing Python backend and entrypoint.
- `image_processor/native_viewer/` - native C++/Qt image-processor UI source.
- `image_processor/native_viewer/src/` - native C++ source files.
- `tools/` - local development/build helper scripts.

## Tests

- `tests/` - detector/analyzer/exporter tests.
- `image_processor/tests/` - image-processor Python tests.

## Generated Or Local Output

- `analysis_outputs/` - generated analysis/export output. Do not delete unless you are sure the experiment outputs are disposable.
- `image_processor/native_viewer/build/` - generated native build tree and executable. It can be rebuilt, but deleting it means the image processor must be rebuilt before launch.
- `tmp/`, `__pycache__/`, `*.log`, `*.obj` - disposable local artifacts.

## Documentation

- `docs/` - project structure notes, maintenance notes, and longer planning/research documents.

