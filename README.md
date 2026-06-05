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

```powershell
python main.py
```

## Configuration

Default detector settings live in:

```text
detection_params.json
```

You can adjust these values before launching the app, or use the app's reload button after editing the file.

## Notes For Data

This repository should contain the application source code only. Keep raw experiment data, exported CSV/PNG/PDF files, and saved session files outside git unless you intentionally want to publish example data.
