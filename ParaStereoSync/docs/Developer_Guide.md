# Developer Guide

Project structure overview

- `ParaStereoSync.py` — main GUI and orchestration
- `eccTrackVideo_Multistep_warp_guess.py` — tracking core
- `triangulatePoints2.py` — stereo triangulation
- `camera_calibration/` — calibration GUI and utilities
- `pickTemplates.py`, `pickPoints.py` — data preparation tools

Coding conventions

- Use descriptive function names; prefer explicitly typed arguments where helpful. Keep GUI code separated from processing logic where possible.

Running and debugging

- To step through processing functions, import them into a small Python script and run under an interactive debugger (e.g., VS Code, PyCharm).
- For unit-like checks, create small video snippets (10–50 frames) and run the tracking functions to validate expected outputs.

Extending the tracker

- The tracking pipeline expects template CSVs with ROI definitions. To add a new tracker algorithm, implement a function with the same input/output signature as `eccTrackVideo()` and call it from `tkStereosync_v2_Parallel.py` or provide as a plugin.

Tests

- There are no formal unit tests in this repo. Recommended: add `pytest` and write tests for `readCamera.py`, `triangulatePoints2.py` and `eccTrackVideo` using small synthetic datasets.
