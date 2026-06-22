# ParaStereoSync & KineVis-3D

Release version: 1.0.0

ParaStereoSync and KineVis3D is a Python toolkit for stereo-video motion measurement, combining camera calibration, synchronized template tracking, 3D triangulation, and 3D displacement visualization.

This repository is organized around two main applications and three supporting tools for publication and experimental workflows.

## Main Software

1. `ParaStereoSync.py`
- Main stereo workflow GUI.
- Loads two videos, calibration files, and template files.
- Runs ECC tracking (including parallel execution options).
- Exports tracking results (`bigTable`) as CSV for downstream analysis.

2. `KineVis3D.py`
- 3D visualization and analysis GUI.
- Loads dense CSV output (including triangulated/displacement columns).
- Provides 3D view, profile extraction, time-series plotting, animation, and rigid-body motion estimation.

## Supporting Tools

1. `VidTrim.py`
- GUI utility for lossless video trimming via FFmpeg.
- Includes single/batch tracking utilities (ECC and ArUco mode) for quick diagnostics.

2. `TempArUco.py`
- Multi-camera ArUco-based template generator.
- Detects common marker IDs across selected camera videos.
- Exports per-camera template CSV files and annotated preview images.

3. `tkCalib2.py` (your requested `tkCalib.py` tool)
- Camera calibration GUI.
- Supports single-image and chessboard workflows.
- Saves reusable camera parameters and calibration initialization state.

## Typical Workflow

1. Calibrate each camera:
```bash
python tkCalib2.py
```
2. Create template files:
- Manually with `pickTemplates.py`, or
- Automatically from ArUco markers with:
```bash
python TempArUco.py
```
3. Run stereo synchronization and tracking:
```bash
python ParaStereoSync.py
```
4. Triangulate matched 2D tracks to 3D points:
```bash
python triangulatePoints2.py
```
5. Explore and export 3D motion results:
```bash
python KineVis3D.py
```

## Installation

Recommended environment (Conda):

```bash
conda create -n stereosync python=3.10 -y
conda activate stereosync
pip install -r requirements.txt
```

Core dependencies are listed in `requirements.txt`:
- `numpy`
- `scipy`
- `pandas`
- `opencv-python`
- `matplotlib`
- `scikit-image`
- `tqdm`

Additional external dependency:
- `ffmpeg` (required for trimming in `VidTrim.py`)

## Inputs and Outputs

### Inputs
- Stereo video files (`.mp4`, `.avi`, ...)
- Camera parameter files written by `writeCamera.py` / `tkCalib2.py`
- Template definition CSV files (`xi, yi, x0, y0, w, h`)

### Main Outputs
- 2D tracking tables (`np.savetxt` CSV from `ParaStereoSync.py` or `eccTrackVideo_Multistep_warp_guess.py`)
- 3D triangulated point tables (via `triangulatePoints2.py` workflow)
- Figures/animations/profile exports from `KineVis3D.py`

### Calibration State Files
`tkCalib2.py` may produce reusable files such as:
- `tkCalib_init_coord3d.npy`
- `tkCalib_init_coordImg.npy`
- `tkCalib_init_imgSize.npy`
- `tkCalib_init_cmatGuess.npy`
- `tkCalib_init_dvecGuess.npy`
- `tkCalib_init_cboardParams.npy`
- `tkCalib_init_calibFlags.npy`

## File Format Notes

1. Camera parameter files
- Read/write helpers: `readCamera.py`, `writeCamera.py`
- Include image size, extrinsics (`rvec`, `tvec`), intrinsics (`cmat`), distortion (`dvec`)

2. Template files
- Common format: one row per target with six values
- Semantic order used by the tracker: `xi, yi, x0, y0, w, h`

3. Tracking tables (`mTable`)
- Stored as 5 columns per tracked point:
- `x`, `y`, `rotation_deg`, `correlation`, `time_sec`

## Documentation Map

Complete documentation is in `docs`:
- `docs/index.md` (entry page)
- `docs/Getting_Started.md`
- `docs/Installation.md`
- `docs/Tutorial_Calibration.md`
- `docs/Tutorial_Tracking.md`
- `docs/Tutorial_Triangulation.md`
- `docs/Tutorial_Visualization.md`
- `docs/API_Reference.md`
- `docs/Developer_Guide.md`
- `docs/manuscript_for_Software_X.md`

## Reproducibility Checklist (Publication)

For each reported experiment, archive:

1. Raw video files (left/right, with frame-rate metadata).
2. Camera calibration files for each camera.
3. Template files for each camera.
4. Tracking output CSV files.
5. Triangulated 3D CSV files.
6. Software commit hash and dependency snapshot.
7. Key GUI settings (frame ranges, tracked point ranges, solver-related options).

## License and Citation

- License file: `LICENSE`
- Citation metadata is provided in `CITATION.cff`
