# Getting Started with ParaStereoSync and KinVis3D

This guide shows a minimal workflow to calibrate cameras, run template-based tracking, triangulate to 3D, and visualize results.

1. Prepare the environment

```bash
conda create -n stereosync python=3.10 -y
conda activate stereosync
pip install -r requirements.txt
```

2. Calibrate cameras

- Run the calibration GUI from the `camera_calibration` folder:

```bash
cd camera_calibration
python tkCalib2.py
```

- Use the GUI to load chessboard images or single calibration images, detect corners and save camera parameters (use `writeCamera.py` to export if needed).

3. Select templates and points

- Run the interactive template picker:

```bash
cd ..
python pickTemplates.py
```

- Save the generated template CSV files and note their paths for the tracking step.

4. Run stereo sync and tracking

```bash
python tkStereosync_v2_Parallel.py
```

- The GUI will prompt for input videos, camera parameter files and template lists. Output tracking CSVs will be saved alongside configurable output directories.

5. Triangulate and visualize

```bash
python triangulatePoints2.py
python Ceiling_Profile_viewer_v3.py
```

Notes
- If you prefer non-interactive batch runs, inspect `createFileList.py` and call the tracking module functions directly from scripts.
- For reproducibility, always save calibration parameter text files and the `tkCalib_init_*.npy` state used during calibration.
