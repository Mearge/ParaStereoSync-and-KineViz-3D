# Tracking Tutorial

This tutorial shows how to prepare templates and run ECC-based tracking for one or two cameras.

1. Prepare templates

```bash
python pickTemplates.py
```

- Use the GUI to zoom and select template ROIs around points of interest. Save the produced CSV file(s).
- Aim for template sizes that are large enough to contain distinguishing texture but small enough to avoid excessive deformation (typical: 32×32 to 128×128 px).

Alternative:
- Use `TempArUco.py` to generate template CSV files automatically from common ArUco markers across multi-camera videos.

2. Run the stereo sync & tracking GUI

```bash
python ParaStereoSync.py
```

- In the GUI: select left/right video files, camera parameter text files, and template CSVs.
- Configure tracking parameters: frame range, pyramid levels, max warp, and quality thresholds.
- Start tracking. For multi-camera processing, enable multiprocessing and choose an appropriate worker count (e.g., number of CPU cores).

3. Batch / programmatic mode
- Import `eccTrackVideo` from `eccTrackVideo_Multistep_warp_guess.py` in a script to run non-interactively. Example:

```python
from eccTrackVideo_Multistep_warp_guess import eccTrackVideo
eccTrackVideo(videoFilepath='data/cam1.mp4', tmpltFilepath='templates/cam1.csv', saveFilepath='results/cam1_tracks.csv')
```

4. Interpreting output
- Output tracking CSVs include frame index, x,y image coordinates per template, and quality metrics. Inspect tracks visually by overlaying points on frames with `drawPoints.py`.

5. Troubleshooting
- If templates drift, try increasing pyramid levels or reducing maximum allowed warp per frame.
- If many templates fail early, check template contrast and cross-check template center alignment.
