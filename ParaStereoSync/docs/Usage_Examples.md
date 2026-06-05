# Usage Examples

## Example 1: Run Main Stereo GUI

```bash
python ParaStereoSync.py
```

Use the GUI to select:
1. Video file 1 and video file 2.
2. Calibration file 1 and calibration file 2.
3. Template file 1 and template file 2.

Then configure ranges and run tracking.

## Example 2: Programmatic Single-Camera ECC Tracking

```python
import numpy as np
from eccTrackVideo_Multistep_warp_guess import eccTrackVideo

video = r"data/cam1.mp4"
templates = r"data/cam1_targets.csv"
frame_range = [0, 500]
tmplt_range = np.arange(0, 20)

eccTrackVideo(
    videoFilepath=video,
    tmpltFilepath=templates,
    tmpltFrameId=0,
    frameRange=frame_range,
    tmpltRange=tmplt_range,
    mTable=None,
    saveFilepath=r"results/cam1_tracks.csv",
)
```

## Example 3: Generate Template CSV Automatically from ArUco Markers

```bash
python TempArUco.py
```

Workflow:
1. Select 2 or more camera videos.
2. Select output folder.
3. Tool exports one target CSV per video and annotated marker images.

## Example 4: Visualize and Analyze 3D Data

```bash
python KineVis3D.py
```

In GUI:
1. Load CSV.
2. Auto-detect points.
3. Render 3D profile.
4. Export time evolution plots/animations.

## Example 5: Trim Source Videos Before Processing

```bash
python VidTrim.py
```

Use `Trim Video` to cut long videos (lossless stream copy) before tracking.

## Output Location Notes

- `ParaStereoSync.py` tracking outputs are saved to user-selected paths.
- `TempArUco.py` writes CSVs and PNG previews to the selected output folder.
- `KineVis3D.py` export files are written to user-selected save paths.
