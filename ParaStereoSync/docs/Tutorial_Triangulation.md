# Triangulation Tutorial

Triangulation converts synchronized 2D tracks from two cameras into 3D object coordinates.

1. Required inputs
- Left and right camera parameter text files (from calibration).
- Synchronized tracking CSVs for left and right cameras (matching frame indices and template ordering).

2. Example command

```bash
python triangulatePoints2.py --left left_tracks.csv --right right_tracks.csv --camL camL.txt --camR camR.txt --out 3d_points.csv
```

3. Input formats
- Camera files: use `readCamera.py` / `writeCamera.py` conventions (imgSize, rvec, tvec, cmat, dvec). See `camera_calibration` utilities for exporting.
- Tracking CSVs: must share the same template ordering and frame indexing for correct triangulation. If necessary, reorder columns or synchronize frames using the outputs from `ParaStereoSync.py`.

4. Output
- `3d_points.csv` contains frame-indexed X,Y,Z object coordinates per tracked point. Use `KineVis3D.py` to visualize displacement over time.

5. Accuracy checks
- Reproject triangulated 3D points back into both cameras and examine reprojection error. Large reprojection residuals indicate mismatched correspondences or calibration errors.
