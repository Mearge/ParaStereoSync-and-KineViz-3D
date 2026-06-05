# Calibration Tutorial

Step-by-step: generate accurate camera intrinsics and extrinsics used by the pipeline.

1. Prepare calibration images
- Use a printed chessboard or symmetric dot grid. Recommended chessboard inner corners: 7x9 or 9x6 depending on print size.
- Acquire images covering the full field of view and different orientations.

2. Launch the calibration GUI

```bash
python tkCalib2.py
```

3. Workflow in the GUI
- Load image set or single image mode.
- Detect chessboard corners (use automatic detection; manually refine if needed).
- Set the real-world square size (in mm) for accurate scale.
- Run optimization and examine reprojection error per image.
- Save camera parameters using the GUI `Save` action. The GUI writes `tkCalib_init_*.npy` state and can export camera parameter text files using `writeCamera.py`.

4. Verify calibration
- Reproject 3D chessboard points onto sample images and visually inspect overlay.
- Check mean reprojection error; values <0.5 px are excellent, <1.5 px typically acceptable for many engineering tasks.

5. Common tips
- If some images have large reprojection errors, remove them and re-run optimization.
- Ensure lens distortion is modeled with enough coefficients — for fisheye lenses, a different model may be required.
