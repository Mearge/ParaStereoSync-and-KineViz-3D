# API Reference

This reference lists the primary public entry points used in workflows and integration scripts.

## Main Software APIs

## `ParaStereoSync.py`

- `ParaStereoSync()`
	- Launches the main stereo synchronization and tracking GUI.

- `parallel_eccTrackCamera(args)`
	- Camera-level multiprocessing wrapper around `eccTrackVideo`.

- `parallel_eccTrackVideo(args)`
	- Legacy wrapper for parallel video-level tracking jobs.

- `parallel_eccTrackPoint(args)`
	- Point-level multiprocessing wrapper for fine-grained tracking.

## `KineVis3D.py`

Class: `CeilingProfileViewer`

Common methods:
- `setup_ui(self)`
- `load_file(self)`
- `process_data(self)`
- `start_point_picking(self)`
- `update_frame(self, value)`
- `show_3d_view(self)`
- `show_2d_profile(self)`
- `show_animation(self)`
- `save_animation(self)`
- `show_selected_profile(self)`
- `show_profile_evolution(self)`
- `save_profile_evolution(self)`
- `plot_time_series(self)`
- `calculate_rigid_body_motion(self)`

## Core Processing Modules

## `eccTrackVideo_Multistep_warp_guess.py`

- `eccTrackVideo(videoFilepath=None, tmpltFilepath=None, tmpltFrameId=0, frameRange=None, tmpltRange=None, mTable=None, saveFilepath=None)`
	- ECC tracker with multistep fallback logic and warp-guess prediction.
	- Output table layout per point: `x, y, rotation, correlation, processing_time`.

## `triangulatePoints2.py`

- `triangulatePoints2(cmat1, dvec1, rvec1, tvec1, cmat2, dvec2, rvec2, tvec2, imgPoints1, imgPoints2)`
	- Returns triangulated 3D points in multiple coordinate frames and reprojection diagnostics.

## Supporting Tool APIs

## `tkCalib2.py`

- `tkCalib()`
	- Main calibration GUI.

- `tkCalib_printMessage(msg: str)`
	- Timestamped logging utility used by the GUI.

## `TempArUco.py`

Main workflow functions:
- `preprocess_for_detection(gray)`
- `detect_markers_in_frame(frame, aruco_dict, detector)`
- `process_video_with_frame_priority(video_path)`

Purpose:
- Detect common ArUco markers across selected camera videos.
- Export per-camera template CSV and annotated marker image.

## `VidTrim.py`

Primary utility functions:
- `trim_video()`
- `track_and_plot()`
- `track_aruco_and_plot()`
- `batch_track_and_plot()`
- `batch_track_aruco_and_plot()`

Purpose:
- Trim videos losslessly with FFmpeg.
- Run quick single/batch tracking diagnostics from a GUI.

## Data I/O Utilities

- `readCamera.py`: `readCamera(...)`
- `writeCamera.py`: `writeCamera(...)`, `cameraParametersToString(...)`
- `readPoints.py`: `readPoints(...)`, chessboard helpers
- `writePoints.py`: `writePoints(...)`
- `pickTemplates.py`: interactive template selection
- `pickPoints.py`: interactive point selection

