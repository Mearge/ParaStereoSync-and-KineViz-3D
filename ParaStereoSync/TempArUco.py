import cv2 
import numpy as np 
import pandas as pd 
import os
from tkinter import Tk, filedialog

# =========================
# USER INPUT (GUI)
# =========================
# Initialize tkinter root window (hidden)
root = Tk()
root.withdraw()  # Hide the main window
root.attributes('-topmost', True)  # Bring dialog to front

print("Please select 2 or more video files (multi-camera)...")
video_paths = list(filedialog.askopenfilenames(
    title="Select Video Files (2+ cameras)",
    filetypes=[
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
        ("All files", "*.*")
    ]
))

if len(video_paths) < 2:
    raise RuntimeError("Please select at least 2 video files. Exiting.")

for idx, path in enumerate(video_paths, start=1):
    print(f"✅ Camera {idx:02d} video: {path}")

print("Please select the output folder for results...")
output_folder = filedialog.askdirectory(
    title="Select Output Folder for CSV and Images"
)

if not output_folder:
    raise RuntimeError("No output folder selected. Exiting.")

print(f"✅ Output folder: {output_folder}")

root.destroy()  # Clean up tkinter

def sanitize_name(name):
    """Keep filenames safe and readable."""
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ('-', '_'):
            keep.append(ch)
        else:
            keep.append('_')
    cleaned = ''.join(keep).strip('_')
    return cleaned if cleaned else "camera"

camera_labels = []
for idx, path in enumerate(video_paths, start=1):
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = sanitize_name(stem)
    camera_labels.append(f"cam{idx:02d}_{safe}")

aruco_dict_type = cv2.aruco.DICT_6X6_1000
template_padding = 5  # pixels

# =========================
# HELPER FUNCTION
# =========================
def preprocess_for_detection(gray):
    """Apply multiple preprocessing techniques for better marker detection"""
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise while preserving edges (reduced strength for faster processing)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=7, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpen the image to enhance marker edges
    kernel_sharpening = np.array([[-1,-1,-1], 
                                   [-1, 9,-1], 
                                   [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpening)
    
    return sharpened

def detect_markers_in_frame(frame, aruco_dict, detector):
    """Detect markers in a single frame using multiple preprocessing methods"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = preprocess_for_detection(gray)
    
    all_corners = []
    all_ids = []
    
    # Try detection with different preprocessing approaches
    # First attempt: with enhanced preprocessing
    corners, ids, _ = detector.detectMarkers(processed)
    if ids is not None:
        all_corners.extend(corners)
        all_ids.extend(ids.flatten())
    
    # Second attempt: with original grayscale
    corners2, ids2, _ = detector.detectMarkers(gray)
    if ids2 is not None:
        for i, marker_id in enumerate(ids2.flatten()):
            if marker_id not in all_ids:
                all_corners.append(corners2[i])
                all_ids.append(marker_id)
    
    # Third attempt: with histogram equalization
    equalized = cv2.equalizeHist(gray)
    corners3, ids3, _ = detector.detectMarkers(equalized)
    if ids3 is not None:
        for i, marker_id in enumerate(ids3.flatten()):
            if marker_id not in all_ids:
                all_corners.append(corners3[i])
                all_ids.append(marker_id)
    
    # Fourth attempt: Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners4, ids4, _ = detector.detectMarkers(otsu)
    if ids4 is not None:
        for i, marker_id in enumerate(ids4.flatten()):
            if marker_id not in all_ids:
                all_corners.append(corners4[i])
                all_ids.append(marker_id)
    
    # Fifth attempt: Adaptive Gaussian thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    corners5, ids5, _ = detector.detectMarkers(adaptive_thresh)
    if ids5 is not None:
        for i, marker_id in enumerate(ids5.flatten()):
            if marker_id not in all_ids:
                all_corners.append(corners5[i])
                all_ids.append(marker_id)
    
    return all_ids, all_corners, processed

def process_video_with_frame_priority(video_path):
    """Process frame 0 first, then scan other frames for missing markers"""
    cap = cv2.VideoCapture(video_path)
    
    # Frame 0 is primary, then scan these for missing markers
    frame_indices = [0, 10, 20, 30, 40, 50, 60, 65]
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

    # Optimized detector parameters
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 100
    params.cornerRefinementMinAccuracy = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 33
    params.adaptiveThreshWinSizeStep = 3
    params.adaptiveThreshConstant = 5
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.5
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.04
    params.minDistanceToBorder = 2
    params.markerBorderBits = 1
    params.minOtsuStdDev = 3.0
    params.perspectiveRemovePixelPerCell = 8
    params.perspectiveRemoveIgnoredMarginPerCell = 0.1
    params.maxErroneousBitsInBorderRate = 0.35
    params.errorCorrectionRate = 0.6

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    
    marker_data = {}  # {marker_id: {'frame': frame_idx, 'corners': corners}}
    frame0 = None
    frame0_marker_ids = set()
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        if frame_idx == 0:
            frame0 = frame
        
        # Detect markers in this frame
        ids, corners, processed = detect_markers_in_frame(frame, aruco_dict, detector)
        
        if len(ids) > 0:
            print(f"  Frame {frame_idx}: Found {len(ids)} markers - IDs: {sorted([int(i) for i in ids])}")
            
            for marker_id, c in zip(ids, corners):
                marker_id = int(marker_id)
                
                # Only add if not already detected (frame 0 has priority)
                if marker_id not in marker_data:
                    # Refine corners
                    cv2.cornerSubPix(processed, c, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
                    
                    marker_data[marker_id] = {
                        'frame': frame_idx,
                        'corners': c.reshape(4, 2)
                    }
                    
                    if frame_idx == 0:
                        frame0_marker_ids.add(marker_id)
    
    cap.release()
    
    all_marker_ids = set(marker_data.keys())
    not_in_frame0 = all_marker_ids - frame0_marker_ids
    
    print(f"  Total unique markers: {len(all_marker_ids)} - IDs: {sorted(all_marker_ids)}")
    print(f"  Markers in frame 0: {len(frame0_marker_ids)} - IDs: {sorted(frame0_marker_ids)}")
    if not_in_frame0:
        print(f"  Markers NOT in frame 0: {len(not_in_frame0)} - IDs: {sorted(not_in_frame0)}")
    
    return marker_data, frame0, frame0_marker_ids

# =========================
# PROCESS ALL VIDEOS
# =========================
camera_results = []

for idx, (video_path, label) in enumerate(zip(video_paths, camera_labels), start=1):
    print(f"\nProcessing camera {idx:02d}: {label}")
    marker_data, frame0, frame0_ids = process_video_with_frame_priority(video_path)
    camera_results.append({
        'index': idx,
        'label': label,
        'video_path': video_path,
        'marker_data': marker_data,
        'frame0': frame0,
        'frame0_ids': frame0_ids,
        'ids': set(marker_data.keys())
    })

# =========================
# FIND COMMON MARKERS ACROSS ALL CAMERAS
# =========================
common_ids = sorted(set.intersection(*[cam['ids'] for cam in camera_results]))

if len(common_ids) == 0:
    raise RuntimeError("No common ArUco markers detected across all selected videos.")

print(f"\n✅ {len(common_ids)} common markers will be included in CSV for every camera")

# Build per-camera marker dictionaries restricted to common IDs
for cam in camera_results:
    cam['markers'] = {}
    cam['not_frame0_ids'] = set()
    for marker_id in common_ids:
        cam['markers'][marker_id] = cam['marker_data'][marker_id]['corners']
        if marker_id not in cam['frame0_ids']:
            cam['not_frame0_ids'].add(marker_id)

# Print frame-origin diagnostics per marker and camera
for marker_id in common_ids:
    marker_sources = []
    for cam in camera_results:
        frame_num = cam['marker_data'][marker_id]['frame']
        marker_sources.append(f"{cam['label']}:F{frame_num}")
    if any(cam['marker_data'][marker_id]['frame'] != 0 for cam in camera_results):
        print(f"  Marker {marker_id}: " + ', '.join(marker_sources))


# =========================
# DRAW MARKERS ON FRAME 0
# =========================
def draw_markers_on_frame0(frame, markers, marker_data, frame0_ids, not_frame0_ids):
    """Draw markers on frame 0 with color coding:
    - Green: markers detected in frame 0
    - Orange: markers NOT detected in frame 0 (found in later frames)"""
    annotated = frame.copy()
    
    # Draw green markers (from frame 0)
    for marker_id in frame0_ids:
        if marker_id in markers:
            pts = markers[marker_id]
            pts_int = pts.astype(int)
            
            # Green polylines and corners
            cv2.polylines(annotated, [pts_int], True, (0, 255, 0), 2)
            for point in pts_int:
                cv2.circle(annotated, tuple(point), 4, (0, 0, 255), -1)
            
            center = pts.mean(axis=0).astype(int)
            text = f"ID: {marker_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Black background
            cv2.rectangle(annotated,
                (center[0] - 5, center[1] - text_height - 5),
                (center[0] + text_width + 5, center[1] + baseline + 5),
                (0, 0, 0), -1)
            
            cv2.putText(annotated, text, tuple(center), font, font_scale, (255, 255, 255), thickness)
    
    # Draw orange markers (NOT from frame 0)
    for marker_id in not_frame0_ids:
        if marker_id in markers:
            pts = markers[marker_id]
            pts_int = pts.astype(int)
            
            # Orange polylines and corners
            cv2.polylines(annotated, [pts_int], True, (0, 165, 255), 2)
            for point in pts_int:
                cv2.circle(annotated, tuple(point), 4, (0, 100, 255), -1)
            
            center = pts.mean(axis=0).astype(int)
            frame_num = marker_data[marker_id]['frame']
            text = f"ID: {marker_id} (F{frame_num})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Orange background
            cv2.rectangle(annotated,
                (center[0] - 5, center[1] - text_height - 5),
                (center[0] + text_width + 5, center[1] + baseline + 5),
                (0, 100, 200), -1)
            
            cv2.putText(annotated, text, tuple(center), font, font_scale, (255, 255, 255), thickness)
    
    return annotated

# Draw on frame 0 for each camera
for cam in camera_results:
    frame0_common = set(common_ids) & cam['frame0_ids']
    cam['annotated'] = draw_markers_on_frame0(
        cam['frame0'],
        cam['markers'],
        cam['marker_data'],
        frame0_common,
        cam['not_frame0_ids']
    )

# =========================
# BUILD CSV DATA
# =========================
def build_rows(frame, markers, ids):
    rows = []
    H, W = frame.shape[:2]

    for marker_id in ids:
        pts = markers[marker_id]

        # Center point
        xi = pts[:, 0].mean()
        yi = pts[:, 1].mean()

        # Template ROI
        x_min = int(np.floor(pts[:, 0].min())) - template_padding
        y_min = int(np.floor(pts[:, 1].min())) - template_padding
        x_max = int(np.ceil(pts[:, 0].max())) + template_padding
        y_max = int(np.ceil(pts[:, 1].max())) + template_padding

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, W - 1)
        y_max = min(y_max, H - 1)

        b = x_max - x_min + 1
        h = y_max - y_min + 1

        rows.append([
            float(xi),
            float(yi),
            float(x_min),
            float(y_min),
            float(b),
            float(h)
        ])

    return rows

for cam in camera_results:
    cam['rows'] = build_rows(cam['frame0'], cam['markers'], common_ids)

# =========================
# EXPORT CSV FILES
# =========================
header_comment = "#  Image points and templates which are picked by user (xi yi x0 y0 w h)"

print("📄 Exported CSV files:")
used_csv_names = set()
for cam in camera_results:
    original_stem = os.path.splitext(os.path.basename(cam['video_path']))[0]
    csv_name = f"{original_stem}_targets.csv"
    if csv_name in used_csv_names:
        suffix = 2
        while True:
            csv_name = f"{original_stem}_targets_{suffix}.csv"
            if csv_name not in used_csv_names:
                break
            suffix += 1
    used_csv_names.add(csv_name)

    csv_path = os.path.join(output_folder, csv_name)
    with open(csv_path, 'w') as f:
        f.write(header_comment + '\n')
        for row in cam['rows']:
            f.write(','.join([f"{val:.18e}" for val in row]) + '\n')
    cam['csv_path'] = csv_path
    print(f"   → {csv_path}")

# =========================
# EXPORT ANNOTATED IMAGES
# =========================
print("\n🖼️  Exported annotated images:")
for cam in camera_results:
    image_path = os.path.join(output_folder, f"{cam['label']}_markers_annotated.png")
    cv2.imwrite(image_path, cam['annotated'])
    cam['image_path'] = image_path
    print(f"   → {image_path}")
print(f"\n✅ All files exported to: {output_folder}")
