def track_and_plot():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from tkinter import messagebox
    video_path = input_path_var.get().strip()
    if not video_path or not os.path.exists(video_path):
        messagebox.showerror("Error", "Please select a valid input video.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open video: {video_path}")
        return
    ret, first_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to read the first frame.")
        cap.release()
        return
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    def improved_template_picker(img, maxW=1200, maxH=800, interpolation=cv2.INTER_LINEAR):
        # First selection (full image)
        scale = min(maxW / img.shape[1], maxH / img.shape[0], 1.0)
        if scale < 1.0:
            disp_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=interpolation)
        else:
            disp_img = img.copy()
        messagebox.showinfo("Pick Point", "Step 1: Select template region. ENTER/SPACE to confirm, ESC to cancel.")
        roi1 = cv2.selectROI("Pick Template (Step 1)", disp_img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Pick Template (Step 1)")
        if scale < 1.0:
            x1, y1, w1, h1 = [int(v/scale) for v in roi1]
        else:
            x1, y1, w1, h1 = roi1
        if w1 == 0 or h1 == 0:
            return None
        # Zoom into selected region for second selection
        zoom_margin = 0  # You can add margin if desired
        x1z = max(0, x1 - zoom_margin)
        y1z = max(0, y1 - zoom_margin)
        x2z = min(img.shape[1], x1 + w1 + zoom_margin)
        y2z = min(img.shape[0], y1 + h1 + zoom_margin)
        zoom_img = img[y1z:y2z, x1z:x2z].copy()
        zoom_scale = min(maxW / zoom_img.shape[1], maxH / zoom_img.shape[0], 1.0)
        if zoom_scale < 1.0:
            disp_zoom_img = cv2.resize(zoom_img, (int(zoom_img.shape[1]*zoom_scale), int(zoom_img.shape[0]*zoom_scale)), interpolation=interpolation)
        else:
            disp_zoom_img = zoom_img.copy()
        messagebox.showinfo("Pick Point", "Step 2: Refine template region. ENTER/SPACE to confirm, ESC to cancel.")
        roi2 = cv2.selectROI("Pick Template (Step 2)", disp_zoom_img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Pick Template (Step 2)")
        if zoom_scale < 1.0:
            x2, y2, w2, h2 = [int(v/zoom_scale) for v in roi2]
        else:
            x2, y2, w2, h2 = roi2
        if w2 == 0 or h2 == 0:
            return None
        # Convert zoomed selection back to original image coordinates
        final_x = x1z + x2
        final_y = y1z + y2
        final_w = w2
        final_h = h2
        # Show feedback
        imgClone = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img.copy()
        color = (0,255,0)
        center = (int(final_x+final_w/2), int(final_y+final_h/2))
        cv2.drawMarker(imgClone, center, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.rectangle(imgClone, (final_x, final_y), (final_x+final_w, final_y+final_h), color, 2)
        cv2.imshow("Template Selected", imgClone)
        cv2.waitKey(500)
        cv2.destroyWindow("Template Selected")
        return final_x, final_y, final_w, final_h
    result = improved_template_picker(first_gray)
    if result is None:
        messagebox.showerror("Error", "No template selected.")
        cap.release()
        return
    x, y, w, h = result
    template = first_gray[y:y+h, x:x+w]
    xs, ys = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
    cropMargin = 100
    guess_x = x
    guess_y = y
    guess_r = 0.0
    c = np.cos(guess_r)
    s = np.sin(guess_r)
    warp_guess = np.array([c, -s, guess_x, s, c, guess_y], dtype=np.float32).reshape(2,3)
    tmplt = template.copy()
    dx = 0
    dy = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_x0 = max(0, int(guess_x - cropMargin))
        crop_y0 = max(0, int(guess_y - cropMargin))
        crop_x1 = min(gray.shape[1], int(guess_x + w + cropMargin))
        crop_y1 = min(gray.shape[0], int(guess_y + h + cropMargin))
        frame_cropped = gray[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        warp_guess_cropped = warp_guess.copy()
        warp_guess_cropped[0, 2] = guess_x - crop_x0
        warp_guess_cropped[1, 2] = guess_y - crop_y0
        eccSuccess = False
        eccFailReason = ""
        warp_matrix = warp_guess_cropped.copy()
        for gaussFiltSize in [9,7,5,3,1,0]:
            try:
                if gaussFiltSize > 1:
                    tmplt_blur = cv2.GaussianBlur(tmplt, (gaussFiltSize, gaussFiltSize), 0)
                    frame_cropped_blur = cv2.GaussianBlur(frame_cropped, (gaussFiltSize, gaussFiltSize), 0)
                else:
                    tmplt_blur = tmplt
                    frame_cropped_blur = frame_cropped
                _, warp_matrix = cv2.findTransformECC(tmplt_blur, frame_cropped_blur, warp_guess_cropped, cv2.MOTION_EUCLIDEAN, criteria, None)
                eccSuccess = True
                break
            except Exception as e:
                eccFailReason = str(e)
        if not eccSuccess:
            motion_model = cv2.MOTION_TRANSLATION
            for gaussFiltSize in [9,7,5,3,1,0]:
                try:
                    if gaussFiltSize > 1:
                        tmplt_blur = cv2.GaussianBlur(tmplt, (gaussFiltSize, gaussFiltSize), 0)
                        frame_cropped_blur = cv2.GaussianBlur(frame_cropped, (gaussFiltSize, gaussFiltSize), 0)
                    else:
                        tmplt_blur = tmplt
                        frame_cropped_blur = frame_cropped
                    _, warp_matrix = cv2.findTransformECC(tmplt_blur, frame_cropped_blur, warp_guess_cropped, motion_model, criteria, None)
                    eccSuccess = True
                    break
                except Exception as e:
                    eccFailReason = str(e)
        xi0 = np.array([dx, dy, 1.], dtype=np.float32).reshape(3, 1)
        xi1 = warp_matrix @ xi0 if eccSuccess else xi0
        left_x = float(xi1[0,0] + crop_x0)
        left_y = float(xi1[1,0] + crop_y0)
        xs.append(np.format_float_positional(left_x, precision=16))
        ys.append(np.format_float_positional(left_y, precision=16))
        rmat33 = np.eye(3, dtype=warp_matrix.dtype)
        rmat33[0:2,0:2] = warp_matrix[0:2,0:2]
        rot_vec, _ = cv2.Rodrigues(rmat33)
        guess_r = rot_vec[2][0]
        c = np.cos(guess_r)
        s = np.sin(guess_r)
        guess_x = left_x
        guess_y = left_y
        warp_guess = np.array([c, -s, guess_x, s, c, guess_y], dtype=np.float32).reshape(2,3)
        print('\b'*100, end='')
        print(f"# Frame {frame_idx} ECC {'completed' if eccSuccess else 'FAILED'}.", end='')
        frame_idx += 1
    cap.release()
    plt.figure(figsize=(14, 8))
    font_title = {'fontsize': 20, 'fontweight': 'bold'}
    font_label = {'fontsize': 16}
    font_tick = {'fontsize': 14}
    plt.subplot(2,1,1)
    plt.plot([float(x) for x in xs], label='X', linewidth=2)
    plt.ylabel('X (pixels)', **font_label)
    plt.title('Template Top-Left Corner Tracking', **font_title)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplot(2,1,2)
    plt.plot([float(y) for y in ys], label='Y', color='orange', linewidth=2)
    plt.ylabel('Y (pixels)', **font_label)
    plt.xlabel('Frame', **font_label)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
def track_aruco_and_plot():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from tkinter import messagebox

    video_path = input_path_var.get().strip()
    if not video_path or not os.path.exists(video_path):
        messagebox.showerror("Error", "Please select a valid input video.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open video: {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to read the first frame.")
        cap.release()
        return

    # --- Detect all ArUco markers in the first frame ---
    aruco_dict_type = aruco_dict_var.get()
    aruco_dict_map = {
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    }
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_map.get(aruco_dict_type, cv2.aruco.DICT_4X4_250))
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners_all, ids_all, _ = detector.detectMarkers(first_frame)
    if ids_all is None or len(ids_all) == 0:
        messagebox.showerror("Error", "No ArUco markers detected in the first frame.\nTry a different dictionary.")
        cap.release()
        return

    # --- Let user pick which marker to track ---
    display = first_frame.copy()
    cv2.aruco.drawDetectedMarkers(display, corners_all, ids_all)
    # Draw ID labels large
    for i, cset in enumerate(corners_all):
        c = cset[0]
        cx = int(np.mean(c[:, 0]))
        cy = int(np.mean(c[:, 1]))
        cv2.putText(display, f"ID={ids_all[i][0]}", (cx - 30, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    maxW, maxH = 1200, 800
    scale = min(maxW / display.shape[1], maxH / display.shape[0], 1.0)
    if scale < 1.0:
        disp_img = cv2.resize(display, (int(display.shape[1]*scale), int(display.shape[0]*scale)))
    else:
        disp_img = display

    selected_id = [None]
    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coords back to original
            ox = int(mx / scale)
            oy = int(my / scale)
            # Find closest marker center
            best_dist = float('inf')
            best_id = None
            for i, cset in enumerate(corners_all):
                c = cset[0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                d = (ox - cx)**2 + (oy - cy)**2
                if d < best_dist:
                    best_dist = d
                    best_id = ids_all[i][0]
            selected_id[0] = best_id

    messagebox.showinfo("Select Marker",
        f"{len(ids_all)} ArUco markers detected (IDs: {sorted([int(x) for x in ids_all.flatten()])}).\n"
        "Click on the marker you want to track, then press ENTER.")
    win_name = "Click on ArUco Marker to Track"
    cv2.imshow(win_name, disp_img)
    cv2.setMouseCallback(win_name, on_mouse)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 or key == 32:  # Enter or Space
            break
        if key == 27:  # Esc
            selected_id[0] = None
            break
    cv2.destroyWindow(win_name)

    if selected_id[0] is None:
        messagebox.showerror("Error", "No marker selected.")
        cap.release()
        return

    target_id = int(selected_id[0])
    messagebox.showinfo("Tracking", f"Tracking ArUco marker ID={target_id}. Please wait...")

    # --- Speed-tuned detector for tracking loop ---
    fast_params = cv2.aruco.DetectorParameters()
    fast_params.adaptiveThreshWinSizeMin = 5
    fast_params.adaptiveThreshWinSizeMax = 21
    fast_params.adaptiveThreshWinSizeStep = 4  # fewer threshold passes
    fast_detector = cv2.aruco.ArucoDetector(aruco_dict, fast_params)

    # Get initial marker size for adaptive crop margin
    init_idx = list(ids_all.flatten()).index(target_id)
    init_corners = corners_all[init_idx][0]
    marker_w = float(np.max(init_corners[:, 0]) - np.min(init_corners[:, 0]))
    marker_h = float(np.max(init_corners[:, 1]) - np.min(init_corners[:, 1]))
    crop_margin = int(max(marker_w, marker_h) * 2)  # 2x marker size
    crop_margin = max(crop_margin, 150)  # minimum 150px

    last_cx = float(np.mean(init_corners[:, 0]))
    last_cy = float(np.mean(init_corners[:, 1]))

    # --- Track the selected marker through all frames ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    xs, ys = [], []
    frame_idx = 0
    lost_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]

        # --- ROI crop around last known position for speed ---
        cx0 = max(0, int(last_cx - crop_margin))
        cy0 = max(0, int(last_cy - crop_margin))
        cx1 = min(fw, int(last_cx + crop_margin))
        cy1 = min(fh, int(last_cy + crop_margin))
        roi = gray[cy0:cy1, cx0:cx1]

        corners_f, ids_f, _ = fast_detector.detectMarkers(roi)
        found = False
        if ids_f is not None:
            for i, mid in enumerate(ids_f):
                if int(mid[0]) == target_id:
                    c = corners_f[i][0]
                    cx = float(np.mean(c[:, 0])) + cx0  # map back to full frame
                    cy = float(np.mean(c[:, 1])) + cy0
                    xs.append(cx)
                    ys.append(cy)
                    last_cx, last_cy = cx, cy
                    found = True
                    break

        # Fallback: full-frame search if ROI missed it
        if not found:
            corners_f, ids_f, _ = fast_detector.detectMarkers(gray)
            if ids_f is not None:
                for i, mid in enumerate(ids_f):
                    if int(mid[0]) == target_id:
                        c = corners_f[i][0]
                        cx = float(np.mean(c[:, 0]))
                        cy = float(np.mean(c[:, 1]))
                        xs.append(cx)
                        ys.append(cy)
                        last_cx, last_cy = cx, cy
                        found = True
                        break

        if not found:
            xs.append(float('nan'))
            ys.append(float('nan'))
            lost_frames += 1
        print('\b'*100, end='')
        print(f"# Frame {frame_idx} ArUco {'OK' if found else 'LOST'}.", end='')
        frame_idx += 1

    cap.release()
    print()
    if lost_frames > 0:
        messagebox.showwarning("Warning", f"Marker ID={target_id} was lost in {lost_frames}/{frame_idx} frames.\nLost frames shown as gaps.")

    # --- Plot ---
    plt.figure(figsize=(14, 8))
    font_title = {'fontsize': 20, 'fontweight': 'bold'}
    font_label = {'fontsize': 16}
    plt.subplot(2,1,1)
    plt.plot(xs, label='X', linewidth=2)
    plt.ylabel('X (pixels)', **font_label)
    plt.title(f'ArUco ID={target_id} Center Tracking', **font_title)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.subplot(2,1,2)
    plt.plot(ys, label='Y', color='orange', linewidth=2)
    plt.ylabel('Y (pixels)', **font_label)
    plt.xlabel('Frame', **font_label)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

import multiprocessing

# Move process_video to top-level
def process_video(video_path, roi):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print(f"Failed to read first frame for {video_path}")
        return
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = roi
    template = first_gray[y:y+h, x:x+w]
    xs, ys = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
    cropMargin = 100
    guess_x = x
    guess_y = y
    guess_r = 0.0
    c = np.cos(guess_r)
    s = np.sin(guess_r)
    warp_guess = np.array([c, -s, guess_x, s, c, guess_y], dtype=np.float32).reshape(2,3)
    tmplt = template.copy()
    dx = 0
    dy = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_x0 = max(0, int(guess_x - cropMargin))
        crop_y0 = max(0, int(guess_y - cropMargin))
        crop_x1 = min(gray.shape[1], int(guess_x + w + cropMargin))
        crop_y1 = min(gray.shape[0], int(guess_y + h + cropMargin))
        frame_cropped = gray[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        warp_guess_cropped = warp_guess.copy()
        warp_guess_cropped[0, 2] = guess_x - crop_x0
        warp_guess_cropped[1, 2] = guess_y - crop_y0
        eccSuccess = False
        eccFailReason = ""
        warp_matrix = warp_guess_cropped.copy()
        for gaussFiltSize in [15,13,11,9,7,5,3,1]:
            try:
                if gaussFiltSize > 1:
                    tmplt_blur = cv2.GaussianBlur(tmplt, (gaussFiltSize, gaussFiltSize), 0)
                    frame_cropped_blur = cv2.GaussianBlur(frame_cropped, (gaussFiltSize, gaussFiltSize), 0)
                else:
                    tmplt_blur = tmplt
                    frame_cropped_blur = frame_cropped
                _, warp_matrix = cv2.findTransformECC(tmplt_blur, frame_cropped_blur, warp_guess_cropped, cv2.MOTION_EUCLIDEAN, criteria, None)
                eccSuccess = True
                break
            except Exception as e:
                eccFailReason = str(e)
        if not eccSuccess:
            motion_model = cv2.MOTION_TRANSLATION
            for gaussFiltSize in [15,13,11,9,7,5,3,1]:
                try:
                    if gaussFiltSize > 1:
                        tmplt_blur = cv2.GaussianBlur(tmplt, (gaussFiltSize, gaussFiltSize), 0)
                        frame_cropped_blur = cv2.GaussianBlur(frame_cropped, (gaussFiltSize, gaussFiltSize), 0)
                    else:
                        tmplt_blur = tmplt
                        frame_cropped_blur = frame_cropped
                    _, warp_matrix = cv2.findTransformECC(tmplt_blur, frame_cropped_blur, motion_model, warp_guess_cropped, criteria, None)
                    eccSuccess = True
                    break
                except Exception as e:
                    eccFailReason = str(e)
        xi0 = np.array([dx, dy, 1.], dtype=np.float32).reshape(3, 1)
        xi1 = warp_matrix @ xi0 if eccSuccess else xi0
        left_x = float(xi1[0,0] + crop_x0)
        left_y = float(xi1[1,0] + crop_y0)
        xs.append(np.format_float_positional(left_x, precision=16))
        ys.append(np.format_float_positional(left_y, precision=16))
        rmat33 = np.eye(3, dtype=warp_matrix.dtype)
        rmat33[0:2,0:2] = warp_matrix[0:2,0:2]
        rot_vec, _ = cv2.Rodrigues(rmat33)
        guess_r = rot_vec[2][0]
        c = np.cos(guess_r)
        s = np.sin(guess_r)
        guess_x = left_x
        guess_y = left_y
        warp_guess = np.array([c, -s, guess_x, s, c, guess_y], dtype=np.float32).reshape(2,3)
        print('\b'*100, end='')
        print(f"# Frame {frame_idx} ECC {'completed' if eccSuccess else 'FAILED'}.", end='')
        frame_idx += 1
    cap.release()
    plt.figure(figsize=(14, 8))
    font_title = {'fontsize': 20, 'fontweight': 'bold'}
    font_label = {'fontsize': 16}
    font_tick = {'fontsize': 14}
    plt.subplot(2,1,1)
    plt.plot([float(x) for x in xs], label='X', linewidth=2)
    plt.ylabel('X (pixels)', **font_label)
    plt.title(f'Template Top-Left Corner Tracking\n{os.path.basename(video_path)}', **font_title)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplot(2,1,2)
    plt.plot([float(y) for y in ys], label='Y', color='orange', linewidth=2)
    plt.ylabel('Y (pixels)', **font_label)
    plt.xlabel('Frame', **font_label)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def process_video_aruco(video_path, target_id, aruco_dict_code):
    """Top-level function for multiprocessing ArUco tracking (optimized)."""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_code)
    fast_params = cv2.aruco.DetectorParameters()
    fast_params.adaptiveThreshWinSizeMin = 5
    fast_params.adaptiveThreshWinSizeMax = 21
    fast_params.adaptiveThreshWinSizeStep = 4
    detector = cv2.aruco.ArucoDetector(aruco_dict, fast_params)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print(f"Failed to read first frame for {video_path}")
        return

    # Detect initial marker position & size for adaptive crop
    gray0 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    corners_init, ids_init, _ = detector.detectMarkers(gray0)
    last_cx, last_cy = gray0.shape[1] / 2, gray0.shape[0] / 2  # default center
    crop_margin = 200
    if ids_init is not None:
        for i, mid in enumerate(ids_init):
            if int(mid[0]) == target_id:
                c = corners_init[i][0]
                last_cx = float(np.mean(c[:, 0]))
                last_cy = float(np.mean(c[:, 1]))
                mw = float(np.max(c[:, 0]) - np.min(c[:, 0]))
                mh = float(np.max(c[:, 1]) - np.min(c[:, 1]))
                crop_margin = max(int(max(mw, mh) * 3), 150)
                break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    xs, ys = [], []
    frame_idx = 0
    lost = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]

        # ROI crop around last known position
        cx0 = max(0, int(last_cx - crop_margin))
        cy0 = max(0, int(last_cy - crop_margin))
        cx1 = min(fw, int(last_cx + crop_margin))
        cy1 = min(fh, int(last_cy + crop_margin))
        roi = gray[cy0:cy1, cx0:cx1]

        corners_f, ids_f, _ = detector.detectMarkers(roi)
        found = False
        if ids_f is not None:
            for i, mid in enumerate(ids_f):
                if int(mid[0]) == target_id:
                    c = corners_f[i][0]
                    cx = float(np.mean(c[:, 0])) + cx0
                    cy = float(np.mean(c[:, 1])) + cy0
                    xs.append(cx); ys.append(cy)
                    last_cx, last_cy = cx, cy
                    found = True
                    break

        # Fallback: full-frame search
        if not found:
            corners_f, ids_f, _ = detector.detectMarkers(gray)
            if ids_f is not None:
                for i, mid in enumerate(ids_f):
                    if int(mid[0]) == target_id:
                        c = corners_f[i][0]
                        cx = float(np.mean(c[:, 0]))
                        cy = float(np.mean(c[:, 1]))
                        xs.append(cx); ys.append(cy)
                        last_cx, last_cy = cx, cy
                        found = True
                        break

        if not found:
            xs.append(float('nan'))
            ys.append(float('nan'))
            lost += 1
        print('\b'*100, end='')
        print(f"# {os.path.basename(video_path)} Frame {frame_idx} ArUco {'OK' if found else 'LOST'}.", end='')
        frame_idx += 1
    cap.release()
    print()
    plt.figure(figsize=(14, 8))
    font_title = {'fontsize': 20, 'fontweight': 'bold'}
    font_label = {'fontsize': 16}
    plt.subplot(2,1,1)
    plt.plot(xs, label='X', linewidth=2)
    plt.ylabel('X (pixels)', **font_label)
    plt.title(f'ArUco ID={target_id} Tracking\n{os.path.basename(video_path)}', **font_title)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.subplot(2,1,2)
    plt.plot(ys, label='Y', color='orange', linewidth=2)
    plt.ylabel('Y (pixels)', **font_label)
    plt.xlabel('Frame', **font_label)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def batch_track_and_plot():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from tkinter import messagebox
    from tkinter import filedialog
    import multiprocessing
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import multiprocessing
    video_paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All Files", "*.*")]
    )
    if not video_paths:
        return
    jobs = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            messagebox.showerror("Error", f"Cannot open video: {video_path}")
            continue
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        def improved_template_picker(img, maxW=1200, maxH=800, interpolation=cv2.INTER_LINEAR):
            # First selection (full image)
            scale = min(maxW / img.shape[1], maxH / img.shape[0], 1.0)
            if scale < 1.0:
                disp_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=interpolation)
            else:
                disp_img = img.copy()
            messagebox.showinfo("Pick Point", f"{os.path.basename(video_path)}: Step 1: Select template region. ENTER/SPACE to confirm, ESC to cancel.")
            roi1 = cv2.selectROI(f"Pick Template (Step 1): {os.path.basename(video_path)}", disp_img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(f"Pick Template (Step 1): {os.path.basename(video_path)}")
            if scale < 1.0:
                x1, y1, w1, h1 = [int(v/scale) for v in roi1]
            else:
                x1, y1, w1, h1 = roi1
            if w1 == 0 or h1 == 0:
                return None
            # Zoom into selected region for second selection
            zoom_margin = 0  # You can add margin if desired
            x1z = max(0, x1 - zoom_margin)
            y1z = max(0, y1 - zoom_margin)
            x2z = min(img.shape[1], x1 + w1 + zoom_margin)
            y2z = min(img.shape[0], y1 + h1 + zoom_margin)
            zoom_img = img[y1z:y2z, x1z:x2z].copy()
            zoom_scale = min(maxW / zoom_img.shape[1], maxH / zoom_img.shape[0], 1.0)
            if zoom_scale < 1.0:
                disp_zoom_img = cv2.resize(zoom_img, (int(zoom_img.shape[1]*zoom_scale), int(zoom_img.shape[0]*zoom_scale)), interpolation=interpolation)
            else:
                disp_zoom_img = zoom_img.copy()
            messagebox.showinfo("Pick Point", f"{os.path.basename(video_path)}: Step 2: Refine template region. ENTER/SPACE to confirm, ESC to cancel.")
            roi2 = cv2.selectROI(f"Pick Template (Step 2): {os.path.basename(video_path)}", disp_zoom_img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(f"Pick Template (Step 2): {os.path.basename(video_path)}")
            if zoom_scale < 1.0:
                x2, y2, w2, h2 = [int(v/zoom_scale) for v in roi2]
            else:
                x2, y2, w2, h2 = roi2
            if w2 == 0 or h2 == 0:
                return None
            # Convert zoomed selection back to original image coordinates
            final_x = x1z + x2
            final_y = y1z + y2
            final_w = w2
            final_h = h2
            # Show feedback
            imgClone = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img.copy()
            color = (0,255,0)
            center = (int(final_x+final_w/2), int(final_y+final_h/2))
            cv2.drawMarker(imgClone, center, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.rectangle(imgClone, (final_x, final_y), (final_x+final_w, final_y+final_h), color, 2)
            cv2.imshow(f"Template Selected: {os.path.basename(video_path)}", imgClone)
            cv2.waitKey(500)
            cv2.destroyWindow(f"Template Selected: {os.path.basename(video_path)}")
            return final_x, final_y, final_w, final_h
        result = improved_template_picker(first_gray)
        if result is None:
            messagebox.showerror("Error", f"No template selected for {video_path}.")
            cap.release()
            continue
        cap.release()
        p = multiprocessing.Process(target=process_video, args=(video_path, result))
        jobs.append(p)
    for p in jobs:
        p.start()
    for p in jobs:
        p.join()

def batch_track_aruco_and_plot():
    import cv2
    import numpy as np
    from tkinter import filedialog, messagebox
    import multiprocessing

    aruco_dict_type = aruco_dict_var.get()
    aruco_dict_map = {
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    }
    aruco_code = aruco_dict_map.get(aruco_dict_type, cv2.aruco.DICT_4X4_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_code)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    video_paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All Files", "*.*")]
    )
    if not video_paths:
        return

    jobs = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            messagebox.showerror("Error", f"Cannot open video: {video_path}")
            continue

        corners_all, ids_all, _ = detector.detectMarkers(first_frame)
        if ids_all is None or len(ids_all) == 0:
            messagebox.showerror("Error", f"No ArUco markers in: {os.path.basename(video_path)}")
            cap.release()
            continue

        display = first_frame.copy()
        cv2.aruco.drawDetectedMarkers(display, corners_all, ids_all)
        for i, cset in enumerate(corners_all):
            c = cset[0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            cv2.putText(display, f"ID={ids_all[i][0]}", (cx - 30, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        maxW, maxH = 1200, 800
        scale = min(maxW / display.shape[1], maxH / display.shape[0], 1.0)
        disp_img = cv2.resize(display, (int(display.shape[1]*scale), int(display.shape[0]*scale))) if scale < 1.0 else display

        selected_id = [None]
        def on_mouse(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                ox, oy = int(mx / scale), int(my / scale)
                best_dist, best_id = float('inf'), None
                for i, cset in enumerate(corners_all):
                    c = cset[0]
                    cx = int(np.mean(c[:, 0]))
                    cy = int(np.mean(c[:, 1]))
                    d = (ox - cx)**2 + (oy - cy)**2
                    if d < best_dist:
                        best_dist, best_id = d, ids_all[i][0]
                selected_id[0] = best_id

        messagebox.showinfo("Select Marker",
            f"{os.path.basename(video_path)}:\n"
            f"IDs found: {sorted([int(x) for x in ids_all.flatten()])}\n"
            "Click on the marker to track, then press ENTER.")
        win_name = f"Select Marker: {os.path.basename(video_path)}"
        cv2.imshow(win_name, disp_img)
        cv2.setMouseCallback(win_name, on_mouse)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (13, 32): break
            if key == 27: selected_id[0] = None; break
        cv2.destroyWindow(win_name)
        cap.release()

        if selected_id[0] is None:
            messagebox.showerror("Error", f"No marker selected for {os.path.basename(video_path)}.")
            continue

        target_id = int(selected_id[0])
        p = multiprocessing.Process(target=process_video_aruco, args=(video_path, target_id, aruco_code))
        jobs.append(p)

    for p in jobs:
        p.start()
    for p in jobs:
        p.join()

import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

def select_input():
    path = filedialog.askopenfilename(
        title="Select Video File",
    )
    # The button creation should be outside of this function and not as an argument
    if path:
        input_path_var.set(path)
        output_path_var.set(os.path.splitext(path)[0] + "_trimmed" + os.path.splitext(path)[1])

def select_output():
    path = filedialog.asksaveasfilename(
        title="Save Trimmed Video As",
        defaultextension=".mp4",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All Files", "*.*")]
    )
    if path:
        output_path_var.set(path)




import re
import subprocess

def trim_video():
    input_path = input_path_var.get().strip()
    output_path = output_path_var.get().strip()
    raw_start = start_entry.get().strip()
    raw_end = end_entry.get().strip()
    use_frames = use_frames_var.get()

    if not input_path or not os.path.exists(input_path):
        messagebox.showerror("Error", "Please select a valid input video.")
        return

    # Helper function to strip everything except numbers and decimals
    def clean_numeric(value):
        if not value: return None
        cleaned = re.sub(r'[^0-9.]', '', value)
        return float(cleaned) if cleaned else None

    try:
        start_val = clean_numeric(raw_start)
        end_val = clean_numeric(raw_end)
    except ValueError:
        messagebox.showerror("Error", "Start/End must be numeric.")
        return

    try:
        if use_frames:
            # --- Robust FPS Detection ---
            fps = 0.0
            try:
                # Try FFprobe first
                cmd_fps = [
                    "ffprobe", "-v", "0", "-of", "csv=p=0",
                    "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", input_path
                ]
                fps_out = subprocess.check_output(cmd_fps).decode().strip()
                if '/' in fps_out:
                    num, den = map(float, fps_out.split('/'))
                    fps = num / den if den != 0 else 0
                elif fps_out:
                    fps = float(fps_out)
            except:
                fps = 0.0

            # Fallback to OpenCV if FFprobe gave 0 or failed
            if fps <= 0:
                import cv2
                temp_cap = cv2.VideoCapture(input_path)
                fps = temp_cap.get(cv2.CAP_PROP_FPS)
                temp_cap.release()

            if fps <= 0:
                raise ValueError("Could not determine video Frame Rate (FPS).")

            # Convert frames to seconds for FFmpeg
            if start_val is not None: start_val /= fps
            if end_val is not None: end_val /= fps

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]
        if start_val is not None: cmd += ["-ss", f"{start_val:.4f}"]
        if end_val is not None: cmd += ["-to", f"{end_val:.4f}"]
        
        # -c copy is lossless; -map 0:a? handles videos without audio
        cmd += ["-i", input_path, "-map", "0:v", "-map", "0:a?", "-c", "copy", output_path]

        subprocess.run(cmd, check=True, capture_output=True)
        messagebox.showinfo("Success", f"Trimmed video saved successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Processing failed: {str(e)}")




# GUI Setup

root = tk.Tk()
root.title("VidTrim")
root.geometry("1150x580")
root.resizable(False, False)

# Variables
input_path_var = tk.StringVar()
output_path_var = tk.StringVar()
use_frames_var = tk.BooleanVar(value=False)
tracking_method_var = tk.StringVar(value="ECC")
aruco_dict_var = tk.StringVar(value="DICT_4X4_250")

# Layout
tk.Label(root, text="Input Video:").pack(anchor="w", padx=10, pady=(10, 0))
tk.Entry(root, textvariable=input_path_var, width=70).pack(padx=10)
tk.Button(root, text="Browse", command=select_input).pack(padx=10, pady=(0, 10))

tk.Label(root, text="Output Video:").pack(anchor="w", padx=10)
tk.Entry(root, textvariable=output_path_var, width=70).pack(padx=10)
tk.Button(root, text="Save As", command=select_output).pack(padx=10, pady=(0, 10))

frame = tk.Frame(root)
frame.pack(pady=10)
tk.Label(frame, text="Start (sec/frame):").grid(row=0, column=0, padx=5)
start_entry = tk.Entry(frame, width=10)
start_entry.grid(row=0, column=1, padx=5)
tk.Label(frame, text="End (sec/frame):").grid(row=0, column=2, padx=5)
end_entry = tk.Entry(frame, width=10)
end_entry.grid(row=0, column=3, padx=5)

tk.Checkbutton(root, text="Use frames instead of seconds", variable=use_frames_var).pack(pady=5)

# --- Tracking Method Selection ---
tracking_frame = tk.LabelFrame(root, text="Tracking Method", font=("Arial", 10, "bold"), padx=10, pady=5)
tracking_frame.pack(padx=10, pady=5, fill="x")

method_row = tk.Frame(tracking_frame)
method_row.pack(anchor="w")
tk.Radiobutton(method_row, text="ECC (Template)", variable=tracking_method_var, value="ECC",
               font=("Arial", 10)).pack(side="left", padx=5)
tk.Radiobutton(method_row, text="ArUco Marker", variable=tracking_method_var, value="ARUCO",
               font=("Arial", 10)).pack(side="left", padx=5)
tk.Label(method_row, text="  ArUco Dict:").pack(side="left", padx=(20, 2))
aruco_dict_options = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]
aruco_dropdown = tk.OptionMenu(method_row, aruco_dict_var, *aruco_dict_options)
aruco_dropdown.config(width=18)
aruco_dropdown.pack(side="left", padx=2)

def dispatch_track():
    if tracking_method_var.get() == "ARUCO":
        track_aruco_and_plot()
    else:
        track_and_plot()

def dispatch_batch_track():
    if tracking_method_var.get() == "ARUCO":
        batch_track_aruco_and_plot()
    else:
        batch_track_and_plot()

# Action buttons frame for better layout
action_frame = tk.Frame(root)
action_frame.pack(pady=15)
tk.Button(action_frame, text="Trim Video", command=trim_video, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=22).pack(side="left", padx=10)
tk.Button(action_frame, text="Track and Plot", command=dispatch_track, bg="#2196F3", fg="white", font=("Arial", 12, "bold"), width=22).pack(side="left", padx=10)
tk.Button(action_frame, text="Batch Track Videos (Parallel)", command=dispatch_batch_track, bg="#FF9800", fg="white", font=("Arial", 12, "bold"), width=30).pack(side="left", padx=10)

tk.Label(root, text="Note: Trimming is lossless and keeps the same format.", fg="gray").pack(pady=5)

root.mainloop()
