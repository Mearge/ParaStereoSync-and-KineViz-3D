import os
import time
import numpy as np
import cv2

from inputs import input2 
from pickTemplates import pickTemplates

def eccTrackVideo(
        videoFilepath=None,
        tmpltFilepath=None,
        tmpltFrameId=0,
        frameRange=None,
        tmpltRange=None,
        mTable=None, # should be (nFrames by 5*nPoints), allocated before calling. 
        saveFilepath=None,
):
    # Create OpenCV VideoCapture object (vid)
    if type(videoFilepath) == str:
        try:
            vid = cv2.VideoCapture(videoFilepath)
        except:
            errMessage = "# Error: eccTrackVideo(): Failed to open video file %s." % videoFilepath
            print(errMessage)
            return (-1, errMessage)
    else:
        errMessage = "# Error: eccTrackVideo(): videoFilepath (str) should be the full path of the video file."
        print(errMessage)
        return (-1, errMessage)
    if vid.isOpened() == False:
        errMessage = "# Error: eccTrackVideo(): Failed to open video file %s." % videoFilepath
        print(errMessage)
        return (-1, errMessage)
    # Print info 
    nFrames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vWidth = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vHeight = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vFps = round(vid.get(cv2.CAP_PROP_FPS))
    print("# eccTrackVideo(): video %s opened. (Height/Width/Frames/Fps)=(%d/%d/%d/%d)." % (videoFilepath, vHeight, vWidth, nFrames, vFps))

    # Get imgInit for template image
    # get initial image for templates (including reading, release, and re-open video)
    for ii in range(tmpltFrameId + 1):
        try:
            ret, imgInit = vid.read()
            imgInit = cv2.cvtColor(imgInit, cv2.COLOR_BGR2GRAY)
        except:
            pass
    vid.release()
    while True:
        vid = cv2.VideoCapture(videoFilepath)
        if vid.isOpened():
            break
        print("# Failed to re-open video %s. Do you want to try again? (Enter n to exit this function)" % videoFilepath)
        if input().strip()[0] == 'n':
            errMessage = "# Error: eccTrackVideo(): videoFilepath (str) cannot be re-opened."
            print(errMessage)
            return (-1, errMessage)
    # 

    # Import templates (from tmpltFilepath) ==> nPoints, tmplts[nPoints, 6]
    try:
        if os.path.exists(tmpltFilepath) == True:
            tmplts = np.loadtxt(tmpltFilepath, delimiter=',')
            nPoints = tmplts.shape[0]
        else:
            # if file does not exist, define by mouse
            print("# Template file %s does not exist. Define it by mouse now." % tmpltFilepath)
            print("  # Enter number of POIs for the video: ")
            nPoints = int(input2())
            results = pickTemplates(
                    img = imgInit, 
                    nPoints=nPoints,
                    savefile=tmpltFilepath, 
                    saveImgfile=os.path.splitext(tmpltFilepath)[0] + '_tmpltPicked' + os.path.splitext(tmpltFilepath)[1])
    except:
        errMessage = "# Error: eccTrackVideo(): Failed to load templates from file %s." % tmpltFilepath
        print(errMessage)
        return (-1, errMessage)
    # print info
    print("# %d templates loaded from %s." % (nPoints, tmpltFilepath))

    # mTable should be pre-allocated. But if it is not, allocate it
    if type(mTable) == type(None):
        mTable = np.ones((nFrames, 5*nPoints), dtype=np.float32) * np.nan

    # Validate tmpltRange
    tmpltRange = np.array(tmpltRange, dtype=np.int32)
    tmpltRange = tmpltRange[tmpltRange < tmplts.shape[0]]

    # Initialize mTable for the first frame
    iFrame = frameRange[0]
    for iPoi in tmpltRange:
        if iPoi >= tmplts.shape[0]:  # Skip if iPoi exceeds the number of templates
            print(f"Warning: iPoi {iPoi} exceeds the number of templates ({tmplts.shape[0]}). Skipping.")
            continue
        mTable[iFrame, iPoi * 5 + 0] = tmplts[iPoi, 0]
        mTable[iFrame, iPoi * 5 + 1] = tmplts[iPoi, 1]
        mTable[iFrame, iPoi * 5 + 2] = 0.0  # Rotation
        mTable[iFrame, iPoi * 5 + 3] = 1.0  # Correlation
        mTable[iFrame, iPoi * 5 + 4] = 0.0  # ECC computation time

    # Start running loop of frames 
    for iFrame in range(nFrames):
        if iFrame >= nFrames:
            break
        # read image
        try:
            ret, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lastReadFrame = frame.copy()
        except:
            print("# Failed to read video %s frame %d. Skipped." % (videoFilepath, iFrame+1))
            frame = lastReadFrame.copy()
        # skip this frame if it is not within the range
        if iFrame <= frameRange[0]:
            continue
        if iFrame > frameRange[1]:
            break
        # ECC tracking - optimized for speed with selective fallbacks
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001)
        crop_margin = 100  # Balanced search area
        
        for iPoi in tmpltRange:
            if iPoi >= nPoints:
                continue
            # Define template
            x0 = round(tmplts[iPoi, 2])
            y0 = round(tmplts[iPoi, 3])
            x1 = round(x0 + tmplts[iPoi, 4])
            y1 = round(y0 + tmplts[iPoi, 5])
            dx = tmplts[iPoi, 0] - x0  # Difference between POI and template top-left corner
            dy = tmplts[iPoi, 1] - y0  # Difference between POI and template top-left corner
            tmplt = imgInit[y0:y1, x0:x1].copy()

            # Define a cropping region around the expected location
            crop_x0 = max(0, int(mTable[iFrame - 1, iPoi * 5 + 0] - dx - crop_margin))
            crop_y0 = max(0, int(mTable[iFrame - 1, iPoi * 5 + 1] - dy - crop_margin))
            crop_x1 = min(frame.shape[1], int(mTable[iFrame - 1, iPoi * 5 + 0] + dx + crop_margin))
            crop_y1 = min(frame.shape[0], int(mTable[iFrame - 1, iPoi * 5 + 1] + dy + crop_margin))
            cropped_frame = frame[crop_y0:crop_y1, crop_x0:crop_x1].copy()

            # Improved warp guess using velocity prediction if possible
            if iFrame > frameRange[0] + 1:
                # Calculate velocity (distance moved between last two frames)
                vx = mTable[iFrame - 1, iPoi * 5 + 0] - mTable[iFrame - 2, iPoi * 5 + 0]
                vy = mTable[iFrame - 1, iPoi * 5 + 1] - mTable[iFrame - 2, iPoi * 5 + 1]
                # Predict next position based on velocity
                guess_x = (mTable[iFrame - 1, iPoi * 5 + 0] + vx) - dx - crop_x0
                guess_y = (mTable[iFrame - 1, iPoi * 5 + 1] + vy) - dy - crop_y0
            else:
                # Fallback to previous frame for the very first step
                guess_x = mTable[iFrame - 1, iPoi * 5 + 0] - dx - crop_x0
                guess_y = mTable[iFrame - 1, iPoi * 5 + 1] - dy - crop_y0
            guess_r = mTable[iFrame - 1, iPoi * 5 + 2] * np.pi / 180.0
            c = np.cos(guess_r)
            s = np.sin(guess_r)
            warp_guess = np.array([c, -s, guess_x, s, c, guess_y], dtype=np.float32).reshape(2, 3)

            # Run ECC on the cropped image
            tic_ecc1 = time.time()
            eccSuccess = False
 
            # Try Euclidean motion model
            if not eccSuccess:
                motion_model = cv2.MOTION_EUCLIDEAN
                for gaussFiltSize in [0, 5, 4, 6, 3, 7, 8, 9, 11, 13, 15]:
                    try:
                        if gaussFiltSize <= 0:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess, motion_model, criteria)
                        else:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess, motion_model, criteria, gaussFiltSize=gaussFiltSize)
                        eccSuccess = True
                        break
                    except:
                        pass

            # Try translation motion model (if Euclidean fails)
            if not eccSuccess:
                motion_model = cv2.MOTION_TRANSLATION
                for gaussFiltSize in [0, 5, 4, 6, 3, 7, 8, 9, 11, 13, 15]:
                    try:
                        if gaussFiltSize <= 0:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess, motion_model, criteria)
                        else:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess, motion_model, criteria, gaussFiltSize=gaussFiltSize)
                        eccSuccess = True
                        break
                    except:
                        pass

            # Try second-order polynomial prediction (if both previous attempts fail)
            if not eccSuccess and iFrame > frameRange[0] + 2:
                # Second-order polynomial extrapolation: 3*p1 - 3*p2 + p3
                # This accounts for acceleration/deceleration
                p1_x = mTable[iFrame - 1, iPoi * 5 + 0]
                p1_y = mTable[iFrame - 1, iPoi * 5 + 1]
                p2_x = mTable[iFrame - 2, iPoi * 5 + 0]
                p2_y = mTable[iFrame - 2, iPoi * 5 + 1]
                p3_x = mTable[iFrame - 3, iPoi * 5 + 0]
                p3_y = mTable[iFrame - 3, iPoi * 5 + 1]
                
                # Predict next position using quadratic extrapolation
                pred_poly_x = 3 * p1_x - 3 * p2_x + p3_x
                pred_poly_y = 3 * p1_y - 3 * p2_y + p3_y
                
                # Create warp guess relative to crop region
                guess_poly_x = pred_poly_x - dx - crop_x0
                guess_poly_y = pred_poly_y - dy - crop_y0
                warp_guess_poly = np.array([c, -s, guess_poly_x, s, c, guess_poly_y], dtype=np.float32).reshape(2, 3)
                
                # Try with Euclidean motion model first
                motion_model = cv2.MOTION_EUCLIDEAN
                for gaussFiltSize in [0, 5, 4, 6, 3, 7, 8, 9, 11, 13, 15]:
                    try:
                        if gaussFiltSize <= 0:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess_poly, motion_model, criteria)
                        else:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess_poly, motion_model, criteria, gaussFiltSize=gaussFiltSize)
                        eccSuccess = True
                        break
                    except:
                        pass
                
                # If still not successful, try translation
                if not eccSuccess:
                    motion_model = cv2.MOTION_TRANSLATION
                    for gaussFiltSize in [0, 5, 4, 6, 3, 7, 8, 9, 11, 13, 15]:
                        try:
                            if gaussFiltSize <= 0:
                                ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess_poly, motion_model, criteria)
                            else:
                                ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, warp_guess_poly, motion_model, criteria, gaussFiltSize=gaussFiltSize)
                            eccSuccess = True
                            break
                        except:
                            pass

            # ADAPTIVE FALLBACK: Only apply expensive operations when ECC fails
            if not eccSuccess:
                # Step 1: Try with histogram equalization (illumination robustness)
                tmplt_eq = cv2.equalizeHist(tmplt)
                cropped_eq = cv2.equalizeHist(cropped_frame)
                
                # Retry ECC with enhanced images
                for gaussFiltSize in [0, 5]:
                    try:
                        if gaussFiltSize <= 0:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt_eq, cropped_eq, warp_guess, 
                                                                            cv2.MOTION_TRANSLATION, criteria)
                        else:
                            ret, warp_matrix_cropped = cv2.findTransformECC(tmplt_eq, cropped_eq, warp_guess, 
                                                                            cv2.MOTION_TRANSLATION, criteria, gaussFiltSize=gaussFiltSize)
                        eccSuccess = True
                        break
                    except:
                        pass
            
            # Step 2: Template matching only if still failing
            if not eccSuccess:
                # Use normalized cross-correlation (robust to illumination)
                res = cv2.matchTemplate(cropped_frame, tmplt, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                # If template match confidence is reasonable, refine with ECC
                if max_val > 0.8:
                    refined_guess = np.array([[1, 0, max_loc[0]], 
                                             [0, 1, max_loc[1]]], dtype=np.float32)
                    try:
                        ret, warp_matrix_cropped = cv2.findTransformECC(tmplt, cropped_frame, refined_guess, 
                                                                        cv2.MOTION_TRANSLATION, criteria)
                        eccSuccess = True
                    except:
                        # Accept template match result
                        warp_matrix_cropped = refined_guess
                        ret = max_val
                        eccSuccess = True
                else:
                    # Low confidence: use template match but mark for review
                    warp_matrix_cropped = np.array([[1, 0, max_loc[0]], 
                                                    [0, 1, max_loc[1]]], dtype=np.float32)
                    ret = max_val
                    eccSuccess = True

            # If ECC succeeded, adjust the warp matrix for the original image
            if eccSuccess:
                # Initialize the warp matrix for the original image
                warp_matrix = np.eye(3, dtype=np.float32)

                # Copy the 2x3 warp matrix from the cropped image
                warp_matrix[:2, :] = warp_matrix_cropped

                # Adjust for the cropping offset
                warp_matrix[0, 2] += crop_x0
                warp_matrix[1, 2] += crop_y0

                # Check if correlation coefficient is less than 0.9, use template matching
                if ret < 0.9:
                    # Use template matching on cropped frame
                    res = cv2.matchTemplate(cropped_frame, tmplt, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    # If template match gives better result, use it
                    if max_val > ret:
                        # Create new warp matrix from template match
                        warp_matrix_cropped = np.array([[1, 0, max_loc[0]], 
                                                        [0, 1, max_loc[1]]], dtype=np.float32)
                        ret = max_val
                        
                        # Update the warp matrix for the original image
                        warp_matrix = np.eye(3, dtype=np.float32)
                        warp_matrix[:2, :] = warp_matrix_cropped
                        warp_matrix[0, 2] += crop_x0
                        warp_matrix[1, 2] += crop_y0

                # Save the correlation value
                mTable[iFrame, iPoi * 5 + 3] = ret  # Correlation
            else:
                print('# ECC failed at frame %d POI:%d (%s)' % (iFrame + 1, iPoi + 1, videoFilepath))
                mTable[iFrame, iPoi * 5 + 3] = 0.0  # Correlation

            # Save the ECC computation time
            toc_ecc1 = time.time()
            mTable[iFrame, iPoi * 5 + 4] = toc_ecc1 - tic_ecc1  # ECC time

            # Calculate image coordinates and rotation of this POI
            xi0 = np.array([dx, dy, 1.0], dtype=np.float32).reshape(3, 1)
            xi1 = warp_matrix @ xi0
            mTable[iFrame, iPoi * 5 + 0] = xi1[0, 0]  # X-coordinate in the original image
            mTable[iFrame, iPoi * 5 + 1] = xi1[1, 0]  # Y-coordinate in the original image
            rmat33 = np.eye(3, dtype=warp_matrix.dtype)
            rmat33[0:2, 0:2] = warp_matrix[0:2, 0:2]
            mTable[iFrame, iPoi * 5 + 2] = cv2.Rodrigues(rmat33)[0][2][0] * 180.0 / np.pi
        # end of for iPoi in tmpltRange
        print('\b'*100, end='')
        print("# Frame %d ECC completed." % (iFrame+1), end='')
    # end of for iFrame in range(frameRange[0], frameRange[1] + 1)
    print('')
    # save mTable 
    if type(saveFilepath) == str:
       np.savetxt(saveFilepath, mTable, delimiter=',')
# end of def eccTrackVideo


if __name__ == '__main__':
    print('# Do you want to run eccTrackVideo demo? (1 to run, other inputs to quit.)')
    toRun = input().strip()
    if toRun == '1':
        videoFilepath = 'F:\\Test04\\C02\\T04C02R11.mp4'
        tmpltFilepath = 'F:\\Test04\\C02\\left_targets.csv'
        tmpltFrameId = 1 # initial frame id
        frameRange = [0, 21000]
        tmpltRange = np.arange(0, 324)
        bigTable = np.ones((5000, 1620), dtype=np.float32) * np.nan
        mTable = bigTable[:, 1:1+5*324]
        saveFilepath= 'F:\\Test04\\C02\\Tracking_results_Multistep_warpguess.csv'
        tmplts = np.loadtxt('F:\\Test04\\C02\\left_targets.csv', delimiter=',')
        print(f"Template shape: {tmplts.shape}")
        eccTrackVideo(
            videoFilepath=videoFilepath,
            tmpltFilepath=tmpltFilepath,
            tmpltFrameId=tmpltFrameId,
            frameRange=frameRange,
            tmpltRange=tmpltRange,
            saveFilepath=saveFilepath
        )
    # end of if toRun ==
