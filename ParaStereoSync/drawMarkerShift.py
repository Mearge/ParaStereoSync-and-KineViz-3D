import numpy as np
import cv2 as cv

def drawMarkerShift(img, position, color=[48,192,48], markerType=cv.MARKER_CROSS,
                    markerSize=20, thickness=2, line_type=8,
                    shift=0, interpolation=cv.INTER_LANCZOS4):
    # imgMarked = img.copy(); 
    # cv.drawMarker(imgMarked, (179,179), 128, cv.MARKER_CROSS,
    #               markerSize=40, thickness = 3); plt.imshow(imgMarked)
    shift = int(shift + 0.5)
    if shift < 0:
        shift = 0
    if shift > 3:
        shift = 3
    scale = 2 ** shift
    # check color (if gray scale image)
    if (len(img.shape) < 3 and type(color) == list and len(color) >= 3):
        color = int((color[0] + color[1] + color[2]) / 3. + 0.5)
    imgResized = np.zeros((img.shape[0] * scale, img.shape[1] * scale), \
                           img.dtype)
    imgResized = cv.resize(img, (-1,-1), imgResized, fx = scale, fy = scale, \
                           interpolation=cv.INTER_CUBIC)
    positionResized = position.copy()
    positionResized[0] = int((position[0] + 0.5) * scale - 0.5 + 0.5)
    positionResized[1] = int((position[1] + 0.5) * scale - 0.5 + 0.5)
    markerSizeResized = int(markerSize * scale + 0.5)
    thicknessResized = int(thickness * scale + 0.5)
    imgResized = cv.drawMarker(imgResized, positionResized, color, 
                               markerType, markerSizeResized, 
                               thicknessResized, line_type)
    img = cv.resize(imgResized, (-1,-1), img, fx=1./scale,
                    fy=1./scale, interpolation=interpolation)
    # plt.imshow(img, cmap='gray')
    return img
    