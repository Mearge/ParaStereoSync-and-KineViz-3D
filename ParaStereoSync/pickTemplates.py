import os 

import numpy as np
import cv2 as cv
from inputs import input2, input3
from pickPoints import pickPoint, pickTmGivenPoint
from drawMarkerShift import drawMarkerShift

def custom_selectROI(winName, imgshow, crosshair_color=(0, 0, 255)):
    """
    Custom ROI selection with colored crosshairs.
    crosshair_color: BGR tuple for crosshair color (default red = (0, 0, 255))
    """
    roi_x, roi_y, roi_w, roi_h = cv.selectROI(winName, imgshow, showCrosshair=True, fromCenter=False)
    return roi_x, roi_y, roi_w, roi_h


def pickTemplates(img=np.zeros((0,0),dtype=np.uint8), 
                  nPoints=0, nZoomIteration = 0, 
                  maxW=1200, maxH=700, 
                  interpolation=cv.INTER_LINEAR, 
                  savefile="", saveImgfile=""):
    """
    

    Parameters
    ----------
    img : TYPE, optional
        DESCRIPTION. The default is np.zeros((0,0),dtype=np.uint8).
    nPoints : TYPE, optional
        DESCRIPTION. The default is 0.
    nZoomIteration : TYPE, optional
        DESCRIPTION. The default is 0.
    maxW : TYPE, optional
        DESCRIPTION. The default is 1200.
    maxH : TYPE, optional
        DESCRIPTION. The default is 700.
    interpolation : TYPE, optional
        DESCRIPTION. The default is cv.INTER_LINEAR.
    savefile : TYPE, optional
        DESCRIPTION. The default is "".
    saveImgfile : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    imgPoints : TYPE
        DESCRIPTION.

    Example:
    --------
    imgPoints = pickTemplates(
        img=cv.imread('examples/2022rcwall/leftPause/DSC02766.JPG'), 
        nPoints=1,
        nZoomIteration=0, # zoom until ESC is pressed
        maxW=1200,
        maxH=700,
        interpolation=cv.INTER_LINEAR,
        savefile=".",
        saveImgfile="."
    )


    """
    # initialize imgPoints as a float array filled with nans
    # 6 columns are for xi, yi, x0, y0, width, height
    imgPoints = np.ones((nPoints,6),dtype=np.float64) * np.nan
    # img: User interaction: ask user to select image file
    while (img.size <= 0):
        print("# Image is not defined yet. Enter image file: ")
        print("#   (or enter \"corner\" for a 256x256 chessboard corner)")
        print("#   (or enter \"aruco\" for a 256x256 aruco marker "
              "(DICT_6X6_250 id 0))")
        print("# For example: examples/pickTemplates/IMG_0001.jpg")
        ufile = input2()
        # User enters "corner" for a 256x256 chessboard corner
        if (ufile.strip() == "corner"):
            img = np.zeros((256,256), dtype=np.uint8)
            img[0:128,0:128] = 255; img[128:256,128:256] = 255
        elif (ufile.strip() == "aruco"):
#            # This may be a deprecated version of Aruco usage in OpenCV
#            img = cv.aruco.drawMarker(
#                  cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250),
#                  id=0,sidePixels=256)
             # This is new version of Aruco in OpenCV
             dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
             img = dict.generateImageMarker(id=0, sidePixels=256)
        elif (os.path.isfile(ufile) == False):
            print("# Input is not a file (%s)." %(ufile))
        else:
            img = cv.imread(ufile)
            if img is None or img.size <= 0:
                print("# Cannot read a valid image from file: ", ufile)
                continue
    # nPoints
    if (nPoints <= 0):
        print("# Number of points: ")
        print("# For example, 3")
        nPoints = input3("", dtype=int, min=1)
        # initialize imgPoints as a float array filled with nans
        # 6 columns are for xi, yi, x0, y0, width, height
        imgPoints = np.ones((nPoints,6),dtype=np.float64) * np.nan
    # ask file to save image points and templates 
    if (savefile == ""):
        print("# Enter the file to save the image points and templates (in csv format):")
        print("#   (or enter a single dot (.) to skip saving.)")
        print("# For example, examples/pickTemplates/try_pickedTemplates.csv")
        savefile = input2().strip()
        if (len(savefile) > 1):
            print("# The templates will be saved in file: %s" % savefile)
        else:
            print("# The templates will not be saved in any file.")\
    # ask file to save the image with markers
    if (saveImgfile == ""):
        print("# Enter the image file to save the markers:")
        print("#   (or enter a single dot (.) to skip saving.)")
        print("# For example, examples/pickTemplates/try_pickedTemplates.JPG")
        saveImgfile = input2().strip()
        if (len(saveImgfile) > 1):
            print("# The marked templates will be saved in file: %s" % saveImgfile)
        else:
            print("# The marked templates will not be saved in any file.")\
    # get points
    while(True):
        # Get index
        while(True):
            print("# Enter the index of this point (from 1 to %d)" 
                   % (nPoints))
            print("#   or 0 to complete the picking.")
            # print("#   For example, 1   (and 2, 3, ... later)")
            # Find the first point index which is not defined yet 
            # (to give example for user)
            # That is, to find the smallest i which imgPoints[i,0] or 
            # imgPoints[i,1] is nan.
            smallestI = -1
            for i in range(nPoints):
                if np.isnan(imgPoints[i,0]) or np.isnan(imgPoints[i,1]):
                    smallestI = i
                    break
            if smallestI == -1:
                print("#  For example (as you have defined all templates): 0")
            else:
                print("#  For example, %d" % (smallestI + 1))
            #
            idx = input3("", dtype=int, min=0, max=nPoints)
            try:
                idx = int(idx)
            except:
                print("# Wrong input. Should be an integer but got: " + idx)
                continue
            if (idx <= 0):
                break
            if (idx < 1 or idx > nPoints):
                print("# Wrong input. Should be between 1 and %d but got %d."
                      " or 0 to complete the picking"
                      % (nPoints, idx))
                continue
            break
        # Finish
        if (idx <= 0):
            print("# Picking completed.")
            break
        # Get an image point
        ptRoi = pickPoint(img=img, nZoomIteration=nZoomIteration,
                          maxW=maxW, maxH=maxH, 
                          interpolation=interpolation, 
                          winName="Pick Point No. %d" % (idx))
        print("# You picked (%9.3f,%9.3f)." % (ptRoi[0], ptRoi[1]))
        imgPoints[idx - 1, 0] = ptRoi[0]
        imgPoints[idx - 1, 1] = ptRoi[1]
        # Get template 
        print("# Then pick the template (ROI) about this point:")
        tmX0, tmY0, tmW, tmH = pickTmGivenPoint(img, ptRoi[0:2], 
                               winName="Pick Template No. %d" % (idx))
        imgPoints[idx - 1, 2] = tmX0
        imgPoints[idx - 1, 3] = tmY0
        imgPoints[idx - 1, 4] = tmW
        imgPoints[idx - 1, 5] = tmH
        print("# stored the point and template to point %d." % (idx))
        # Draw marker cross, number, and template 
        if len(img.shape) == 2:
            # gray-scale (single channel) - green
            color = 100  # Green in grayscale
        else:
            # color image (multi-channel) - BGR format
            color = [0, 255, 0]  # Green in BGR (B, G, R)
        imgClone = drawMarkerShift(img, ptRoi[0:2], color=color)
        fontScale = 2.0
        cv.putText(imgClone, "%d" % (idx), 
                   [int(ptRoi[0]+0.5), int(ptRoi[1]+0.5)], 
                   cv.FONT_HERSHEY_PLAIN, 
                   fontScale, color)
        pt1 = [tmX0, tmY0]
        pt2 = [tmX0 + tmW - 1, tmY0]
        pt3 = [tmX0 + tmW - 1, tmY0 + tmH - 1]
        pt4 = [tmX0, tmY0 + tmH - 1]           
        cv.line(imgClone, pt1, pt2, color)
        cv.line(imgClone, pt2, pt3, color)
        cv.line(imgClone, pt3, pt4, color)
        cv.line(imgClone, pt4, pt1, color)
        np.copyto(img, imgClone)        
    # save image points and templates to file
    if (len(savefile) > 1):
        np.savetxt(savefile, imgPoints, fmt='%26.18e', delimiter=' , ',
               header=' Image points and templates which '
               'are picked by user (xi yi x0 y0 w h)')
    # save image with markers to new image
    if (len(saveImgfile) > 1):
        cv.imwrite(saveImgfile, img)
    # return result
    return imgPoints