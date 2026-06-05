import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from inputs import input2, input3
from drawMarkerShift import drawMarkerShift

def pickPoints(img=np.zeros((0,0),dtype=np.uint8), nPoints=0, 
               nZoomIteration = 0, 
               maxW=1200, maxH=700, 
               interpolation=cv.INTER_LINEAR, 
               savefile="", saveImgfile=""):
    """
    Shows the given image, allows user to pick points by mouse, and save the 
    picked points in csv file and a new image with picked points marked. 
    Example: 
    imgPoints = pickPoints("./examples/pickPoints/IMG_0001.jpg", 
                           nPoints=3, 
                           savefile="./examples/pickPoints/picked.csv", 
                           saveImgfile="./examples/pickPoints/picked.jpg")

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

    """
    # check
    if type(img) == str:
        img = cv.imread(img)
        if type(img) == type(None):
            print("# The image file you entered cannot be recognized.")
            img = np.zeros((0,0,0), dtype=np.uint8)
    # initialize imgPoints as a float array filled with nans
    imgPoints = np.ones((nPoints,2),dtype=np.float64) * np.nan
    # img: ask user to select image file
    while (img.size <= 0):
        print("# Enter image file: ")
        print("#   (or enter corner for a 256x256 chessboard corner)")
        print("#   (or enter aruco for a 256x256 aruco marker "
              "(DICT_6X6_250 id 0))")
        print("# For example: examples/pickPoints/IMG_0001.jpg")
        ufile = input2()
        # User enters "corner" for a 256x256 chessboard corner
        if (ufile.strip() == "corner"):
            img = np.zeros((256,256), dtype=np.uint8)
            img[0:128,0:128] = 255; img[128:256,128:256] = 255
        elif (ufile.strip() == "aruco"):
#            img = cv.aruco.drawMarker(
#                  cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250),
#                  id=0,sidePixels=256)
             dict=cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
             img = dict.generateImageMarker(id=0, sidePixels=256)
        elif (os.path.isfile(ufile) == False):
            print("# Input is not a file (%s)." %(ufile))
        else:
            img = cv.imread(ufile)
            if type(img) == type(None) or img.size <= 0:
                print("# Cannot read a valid image from file: ", ufile)
                continue
    # nPoints
    if (nPoints <= 0):
        print("# Number of points (must be >= 1): ")
        print("# For example, 3")
        nPoints = input3("", dtype=int, min=1)
        imgPoints = np.ones((nPoints,2),dtype=float) * np.nan
    # ask file name to save image points 
    if (savefile == ""):
        print("# Enter the file to save the image points (in csv format):")
        print("#   (or enter a single dot (.) to skip saving.)")
        print("# For example, examples/pickPoints/try_pickedPoints.csv")
        savefile = input2().strip()
        if (len(savefile) > 1):
            print("# The picked points will be saved in file: %s" % savefile)
        else:
            print("# The picked points will not be saved in any file.")
    # ask image file name to save image with markers
    if (saveImgfile == ""):
        print("# Enter the image file to save the markers (in OpenCV supported image format):")
        print("#   (or enter a single dot (.) to skip saving.)")
        print("# For example: examples/pickPoints/try_pickedPoints.JPG")
        saveImgfile = input2().strip()
        if (len(saveImgfile) > 1):
            print("# The marked image will be saved in file: %s" % saveImgfile)
        else:
            print("# The marked image will not be saved in any file.")
    # Start working (from this line, no more argument asking)        
    # get points
    while(True):
        # Get index
        while(True):
            print("# Enter the index of this point (must be between 1 and %d)" 
                   % (nPoints))
            print("#   or 0 to complete the picking.")
            # print("# For example, 1")
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
                print("# For example (as you have defined all points): 0")
            else:
                print("# For example, %d" % (smallestI + 1))
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
        print("# stored to point %d." % (idx))
        # Draw marker cross and number on the point
        if len(img.shape) == 2:
            # gray-scale (single channel)
            color = 100  # Blue in grayscale
        else:
            # color image (multi-channel) - BGR format
            color = [255, 0, 0]  # Blue in BGR (B, G, R)
        imgClone = drawMarkerShift(img, ptRoi[0:2], color=color)
        # Text color - green in BGR format
        text_color = [0, 255, 0]  # Green in BGR
        fontScale = 3.0
        cv.putText(imgClone, "%d" % (idx), 
                   [int(ptRoi[0]+0.5), int(ptRoi[1]+0.5)], 
                   cv.FONT_HERSHEY_PLAIN, 
                   fontScale, text_color)
        np.copyto(img, imgClone)
    # save to file
    if (savefile != "."):
        np.savetxt(savefile, imgPoints, fmt='%24.16e', delimiter=' , ',
                   header=' Image points which are picked by user (xi yi)')
    # save marked image to file
        if (saveImgfile != "."):
            cv.imwrite(saveImgfile, img)
    # return result
    return imgPoints


def pickPoint(img=np.zeros((0,0),dtype=np.uint8), 
              nZoomIteration = 0, 
              maxW=1200, maxH=700,
              interpolation=cv.INTER_LINEAR, 
              winName="Select ROI. Repeat Space. ESC to confirm."):
    x, y, x0, y0 = 0, 0, 0, 0
    # User interaction: ask user to select image file
    while (img.size <= 0):
        print("# Image is not defined yet. Enter image file: ")
        print("#   (or enter \"corner\" for a 256x256 chessboard corner)")
        print("#   (or enter \"aruco\" for a 256x256 aruco marker "
              "(DICT_6X6_250 id 0))")
        print("# For exmaple, c:/test/IMG_0001.BMP")
        ufile = input2()
        # User enters "corner" for a 256x256 chessboard corner
        if (ufile.strip() == "corner"):
            img = np.zeros((256,256), dtype=np.uint8)
            img[0:128,0:128] = 255; img[128:256,128:256] = 255
        elif (ufile.strip() == "aruco"):
            img = cv.aruco.drawMarker(
                  cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250),
                  id=0,sidePixels=256)
        elif (os.path.isfile(ufile) == False):
            print("# Input is not a file (%s)." %(ufile))
        else:
            img = cv.imread(ufile)
            if img is None or img.size <= 0:
                print("# Cannot read a valid image from file: ", ufile)
                continue
    x1, y1 = img.shape[1], img.shape[0] 
    # winName = "Select roi. Press ESC to accept the last pick."
    # first time
    #   Select scale
    if nZoomIteration <= 0:
        nZoomIteration = 999
    print("# Zoom-in by selecting and pressing [Enter] or [Space]")    
    print("# Iteratively zoom-in until you satisfy the precision")
    print("# Press ESC or c to accept the last pick (of crosshair)")
    for i in range(nZoomIteration):
        scale_x = maxW * 1.0 / (x1 - x0)
        scale_y = maxH * 1.0 / (y1 - y0)
        scale_x = min(scale_x, scale_y)
        scale_y = scale_x
        #   new image
        imgshow = cv.resize(img[y0:y1,x0:x1], dsize=(-1,-1), fx=scale_x, fy=scale_y, \
                            interpolation= interpolation)
        scale_x = imgshow.shape[1] * 1.0 / (x1 - x0)
        scale_y = imgshow.shape[0] * 1.0 / (y1 - y0)
        #   get point
        roi_x, roi_y, roi_w, roi_h = cv.selectROI( \
            winName, imgshow, showCrosshair=True, fromCenter=False)
        if (roi_w == 0 or roi_h == 0): 
            cv.destroyWindow(winName)
            # print("# ROI width or height is zero")
            # return [-1,-1,-1,-1,-1,-1]
            break
        #   calculate position (not sure why needs to add -0.5 at the end)
        # x = (x0 - 0.5 + 0.5 / scale_x) + (roi_x + 0.5 * (roi_w - 1.)) / scale_x 
        # y = (y0 - 0.5 + 0.5 / scale_y) + (roi_y + 0.5 * (roi_h - 1.)) / scale_y
        x = x0 - 0.5 + (roi_x + 0.5 * roi_w) / scale_x 
        y = y0 - 0.5 + (roi_y + 0.5 * roi_h) / scale_y 
        # print("This: ", "; x=", x, ";x0=", x0, ";x1=", x1, ";roi_x=", roi_x, ";roi_w=", roi_w, ";scale_x=", scale_x)
        x0 = int(x - ((roi_w - 1) * 0.5) / scale_x + 0.5)
        y0 = int(y - ((roi_h - 1) * 0.5) / scale_y + 0.5)
        x1 = int(x + ((roi_w - 1) * 0.5) / scale_x + 0.5) + 1
        y1 = int(y + ((roi_h - 1) * 0.5) / scale_y + 0.5) + 1
        # print("Next: ", "; x=", x, ";x0=", x0, ";x1=", x1, ";roi_x=", roi_x, ";roi_w=", roi_w, ";scale_x=", scale_x)
        cv.destroyWindow(winName)
    return [x, y, x0, y0, (x1 - x0), (y1 - y0)]


def pickTmGivenPoint(img, pt, markColor= [], 
                     markerType = cv.MARKER_CROSS, \
                     winName="Pick Template", \
                     markerSize = 20, markerThickness = 2, 
                     markerLineType = 8, \
                     markerShift = 1):
    # draw pt on img
    imgClone = img.copy()
    if (len(markColor) == 0):
        if (len(imgClone.shape) == 2): # gray scale
            markColor = 224
        else:
            markColor = [64,255,64]
    imgClone = drawMarkerShift(imgClone, pt, markColor, markerType, \
                               markerSize, markerThickness, \
                               shift = markerShift)
    ptAndBox = pickPoint(imgClone, winName=winName)
    return ptAndBox[2:]


def pickPointAndTm(img):
    ptAndBox = pickPoint(img)
    tmX0, tmY0, tmW, tmH = pickTmGivenPoint(img, ptAndBox[0:2])
    return [ptAndBox[0], ptAndBox[1], tmX0, tmY0, tmW, tmH]


def pickPoint_example(imgSize = 360, imgRot = 30.0):
    # make sure imgSize is an even
    imgSize = int(imgSize/2) * 2
    # generate image with a black-white rectangular corner
    img = np.zeros((imgSize, imgSize), dtype = np.uint8)
    img[0:int(imgSize/2), 0:int(imgSize/2)] = 200;
    img[int(imgSize/2):imgSize, int(imgSize/2):imgSize] = 200;
    # cv.imshow("TETT", img); cv.waitKey(0);
    # cv.destroyWindow("TETT")
    (px,py,x0,y0,w,h) = pickPoint(img)
    print((px,py,x0,y0,w,h))

    
def pickPointAndTm_example(imgSize = 360, imgRot = 30.0):
    # make sure imgSize is an even
    imgSize = int(imgSize/2) * 2
    # generate image with a black-white rectangular corner
    img = np.zeros((imgSize, imgSize), dtype = np.uint8)
    img[0:int(imgSize/2), 0:int(imgSize/2)] = 200
    img[int(imgSize/2):imgSize, int(imgSize/2):imgSize] = 200
    (px,py,x0,y0,w,h) = pickPointAndTm(img)
    print((px,py,x0,y0,w,h))
    plt.imshow(img[x0:x0+w,y0:y0+h], cmap='gray')
    



