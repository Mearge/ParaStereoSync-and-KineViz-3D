import os, re
import numpy as np
import cv2 as cv


def input2(prompt=""):
    """
    This function is similar to Python function input() but if the returned
    string starts with a hashtag (#) this function ignores the line of the
    strin and runs the input() function again.
    The head spaces and tail spaces are removed as well.
    This function only allows user to edit a script for a series of input,
    but also allows user to put comments by starting the comments with a
    hashtag, so that the input script is earier to understand.
    For example, a BMI converter could run in this way:
    /* -------------------------------
    1.75  (user's input)
    70    (user's input)
    The BMI is 22.9
    --------------------------------- */
    The user can edit a file for future input:
    /* ---------------------------------
    # This is an input script for a program that calculates BMI
    # Enter height in unit of meter
    1.75
    # Enter weight in unit of kg
    70

    Parameters
        prompt  A String, representing a default message before the input.
    --------------------------------- */
    """
    theInput = ""
    if len(prompt) == 0:
        thePrompt = ""
    else:
        thePrompt = "# " + prompt
        print(thePrompt)
    # run the while loop of reading
    while(True):
        theInput = input()
        theInput = theInput.strip()
        if (len(theInput) == 0):
            continue
        if (theInput[0] == '#'):
            continue
        break
    # remove everything after the first #
    if theInput.find('#') >= 0:
        theInput = theInput[0:theInput.find('#')]
    return theInput


def imread2(filename="", flags=cv.IMREAD_COLOR, example=""):
    ufile = filename.strip()
#    img = np.array([])
    img = cv.imread(ufile)
    while (type(img) != np.ndarray or img.size == 0):
        print("# Enter image file: ")
        print("#   (or enter corner for a 256x256 chessboard corner)")
        print("#   (or enter aruco for a 256x256 aruco marker (DICT_6X6_250 id 0))")
        print("#   (or enter aruco3c for a 3-channel (BGR) 256x256 aruco marker (DICT_6X6_250 id 0))")
        if len(example) <= 1:
            print("# For example: examples/pickPoints/IMG_0001.jpg")
        else:
            print("# For example: %s" % (example))
        ufile = input2()
        # User enters "corner" for a 256x256 chessboard corner
        if (ufile.strip() == "corner"):
            img = np.zeros((256,256), dtype=np.uint8)
            img[0:128,0:128] = 255; img[128:256,128:256] = 255
        elif (ufile.strip() == "aruco"):
            img = cv.aruco.drawMarker(
                cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250),
                id=0,sidePixels=256)
        elif (ufile.strip() == "aruco3c"):
            img = cv.aruco.drawMarker(
                cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250),
                id=0,sidePixels=256)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif (os.path.isfile(ufile) == False):
            print("# Error: Your input (%s) is not a file." %(ufile))
        else:
            img = cv.imread(ufile, flags)
            if img is None or img.size <= 0:
                print("# Error Cannot read a valid image from file: ", ufile)
    return img


def chessboardPts2d(nRows=7, nCols=7, cellSize=1.0):
    pts2d = np.zeros((nRows * nCols, 2), dtype=float)
    for i in range(nRows):
        for j in range(nCols):
            pts2d[i * nCols + j, 0] = i * cellSize
            pts2d[i * nCols + j, 1] = j * cellSize
    return pts2d


def chessboardPts3d(nRows=7, nCols=7, cellSize=1.0):
    pts3d = np.zeros((nRows * nCols, 3), dtype=float)
    for i in range(nRows):
        for j in range(nCols):
            pts3d[i * nCols + j, 0] = i * cellSize
            pts3d[i * nCols + j, 1] = j * cellSize
            pts3d[i * nCols + j, 2] = 0.0
    return pts3d


def readPoints(filename=""):
    pts = np.array([])
    # if filename is given
    if (len(filename) >= 1):
        try:
            pts = np.loadtxt(filename, delimiter=',')
            return pts
        except Exception as e:
            print("# Error: readPoints() cannot read points from file (%s)."
                   % (filename))
            print("# Exception is", e)
    # ask user if filename is empty
    while (pts.size == 0):
        print("# How do you want to input points:")
        print("#  file: Enter file name of csv format points.")
        print("#   # comments")
        print("#     x1,y1 (,z1)")
        print("#     x2,y2 (,z2)")
        print("#     ...")
        print("#            For example: ")
        print("#            file")
        print("#            .\\examples\\pickPoints\\picked_IMG_0001.csv")
        print("#  chessboard2d m n: Points of m by n chessboard 2d points")
        print("#                     For example: chessboard2d 7 7")
        print("#  chessboard3d m n: Points of m by n chessboard 3d points")        
        print("#                    For example: chessboard3d 7 7")
        print("#  manual2d n: Manually type-in n image points:")
        print("#              For example: ")
        print("#                 manual2d 3")
        print("#                 x1,y1")
        print("#                 x2,y2")
        print("#                 x3,y3")
        print("#  manual3d n: Manually type-in n image points:")
        print("#              For example: ")
        print("#                 manual3d 3")
        print("#                 x1,y1,z1")
        print("#                 x2,y2,z2")
        print("#                 x3,y3,z3")
        uInput = input2().strip()
        if (uInput == "file"):
            try:
                print("# Enter file name of points:")
                filename = input2()
                pts = np.loadtxt(filename, delimiter=',')
            except Exception as e:
                print("# Error: readPoints() cannot read points from file (%s)."
                       % (filename))
                print("# Exception is", e)
                print("# Try again.")
                continue
        if (uInput[0:12] == "chessboard2d"):
            m = int(re.split(',| ', uInput.strip())[1])
            n = int(re.split(',| ', uInput.strip())[2])
            pts = chessboardPts2d(m, n, 1.0)
        if (uInput[0:12] == "chessboard3d"):
            m = int(re.split(',| ', uInput.strip())[1])
            n = int(re.split(',| ', uInput.strip())[2])
            pts = chessboardPts3d(m, n, 1.0)
        if (uInput[0:8] == 'manual2d'):
            nPoints = int(re.split(',| ', uInput.strip())[1])
            pts = np.ones((nPoints,2), dtype=float) * np.nan
            for i in range(nPoints):
                datInput = input2("").strip()
                datInput = re.split(',| ', datInput)
                pts[i,0] = float(datInput[0])
                pts[i,1] = float(datInput[1])
        if (uInput[0:8] == 'manual3d'):
            nPoints = int(re.split(',| ', uInput.strip())[1])
            pts = np.ones((nPoints,3), dtype=float) * np.nan
            for i in range(nPoints):
                datInput = re.split(',| ', datInput)
                pts[i,0] = float(datInput[0])
                pts[i,1] = float(datInput[1])
                pts[i,2] = float(datInput[2])
        if (pts.size > 0):
            print("# Read %d points" % (pts.shape[0]))
            print("# The first point is ", pts[0])
            print("# The last point is ", pts[-1])
    return pts


def drawPoints(img=None, pts=None, color=[0,255,0], 
               markerType=cv.MARKER_CROSS, markerSize=20, 
               thickness=2, lineType=8, 
               fontScale=0.0, 
               savefile=""):
    """

    Parameters
    ----------
    img : str, optional
        The image (np.ndarray, shape:(h,w) or (h,w,depth))
        If img is not given, this function asks through console (print/input)
    pts : np.ndarray (n-by-2, n is # of points), optional
        DESCRIPTION. The default is None.
    color : TYPE, optional
        DESCRIPTION. The default is [0,255,0].
    markerType : TYPE, optional
        DESCRIPTION. The default is cv.MARKER_CROSS.
    markerSize : TYPE, optional
        DESCRIPTION. The default is 20.
    thickness : TYPE, optional
        DESCRIPTION. The default is 2.
    lineType : TYPE, optional
        DESCRIPTION. The default is 8.
    fontScale : TYPE, optional
        DESCRIPTION. The default is 0.
    savefile : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    
    
    # image
    if (type(img) == type(None)):
        print("# Enter image file name: ")
        print("#  For example (for drawPoints):"
              " examples/drawPoints/IMG_0001.jpg")
        img = imread2()
        print("# Image size: %d x %d (width x height)"
              % (img.shape[1], img.shape[0]))
    # pts
    if (type(pts) == type(None)):
        print("# Enter file of image points (in csv format)")
        print("#  For example (for drawPoints):")
        print("#       file")
        print("#       examples/drawPoints/picked_IMG_0001.csv")
        pts = readPoints()
    # savefile
    if (savefile == ""):
        print("# Enter the image file to save:")
        print("#  or enter a single character to skip file saving.")
        print("#  For example, examples/drawPoints/try_marked_IMG_0001.jpg")
        savefile = input2()
    # draw points one by one
    pts2 = pts.reshape(-1, pts.shape[-1])
    nPts = pts2.shape[0]
    for i in range(nPts):
        thisPt = pts2[i]
        if np.isnan(thisPt[0]) or np.isnan(thisPt[1]):
            continue
        thisPtInt = [int(thisPt[0]+0.5), int(thisPt[1]+0.5)]
        img = cv.drawMarker(img,
                      thisPtInt,
                      color=color, 
                      markerType=markerType, 
                      markerSize=markerSize, 
                      thickness=thickness, line_type=lineType)
#        fontScale = 2.0
        if fontScale > 0.0:
            cv.putText(img, "%d" % (i + 1), 
                       thisPtInt,
                       cv.FONT_HERSHEY_PLAIN, 
                       fontScale, color)
    # save 
    if (len(savefile) > 1):
        cv.imwrite(savefile, img)
        print("# Image with markers saved to %s" % (savefile))
    return img

