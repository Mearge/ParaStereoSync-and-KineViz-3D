import numpy as np
import cv2 as cv
import os

from inputs import input2

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