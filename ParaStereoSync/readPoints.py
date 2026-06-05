import numpy as np
import re

from inputs import input2

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

def chessboardPts3d(nRows=7, nCols=7, cellSize=1.0):
    pts3d = np.zeros((nRows * nCols, 3), dtype=float)
    for i in range(nRows):
        for j in range(nCols):
            pts3d[i * nCols + j, 0] = i * cellSize
            pts3d[i * nCols + j, 1] = j * cellSize
            pts3d[i * nCols + j, 2] = 0.0
    return pts3d


def chessboardPts2d(nRows=7, nCols=7, cellSize=1.0):
    pts2d = np.zeros((nRows * nCols, 2), dtype=float)
    for i in range(nRows):
        for j in range(nCols):
            pts2d[i * nCols + j, 0] = i * cellSize
            pts2d[i * nCols + j, 1] = j * cellSize
    return pts2d
