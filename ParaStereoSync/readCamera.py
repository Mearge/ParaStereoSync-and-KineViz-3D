import numpy as np
import re
import cv2 as cv
from inputs import input2
from writeCamera import cameraParametersToString

def readCamera(filename=None, example=None):
    """
    Reads camera parameters (rvec, tvec, cmat, dvec) from file. Data should be
    in 1-column vector in text format composed with rvec (3 elements), tvec (3
    elements), cmat (9 (3 by 3) elements), and dvec (4, 5, 8, 12, or 14 
    elements) all combined in a single column vector.  

    Parameters
    ----------
    filename : str
        file name that contains the rvec, tvec, cmat, and dvec. 
        If not given, this function asks the user.

    Returns
    -------
    imgSize : np.ndarray (shape: 2)
    rvec : np.ndarray (shape: 3)
        rvec rotational vector of the extrinsic parameters
    tvec : np.ndarray (shape: 3)
        tvec translational vector of the extrinsic parameters
    cmat : np.ndarray (shape: (3,3))
        cmat the camera matrix 
    dvec : np.ndarray (shape: n, n can be 4, 5, 8, 12, or 14)
        dvec the distortion coefficients (k1, k2, p1, p2[, k3, k4, k5, k6[
            , s1, s2, s3, s4[, taux, tauy]]]) 
    Example:
    --------
    Here is an example of the file of a camera:
    # Start of the example of the file
    # An example of camera parameters
    3840
    2160
    1.123
    -0.135
    0.135
    -700.
    800.
    5400.
    8000.
    0.
    2999.5
    0.
    8000.
    1999.5
    0.
    0.
    1.
    -0.02
    0.
    0.
    0.
    # End of the example of the file
    imgSize, rvec, tvec, cmat, dvec = readCamera('camera_1.txt')
    

    """
    imgSize = np.array([]) # mat[0:2]
    rvec = np.array([])   # mat[2:5]
    tvec = np.array([])   # mat[5:8]
    cmat = np.array([])   # mat[8:17]
    dvec = np.array([])   # mat[17:]
    # if filename is given
    if type(filename) == str:
        try:
            mat = np.loadtxt(filename, delimiter=',')
            mat = mat.flatten()
            if mat.size < 21:
                print("# Error. readCamera(): Matrix size is too small for camera parameters.")
                print("#  Matrix size should be >= 21")
                print("#  First 2 [0:2] for image size (width, height), ")
                print("#  followed by [2:8] for rvec and tvec,  ")
                print("#  3x3 (9) [8:17] for camera matrix, and at least 4 for")
                print("#  distortion coefficients [17:21 or more] (k1,k2,p1,p2[,k3[")
                print("#                                ,k4,k5,k6[")
                print("#                                ,s1,s2,s3,s4[")
                print("#                                ,taux,tauy]]]]")
            else:
                imgSize = mat[0:2].copy()
                rvec = mat[2:5].copy()
                tvec = mat[5:8].copy()
                cmat = mat[8:17].copy().reshape(3,3)
                dvec = mat[17:].copy()
                return imgSize.astype(int).flatten(), rvec.reshape(3, 1), tvec.reshape(3, 1), cmat.reshape(3, 3), dvec.reshape(1, -1)
        except Exception as e:
            print("# Error: readCamera() cannot read data from file (%s)."
                   % (filename))
            print("# Exception is", e)
            print("# The file needs to be in text format.")
    # ask user if filename is empty
    while (imgSize.size != 2 or rvec.size != 3 or tvec.size != 3 \
           or cmat.size != 9 or dvec.size < 4):
        print("\n# How do you want to input camera parameters:")
        print("#     file: reading matrix from file (text format)")
        print("#       which matrix size should be >= 19")
        print("#  First 2 [0:2] for image size (width, height), ")
        print("#  followed by [2:8] for rvec and tvec,  ")
        print("#  3x3 (9) [8:17] for camera matrix, and at least 4 for")
        print("#  distortion coefficients [17:21 or more] (k1,k2,p1,p2[,k3[")
        print("#                                ,k4,k5,k6[")
        print("#                                ,s1,s2,s3,s4[")
        print("#                                ,taux,tauy]]]]")
        print("#       For example: ")
        print("#           file")
        if (type(example) == type(None)):
            print("#           ..\\..\\examples\\camera_1.txt")
        else:
            print("#           %s" % (example))
        print("#    manual")
        print("#       for example:")
        print("#         manual")
        print("#           # imgSize")
        print("#           3840, 2160 ")
        print("#           # rvec")
        print("#           0, 0, 0 ")
        print("#           # tvec")
        print("#           0, 0, 0 ")
        print("#           # camera matrix")
        print("#           1800, 0, 959.5")
        print("#           0, 1800, 539.5")
        print("#           0,0,1")
        print("#           # distortion coefficients")
        print("#           0.0321, 0, 0, 0")
        uInput = input2().strip()
        if (uInput == 'file'):
            try:
                print("#  Enter file of camera parameters:")
                filename = input2()
                mat = np.loadtxt(filename, delimiter=',')
                mat = mat.flatten()
                if mat.size < 21:
                    print("# Error. readCamera(): Matrix size is too small for camera parameters.")
                    print("#  Matrix size should be >= 19")
                    print("#  First 2 [0:2] for image size (width, height), ")
                    print("#  followed by [2:8] for rvec and tvec,  ")
                    print("#  3x3 (9) [8:17] for camera matrix, and at least 4 for")
                    print("#  distortion coefficients [17:21 or more] (k1,k2,p1,p2[,k3[")
                    print("#                                ,k4,k5,k6[")
                    print("#                                ,s1,s2,s3,s4[")
                    print("#                                ,taux,tauy]]]]")
                else:
                    imgSize = mat[0:2].copy()
                    rvec = mat[2:5].copy()
                    tvec = mat[5:8].copy()
                    cmat = mat[8:17].copy().reshape(3,3)
                    dvec = mat[17:].copy()
            except Exception as e:
                print("# Error: readCamera() cannot read matrix from file (%s)."
                       % (filename))
                print("# Exception is", e)
                print("# Try again.")
                continue
        if (uInput[0:6] == 'manual'):
            print("#  Enter 2 integers for image size (width and height):")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            imgSize = np.zeros(2, dtype=float)
            print("#  Enter 3 reals for rvec:")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            rvec = np.zeros(3, dtype=float)
            for i in range(3):
                rvec[i] = float(datInput[i])
            print("#  Enter 3 reals for tvec:")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            tvec = np.zeros(3, dtype=float)
            for i in range(3):
                tvec[i] = float(datInput[i])
            print("#  Enter 3x3 reals for camera matrix:")
            print("#    three elements (one row) in a line:")
            print("#    For example:")
            print("#           1800, 0, 959.5")
            print("#           0, 1800, 539.5")
            print("#           0,0,1")
            cmat = np.eye(3, dtype=float)
            for i in range(3):
                datInput = input2("").strip()
                datInput = re.split(',| ', datInput)
                datInput = list(filter(None, datInput))
                for j in range(3):
                    cmat[i,j] = datInput[j]
            print("#  Enter n reals for distortion coefficients (n can be 4, 5, 8, 12, or 14):")
            print("#  distortion coefficients: k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,taux,tauy]]]]")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            nDistCoeffs = len(datInput)
            dvec = np.zeros(nDistCoeffs, dtype=float)
            for i in range(nDistCoeffs):
                dvec[i] = float(datInput[i])
        if (imgSize.size == 2 and rvec.size == 3 and tvec.size == 3 
            and cmat.size == 9 and dvec.size >= 4):
            theStr = cameraParametersToString(imgSize, rvec, tvec, cmat, dvec)
            print(theStr)
            break
        else:
            print("# readCamera(): Invalid size(s) of parameters.")
            print(imgSize.astype(int))
            print(rvec.reshape(3, 1))
            print(tvec.reshape(3, 1))
            print(cmat.reshape(3, 3))
            print(dvec.reshape(1, -1))
    return imgSize.astype(int).flatten(), rvec.reshape(3, 1), tvec.reshape(3, 1), cmat.reshape(3, 3), dvec.reshape(1, -1)
#        

