import numpy as np
import cv2 as cv
from improMisc import npFromString


def validsOfPoints(points: np.ndarray):
    """
    This function allows user to input an array of points which some of them 
    have nan values, and returns a vector that indicates which are valid points.
    For example, 
    x = np.array([
           [ 1.,  2., np.nan],
           [ 3.,  4.,  5.],
           [np.nan,  6., np.nan],
           [ 4.,  5.,  7.],
           [ 3., np.nan,  6.],
           [ 4.,  3.,  2.],
           [ 5.,  3.,  5.],
           [ 4.,  3.,  2.]])
    validsOfPoints(x) returns array([0, 1, 0, 1, 0, 1, 1, 1], dtype=uint8)

    Parameters
    ----------
    points : np.ndarray (typcailly n-by-2 or n-by-3, n is number of points)

    Returns
    -------
    valids : TYPE
        DESCRIPTION.

    """
    valids = np.ones(points.shape[0], dtype=np.uint8)
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            if np.isnan(points[i, j]):
                valids[i] = 0
                break
    return valids



def newIndexOfValidsOfPoints(points: np.ndarray):
    """
    This function allows user to input an array of points which some of them 
    have nan values, and returns new indices of valid points.
    The index is zero based. That is, 0 means the first valid point, 1 means
    it is the 2nd valid point. -1 means it is an invalid point.
    For example, 
    x = np.array([
           [ 1.,  2., np.nan],
           [ 3.,  4.,  5.],
           [np.nan,  6., np.nan],
           [ 4.,  5.,  7.],
           [ 3., np.nan,  6.],
           [ 4.,  3.,  2.],
           [ 5.,  3.,  5.],
           [ 4.,  3.,  2.]])
    newIndexOfValidsOfPoints(x) returns 
    array([-1,0,-1,1,-1,2,3,4], dtype=int)

    Parameters
    ----------
    points : np.ndarray
        DESCRIPTION.

    Returns
    -------
    valids : TYPE
        DESCRIPTION.

    """
    validIndices = np.ones(points.shape[0], dtype=int) * -1
    validCount = 0
    isValid = True
    for i in range(points.shape[0]):
        isValid = True
        for j in range(points.shape[1]):
            if np.isnan(points[i, j]):
                isValid = False
                break
        if isValid == True:
            validIndices[i] = validCount
            validCount += 1
    return validIndices                


def countCalibPoints3d(matPoints3d):
    # assuming matPoints3d is a float N-by-3 ndarray
    if type(matPoints3d) != np.ndarray:
        print("# Error: countCalibPoints3d: matPoints3d must be a np.ndarray")
        return
    if matPoints3d.shape[1] != 3:
        print("# Error: countCalibPoints3d: matPoints3d must be N-by-3.")
        return
    # init return variables
    validsOfPoints3d = np.zeros(0)
    # if matPoints3d size is a multiple of 3, reshape it to Nx3 
    if matPoints3d.size % 3 == 0:
        npts3d = int(matPoints3d.size / 3)
        matPoints3d = matPoints3d.reshape([-1, 3])
        validsOfPoints3d = validsOfPoints(matPoints3d)
        nval3d = np.sum(validsOfPoints3d)
    else:
        npts3d = 0
        validsOfPoints3d = []
        nval3d = 0
    # return
    return validsOfPoints3d


def countCalibPoints2d(points2d):
    # init return variables
    validsOfPoints2d = np.zeros(0)
    # if points2d is a string, convert string to numpy array.
    try:
        if type(points2d) == str:
            matPoints2d = npFromString(points2d)
    except:
        return
    if type(points2d) == np.ndarray:
        matPoints2d = points2d
    # if matPoints2d size is a multiple of 2, reshape it to Nx2
    if matPoints2d.size % 2 == 0:
        npts2d = int(matPoints2d.size / 2)
        matPoints2d = matPoints2d.reshape([-1, 2])
        validsOfPoints2d = validsOfPoints(matPoints2d)
        nval2d = np.sum(validsOfPoints2d)
    else:
        npts2d = 0
        validsOfPoints2d = []
        nval2d = 0
    return validsOfPoints2d


def countCalibPoints(points3d: np.ndarray, points2d: np.ndarray):
    """
    This function counts the 3D points, image (2D) pointns, and valid points 
    for calibration.
    For example, 
        points3d = np.fromstring('0 0 0 \n 1 0 0 \n 1 1 0 \n 0 1 0 \n nan nan nan', sep=' ').reshape((-1, 3))
        (i.e., valid points: 0 1 2 3. invalid point: 4. total: 5 points)
        points2d = np.fromstring('1. 1. \n 2. 2. nan nan \n \n 1. 2. \n 3. 3.', sep=' ').reshape((-1, 2))
        (i.e., valid points: 0 1 3 4 invalid point: 4. total: 5 points)
        
        validsOfPoints3d, validsOfPoints2d, validCalibPoints,\
            idxAllToValid, idxValidToAll, validPoints3d, validPoints2d =\
        countCalibPoints(points3d, points2d) returns ==> 
        (where indices are zero-based)
        validsOfPoints3d would be array([1, 1, 1, 1, 0], dtype=uint8)
        validsOfPoints2d would be array([1, 1, 0, 1, 1], dtype=uint8)
        validCalibPoints would be array([1, 1, 0, 1, 0], dtype=uint8)
        idxAllToValid would be array([0,1,-1,2,-1], dtype=uint8)
        idxValidToAll would be array([0,1,3], dtype=uint8)
        validPoints3d would be an N-by-3 array of valid 3d points where N is 
                      number of valid points
        validPoints2d would be an N-by-2 array of valid image points

    Parameters
    ----------
    points3d : np.ndarray
        DESCRIPTION.
    points2d : np.ndarray
        DESCRIPTION.

    Returns
    -------
    validsOfPoints3d : TYPE
        DESCRIPTION.
    validsOfPoints2d : TYPE
        DESCRIPTION.
    validCalibPoints : TYPE
        DESCRIPTION.
    idxAllToValid : TYPE
        DESCRIPTION.
    idxValidToAll : TYPE
        DESCRIPTION.
    validPoints3d : TYPE
        DESCRIPTION.
    validPoints2d : TYPE
        DESCRIPTION.

    """
    # initialize return arrays
    validsOfPoints3d = np.zeros(0)
    validsOfPoints2d = np.zeros(0)
    validCalibPoints = np.zeros(0)
    idxAllToValid = np.zeros(0)
    idxValidToAll = np.zeros(0)
    validPoints3d = np.zeros(0)
    validPoints2d = np.zeros(0)    
    # check points3d
    if type(points3d) != np.ndarray or points3d.shape[1] != 3:
        print("# Error: countCalibPoints: points3d must be an N-by-3 array.")
        return 
    npts3d = points3d.shape[0]
    validsOfPoints3d = validsOfPoints(points3d)
    npts3d = int(points3d.size / 3)
    # check points2d 
    if type(points2d) != np.ndarray or points2d.shape[1] != 2:
        print("# Error: countCalibPoints: points2d must be an N-by-2 array.")
        return 
    npts2d = points2d.shape[0]
    validsOfPoints2d = validsOfPoints(points2d)
    # valid calibration points
    # validsOf3d2d is True only if npts2d == npts3d, and 
    # both validsOfPoints2d and validsOfPoints3d are true (1)
    # for example: validsOfPoints3d: [1, 1, 1, 1, 0]
    #              validsOfPoints2d: [1, 1, 0, 1, 1]
    #              validCalibPoints: [1, 1, 0, 1, 0]
    #              idxAllToValid:    [0, 1,-1, 2,-1]
    #              idxValidToAll:    [0, 1, 3]
    validCalibPoints = np.zeros(0, dtype=np.uint8)
    nValidCalibPoints = 0
    if npts2d == npts3d:
        validCalibPoints =np.zeros(npts2d, dtype=np.uint8)
        for i in range(npts2d):
            if validsOfPoints2d[i] >= 1 and validsOfPoints3d[i] >= 1:
                validCalibPoints[i] = 1
                nValidCalibPoints += 1
        # idxAllToValid would be array([0,1,-1,2,-1], dtype=uint8)
        # idxValidToAll would be array([0,1,3])
        idxAllToValid = np.ones(npts2d, dtype=np.int16) * (-1)
        idxValidToAll = np.ones(nValidCalibPoints, dtype=np.int16) * (-1)
        validCount = 0
        for i in range(npts2d):
            if validCalibPoints[i] >= 1:
                idxAllToValid[i] = validCount
                idxValidToAll[validCount] = i
                validCount += 1
        # validPoints3d would be valid calibration points of matPoints3d
        # validPoints2d would be valid calibration points of matPoints2d
        validPoints3d = np.zeros((nValidCalibPoints, 3), dtype=float)
        validPoints2d = np.zeros((nValidCalibPoints, 2), dtype=float)
        for i in range(nValidCalibPoints):
            k = idxValidToAll[i]
            validPoints3d[i,:] = points3d[k,:]
            validPoints2d[i,:] = points2d[k,:]
    # print message
    # return          
    return validsOfPoints3d, validsOfPoints2d, validCalibPoints, \
           idxAllToValid, idxValidToAll, validPoints3d, validPoints2d

    
def newArrayByMapping(oriArray, newIdx):
    """
    This function returns a new 2D array that maps to the original 2D array, 
    given the indices of the new 2D array.
    For example, 
    oriArray = np.array(
           [[0., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    nexIdx = np.array([ 0,  1, -1,  2, -1])
    The newArrayByMapping(oriArray, newIdx) would be:
        np.array([[ 0.,  0.,  0.],
                  [ 1.,  0.,  0.],
                  [nan, nan, nan],
                  [ 0.,  1.,  0.],
                  [nan, nan, nan]])
    For example, 
    oriArray = np.array([[ 0.,  0.,  0.],
                         [ 1.,  0.,  0.],
                         [np.nan, np.nan, np.nan],
                         [ 0.,  1.,  0.],
                         [np.nan, np.nan, np.nan]])
    newIdx = np.array([0, 1, 3])
    The newArrayByMapping(oriArray, newIdx) would be:
            array([[0., 0., 0.],
                   [1., 0., 0.],
                   [0., 1., 0.]])
    Parameters
    ----------
    oriArray : TYPE
        DESCRIPTION.
    newIdx : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    newIdx = newIdx.flatten()
    maxIdx = np.max(newIdx)
    if oriArray.shape[0] <= maxIdx:
        print("# Error: newArrayByMapping: oriArray does not have sufficient rows to map to new indices.")
        print("     new indices are ", newIdx)
        print("     oriArray has only %d rows." % (oriArray.shape[0]))
#        return
    nNewRows = newIdx.size
    newArray = np.ones((nNewRows, oriArray.shape[1]), oriArray.dtype) * np.nan
    for i in range(newIdx.size):
        if newIdx[i] >= 0:
            newArray[i,:] = oriArray[newIdx[i]]
    return newArray

