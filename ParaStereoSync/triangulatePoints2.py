import cv2 as cv
import numpy as np 
from inputs import input2, input3
from readPoints import readPoints
from writePoints import writePoints
from readCamera import readCamera
from drawPoints import drawPoints
from createFileList import createFileList

def triangulatePoints2(
        cmat1, dvec1, rvec1, tvec1, 
        cmat2, dvec2, rvec2, tvec2, 
        imgPoints1, imgPoints2):
    """
    This function triangulates points from given two sets of image 
    coordinates of N points, intrinsic and extrinsic parameters of 
    two cameras.   
    
    Parameters
    ----------
    cmat1 : numpy ndarray, a 3x3 np.float64 numpy matrix
    dvec1 : numpy ndarray, a 1D np.floatr64 numpy matrix
    rvec1 : numpy ndarray, 3-element np.float64 numpy matrix
    tvec1 : numpy ndarray, 3-element np.float64 numpy matrix
    cmat2 : numpy ndarray, a 3x3 np.float64 numpy matrix
    dvec2 : numpy ndarray, a 1D np.floatr64 numpy matrix
    rvec2 : numpy ndarray, 3-element np.float64 numpy matrix
    tvec2 : numpy ndarray, 3-element np.float64 numpy matrix
    imgPoints1 : numpy ndarray, Nx2 2D array of N points 
        image coordinates of N points in camera 1 (in original photo)
    imgPoints2 : numpy ndarray, Nx2 2D array of N points 
        image coordinates of N points in camera 2 (in original photo)

    Returns
    -------
    objPoints : numpy ndarray, Nx3 np.float64 numpy matrix
        object points triangulated, in world coordinate
    objPoints1 : numpy ndarray, Nx3 np.float64 numpy matrix
        object points triangulated, in camera-1 coordinate
    objPoints2 : numpy ndarray, Nx3 np.float64 numpy matrix
        object points triangulated, in camera-2 coordinate
    prjPoints1 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected points in camera-1 image coordinate
    prjPoints2 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected points in camera-2 image coordinate
    prjErrors1 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected errors in camera-1 image coordinate
        i.e., prjPoints1 - imgPoints1
    prjErrors2 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected errors in camera-2 image coordinate
        i.e., prjPoints2 - imgPoints2


    """
    # force reshape and type conversion 
    imgPoints1 = np.array(imgPoints1, dtype=np.float64).reshape(-1,2)
    imgPoints2 = np.array(imgPoints2, dtype=np.float64).reshape(-1,2)
    # initialization (to zero-size arrays)
    objPoints = np.zeros((0))
    prjPoints1 = np.zeros((0))
    prjPoints2 = np.zeros((0))
    prjErrors1 = np.zeros((0))
    prjErrors2 = np.zeros((0))
    # check 
    nPoints = imgPoints1.shape[0]
    if imgPoints2.shape[0] != nPoints:
        print("# Error: triangulatePoints2(): imgPoints1 and 2 have"
              " different number of points.")
        return objPoints, prjPoints1, prjPoints2, prjErrors1, prjErrors2 
    # memory allocation
    objPoints = np.ones((nPoints, 3), dtype=np.float64) * np.nan
#    prjPoints1 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
#    prjPoints2 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
#    prjErrors1 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
#    prjErrors2 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
    prjMat1 = np.zeros((3,4), dtype=np.float64)
    prjMat2 = np.zeros((3,4), dtype=np.float64)
    rctMat1 = np.zeros((3,3), dtype=np.float64)
    rctMat2 = np.zeros((3,3), dtype=np.float64)
    qMat = np.zeros((3,4), dtype=np.float64)
    undPoints1 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
    undPoints2 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
    # Calculate rmat, tvec from coord of left to right 
    r44L = np.eye(4, dtype=np.float64)
    r44R = np.eye(4, dtype=np.float64)
    r44L[0:3,0:3] = cv.Rodrigues(rvec1)[0]
    r44L[0:3,3] = tvec1.flatten()
    r44R[0:3,0:3] = cv.Rodrigues(rvec2)[0]
    r44R[0:3,3] = tvec2.flatten()
    r44 = np.matmul(r44R, np.linalg.inv(r44L))
    r33 = r44[0:3,0:3].copy()
    rvec = cv.Rodrigues(r33)[0]
    tvec = r44[0:3,3].copy()
    # stereo rectify
    rctMat1, rctMat2, prjMat1, prjMat2, qMat, dum1, dum2 = \
        cv.stereoRectify(cmat1, dvec1, cmat2, dvec2, (1000,1200), r33, tvec)
    # undistortion
    #    cv::undistortPoints(xLC2, uL, camMatrix1, distVect1, rcL, pmL);
    #    cv::undistortPoints(xRC2, uR, camMatrix2, distVect2, rcR, pmR);
    undPoints1 = cv.undistortPoints(imgPoints1, cmat1, dvec1, undPoints1, 
                                    rctMat1, prjMat1).reshape(-1,2)
    undPoints2 = cv.undistortPoints(imgPoints2, cmat2, dvec2, undPoints2, 
                                    rctMat2, prjMat2).reshape(-1,2)
    # triangulation 
    objPoints = cv.triangulatePoints(prjMat1, prjMat2, undPoints1.transpose(), 
                                     undPoints2.transpose())
    # coordinate transformation to cam-1 coord.
    rctInv1 = np.eye(4, dtype=np.float64)
    rctInv1[0:3,0:3] = np.linalg.inv(rctMat1)
    objPoints = np.matmul(rctInv1, objPoints)
    # object points in cam1, cam2, world coordinate
    objPoints1 = objPoints.copy()
    objPoints2 = np.matmul(r44, objPoints1)
    objPoints = np.matmul(np.linalg.inv(r44L), objPoints).transpose()
    objPoints1 = objPoints1.transpose()
    objPoints2 = objPoints2.transpose()
    for iPt in range(objPoints.shape[0]):
        for ix in range(3):
            objPoints[iPt,ix] /= objPoints[iPt,3]
            objPoints1[iPt,ix] /= objPoints1[iPt,3]
            objPoints2[iPt,ix] /= objPoints2[iPt,3]
    objPoints = objPoints[:,0:3]
    objPoints1 = objPoints1[:,0:3]
    objPoints2 = objPoints2[:,0:3]
    # project points 
    prjPoints1 = cv.projectPoints(objPoints, rvec1, tvec1, cmat1, dvec1)[0].reshape(-1,2)
    prjPoints2 = cv.projectPoints(objPoints, rvec2, tvec2, cmat2, dvec2)[0].reshape(-1,2)
    # projection errors
    prjErrors1 = prjPoints1 - imgPoints1
    prjErrors2 = prjPoints2 - imgPoints2
    
    return objPoints, objPoints1, objPoints2, prjPoints1, prjPoints2, prjErrors1, prjErrors2







if __name__ == '__main__':
    rvec1, tvec1, cmat1, dvec1 = readCamera('examples/triangulatePoints2/camera_left.csv')
    rvec2, tvec2, cmat2, dvec2 = readCamera('examples/triangulatePoints2/camera_right.csv')
    imgPoints1 = readPoints('examples/triangulatePoints2/picked30_left_0000.csv')[:,0:2]
    imgPoints2 = readPoints('examples/triangulatePoints2/picked30_right_0000.csv')[:,0:2]
    objPoints, objPoints1, objPoints2, \
        prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
        triangulatePoints2(cmat1, dvec1, rvec1, tvec1, 
                           cmat2, dvec2, rvec2, tvec2, imgPoints1, imgPoints2)

    c6x = np.zeros((1,3), dtype=np.float64)
    c6x[:] += (objPoints2[4,:] - objPoints2[0,:]) + (objPoints2[9,:] - objPoints2[5,:])
    c6x[:] += (objPoints2[15,:] - objPoints2[12,:]) + (objPoints2[19,:] - objPoints2[16,:])
    c6x /= np.linalg.norm(c6x)
    c6y = np.zeros((1,3), dtype=np.float64)
    for i in range(5):
        c6y[:] += (objPoints2[0 + i,:] - objPoints2[5 + i,:])
    c6y /= np.linalg.norm(c6y)
    c6z = np.cross(c6x,c6y)
    c6z /= np.linalg.norm(c6z)
    c6y = np.cross(c6z, c6x)
    c6y /= np.linalg.norm(c6y)
    r44_c6 = np.eye(4, dtype=np.float64)
    r44_c6[0:3,0] = c6x
    r44_c6[0:3,1] = c6y
    r44_c6[0:3,2] = c6z
    r44_c6[0:3,3] = objPoints[0,:]
    r44_c6_inv = np.linalg.inv(r44_c6)

    # draw 
    img1 = cv.imread("examples/triangulatePoints2/samples/brb1_left_0000.JPG")
    img1 = drawPoints(img1, imgPoints1, color=[0,0,0], thickness=4, savefile=".")
    img1 = drawPoints(img1, imgPoints1, color=[255,0,0], thickness=2, savefile=".")
    img1 = drawPoints(img1, prjPoints1, color=[0,0,0], thickness=4, savefile=".")
    img1 = drawPoints(img1, prjPoints1, color=[0,0,255], thickness=2, savefile="examples/triangulatePoints2/samples/brb1_prj.JPG") 
    img2 = cv.imread("examples/triangulatePoints2/samples/brb1_right_0000.JPG")
    img2 = drawPoints(img2, imgPoints2, color=[0,0,0], thickness=4, savefile=".")
    img2 = drawPoints(img2, imgPoints2, color=[255,0,0], thickness=2, savefile=".")
    img2 = drawPoints(img2, prjPoints2, color=[0,0,0], thickness=4, savefile=".")
    img2 = drawPoints(img2, prjPoints2, color=[0,0,255], thickness=2, savefile="examples/triangulatePoints2/samples/brb2_prj.JPG") 


### Matlab (with Bouquet's toolbox) check
# cmat1 = [[1.98888802e+03 0.00000000e+00 2.02921156e+03]
#  [0.00000000e+00 1.98210177e+03 1.50853102e+03]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]];
# dvec1 = [-0.00659889  0.          0.          0.          0.        ];
# dvec1 = [-0.00659889  0.          0.          0.          0.        ];
# rvec1=[0 0 0 ]; tvec1=[0 0 0];
# cmat2=[[1.92745626e+03 0.00000000e+00 2.02121427e+03]
#  [0.00000000e+00 1.92639892e+03 1.51642305e+03]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]];
# dvec2=[-0.00291725  0.          0.          0.          0.        ];
# rvec2=[-0.02618424 -0.01501711  0.00072262];
# tvec2=[-40.46213113  72.02069942 -31.14572627];
# imgPoints1=[[2312.82994924 1931.22357143]
#  [2318.34864865 1773.27428571]
#  [2319.88673139 1616.51428571]
#  [2327.61450382 1449.57428571]
#  [2326.87883959 1278.205     ]];
# imgPoints2=[[2049.38190955 2411.83214286]
#  [2051.91377091 2245.1       ]
#  [2050.73699422 2078.48214286]
#  [2056.46474635 1903.02      ]
#  [2053.13297872 1722.72857143]];
 
# xL = imgPoints1'; 
# xR = imgPoints2';
# om = rvec2';
# T = tvec2';
# fc_left = [cmat1(1,1) cmat1(2,2)];
# cc_left = [cmat1(1,3) cmat1(2,3)];
# kc_left = dvec1;
# alpha_c_left = 0.0;
# fc_right = [cmat2(1,1) cmat2(2,2)];
# cc_right = [cmat2(1,3) cmat2(2,3)];
# kc_right = dvec2;
# alpha_c_right = 0.0;
# [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right); 



# triangulatePoints_test()





