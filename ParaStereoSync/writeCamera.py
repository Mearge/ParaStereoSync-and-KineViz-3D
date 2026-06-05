import os
import numpy as np
import cv2 as cv

def writeCamera(savefile, imgSize, rvec, tvec, cmat, dvec):
    imgSize = np.array(imgSize).flatten()
    rvec = np.array(rvec).reshape(-1, 1)
    tvec = np.array(tvec).reshape(-1, 1)
    cmat = np.array(cmat).reshape(3, 3)
    dvec = np.array(dvec).reshape(1, -1)
    if imgSize.size != 2 or rvec.size != 3 or tvec.size != 3 \
        or cmat.size != 9 or dvec.size < 4:
        print("# Error. writeCamera(): Invalid sizes of parameters.")
        print("# imgSize size should be 2 but is %d" % np.array(imgSize).size)
        print("# rvec size should be 3 but is %d" % (rvec.size))
        print("# tvec size should be 3 but is %d" % (tvec.size))
        print("# cmat size should be 9 but is %d" % (cmat.size))
        print("# dvec size should be >=4 but is %d" % (dvec.size))
        return
    # generate a string that is easy to read
    # if savefile is cout or stdout, print to screen
    if savefile == 'cout' or savefile == 'stdout':
        _savefile = '__temp_writeCamera__.txt'
    else: 
        _savefile = savefile
    camAll = np.zeros(2+3+3+9+dvec.size, dtype=float)
    camAll[0:2] = np.array(imgSize).flatten()
    camAll[2:5] = rvec.flatten();
    camAll[5:8] = tvec.flatten();
    camAll[8:17] = cmat.flatten();
    camAll[17:] = dvec;
    theStr = cameraParametersToString(imgSize, rvec, tvec, cmat, dvec)
    np.savetxt(_savefile, camAll, fmt='%24.16e', delimiter=' , ',
            header='Camera paramters. imgSize, rvec, tvec, cmat (flatten), dvec', 
            footer=theStr)
    if savefile == 'cout' or savefile == 'stdout':
        with open(_savefile, 'r') as file:
            content = file.read()
            print(content)
#            os.remove(_savefile)   
    return         
        
def cameraParametersToString(imgSize, rvec, tvec, cmat, dvec):
    # generate a string that is easy to read
    imgSize = np.array(imgSize).flatten()
    r33, jmat= cv.Rodrigues(rvec.reshape(-1))
    r44 = np.eye(4, dtype=float)
    r44[0:3,0:3] = r33
    r44[0:3,3] = tvec.reshape(-1)
    r44inv = np.linalg.inv(r44)
    theStr = ''
    theStr += '# Image size (width, height) is (%d, %d)\n' % (imgSize[0], imgSize[1])
    theStr += '# rvec_x = %24.16e\n' % (rvec[0])
    theStr += '# rvec_y = %24.16e\n' % (rvec[1])
    theStr += '# rvec_z = %24.16e\n' % (rvec[2])
    theStr += '# tvec_x = %24.16e\n' % (tvec[0])
    theStr += '# tvec_y = %24.16e\n' % (tvec[1])
    theStr += '# tvec_z = %24.16e\n' % (tvec[2])
    theStr += '# r44 matrix:\n'
    for i in range(4):
        theStr += '#   %24.16e %24.16e %24.16e %24.16e\n' % \
                  (r44[i,0], r44[i,1], r44[i,2], r44[i,3])
    theStr += '# inverse of r44 matrix:\n'
    for i in range(4):
        theStr += '#   %24.16e %24.16e %24.16e %24.16e\n' % \
                  (r44inv[i,0], r44inv[i,1], r44inv[i,2], r44inv[i,3])
    theStr += "# fx = %24.16e (pixels)\n" % (cmat[0,0])
    theStr += '# fy = %24.16e (pixels)\n' % (cmat[1,1])
    theStr += '# cx = %24.16e (pixels)\n' % (cmat[0,2])
    theStr += '# cy = %24.16e (pixels)\n' % (cmat[1,2])
    distStr = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 
               's1', 's2', 's3', 's4', 'taux', 'tauy']
    for i in range(dvec.reshape(-1).size):
        if i <= 3 or dvec.reshape(-1)[i] != 0.0:
            theStr += '# %s = %24.16e\n' % (distStr[i], dvec.reshape(-1)[i])
    return theStr