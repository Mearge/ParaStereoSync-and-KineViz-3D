import numpy as np
import cv2 as cv
import re
from inputs import input2, str2Ndarray
from imread2 import imread2
from readCamera import readCamera

def draw3dMesh(img=None, cmat=None, dvec=None, rvec=None, tvec=None,
               meshx=None, meshy=None, meshz=None, 
               color=None, thickness=None, shift=None, 
               savefile=None):
    """

    Parameters
    ----------
    img : TYPE, optional
        DESCRIPTION. The default is None.
    cmat : TYPE, optional
        DESCRIPTION. The default is None.
    dvec : TYPE, optional
        DESCRIPTION. The default is None.
    rvec : TYPE, optional
        DESCRIPTION. The default is None.
    tvec : TYPE, optional
        DESCRIPTION. The default is None.
    meshx : TYPE, optional
        DESCRIPTION. The default is None.
    meshy : TYPE, optional
        DESCRIPTION. The default is None.
    meshz : TYPE, optional
        DESCRIPTION. The default is None.
    color : TYPE, optional
        DESCRIPTION. The default is None.
    thickness : TYPE, optional
        DESCRIPTION. The default is None.
    shift : TYPE, optional
        DESCRIPTION. The default is None.
    savefile : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    imgCopy : TYPE
        DESCRIPTION.

    """
    # image
    if type(img) == type(None):
        print("# Enter image file:")
        img = imread2(example="examples/draw3dMesh/brb2_cam6.JPG")
    if type(img) == str:
        imgStr = img
        img = imread2(imgStr)
    # camera parameters
    if type(cmat) == type(None) or type(dvec) == type(None) or\
       type(rvec) == type(None) or type(tvec) == type(None):
        print("# Enter camera parameters:")
        rvec, tvec, cmat, dvec = readCamera(
            example='examples/draw3dMesh/brb2_cam6.csv')
    # mesh 
    if type(meshx) == type(None):
        print("# Enter a Numpy array for X dimension:")
        print("#  For example: ")
        print("#    [-300,-200,-100,0,100,200,300], or ")
        print("#    np.linspace(-300,300,6), or ")
        print("#    0.0 (single value if this dimension has single layer")
        meshx = str2Ndarray(input2()).flatten()
    if type(meshy) == type(None):
        print("# Enter a Numpy array for Y dimension:")
        print("#  For example: ")
        print("#    [-300,-200,-100,0,100,200,300], or ")
        print("#    np.linspace(-300,300,6), or ")
        print("#    0.0 (single value if this dimension has single layer")
        meshy = str2Ndarray(input2()).flatten()
    if type(meshz) == type(None):
        print("# Enter a Numpy array for Z dimension:")
        print("#  For example: ")
        print("#    [-300,-200,-100,0,100,200,300], or ")
        print("#    np.linspace(-300,300,6), or ")
        print("#    0.0 (single value if this dimension has single layer")
        meshz = str2Ndarray(input2()).flatten()
    # color
    if type(color) == type(None):
        print("# Enter color you want to draw mesh (in BGR format):")
        print("# For example: ")
        print("#   (0, 255, 0) for green")
        datInput = input2("").strip()
        color = eval(datInput)
    # thickness
    if type(thickness) == type(None):
        print("# Enter line thickness you want to draw mesh (in pixel):")
        print("# For example: ")
        print("#   3")
        datInput = input2("").strip()
        thickness = eval(datInput)
    # shift
    if type(shift) == type(None):
        shift = 0
    # savefile
    if (type(savefile) == type(None)):
        print("# Enter the file to save the image:")
        print("#   (or enter a single dot (.) to skip saving.)")
        print("# For example, examples/draw3dMesh/brb2_cam6_meshed.JPG")
        savefile = input2("").strip()
        if (len(savefile) > 1):
            print("# The drawn image will be saved in file: %s" % savefile)
        else:
            print("# The drawn image will not be saved in any file.")
        
        
    # generate 3d mesh
    meshx = meshx.flatten()
    meshy = meshy.flatten()
    meshz = meshz.flatten()
    nx = meshx.shape[0]
    ny = meshy.shape[0]
    nz = meshz.shape[0]
    objPoints = np.ones((nx, ny, nz, 3), dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                objPoints[ix,iy,iz,0] = meshx[ix]
                objPoints[ix,iy,iz,1] = meshy[iy]
                objPoints[ix,iy,iz,2] = meshz[iz]
    # find projected points in image
    objPoints3f = objPoints.reshape((-1,1,3))
    imgPoints2f, jacobian = cv.projectPoints(objPoints3f, 
                                             rvec, tvec, cmat, dvec)
    imgPoints = imgPoints2f.reshape((nx, ny, nz, 2))
    imgCopy = img.copy()
    lineType = cv.LINE_AA
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # draw mesh along x direction (to next ix)
                if ix < nx - 1:
                    pt1 = np.array(imgPoints[ix,iy,iz,:] + 0.5, dtype=int)
                    pt2 = np.array(imgPoints[ix+1,iy,iz,:] + 0.5, dtype=int)
                    cv.line(imgCopy, pt1, pt2, (0,0,0), thickness + 2, 
                            lineType, shift)
                    cv.line(imgCopy, pt1, pt2, color, thickness, 
                            lineType, shift)
                if iy < ny - 1:
                    pt1 = np.array(imgPoints[ix,iy,iz,:] + 0.5, dtype=int)
                    pt2 = np.array(imgPoints[ix,iy+1,iz,:] + 0.5, dtype=int)
                    cv.line(imgCopy, pt1, pt2, (0,0,0), thickness + 2, 
                            lineType, shift)
                    cv.line(imgCopy, pt1, pt2, color, thickness, 
                            lineType, shift)
                if iz < nz - 1:
                    pt1 = np.array(imgPoints[ix,iy,iz,:] + 0.5, dtype=int)
                    pt2 = np.array(imgPoints[ix,iy,iz+1,:] + 0.5, dtype=int)
                    cv.line(imgCopy, pt1, pt2, (0,0,0), thickness + 2, 
                            lineType, shift)
                    cv.line(imgCopy, pt1, pt2, color, thickness, 
                            lineType, shift)
    # save file 
    if (len(savefile) > 1):
        cv.imwrite(savefile, imgCopy)
    
    # return
    return imgCopy


def test_draw3dMesh():
    img = cv.imread('examples/draw3dMesh/brb2_cam6.jpg')
    rvec,tvec,cmat,dvec = readCamera('examples/draw3dMesh/brb2_cam6.csv')
    imgGrid = draw3dMesh('examples/draw3dMesh/brb2_cam6.jpg', 
                         cmat, dvec, rvec, tvec, 
                         np.linspace(-300,300,11), 
                         np.linspace(-300,300,11), 
                         np.array([0]), 
                         thickness=3, 
                         color=(0,255,0),
                         savefile='examples/draw3dMesh/brb2_cam6_meshed.JPG')
    # resize (to be able to screen)
    imgSmall = cv.resize(imgGrid, dsize=(imgGrid.shape[1] // 4, imgGrid.shape[0] // 4))
    cv.imshow("Small mesh image", imgSmall)
    ikey = cv.waitKey(0)
    cv.destroyWindow("Small mesh image")
#p    cv.imshow("Image with grid", imgGrid); 
#    cv.waitKey(0)
#    cv.destroyWindow('Image with grid')
        

if __name__ == '__main__':
    test_draw3dMesh()