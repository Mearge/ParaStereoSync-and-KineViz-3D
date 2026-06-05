import os, sys, time
import glob
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def isImreadSupported(file):
    """
    This function returns if the given file is supported by OpenCV imread(),
    by (only) checking the extension file name.
    For example,
        isImreadSupported('apple.jpg') returns True.
        isImreadSupported('pineapple.gif') returns False. (as OpenCV imread()
            does not read GIF file.)
    """
    supportedExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',\
                    '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.sr',\
                    '.ras', '.tif', '.tiff', '.exr', '.hdr', '.pic',\
                    '.BMP', '.DIB', '.JPEG', '.JPG', '.JPE', '.JP2', '.PNG',\
                    '.WEBP', '.PBM', '.PGM', '.PPM', '.PXM', '.PNM', '.SR',\
                    '.RAS', '.TIF', '.TIFF', '.EXR', '.HDR', '.PIC'];
    if type(file) != str:
        return False
    ext = file[file.rfind('.'):]
    if ext in supportedExt:
        return True
    else:
        return False


def imreadSupportedFiles(files):
    """
    This function returns a list of files which are supported by OpenCV imread()
    by (only) checking their extension file names.
    For example,
        imreadSupportedFiles(["apple.jpg", "pineapple.gif", "orange.tif"])
        returns ["apple.jpg", "orange.tif"]
        (as OpenCV imread() does not read GIF file.)
    """
    if type(files) != list:
        return []
    nfile = len(files)
    returnFiles = []
    for ifile in range(nfile):
        file = files[ifile]
        if (isImreadSupported(file) == True):
            returnFiles.append(file)
    return returnFiles


def xcorr(x1, x2):
    """
    Cross-correlation function estimates.
    """
    corr = np.correlate(x1, x2, mode='full')
    lags = np.arange(1 - x1.size, x2.size)
    return corr, lags


def camPos(rvec, tvec):
    """
    This function returns camera position given rvec and tvec
    """
    r44 = np.eye(4, dtype=np.float64)
    if rvec.size == 3:
        r44[0:3,0:3] = cv.Rodrigues(rvec)[0]
    elif rvec.size == 9:
        r44[0:3,0:3] = rvec.reshape((3,3))
    else:
        print("# Warning: camPos: rvec is neither 3x1 or 3x3. Returned zeros.")
        return np.zeros(3, dtype=np.float64)
    r44[0:3,3] = tvec.reshape(3)
    r44inv = np.linalg.inv(r44)
    return r44inv[0:3,3].reshape(3)


def inputdlg(prompt: list, name: str='Input Dialog', numlines: int =1, defaultanswer=[]):
    """
    ThiAs function mimics matlab function inputdlg but only supports limited 
    functionality. Only arguments prompt, name, and defaultanswer are supported.
    The arguments numlines and other arguments of matlab inputdlg are
    not supported yet. 

    Example:
        prompt=['Enter the matrix size for x^2:','Enter the colormap name:'];
        name='Input for Peaks function';
        numlines = 1;
        defaultanswer = ['20', 'hsv']
        answers = inputdlg(prompt, name, numlines, defaultanswer)
        if len(answers) > 0:
            print("The inputs are:")
            print(answers)
        else:
            print("Input dialog is cancelled.")
    """
    if type(prompt) == str:
        prompt = [prompt]    
    answers = []
    edits = []
    window = tk.Tk()
    window.title(name)
    nrow = len(prompt)
    for i in range(nrow):
        frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 0)
        frame.grid(row=i, column=0)
        label = tk.Label(master=frame, text=prompt[i])
        label.pack()
        #
        frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
        frame.grid(row=i, column=1)
        edits.append(tk.Entry(master=frame, width=100))
        if i < len(defaultanswer): 
            edits[i].insert(0, defaultanswer[i])
        edits[i].pack()
    frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
    frame.grid(row=nrow, column=0)
    bt1 = tk.Button(master=frame, text='OK')
    bt1.pack()
    frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
    frame.grid(row=nrow, column=1)
    bt2 = tk.Button(master=frame, text='Cancel')
    bt2.pack()
    def eventOK(e):
        for i in range(nrow):
            answers.append(edits[i].get())
        window.destroy()
        window.quit()
    def eventCancel(e):
        answers = []
        window.destroy()
        window.quit()
    bt1.bind('<Button>', eventOK)
    bt2.bind('<Button>', eventCancel)
    window.mainloop()
    return answers


def uigetfile(fileDialogTitle='Select the file to open', initialDirectory='/', fileTypes = (('All files', '*.*'), ('TXT files', '*.txt;*.TXT'), ('JPG files', '*.jpg;*.JPG;*.JPEG;*.jpeg'), ('BMP files', '*.bmp;*.BMP'), ('Csv files', '*.csv'), ('opencv-supported images', '*.bmp;*.BMP;*.pbm;*.PBM;*.pgm;*.PGM;*.ppm;*.PPM;*.sr;*.SR;*.ras;*.RAS;*.jpeg;*.JPEG;*.jpg;*.JPG;*.jpe;*.JPE;*.jp2;*.JP2;*.tif;*.TIF;*.tiff;*.TIFF'), )):
    filePath = []
    fileName = []    
    tmpwin = tk.Tk()
    tmpwin.lift()    
    #window.iconify()  # minimize to icon
    #window.withdraw()  # hide it 
    fullname = filedialog.askopenfilename(title=fileDialogTitle, initialdir=initialDirectory, filetypes=fileTypes)        
    tmpwin.destroy()
    if fullname:
        allIndices = [i for i, val in enumerate(fullname) if val == '/']
        filePath = fullname[0 : 1+max(allIndices)]
        fileName = fullname[1+max(allIndices) : ]
    return filePath, fileName


def uigetfiles(fileDialogTitle='Select the files to open', initialDirectory='/', fileTypes = (('All files', '*.*'), ('TXT files', '*.txt;*.TXT'), ('JPG files', '*.jpg;*.JPG;*.JPEG;*.jpeg'), ('BMP files', '*.bmp;*.BMP'), ('Csv files', '*.csv'), ('opencv-supported images', '*.bmp;*.BMP;*.pbm;*.PBM;*.pgm;*.PGM;*.ppm;*.PPM;*.sr;*.SR;*.ras;*.RAS;*.jpeg;*.JPEG;*.jpg;*.JPG;*.jpe;*.JPE;*.jp2;*.JP2;*.tif;*.TIF;*.tiff;*.TIFF'), )):
    filePaths = []
    fileNames = []
    tmpwin = tk.Tk()
    tmpwin.lift()
    #window.iconify()  # minimize to icon
    #window.withdraw()  # hide it 
    fullnames = filedialog.askopenfilenames(title=fileDialogTitle, initialdir=initialDirectory, filetypes=fileTypes)        
    tmpwin.destroy()
    for i in range(len(fullnames)):
        ops = os.path.split(fullnames[i])
        filePaths.append(ops[0])           
        fileNames.append(ops[1])           
    return filePaths, fileNames


def uigetfiles_tupleFullpath(fileDialogTitle='Select the files to open', initialDirectory='/', fileTypes = (('All files', '*.*'), ('TXT files', '*.txt;*.TXT'), ('JPG files', '*.jpg;*.JPG;*.JPEG;*.jpeg'), ('BMP files', '*.bmp;*.BMP'), ('Csv files', '*.csv'), ('opencv-supported images', '*.bmp;*.BMP;*.pbm;*.PBM;*.pgm;*.PGM;*.ppm;*.PPM;*.sr;*.SR;*.ras;*.RAS;*.jpeg;*.JPEG;*.jpg;*.JPG;*.jpe;*.JPE;*.jp2;*.JP2;*.tif;*.TIF;*.tiff;*.TIFF'), )):
    tmpwin = tk.Tk()
    tmpwin.lift()
    #window.iconify()  # minimize to icon
    #window.withdraw()  # hide it 
    fullnames = filedialog.askopenfilenames(title=fileDialogTitle, initialdir=initialDirectory, filetypes=fileTypes)        
    tmpwin.destroy()
    # fullnames is a tuple and could be ('c:/d1/f1.ext', 'c:/d1/f2.ext')
    return fullnames


def uiputfile(fileDialogTitle='Select the file to save', initialDirectory='/', fileTypes = (('All files', '*.*'), ('TXT files', '*.txt;*.TXT'), ('JPG files', '*.jpg;*.JPG;*.JPEG;*.jpeg'), ('BMP files', '*.bmp;*.BMP'), ('Csv files', '*.csv'), ('opencv-supported images', '*.bmp;*.BMP;*.pbm;*.PBM;*.pgm;*.PGM;*.ppm;*.PPM;*.sr;*.SR;*.ras;*.RAS;*.jpeg;*.JPEG;*.jpg;*.JPG;*.jpe;*.JPE;*.jp2;*.JP2;*.tif;*.TIF;*.tiff;*.TIFF'), )):
    filePath = []
    fileName = []    
    tmpwin = tk.Tk()
    tmpwin.lift()    
    #window.iconify()  # minimize to icon
    #window.withdraw()  # hide it 
    fullname = filedialog.asksaveasfilename(title=fileDialogTitle, initialdir=initialDirectory, filetypes=fileTypes)        
    tmpwin.destroy()
    if fullname:
        allIndices = [i for i, val in enumerate(fullname) if val == '/']
        filePath = fullname[0 : 1+max(allIndices)]
        fileName = fullname[1+max(allIndices) : ]
    return filePath, fileName


def npFromString(theStr):
    if type(theStr) != str:
        return np.array([])
    _str = theStr
    _str = _str.replace(',', ' ').replace(';', ' ').replace('[', ' ')
    _str = _str.replace(']', ' ').replace('na ', 'nan').replace('\n',' ')
    _str = _str.replace('n/a', 'nan').replace('#N/A', 'nan')
    mat = np.fromstring(_str, dtype=float, sep=' ')
    return mat

def trainImgFilesToImgArray(files: list):
    """
    This function converts files (list of file names) to single image array.
    
    Example
    -------
    # In this case, imgFiles would be ['xxx/CFD_001.JPG', 'xxx/CFD_002.JPG', 'xxx/CFD_003.JPG']
    files = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:3]
    # In this case, imgArray would be an np.ndarray (shape=(3,448,448,3), dtype=np.uint8)
    imgArray = trainImgFilesToImgArray(files)
    """
    img = cv.imread(files[0])
    if type(img) == type(None):
        print("# Error: trainImgFilesToImgArray(): Cannot read image from %s" % files[0])
        return
    # get image dimension (imgH, imgW, imgNc) (even if it has only one channel)
    img = img.reshape(img.shape[0], img.shape[1], -1)
    (imgH, imgW, imgNc) = img.shape
    # allocate image array
    imgArray = np.zeros((len(files), imgH, imgW, imgNc), dtype=np.uint8)
    # read image and fill in imgArray
    for i in range(len(files)):
        img = cv.imread(files[i])
        if type(img) == type(None):
            print("# Error: trainImgFilesToImgArray(): Cannot read image from %s" % files[i])
            return
        img = img.reshape(img.shape[0], img.shape[1], -1)
        (imgHi, imgWi, imgNci) = img.shape
        if imgHi != imgH or imgWi != imgW or imgNci != imgNc:
            print("# Warning: The %d-th file has inconsistent image size (or number of channels)." % (i + 1))
            continue
        imgArray[i,:,:,:] = img[:,:,:]
    return imgArray        


def imgGridSlice(images: np.ndarray, cellHeight: int, cellWidth: int):
    """
    This function slices images into grids of images. For a 448-by-448-pixel image is to 
    be sliced into 49 64-by-64-pixel images (or 49 cells). 

    Parameters
    ----------
    images : np.ndarray shape=(N, imgHeight, imgWidth, nChannel) or 
             shape=(N, cellHeight, cellWidth) if it is gray image
        images to be sliced
    cellHeight : int
        the height of each cell to be sliced
    cellWidth : int
        the width of each cell to be sliced.
    
    Return
    ------
    slicedImgs : np.ndarray shape=(N * n1 * n2, cellHeight, cellWidth, nChannel)
        where n1 = imgHeight // cellHeight, n2 = imgWidth // cellWidth
    
    Example
    -------
    files = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:3]
    imgArray = trainImgFilesToImgArray(files)
    slicedImgs = imgGridSlice(imgArray, 64, 64)
    """
    if len(images.shape) == 4:
        (N, imgHeight, imgWidth, nChannel) = images.shape
    elif len(images.shape) == 3:
        (N, imgHeight, imgWidth) = images.shape
    else:
        print("# Error: imgGridSlice() requires images to be either 3D or 4D ndarray.")
        return np.array([])
    images4d = images.reshape(N, imgHeight, imgWidth, -1)
    # n1, n2, slicedImgs
    n1 = imgHeight // cellHeight
    n2 = imgWidth // cellWidth
    slicedImgs = np.zeros((N * n1 * n2, cellHeight, cellWidth, nChannel), dtype=np.uint8)
    # loop each image
    idx1 = 0
    y0, x0 = (imgHeight - n1 * cellHeight) // 2, (imgWidth - n2 * cellWidth)
    for iN in range(N):
        for iN1 in range(n1):
            for iN2 in range(n2):
                slicedImgs[idx1, :, :, :] \
                    = images[iN, y0 + iN1 * cellHeight: y0 + (iN1 + 1) * cellHeight, 
                                   x0 + iN2 * cellWidth: x0 + (iN2 + 1) * cellWidth, 0:nChannel].reshape(
                                    1, cellHeight, cellWidth, nChannel)
                # cv.imshow("TEST %d %d" % (iN1, iN2), slicedImgs[idx1, :, :, :])
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                idx1 += 1
    return slicedImgs


def slicedMasksToCategorical(slicedMasks: np.ndarray):
    """
    This function converts sliced to categorical "With" and "Without." 
    
    Example
    -------
    mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')[0:5]
    # mskFiles would be ['.../CFG_001.JPG', '.../CFG_002.JPG', ..., '.../CFG_005.JPG']
    mskArray = trainImgFilesToImgArray(mskFiles)
    # mskArray would be an np.ndarray shaped (5, 448, 448, 3), where the 5 is number of files
    slicedMsks = imgGridSlice(mskArray, 64, 64)
    # slicedMsks would be an np.ndarray shaped (245, 64, 64, 3), where 245 is 5 * (448//64) * (448//64)
    categ = slicedMasksToCategorical(slicedMsks)
    # categ would be an np.array shaped (147, 2) where 147 is the number of sliced images, and the
    # [0] is the probability of being black (zero, or has no, or without) in type np.float32
    # [1] is the probability of being white (255, or has, or with), in type np.float32
    """
    nimg = slicedMasks.shape[0]
    categ = np.zeros((nimg, 2), dtype=np.float32)
    for i in range(nimg):
        countZero = np.sum(slicedMasks[i,:,:,:] == 0)
        countNonz = np.sum(slicedMasks[i,:,:,:] != 0)
        categ[i,0] = countZero / (countZero + countNonz)
        categ[i,1] = countNonz / (countZero + countNonz)
    return categ


def trainImgMskFilesToImgArrayCateg(imgFiles, mskFiles, cellHeight, cellWidth):
    """
    This function converts files of images and masks to sliced categorical training data x/y.
    That is, imgFiles --> slicedImgs, mskFiles --> slicedMsks and categ

    Example:
    --------
    imgFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:5]
    mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')[0:5]
    imgArray = trainImgFilesToImgArray(imgFiles)
    mskArray = trainImgFilesToImgArray(mskFiles)
    slicedImgs = imgGridSlice(imgArray, 64, 64)
    slicedMsks = imgGridSlice(mskArray, 64, 64)
    categ = slicedMasksToCategorical(slicedMsks)

    """
    imgArray = trainImgFilesToImgArray(imgFiles)
    mskArray = trainImgFilesToImgArray(mskFiles)
    slicedImgs = imgGridSlice(imgArray, cellHeight, cellWidth)
    slicedMsks = imgGridSlice(mskArray, cellHeight, cellWidth)
    categ = slicedMasksToCategorical(slicedMsks)
    return slicedImgs, slicedMsks, categ


def showImgMskCateg(slicedImgs, slicedMsks, categ, showDim):
    """
    Example:
    --------
    imgFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:5]
    mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')[0:5]
    imgArray = trainImgFilesToImgArray(imgFiles)
    mskArray = trainImgFilesToImgArray(mskFiles)
    slicedImgs = imgGridSlice(imgArray, 64, 64)
    slicedMsks = imgGridSlice(mskArray, 64, 64)
    categ = slicedMasksToCategorical(slicedMsks)
    showImgMskCateg(slicedImgs[0:49,:,:,:], slicedMsks[0:49,:,:,:], categ, (7, 7))
    """
#    if showDim[0] > 9 or showDim[1] > 9:
#        print("# Error: showImgMskCateg() So far we only support dimension <= 9")
#        return
    # check
    if slicedImgs.shape[0] != slicedMsks.shape[0]:
        print("# Error: showImgMskCateg(): slicedImgs and slicedMsks must have the same dimension.")
        return
    #
    global ipage, npage
    ipage = 0
    npage = slicedImgs.shape[0] // (showDim[0] * showDim[1])
    figImg = plt.figure()
    figMsk = plt.figure()
    listAx = []
    print("In showImgMskCateg()1 ipage is %d" % ipage)
    def plotImgMsk():
        global ipage, npage
        figImg.clf()
        figMsk.clf()
        print("In plotImgMsk() ipage is %d" % ipage)
        staIdx = ipage * (showDim[0] * showDim[1])
        endIdx = (ipage + 1) * (showDim[0] * showDim[1])
        figImg.suptitle('Images (%d-%d)' % (staIdx, endIdx - 1))
        for i in range(showDim[0]):
            for j in range(showDim[1]):
                # show image
                idx = ipage * (showDim[0] * showDim[1]) + i * showDim[1] + j
                if idx < slicedImgs.shape[0]:
                    ax = figImg.add_subplot(showDim[0],showDim[1], i * showDim[1] + j + 1)
                    ax.imshow(cv.cvtColor(slicedImgs[idx, :, :, :], cv.COLOR_BGR2RGB))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
        figMsk.suptitle('Masks (%d-%d)' % (staIdx, endIdx - 1))
        for i in range(showDim[0]):
            for j in range(showDim[1]):
                # show image
                idx = ipage * (showDim[0] * showDim[1]) + i * showDim[1] + j
                if idx < slicedMsks.shape[0]:
                    ax = figMsk.add_subplot(showDim[0],showDim[1], i * showDim[1] + j + 1)
                    ax.imshow(cv.cvtColor(slicedMsks[idx, :, :, :], cv.COLOR_BGR2RGB))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    listAx.append(ax)
        figImg.canvas.draw()
        figMsk.canvas.draw()
    print("In showImgMskCateg()2 ipage is %d" % ipage)
    plotImgMsk()
    def onclick(event):
        global ipage, npage
        print("In onclick()1 ipage is %d" % ipage)
        thisAx = -1
        staIdx = ipage * (showDim[0] * showDim[1])
        endIdx = (ipage + 1) * (showDim[0] * showDim[1])
        for i in range(len(listAx)):
            if event.inaxes == listAx[i]:
                thisAx = i
        if thisAx >= 0:
            figMsk.suptitle('Masks: (%d-%d). Index=%d(1-based); P of black= %.3f ; P of white= %.3f' % 
                (staIdx, endIdx - 1, thisAx + 1 , categ[thisAx,0], categ[thisAx,1]))
            figMsk.canvas.draw()
        else:
            ipage = ipage + 1
            if ipage >= npage:
                ipage = 0
            plotImgMsk()
        print("In onclick()2 ipage is %d" % ipage)
    cid = figMsk.canvas.mpl_connect('button_press_event', onclick)
#    cid = figMsk.canvas.mpl_connect('button_press_event', lambda event: onclick(event, listAx, figMsk, categ, ipage))
    plt.show(block=True)
#    fig.canvas.mpl_disconnect(cid)

def demoVideoFrameCount():
    theAns = inputdlg(prompt=['Path of videos', 'Video ext'],
                    name='Video frame count and first-last frame', 
                    numlines=1,
                    defaultanswer=['E:\\DCIM\\100MEDIA', 'MP4'])
    if len(theAns) > 0:
        vpath = theAns[0]
        ext = theAns[1]
        mp4Files = glob.glob(os.path.join(vpath, '*.' + ext))
        for i in range(len(mp4Files)):
            vid = cv.VideoCapture(mp4Files[i])
            nframe = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
            imgh = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
            imgw = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
            for j in range(nframe):
                okSetToFrame0 = vid.set(cv.CAP_PROP_POS_FRAMES, j)
                okReadFrame0, img = vid.read()
                if type(img) != type(None):
                    okWriteFrame0 = cv.imwrite(os.path.join(
                        vpath, 'V%03d_%05d.JPG' % (i + 1, j + 1)), img)
                    okSetToFrame0 = vid.set(cv.CAP_PROP_POS_FRAMES, j + 1)
                    okReadFrame0, img = vid.read()
                    okWriteFrame0 = cv.imwrite(os.path.join(
                        vpath, 'V%03d_%05d.JPG' % (i + 1, j + 2)), img)
                    break
            for j in reversed(range(nframe)):
                okSetToFramen = vid.set(cv.CAP_PROP_POS_FRAMES, j)
                okReadFramen, img = vid.read()
                if type(img) != type(None):
                    okWriteFramen = cv.imwrite(os.path.join(
                        vpath, 'V%03d_%05d.JPG' % (i + 1, j + 1)), img)
                    break
                        

# From colon-separate integer range to list of integer 
# Range is in Matlab style, not Python style.
# That is, "2:5" mean 2 3 4 5,　not 2 3 4. 
# For example, colonRangeToIntList("2:5  10  15  17:20") would be [2, 3, 4, 5, 10, 15, 17, 18, 19, 20]
def colonRangeToIntList(theStr):
    sequences = []
    for part in theStr.split():
        try:
            # Try converting the part to an integer (single number)
            num = int(part)
            sequences.append(num)
        except ValueError:
            # If conversion fails, assume it's a colon-separated range
            start, end = map(int, part.split(":"))
            sequences.extend(range(start, end+1))
    return sequences


# imgFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:5]
# mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')[0:5]
# imgArray = trainImgFilesToImgArray(imgFiles)
# mskArray = trainImgFilesToImgArray(mskFiles)
# slicedImgs = imgGridSlice(imgArray, 64, 64)
# slicedMsks = imgGridSlice(mskArray, 64, 64)
# categ = slicedMasksToCategorical(slicedMsks)
#showImgMskCateg(slicedImgs[0:49,:,:,:], slicedMsks[0:49,:,:,:], categ, (7, 7))


