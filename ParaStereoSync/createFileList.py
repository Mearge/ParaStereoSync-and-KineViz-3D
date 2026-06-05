import os
import glob
from inputs import input2, input3, float2, int2

def ufilesToFileList(ufiles: str):
    """
    Examples
    --------
    ufilesToFileList('d:/pics/DSC*.JPG') --> list of existing files 
    ufilesToFileList('d:/pics/DSC%02d.JPG,5,3') --> 
        ['d:/pics\\DSC05.JPG', 'd:/pics\\DSC06.JPG', 'd:/pics\\DSC07.JPG']
    ufilesToFileList('d:/pics/DSC%02d.JPG,2,8,2') --> 
        ['d:/pics\\DSC02.JPG', 'd:/pics\\DSC04.JPG', 'd:/pics\\DSC06.JPG']
    """
    if ufiles.find('*') >= 0 or ufiles.find('?') >= 0:
        files = glob.glob(ufiles)
        files.sort()
        return files
    elif ufiles.find('%') >= 0:
        ufiles_split = ufiles.split(',')
        if len(ufiles_split) == 3:
            cfiles = ufiles_split[0].strip()
            cfilesStart = int(ufiles_split[1])
            cfilesCount = int(ufiles_split[2])
            files = []
            for i in range(cfilesCount):
                files.append(cfiles % (i + cfilesStart))
            return files
        elif len(ufiles_split) == 4:
            cfiles = ufiles_split[0].strip()
            cfilesStart = int(ufiles_split[1])
            cfilesEnd = int(ufiles_split[2])
            cfilesIncr= int(ufiles_split[3])
            files = []
            for i in range(cfilesStart, cfilesEnd, cfilesIncr):
                files.append(cfiles % (i))
            return files
        else:
            return []
    else:
        # since ufiles have neither % nor wildcard(*,?), it could be 
        # just the file name of a single file
        return [ufiles]


def createFileList(files="", savefile="", cStartIdx=-1, cNumFiles=-1):
    """
    This funcion returns a list of file names, and saves them to a 
    text file, given one of the formats:
    (1) file name with wildcard and the output file name, 
    (2) file name with C specifier, start index, number of files, and
        the output file name, 
    (3) list of file name
    For example, 
        createFileList(".\images\IMG_????.tif", 
                ".\files.txt"), or 
        createFileList(".\images\IMG_%04d.tif", 
                ".\files.txt", 
                cStartIdx=3, cNumFiles=4), or
        createFileList([".\images\IMG_0003.tif", 
                 ".\images\IMG_0004.tif",
                 ".\images\IMG_0005.tif", 
                 ".\images\IMG_0006.tif"], 
                 ".\files.txt")

    Parameters
    ----------
    files : str
        A file name that contains wildcard or C specifier that 
        describes the files,
        e.g., "c:\images\DCIM????.BMP" or "c:\images\DCIM%04d.BMP"
        Use backslash in Windows and forward slash in other systems.
    cStartIdx : int
        if C specifier is used, cStartIdx is the starting index for the 
        %d specifier. 
    cNumFiles : int
        if C specifier is used, cNumFiles is the number of files to 
        generate in the file list. For example, files of "IMG_%04d.BMP",
        cStartIdx of 3 and cNumFiles of 3 would generate IMG_0003.BMP, 
        IMG_0004.BMP, and IMG_0005.BMP. 
    savefile : str
        A file to save that contains all files that match wfile
        e.g., "c:\analysis\files.txt"
        If the length of savefile is "", it asks you to enter by keyboard.
        If the length of savefile is == 1, it skips the file saving.

    Returns
    -------
    theFiles : list of strings
        the file sequence in format of list of strings
    """
    # Check argument files, if necessary, ask user
    while (files == ""):
        print("# Enter files by wildcard or c specifier format:")
        print("#  E.g., examples/createFileList/images/IMG_????.tif ")
        print("# or c specifier file, starting index, number of files.")
        print("#  E.g., examples/createFileList/images/IMG_%04d.tif 3 4")
        files = input2().strip()
        files = files.split()
        if len(files) >= 3:
            cNumFiles = int2(files[2])
        if len(files) >= 2:
            cStartIdx = int2(files[1])
        files = files[0]
    if (savefile == ""):
        print("# Enter file to save the list of files:")
        print("#  or enter a single character to skip file saving.")
        print("#  For example, examples/createFileList/try_files.txt")
        savefile = input2().strip()
    if type(files) == str and (files.find('*') >= 0 or files.find('?') >= 0):
        # wildcard format
        return fileSeqByWildcard(files, savefile)
    elif type(files) == str and files.find('%') >= 0:
        # c specifier
        if (cStartIdx < 0):
            print("# You are trying to use C specifier to define"
                  " a file sequence but did not give a proper starting" 
                  " index (must be >= 0).")
            print("# Enter start index:")
            print("#   For example: 3")
            cStartIdx = input3("", dtype=int, min=0)
        if (cNumFiles <= 0):
            print("# You are trying to use C specifier to define"
                  " a file sequence but did not give a proper number of " 
                  " files (must be >= 1).")
            print("# Enter number of files:")
            print("#   For example: 4")
            cNumFiles = input3("", dtype=int, min=1)
        return fileSeqByCspec(files, cStartIdx, cNumFiles, savefile)
    elif type(files) == list:
        # list of file names
        # save file list to a file
        if (len(savefile) > 1):
            # save file sequence to the file
            with open(savefile, 'w') as fp:
                for item in files:
                    # write each item on a new line
                    fp.write("%s\n" % item)
        return files


def fileSeqByWildcard(wfile, savefile=""):
    """
    This funcion returns a list of files by wildcard file name format, and
    save the (full-path) file names in a text file.
    For example, fileSeqByWildcard(".\images\IMG_????.tif", 
                                   ".\files.txt")
    could return ['.\images\IMG_0003.tif', '.\images\IMG_0004.tif', 
                  '.\images\IMG_0005.tif', '.\images\IMG_0006.tif']
    if these are files that match the specific pattern. 
    The file content of "c:\analysis\files.txt" would be
    .\images\IMG_0003.tif 
    .\images\IMG_0004.tif
    .\images\IMG_0005.tif
    .\images\IMG_0006.tif

    Parameters
    ----------
    wfile : str
        A file name that contains wildcard that describes the files,
        e.g., ".\images\IMG_????.tif"
        Note that every backslash would be converted to forward slash.
    savefile : str
        A file to save that contains all files that match wfile
        e.g., ".\files.txt"
        If the length of savefile is <= 1, it skips the file saving.

    Returns
    -------
    theFiles : list of strings
        the file sequence in format of list of strings
    """
    # get list of files
    theFiles = glob.glob(wfile)
    #for i in range(len(theFiles)):
    #    theFiles[i] = theFiles[i].replace('\\', '/')
    theFiles.sort()
    # save to a file
    if (len(savefile) > 1):
        # save file sequence to the file
        with open(savefile, 'w') as fp:
            for item in theFiles:
                # write each item on a new line
                fp.write("%s\n" % item)
    # return a list of (full-path) file names
    return theFiles


def fileSeqByCspec(cfile, startIdx, nFiles, savefile=""):
    """
    This funcion returns a list of files by c specifier (%d) file name format.
    Example: fileSeqByCspec("./images/img_%04d.bmp", 2, 3, 
                            "./analysis/files.txt")
    It would return
    ['./images/img_0002.bmp', 'c:/images/img_0003.bmp', 
     './images/img_0004.bmp']
    The file content of "c:/analysis/files.txt" would be
    ./images/DCIM0002.BMP 
    ./images/DCIM0003.BMP
    ./images/DCIM0004.BMP

    Parameters
    ----------
    cfile : str
        A (full-path) file name that describes the files by using C specifier,
        e.g., "c:\images\img_%04d.bmp"
        Every backslash will be converted to a forward slash.
    startIdx : int
        The integer that describes the first file in the cfile, e.g.,
        for cfile of "DCIM%04d.BMP" and startIdx of 1, the first file
        would be DCIM0001.BMP
    nFiles : int
        Number of files in the file sequence
    savefile : str
        A file to save that contains all files that match wfile
        e.g., "c:\analysis\files.txt"
        If the length of savefile is <= 1, it skips the file saving.

    Returns
    -------
    theFiles : list of strings
        the file sequence in format of list of strings
    """
    theFiles = []
    # check
    if cfile.find('/') < 0 and cfile.find('\\') < 0:
        print("# Warning: fileSeqByCspec(): Argument cfile"
              " should have one or more slash or backslash.")
    if cfile.find('%') < 0:
        print("# Warning: fileSeqByCspec(): Argument cfile"
              " should have a % for c-style operations.")
    if startIdx < 0:
        print("# Warning: fileSeqByCspec(): Argument startIdx"
              " should be non-negative.")
    if nFiles < 0:
        print("# Warning: fileSeqByCspec(): Argument nFiles"
              " should be positive.")
    # theDir
    #cfile2 = cfile.replace('\\', '/')
    # files
    for i in range(nFiles):
        theFiles.append(cfile % (i + startIdx))
    # save to a file
    if (len(savefile) > 1):
        # save file sequence to the file
        with open(savefile, 'w') as fp:
            for item in theFiles:
                # write each item on a new line
                fp.write("%s\n" % item)
    # return a list of (full-path) file names
    return theFiles