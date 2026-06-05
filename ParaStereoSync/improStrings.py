import numpy as np


def npFromTupleNp(theTupleNp):
    try:
        theShape = theTupleNp[0].shape
        theLen = len(theTupleNp)
        newShape = (theLen,) + theShape
        theMat = np.zeros(newShape, float)
        for i in range(theLen):
            theMat[i] = theTupleNp[i]
    except:
        theMat = np.array([])
    return theMat
    

def npFromStrings(theStrList: list):
    """
    Converts a list of strings (which only contains floats) to a numpy 
    float array (in 1D). The separator can be ',', ' ', '\t', '\n', 
    '(', ')', '[', ']', '{', '}'. The 'nan' or 'na' would be considered 
    as np.nan. 
    The returned numpy will be in 1D. 
    For example:
        npFromStrings(['1.2 , 2.3', 
                       'nan \n 4.5'])
            returns array([1.2, 2.3, nan, 4.5])
    """
    theType = type(theStrList)
    if theType == str:
        return npFromString(theStrList.strip())
#   elif theType == list or theType == tuple:
    else: 
        # assuming the type is list or tuple, or 
        # other types that runs with the following for loop
        _str = ''
        for i in theStrList:
            _str += (str(i).strip() + '\n')
        return npFromString(_str)

def npFromString(theStr: str):
    """
    Converts a string (which only contains floats) to a numpy 
    float array (in 1D). The separator can be ',', ' ', '\t', '\n', 
    '(', ')', '[', ']', '{', '}'. The 'nan' or 'na' would be considered 
    as np.nan. 
    The returned numpy will be in 1D. 
    For example:
        npFromString('1.2 , 2.3 \t nan \n 4.5')
            returns array([1.2, 2.3, nan, 4.5])
    """
    if type(theStr) == str:
        _str = theStr.strip()
    elif type(theStr) == list or type(theStr) == tuple:
        return npFromStrings(theStr)
    else: 
        _str = str(theStr)
    _str = _str.replace(',', ' ').replace(';', ' ').replace('[', ' ')
    _str = _str.replace(']', ' ').replace('na ', 'nan').replace('\n',' ')
    _str = _str.replace('(', ' ').replace(')', ' ')
    _str = _str.replace('{', ' ').replace('}', ' ')
    _str = _str.replace('n/a', 'nan').replace('#N/A', 'nan')
    _str = _str.replace('np.nan', 'nan').replace('numpy.nan', 'nan')
    theMat = np.fromstring(_str, sep=' ')
    return theMat   


def stringFromNp(theMat: np.ndarray, ftype='txtf', sep=' '):
#    if type(theMat) == tuple:
#        theMat = npFromTupleNp(theMat)
    if ftype=='txtf':
        # convert the matrix into a string by using np.array2string
        np.set_printoptions(threshold=np.inf)
        theStr = np.array2string(theMat,
                                 max_line_width=10000,
                                 separator=sep,
                                 formatter={'float_kind':lambda x: "%24.16e" % x})
        np.set_printoptions(threshold=None)
        # replace some characters 
        theStr = theStr.replace('  ',' ')
        theStr = theStr.replace(' [ ','')
        theStr = theStr.replace(' [','')
        theStr = theStr.replace('[ ','')
        theStr = theStr.replace('[','')
        theStr = theStr.replace(' [ ','')
        theStr = theStr.replace(' ]','')
        theStr = theStr.replace('] ','')
        theStr = theStr.replace(']','')
        theStr = theStr.replace('\n ', '\n')
        for i in range(100):
            theStr = theStr.replace('  ', ' ')
        theStr = theStr.replace('\n ', '\n')
    return theStr.strip()


