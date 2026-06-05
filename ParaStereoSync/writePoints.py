import numpy as np

def writePoints(savefile, points, header=''):
    if (savefile == '.'):
        return
    np.savetxt(savefile, points, fmt='%24.16e', delimiter=' , ',
               header=header)
        
