import numpy as np
import os
import re

def input2(prompt=""):
    """
    This function is similar to Python function input() but if the returned
    string starts with a hashtag (#) this function ignores the line of the
    strin and runs the input() function again.
    The head spaces and tail spaces are removed as well.
    This function only allows user to edit a script for a series of input,
    but also allows user to put comments by starting the comments with a
    hashtag, so that the input script is earier to understand.
    For example, a BMI converter could run in this way:
    /* -------------------------------
    1.75  (user's input)
    70    (user's input)
    The BMI is 22.9
    --------------------------------- */
    The user can edit a file for future input:
    /* ---------------------------------
    # This is an input script for a program that calculates BMI
    # Enter height in unit of meter
    1.75
    # Enter weight in unit of kg
    70

    Parameters
        prompt  A String, representing a default message before the input.
    --------------------------------- */
    """
    theInput = ""
    if len(prompt) == 0:
        thePrompt = ""
    else:
        thePrompt = "# " + prompt
        print(thePrompt, end='')
    # run the while loop of reading
    while(True):
        theInput = input()
        theInput = theInput.strip()
        if (len(theInput) == 0):
            continue
        if (theInput[0] == '#'):
            continue
        break
    # remove everything after the first #
    if theInput.find('#') >= 0:
        theInput = theInput[0:theInput.find('#')]
    return theInput.strip()


def input3(prompt="", dtype=str, min=-1.e8, max=1.e8):
    """
    

    Parameters
    ----------
    prompt : TYPE, optional
        DESCRIPTION. The default is "".
    dtype : TYPE, optional
        DESCRIPTION. The default is str.
    min : TYPE, optional
        DESCRIPTION. The default is -1.e8.
    max : TYPE, optional
        DESCRIPTION. The default is 1.e8.

    Returns
    -------
    uInput : TYPE
        DESCRIPTION.

    """
    while(True):
        uInput = input2(prompt)
        if dtype == str:
            return uInput
        if dtype == float:
            try:
                uInput = float(uInput)
            except:
                print("# Input should be a float but got ", uInput)
                print("# Try to input again.")
                continue
            if uInput < min or uInput > max:
                print("# Input should be between %f and %f but got"
                      " %f." % (min, max, uInput))
                print("# Try to input again.")
                continue
        if dtype == int:
            try:
                uInput = int(uInput)
            except:
                print("# Input should be an int but got ", uInput)
                print("# Try to input again.")
                continue
            if uInput < int(min) or uInput > int(max):
                print("# Input should be between %d and %d but got"
                      " %d." % (min, max, uInput))
                print("# Try to input again.")
                continue
        break
    return uInput

def inputs(prompt=''):
    """
    This function (inputs) is similar to input() but it allows multiple
    lines of input. 
    The key [Enter] does not ends the input. This function reads 
    inputs line by line until a Ctrl-D (or Ctrl-Z) is entered. 
    This function returns a list of strings. 
    
    Parameters
    ----------
    prompt : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    contents : List of strings
        A list of strings. Each string is a line of input.

    """
    print(prompt, end='')
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    return contents

            
def float2(x):
    """
    Returns float(x) but if exception occurs it returns np.nan
    For example, 
        float2('3.4') returns 3.4
        float2('1 + 1') returns 2.0
        float2('three') returns np.nan
        float2('[3.3, 4.4]') returns 3.3
        float2(np.array([3.3, 4.4])) returns 3.3

    Parameters
    ----------
    x : int, float, string, or other type that float() accepts.
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if type(x) == float:
        return x
    elif type(x) == int:
        return float(x)
    elif type(x) == str:
        try:
            return float(eval(x))
        except:
            try:
                return float2(eval(x))
            except:
                return np.nan
    elif type(x) == list:
        return float2(x[0])
    elif type(x) == tuple:
        return float2(x[0])
    elif type(x) == np.ndarray:
        try:
            return x.flatten()[0]
        except:
            return np.nan
    else:
        try:
            return float(x)
        except:
            return np.nan


def int2(x):
    y = float2(x)
    if np.isnan(y):
        return np.iinfo(int).min # which is -2147483648
    else:
        return int(y)

def str2Ndarray(theStr=""):
    """
    Converts a string to a numpy array. For example:
        str2Ndarray("(1, 1.5, 2, 2.5, 3)")
        str2Ndarray("[1, 1.5, 2, 2.5, 3]")
        str2Ndarray("np.linspace(1,3,5)")
        str2Ndarray("np.loadtxt('yourfile.csv', delimiter=',')")        
    If user inputs cannot be parsed, this function asks the user to 
    input through keyboard. However if parsing fails more than 10 
    times, this function returns np.array([]). 
    You can use this function in this way:
        uInput = input("# Enter a Numpy array:")
        mat = str2Ndarray(uInput)       

    Parameters
    ----------
    theStr : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    np.ndarray
        DESCRIPTION.

    """
    nErr = 0; maxErr = 10
    while(True):
        if (theStr == ""):
            print("# Enter a numpy array (list, tuple, or Numpy statement): ")
            theStr = input2()
        try:
            mat = np.array(eval(theStr))
            if len(mat.shape) <= 0:
                mat = mat.flatten()
            return mat
        except:
            print("# Failed to parse your input.")
            print("#  Try numpy form. For example:") 
            print("#   (1, 1.5, 2, 2.5, 3)  (tuple)")
            print("#   [1, 1.5, 2, 2.5, 3]  (list)")
            print("#   np.linspace(1,3,5)   (np form)")
            print("#   np.loadtxt('examples/pickPoints/picked_IMG_0001.csv', delimiter=',')")
            theStr = ""
            nErr += 1
            if nErr >= maxErr:
                break
        continue 
    print("# Error: str2Ndarray(): Got errors too many times.")
    return np.array([])          
