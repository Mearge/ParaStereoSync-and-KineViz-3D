# The function chessboard_object_points() returns 3D object points of a 
# chessboard pattern given the number of corners in the x and y directions 
# and the size of the squares.
# 
# Input: 
#   n1: number of corners in the 1st direction. If the face is 'xy', n1 is number of corners in the x direction.
#   n2: number of corners in the 2nd direction. If the face is 'xy,, n2 is number of corners in the y direction.
#   d1: size of the squares in the 1st direction. If the face is 'xy', d1 is size of the squares in the x direction.
#   d2: size of the squares in the 2nd direction. If the face is 'xy', d2 is size of the squares in the y direction.
#   face: the orientation of the chessboard pattern. The default is 'xy'.
#         'xy': the chessboard pattern is on the xy plane.
#         'yx': the chessboard pattern is on the yx plane, while corners go y axis first.
#         'yz': the chessboard pattern is on the yz plane.
#         'zy': the chessboard pattern is on the zy plane, while corners go z axis first.
#         'zx': the chessboard pattern is on the zx plane.
#         'xz': the chessboard pattern is on the xz plane, while corners go x axis first.
# Output:
#   object_points: 3D object points of the chessboard pattern, which is a
#                  n1*n2*3 numpy array. The first column is the x coordinate, and so on.
# Example:
#   chessboard_object_points(n1=3, n2=2, d1=1.0, d2=2.0, face='xy') returns
#     [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 2., 0.], [1., 2., 0.], [2., 2., 0.]]
#   chessboard_object_points(n1=3, n2=2, d1=1.0, d2=2.0, face='yx') returns
#     [[0., 0., 0.], [0., 1., 0.], [0., 2., 0.], [2., 0., 0.], [2., 1., 0.], [2., 2., 0.]]
#   chessboard_object_points(n1=3, n2=2, d1=1.0, d2=2.0, face='yz') returns
#     [[0., 0., 0.], [0., 1., 0.], [0., 2., 0.], [0., 0., 2.], [0., 1., 2.], [0., 2., 2.]]
#   chessboard_object_points(n1=3, n2=2, d1=1.0, d2=2.0, face='zy') returns
#     [[0., 0., 0.], [0., 0., 1.], [0., 0., 2.], [0., 2., 0.], [0., 2., 1.], [0., 2., 2.]]
#   chessboard_object_points(n1=3, n2=2, d1=1.0, d2=2.0, face='zx') returns
#     [[0., 0., 0.], [0., 0., 1.], [0., 0., 2.], [2., 0., 0.], [2., 0., 1.], [2., 0., 2.]]
#   chessboard_object_points(n1=3, n2=2, d1=1.0, d2=2.0, face='xz') returns
#     [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 0., 2.], [1., 0., 2.], [2., 0., 2.]]

import numpy as np

def chessboard_object_points(n1, n2, d1, d2, face='xy'):
    if face == 'xy':
        x = np.arange(n1)*d1
        y = np.arange(n2)*d2
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.zeros(n1*n2)
        object_points = np.vstack((x, y, z)).T
    elif face == 'yx':
        y = np.arange(n1)*d1
        x = np.arange(n2)*d2
        y, x = np.meshgrid(y, x)
        x = x.flatten()
        y = y.flatten()
        z = np.zeros(n1*n2)
        object_points = np.vstack((x, y, z)).T
    elif face == 'yz':
        y = np.arange(n1)*d1
        z = np.arange(n2)*d2
        y, z = np.meshgrid(y, z)
        y = y.flatten()
        z = z.flatten()
        x = np.zeros(n1*n2)
        object_points = np.vstack((x, y, z)).T
    elif face == 'zy':
        z = np.arange(n1)*d1
        y = np.arange(n2)*d2
        z, y = np.meshgrid(z, y)
        y = y.flatten()
        z = z.flatten()
        x = np.zeros(n1*n2)
        object_points = np.vstack((x, y, z)).T
    elif face == 'zx':
        z = np.arange(n1)*d1
        x = np.arange(n2)*d2
        z, x = np.meshgrid(z, x)
        x = x.flatten()
        z = z.flatten()
        y = np.zeros(n1*n2)
        object_points = np.vstack((x, y, z)).T
    elif face == 'xz':
        x = np.arange(n1)*d1
        z = np.arange(n2)*d2
        x, z = np.meshgrid(x, z)
        x = x.flatten()
        z = z.flatten()
        y = np.zeros(n1*n2)
        object_points = np.vstack((x, y, z)).T
    else:
        print('Error: face is not supported.')
        object_points = None
    return object_points

# Test the function
if __name__ == '__main__':
    n1 = 3
    n2 = 2
    d1 = 1
    d2 = 2
    object_points = chessboard_object_points(n1, n2, d1, d2, face='xy')
    print("(n1=3,n2=2,d1=1.0,d2=2.0,face='xy':\n", object_points)
    object_points = chessboard_object_points(n1, n2, d1, d2, face='yx')
    print("(n1=3,n2=2,d1=1.0,d2=2.0,face='yx':\n", object_points)
    object_points = chessboard_object_points(n1, n2, d1, d2, face='yz')
    print("(n1=3,n2=2,d1=1.0,d2=2.0,face='yz':\n", object_points)
    object_points = chessboard_object_points(n1, n2, d1, d2, face='zy')
    print("(n1=3,n2=2,d1=1.0,d2=2.0,face='zy':\n", object_points)
    object_points = chessboard_object_points(n1, n2, d1, d2, face='zx')
    print("(n1=3,n2=2,d1=1.0,d2=2.0,face='zx':\n", object_points)
    object_points = chessboard_object_points(n1, n2, d1, d2, face='xz')
    print("(n1=3,n2=2,d1=1.0,d2=2.0,face='xz':\n", object_points)
