import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg
import tools
import SeqWriter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img1 = plt.imread("pokemon_10.jpg")
    img2 = plt.imread("pokemon_19.jpg")

    C = [[473.4, 507.2, 738.3, 736.2],
         [500.7, 346.0, 347.7, 505.7]]

    C2 = [[571.6, 384.9, 571.4, 755.5],
          [541.8, 420.6, 333.8, 434.6]]

    C_outer = [[119.1, 463.7, 988.8, 798.0],
               [597.9, 218.7, 321.2, 813.5]]

    C2_outer = [[397.4, 153.8, 701.0, 1104.9],
                [822.7, 345.3, 241.6, 582.8]]

    plt.plot(C[0], C[1], 'ro')
    plt.plot(C_outer[0], C_outer[1], 'ro')
    plt.imshow(img1)
    plt.show()
    
    plt.plot(C2[0], C2[1], 'bo')
    plt.plot(C2_outer[0], C2_outer[1], 'bo')
    plt.imshow(img2)
    plt.show()
