import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg


def u2F(u1, u2):
    # computes the fundamental matrix using the seven-point algorithm from 7
    # euclidean correspondences u1, u2, measured in two images. For
    # constructing the third order polynomial from null space matrices G1 and G2
    print(u1)
    for inx in itertools.combinations(range(0, 12), 7):
        u1_current = u1[:, inx]
        u2_current = u2[:, inx]
        print(inx)
        u2F_polynom()


def u2F_polynom(g1: np.array, g2: np.array):
    # %U2F_POLYNOM  Coefficients of polynomial for 7-point algorithm
    # %
    # %   [ a0, a1, a2, a3 ] = u2f_polynom( G1, G2 )
    #
    # % (c) 2020-04-14 Martin Matousek
    # % Last change: $Date$
    # %              $Revision$

    a3 = np.linalg.det(g2)

    a2 = g2[2, 1] * g2[3, 2] * g1[1, 3] - g2[2, 1] * g2[1, 2] * g1[3, 3] \
         + g2[1, 1] * g2[2, 2] * g1[3, 3] + g2[3, 1] * g1[1, 2] * g2[2, 3] \
         + g2[3, 1] * g2[1, 2] * g1[2, 3] - g2[1, 1] * g1[3, 2] * g2[2, 3] \
         - g2[3, 1] * g1[2, 2] * g2[1, 3] - g2[3, 1] * g2[2, 2] * g1[1, 3] \
         - g2[1, 1] * g2[3, 2] * g1[2, 3] + g1[2, 1] * g2[3, 2] * g2[1, 3] \
         + g2[2, 1] * g1[3, 2] * g2[1, 3] + g1[3, 1] * g2[1, 2] * g2[2, 3] \
         - g1[2, 1] * g2[1, 2] * g2[3, 3] - g1[1, 1] * g2[2, 3] * g2[3, 2] \
         - g2[2, 1] * g1[1, 2] * g2[3, 3] + g2[1, 1] * g1[2, 2] * g2[3, 3] \
         + g1[1, 1] * g2[2, 2] * g2[3, 3] - g1[3, 1] * g2[2, 2] * g2[1, 3]

    a1 = + g1[1, 1] * g1[2, 2] * g2[3, 3] + g1[1, 1] * g2[2, 2] * g1[3, 3] \
         + g2[3, 1] * g1[1, 2] * g1[2, 3] - g1[2, 1] * g1[1, 2] * g2[3, 3] \
         - g2[1, 1] * g1[3, 2] * g1[2, 3] - g2[2, 1] * g1[1, 2] * g1[3, 3] \
         - g2[3, 1] * g1[2, 2] * g1[1, 3] + g2[1, 1] * g1[2, 2] * g1[3, 3] \
         + g1[2, 1] * g1[3, 2] * g2[1, 3] + g1[2, 1] * g2[3, 2] * g1[1, 3] \
         + g1[3, 1] * g2[1, 2] * g1[2, 3] - g1[2, 1] * g2[1, 2] * g1[3, 3] \
         - g1[3, 1] * g2[2, 2] * g1[1, 3] + g2[2, 1] * g1[3, 2] * g1[1, 3] \
         - g1[1, 1] * g2[3, 2] * g1[2, 3] - g1[3, 1] * g1[2, 2] * g2[1, 3] \
         + g1[3, 1] * g1[1, 2] * g2[2, 3] - g1[1, 1] * g1[3, 2] * g2[2, 3]
    a0 = np.linalg.det(g1)
    return [a0, a1, a2, a3]


if __name__ == "__main__":
    img1 = plt.imread('daliborka_01.jpg')
    img2 = plt.imread('daliborka_23.jpg')
    f = sio.loadmat("daliborka_01_23-uu.mat")

    # indices of a points, that form an edge
    edges = f["edges"]
    # list of 12 indices of points ix
    ix = f["ix"][0]
    # There is a set of point matches between the images above.
    u01 = f["u01"]
    u23 = f["u23"]

    # Step 1. Find the fundamental matrix F relating the images above: generate
    # all 7-tuples from the selected set of 12 correspondences, estimate F for
    # each of them and chose the one, that minimizes maximal epipolar error
    # over all matches.

    u1 = u01[:, ix]
    u2 = u23[:, ix]

    F = u2F(u1, u2)
    
    #  Step 2. Draw the 12 corresponding points in different colour in the two
    #  images. Using the best F, compute the corresponding epipolar lines and
    #  draw them into the images in corresponding colours (a line segment given
    #  by the intersection of the image area and a line must me computed).
    #  Export as 08_eg.pdf.

    # Step 3. Draw graphs of epipolar errors d1_i and d2_i for all points
    # (point index on horizontal axis, the error on vertical axis). Draw both
    # graphs into single figure (different colours) and export as 08_errors.pdf.

    # Step 4. Save all the data into 08_data.mat: the input data u1, u2, ix,
    # the indices of the 7 points used for computing the optimal F as point_sel
    # and the matrix F.

    # plt.plot(u01[0], u01[1], 'r.')
    # print(u01)
    # plt.imshow(img1)
    # plt.imshow(img2)
    # plt.show()
    # print(f)
