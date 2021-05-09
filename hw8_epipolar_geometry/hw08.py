import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg


def u2F(u1, u2):
    # computes the fundamental matrix using the seven-point algorithm from 7
    # euclidean correspondences u1, u2, measured in two images. For
    # constructing the third order polynomial from null space matrices G1 and G2
    for inx in itertools.combinations(range(0, 12), 7):
        u1_current = u1[:, inx]
        u2_current = u2[:, inx]
        tmp_G = list()
        for counter in range(7):
            a1 = np.array([u1_current[0][counter], u1_current[1][counter], 1])
            a2 = np.array([u2_current[0][counter], u2_current[1][counter], 1])
            tmp_G.append(np.c_[a2[0] * a1[0], a2[0] * a1[1], a2[0] * a1[2],
                               a2[1] * a1[0], a2[1] * a1[1], a2[1] * a2[2],
                               a2[2] * a1[0], a2[2] * a1[1], a2[2] * a1[2]])
        tmp_G = np.array(tmp_G)
        tmp_G = tmp_G.reshape(7, 9)
        result = scipy.linalg.null_space(tmp_G)
        G1 = result.T[0].reshape(3, 3)
        G2 = result.T[1].reshape(3, 3)
        polynom = u2F_polynom(G1, G2)
        roots = np.roots(polynom)
        for root in roots:
            if np.iscomplex(root):
                continue
            else:
                G = G1 + root * G2
                if np.array_equal(G, G2):
                    continue
                else:
                    for counter in range(12):
                        a1 = np.array([u1[0][counter],
                                       u1[1][counter], 1])
                        a2 = np.array([u2[0][counter],
                                       u2[1][counter], 1])
                        ep1 = G @ a1
                        ep2 = G.T @ a2
                        # print(ep1)
                        # print(ep2)

def u2F_polynom(q: np.array, p: np.array):
    # %U2F_POLYNOM  Coefficients of polynomial for 7-point algorithm
    # %
    # %   [ a0, a1, a2, a3 ] = u2f_polynom( G1, G2 )
    #
    # % (c) 2020-04-14 Martin Matousek
    # % Last change: $Date$
    # %              $Revision$
    # ".replace("1", "0").replace("2", "1").replace("3", "2")
    a3 = np.linalg.det(p)

    a2 = p[1, 0] * p[2, 1] * q[0, 2] - p[1, 0] * p[0, 1] * q[2, 2] + p[0, 0] * \
         p[1, 1] * q[2, 2] + p[2, 0] * q[0, 1] * p[1, 2] + p[2, 0] * p[0, 1] * \
         q[1, 2] - p[0, 0] * q[2, 1] * p[1, 2] - p[2, 0] * q[1, 1] * p[0, 2] - \
         p[2, 0] * p[1, 1] * q[0, 2] - p[0, 0] * p[2, 1] * q[1, 2] + q[1, 0] * \
         p[2, 1] * p[0, 2] + p[1, 0] * q[2, 1] * p[0, 2] + q[2, 0] * p[0, 1] * \
         p[1, 2] - q[1, 0] * p[0, 1] * p[2, 2] - q[0, 0] * p[1, 2] * p[2, 1] - \
         p[1, 0] * q[0, 1] * p[2, 2] + p[0, 0] * q[1, 1] * p[2, 2] + q[0, 0] * \
         p[1, 1] * p[2, 2] - q[2, 0] * p[1, 1] * p[0, 2]
    a1 = q[0, 0] * q[1, 1] * p[2, 2] + q[0, 0] * p[1, 1] * q[2, 2] + p[2, 0] * \
         q[0, 1] * q[1, 2] - q[1, 0] * q[0, 1] * p[2, 2] - p[0, 0] * q[2, 1] * \
         q[1, 2] - p[1, 0] * q[0, 1] * q[2, 2] - p[2, 0] * q[1, 1] * q[0, 2] + \
         p[0, 0] * q[1, 1] * q[2, 2] + q[1, 0] * q[2, 1] * p[0, 2] + q[1, 0] * \
         p[2, 1] * q[0, 2] + q[2, 0] * p[0, 1] * q[1, 2] - q[1, 0] * p[0, 1] * \
         q[2, 2] - q[2, 0] * p[1, 1] * q[0, 2] + p[1, 0] * q[2, 1] * q[0, 2] - \
         q[0, 0] * p[2, 1] * q[1, 2] - q[2, 0] * q[1, 1] * p[0, 2] + q[2, 0] * \
         q[0, 1] * p[1, 2] - q[0, 0] * q[2, 1] * p[1, 2]

    a0 = np.linalg.det(q)
    return [a0, a1, a2, a3]


if __name__ == "__main__":
    img1 = plt.imread('daliborka_01.jpg')
    img2 = plt.imread('daliborka_23.jpg')
    f = sio.loadmat("daliborka_01_23-uu.mat")

    # indices of a points, that form an edge
    edges = f["edges"]  # - 1
    # list of 12 indices of points ix
    ix = f["ix"][0] - 1
    # There is a set of point matches between the images above.
    u01 = f["u01"]
    u23 = f["u23"]
    u01 = np.array([u01[0] - 1, u01[1] - 1])
    u23 = np.array([u23[0] - 1, u23[1] - 1])
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
