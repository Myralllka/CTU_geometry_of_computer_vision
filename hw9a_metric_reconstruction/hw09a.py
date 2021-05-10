import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg


def shortest_distance(p, lnn) -> float:
    return abs((lnn[0] * p[0] + lnn[1] * p[1] + lnn[2])) / (
        np.sqrt(lnn[0] ** 2 + lnn[1] ** 2))


def step2(u1, u2):
    fig = plt.figure()
    fig.clf()
    fig.suptitle('The epipolar lines using Fx')
    plt.subplot(121)

    i = 0
    for x_p1, y_p1, x_p2, y_p2 in zip(u1[0], u1[1], u2[0], u2[1]):
        plt.plot([int(x_p1)], [int(y_p1)], color=colors[i], marker="X",
                 markersize=10)
        point2 = np.c_[x_p2, y_p2, 1].reshape(3, 1)

        x = np.linspace(1, 1200, 1200)
        ep1 = F.T @ point2
        y = -((ep1[2] / ep1[1]) + x * ep1[0] / ep1[1])
        plt.plot(x, y, color=colors[i])
        i += 1
    plt.imshow(img1)

    plt.subplot(122)

    i = 0
    for x_p1, y_p1, x_p2, y_p2 in zip(u1[0], u1[1], u2[0], u2[1]):
        plt.plot([int(x_p2)], [int(y_p2)],
                 color=colors[i],
                 marker="X",
                 markersize=10)
        point1 = np.c_[x_p1, y_p1, 1].reshape(3, 1)

        x = np.linspace(1, 1200, 1200)
        point1 = point1.reshape(3, 1)
        ep2 = F @ point1
        y = -((ep2[2] / ep2[1]) + x * ep2[0] / ep2[1])
        plt.plot(x, y, color=colors[i])
        i += 1
    plt.imshow(img2)
    fig.savefig("09_egx.pdf")
    plt.show()


def step3(U1, U2, F):
    fig = plt.figure()
    fig.clf()
    d1_arr, d2_arr = [], []
    for i in range(len(U1[0])):
        a1 = np.array([U1[0][i],
                       U1[1][i], 1])
        a2 = np.array([U2[0][i],
                       U2[1][i], 1])
        ep1 = F.T @ a2
        ep2 = F @ a1
        d1_arr.append(shortest_distance(a1, ep1))
        d2_arr.append(shortest_distance(a2, ep2))
    plt.title('The epipolar error for all points')
    plt.xlabel('point index')
    plt.ylabel('epipolar err [px]')
    plt.plot(d1_arr, color="b", label="image 1")
    plt.plot(d2_arr, color='g', label='image 2')
    plt.legend(loc='best')
    plt.savefig("09_errorsx.pdf")
    plt.show()


if __name__ == "__main__":
    # Data preprocessing
    img1 = plt.imread('daliborka_01.jpg')
    img2 = plt.imread('daliborka_23.jpg')

    f = sio.loadmat("daliborka_01_23-uu.mat")

    U1 = f["u01"]
    U2 = f["u23"]

    # indices of a points, that form an edge
    edges = f["edges"]  # - 1
    edges = np.array([edges[0] - 1, edges[1] - 1])
    # list of 12 indices of points ix
    ix = f["ix"][0] - 1
    # There is a set of point matches between the images above.
    u01 = f["u01"]
    u23 = f["u23"]
    u01 = np.array([u01[0] - 1, u01[1] - 1])
    u23 = np.array([u23[0] - 1, u23[1] - 1])

    u1 = u01[:, ix]
    u2 = u23[:, ix]

    colors = ["dimgray", "rosybrown", "maroon", "peru",
              "moccasin", "yellow", "olivedrab", "lightgreen",
              "navy", "royalblue", "indigo", "hotpink"]

    # Step 1
    # Compute essential matrix Ex using your best fundamental matrix F
    # estimated in HW-08 and internal calibration from HW-04. Compute also
    # the fundamental matrix Fx consistent with K from Ex and K

    F = sio.loadmat("08_data.mat")["F"]
    K = sio.loadmat("K.mat")["K"]

    Ex = K.T @ F @ K
    Fx = np.linalg.inv(K).T @ Ex @ np.linalg.inv(K)

    # Step 2
    # Draw the 12 corresponding points IX (from HW-08) in different colour in
    # the two images. Using Fx, compute the corresponding epipolar lines and
    # draw them into the images in corresponding colours. Export as 09_egx.pdf.

    step2(u1, u2)

    # Step 3
    # Draw graphs of epipolar errors d1_i and d2_i w.r.t Fx for all points.
    # Draw both graphs into single figure (different colours) and export as
    # 09_errorsx.pdf.

    step3(U1, U2, Fx)

    # Step 4
    # Find essential matrix E by minimizing the maximum epipolar error of the
    # respective fundamental matrix Fe consistent with K using the same
    # correspondences:

    ### Generate all 7-tuples from the set of 12 correspondences and estimate f
    # undamental matrix F for each of them.

    ### For each tested F, compute essential matrix E using internal
    # calibration K.

    ### Compute fundamental matrix Fe consistent with K from E and K and its
    # epipolar error over all matches.

    ### Choose such Fe and E that minimize maximal epipolar error over all
    # matches.

    for inx in itertools.combinations(range(0, len(U1[0])), 7):
        pass
    # Step 5
    # Draw the 12 corresponding points in different colour in the two images.
    # Using Fe, compute the corresponding epipolar lines and draw them into the
    # images in corresponding colours. Export as 09_eg.pdf.

    # Step 6
    # raw graphs of epipolar errors d1_i and d2_i w.r.t Fe for all points.
    # Draw both graphs into single figure (different colours) and export as
    # 09_errors.pdf

    #  Step 7
    # Save F, Ex, Fx, E, Fe and u1, u2, point_sel_e (indices of seven points
    # used for computing Fe) as 09a_data.mat.
