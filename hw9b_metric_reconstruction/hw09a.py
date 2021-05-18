import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg
from hw08 import u2F, shortest_distance


def step2(u1_step2, u2_step2, F_step2, filename="09_egx.pdf",
          header='The epipolar lines using Fx'):
    fig = plt.figure()
    fig.clf()
    fig.suptitle(header)
    plt.subplot(121)

    i = 0
    for x_p1, y_p1, x_p2, y_p2 in zip(u1_step2[0], u1_step2[1], u2_step2[0],
                                      u2_step2[1]):
        plt.plot([int(x_p1)], [int(y_p1)], color=colors[i], marker="X",
                 markersize=10)
        point2_step2 = np.c_[x_p2, y_p2, 1].reshape(3, 1)

        x = np.linspace(1, 1200, 1200)
        ep1_step2 = F_step2.T @ point2_step2
        y = -((ep1_step2[2] / ep1_step2[1]) + x * ep1_step2[0] / ep1_step2[1])
        plt.plot(x, y, color=colors[i])
        i += 1
    plt.imshow(img1)

    plt.subplot(122)

    i = 0
    for x_p1, y_p1, x_p2, y_p2 in zip(u1_step2[0], u1_step2[1], u2_step2[0],
                                      u2_step2[1]):
        plt.plot([int(x_p2)], [int(y_p2)],
                 color=colors[i],
                 marker="X",
                 markersize=10)
        point1_step2 = np.c_[x_p1, y_p1, 1].reshape(3, 1)

        x = np.linspace(1, 1200, 1200)
        point1_step2 = point1_step2.reshape(3, 1)
        ep2_step2 = F_step2 @ point1_step2
        y = -((ep2_step2[2] / ep2_step2[1]) + x * ep2_step2[0] / ep2_step2[1])
        plt.plot(x, y, color=colors[i])
        i += 1
    plt.imshow(img2)
    plt.show()
    fig.savefig(filename)


def step3(u1_step3, u2_step3, F_step3, filename="09_errorsx.pdf",
          header="The epipolar error for all points (Fx)"):
    fig = plt.figure()
    fig.clf()
    d1_arr, d2_arr = [], []
    for i in range(len(u1_step3[0])):
        a1_step3 = np.array([u1_step3[0][i],
                             u1_step3[1][i], 1])
        a2_step3 = np.array([u2_step3[0][i],
                             u2_step3[1][i], 1])
        ep1_step3 = F_step3.T @ a2_step3
        ep2_step3 = F_step3 @ a1_step3
        d1_arr.append(shortest_distance(a1_step3, ep1_step3))
        d2_arr.append(shortest_distance(a2_step3, ep2_step3))
    plt.title(header)
    plt.xlabel('point index')
    plt.ylabel('epipolar err [px]')
    plt.plot(d1_arr, color="b", label="image 1")
    plt.plot(d2_arr, color='g', label='image 2')
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # Data preprocessing
    img1 = plt.imread('daliborka_01.jpg')
    img2 = plt.imread('daliborka_23.jpg')

    f = sio.loadmat("daliborka_01_23-uu.mat")
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
    u, d, vh = np.linalg.svd(Ex)
    Ex = u @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ vh
    Fx = np.linalg.inv(K).T @ Ex @ np.linalg.inv(K)

    # Step 2
    # Draw the 12 corresponding points IX (from HW-08) in different colour in
    # the two images. Using Fx, compute the corresponding epipolar lines and
    # draw them into the images in corresponding colours. Export as 09_egx.pdf.

    step2(u1, u2, Fx)

    # Step 3
    # Draw graphs of epipolar errors d1_i and d2_i w.r.t Fx for all points.
    # Draw both graphs into single figure (different colours) and export as
    # 09_errorsx.pdf.

    step3(u01, u23, Fx)

    # Step 4
    # Find essential matrix E by minimizing the maximum epipolar error of the
    # respective fundamental matrix Fe consistent with K using the same
    # correspondences:

    result_Fe_errors_inxs = []
    ### Generate all 7-tuples from the set of 12 correspondences and estimate
    # fundamental matrix F for each of them.
    for inx in itertools.combinations(range(0, len(u1[0])), 7):
        u1_current = u1[:, inx]
        u2_current = u2[:, inx]
        Fs_step4 = u2F(u1_current, u2_current)
        for each_F in Fs_step4:
            ### For each tested F, compute essential matrix E using internal
            # calibration K.

            E1_step4 = K.T @ each_F @ K
            u, d, vh = np.linalg.svd(E1_step4)
            E_step4 = u @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ vh
            ### Compute fundamental matrix Fe consistent with K from E and K and its
            # epipolar error over all matches.

            Fe = np.linalg.inv(K).T @ E_step4 @ np.linalg.inv(K)

            point_errors = []
            for counter in range(len(u01[0])):
                a1 = np.array([u01[0][counter],
                               u01[1][counter],
                               1])
                a2 = np.array([u23[0][counter],
                               u23[1][counter],
                               1])
                ep1 = Fe.T @ a2
                ep2 = Fe @ a1
                d1_i = shortest_distance(a1, ep1)
                d2_i = shortest_distance(a2, ep2)
                point_errors.append(d1_i + d2_i)
            result_Fe_errors_inxs.append(
                    [max(point_errors), Fe, E_step4, inx])
    ### Choose such Fe and E that minimize maximal epipolar error over all
    # matches.
    result_Fe_errors_inxs.sort(key=lambda x: x[0])
    Fe = result_Fe_errors_inxs[0][1]
    point_sel_e = [ix[i] for i in range(12) if
                   i in result_Fe_errors_inxs[0][3]]
    E = result_Fe_errors_inxs[0][2]

    # Step 5
    # Draw the 12 corresponding points in different colour in the two images.
    # Using Fe, compute the corresponding epipolar lines and draw them into the
    # images in corresponding colours. Export as 09_eg.pdf.

    step2(u1, u2, Fe, "09_eg.pdf", "The epipolar lines using Fe")

    # Step 6
    # raw graphs of epipolar errors d1_i and d2_i w.r.t Fe for all points.
    # Draw both graphs into single figure (different colours) and export as
    # 09_errors.pdf

    step3(u01, u23, Fe, "09_errors.pdf",
          "The epipolar error for all points (Fe)")

    #  Step 7
    # Save F, Ex, Fx, E, Fe and u1, u2, point_sel_e (indices of seven points
    # used for computing Fe) as 09a_data.mat.
    point_sel = sio.loadmat("08_data.mat")["point_sel"]
    sio.savemat('09a_data.mat', {
        'u1': u01,
        'u2': u23,
        "point_sel_e": point_sel_e,
        "point_sel": point_sel,
        'F': F,
        "Ex": Ex,
        "Fe": Fe,
        "Fx": Fx,
        "E": E,
        })
