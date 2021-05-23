import numpy as np  # for matrix computation and linear algebra
import scipy.linalg
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg


def step_56(u1_c, u2_c, X_c, P1_c, P2_c, angls):
    fig = plt.figure()
    fig.clf()
    plt.title('Model reconstruction')
    ax = plt.axes(projection='3d')
    d1_arr, d2_arr = [], []
    for each_X, i in zip(X_c, range(len(X_c))):
        X_loop = np.array([each_X[0], each_X[1], each_X[2], 1])
        reprojected = P1_c @ X_loop
        reprojected = (reprojected / reprojected[-1])
        d1_arr.append(reprojected)
        reprojected = P2_c @ X_loop
        reprojected = (reprojected / reprojected[-1])
        d2_arr.append(reprojected)
    d1_arr = np.array(d1_arr).T
    d2_arr = np.array(d2_arr).T

    # ax.plot3D(d1_arr[0], d1_arr[1], d1_arr[2], 'r.')
    # ax.plot3D(d2_arr[0], d2_arr[1], d2_arr[2], 'g.')
    X_c = np.array(X_c).T
    for i in range(len(X_c[0])):
        ax.plot([d1_arr[0, i], X_c[0, i]], [d1_arr[1, i], X_c[1, i]],
                [d1_arr[2, i], X_c[2, i]], color="yellow")
    for i in range(len(X[0])):
        ax.plot([d2_arr[0, i], X_c[0, i]], [d2_arr[1, i], X_c[1, i]],
                [d2_arr[2, i], X_c[2, i]], color="green")

    # for i, j in zip(angls[0], angls[1]):
    #     ax.plot([d1_arr[0, i], d1_arr[0, j]], [d1_arr[1, i], d1_arr[1, j]], [d1_arr[2, i], d1_arr[2, j]],
    #             color="black")
    # for i, j in zip(angls[0], angls[1]):
    #     ax.plot([d2_arr[0, i], d2_arr[0, j]], [d2_arr[1, i], d2_arr[1, j]],
    #             [d2_arr[2, i], d2_arr[2, j]],color="black")
    # for i, j in zip(angls[0], angls[1]):
    #     ax.plot([X[0, i], X[0, j]], [X[1, i], X[1, j]], [X[2, i], X[2, j]],
    #             color="black")

    # borders = 30
    # ax.axes.set_xlim3d(left=-borders, right=borders)
    # ax.axes.set_ylim3d(bottom=-borders, top=borders)
    # ax.axes.set_zlim3d(bottom=-borders, top=borders)

    ax.plot3D(X_c[0], X_c[1], X_c[2], 'b.')

    plt.show()


def step5(u1_c, u2_c, X, P1_c, P2_c):
    fig = plt.figure()
    fig.clf()
    d1_arr, d2_arr = [], []
    for each_X, i in zip(X, range(len(X))):
        X_loop = np.array([each_X[0], each_X[1], each_X[2], 1])
        reprojected = P1_c @ X_loop
        reprojected = (reprojected / reprojected[-1])[:-1]
        d1_arr.append(
                np.linalg.norm(
                    np.array([u1_c[0][i], u1_c[1][i]]) - reprojected))
        reprojected = P2_c @ X_loop
        reprojected = (reprojected / reprojected[-1])[:-1]
        d2_arr.append(
                np.linalg.norm(
                    np.array([u2_c[0][i], u2_c[1][i]]) - reprojected))

    plt.title("Reprojection errors")
    plt.xlabel('point index')
    plt.ylabel('Reprojection err [px]')
    plt.plot(d1_arr, color="b", label="image 1")
    plt.plot(d2_arr, color='g', label='image 2')
    plt.legend(loc='best')
    # plt.savefig("09_errorsr.pdf")
    plt.show()


def step4(u1_c, u2_c, X, P1_c, P2_c, angls):
    fig = plt.figure()
    fig.clf()

    fig.suptitle("The corresponding points and edges")

    plt.subplot(121)
    for i, j in zip(angls[0], angls[1]):
        plt.plot([u1_c[0][i], u1_c[0][j]], [u1_c[1][i], u1_c[1][j]], 'y-')
    for each_X in X:
        reprojected = P1_c @ np.array([each_X[0], each_X[1], each_X[2], 1])
        reprojected /= reprojected[-1]
        plt.plot(reprojected[0], reprojected[1], 'ro')
    plt.plot(u1_c[0], u1_c[1], 'b.')
    # plt.imshow(img1)
    ##########################################
    plt.subplot(122)
    for i, j in zip(angls[0], angls[1]):
        plt.plot([u2_c[0][i], u2_c[0][j]], [u2_c[1][i], u2_c[1][j]], 'y-')
    for each_X in X:
        reprojected = P2_c @ np.array([each_X[0], each_X[1], each_X[2], 1])
        reprojected /= reprojected[-1]
        plt.plot(reprojected[0], reprojected[1], 'ro')
    plt.plot(u2_c[0], u2_c[1], 'b.')
    # plt.imshow(img2)
    plt.show()
    # fig.savefig("09_reprojection.pdf")


def e2rc(E_e2rc):
    # E = R cross C, decomposition:
    Rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    U, S, V = np.linalg.svd(E_e2rc)
    R1_c = U @ Rz @ V
    R2_c = U @ Rz.T @ V
    C1_c = V[-1] * S[0]
    C2_c = -C1_c
    return R1_c, R2_c, C1_c, C2_c


def compute_Xs(u_current_1, u_current_2, P1_c, P2_c):
    res_X = []
    for cnt in range(len(u_current_1[0])):
        u_tmp = np.array([u_current_1[0][cnt],
                          u_current_1[1][cnt], 1]).reshape(3, 1)
        v_tmp = np.array([u_current_2[0][cnt],
                          u_current_2[1][cnt], 1]).reshape(3, 1)
        M = np.vstack([np.hstack([u_tmp, np.zeros((3, 1)), -P1_c]),
                       np.hstack([np.zeros((3, 1)), v_tmp, -P2_c])])
        _, _, Vh = np.linalg.svd(M)
        res_X.append(Vh[-1, 2:5] / Vh[-1, -1])
    return res_X


# def triangulation(u_current_1, u_current_2, P1_c, P2_c):
#     res_X = []
#     p11, p12, p13 = P1_c
#     p21, p22, p23 = P2_c
#     for cntr in range(len(u_current_1[0])):
#         u_tmp = np.array([u_current_1[0][cntr],
#                           u_current_1[1][cntr],
#                           1]).reshape(3, 1)
#         v_tmp = np.array([u_current_2[0][cntr],
#                           u_current_2[1][cntr],
#                           1]).reshape(3, 1)
#         r1 = ((u_tmp[0] * p13) - p11)
#         r2 = ((u_tmp[1] * p13) - p12)
#         r3 = ((v_tmp[0] * p23) - p21)
#         r4 = ((v_tmp[1] * p23) - p22)
#         M = np.vstack([r1, r2, r3, r4])
#         print(M)
#         _, _, Vh = np.linalg.svd(M)
#         res_X.append(Vh[-1] / Vh[-1, -1])
#     return res_X


def optimize_rc(R1_current, C1_current, R2_current, C2_current, K_current,
                u_current_1, u_current_2):
    result_index_R_C_Ps = []
    # P1_c = np.append(K, np.array([0, 0, 0]).reshape(3, 1), axis=1)
    P1_c = np.hstack((np.eye(3), np.zeros((3, 1))))
    for R_loop, C_loop in [[R1_current, C1_current],
                           [R1_current, C2_current],
                           [R2_current, C1_current],
                           [R2_current, C2_current]]:
        counter = 0
        P2_c = np.append(K_current @ R_loop,
                         (- K_current @ R_loop @ C_loop).reshape(3, 1), axis=1)
        X = compute_Xs(u_current_1, u_current_2, P1_c, P2_c)
        for each, i in zip(X, range(len(X))):
            u_tmp = np.array([u_current_1[0][i],
                              u_current_1[1][i], 1]).reshape(3, 1)
            v_tmp = np.array([u_current_2[0][i],
                              u_current_2[1][i], 1]).reshape(3, 1)
            if (np.dot(each, u_tmp) / (
                    np.linalg.norm(each) * np.linalg.norm(u_tmp)) > 0) and \
                    (np.dot(each, v_tmp) / (
                            np.linalg.norm(each) * np.linalg.norm(v_tmp)) > 0):
                counter += 1
        print(counter)
        result_index_R_C_Ps.append([counter, R_loop, C_loop, P1_c, P2_c])
    return sorted(result_index_R_C_Ps, key=lambda x: x[0])[-1]


if __name__ == "__main__":
    # Data preprocessing
    img1 = plt.imread('daliborka_01.jpg')
    img2 = plt.imread('daliborka_23.jpg')

    f = sio.loadmat("daliborka_01_23-uu.mat")

    K = sio.loadmat('K.mat')["K"]

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

    # Step 1
    # Decompose the best E into relative rotation R and translation C
    # (four solutions). Choose such a solution that reconstructs (most of)
    # the points in front of both (computed) cameras.

    E = sio.loadmat("09a_data.mat")["E"]

    R1, C1, R2, C2 = e2rc(E)

    # Step 2
    # Construct projective matrices P1, P2 (including K).
    _, R, C, P1, P2 = optimize_rc(R1, R2, C1, C2, K, u01, u23)

    # Step 3
    # Compute scene points X.

    X = compute_Xs(u01, u23, P1, P2)

    # Step 4
    # Display the images, draw the input points as blue dots and the scene points X projected by appropriate P_i as red circles. Draw also the edges, connecting the original points as yellow lines. Export as 09_reprojection.pdf.

    step4(u01, u23, X, P1, P2, edges)

    # Step 5
    # Draw graph of reprojection errors and export as 09_errorsr.pdf.

    # step5(u01, u23, X, P1, P2)

    # step_56(u01, u23, X, P1, P2, edges)
    # Step 6
    # Draw the 3D point set (using 3D plotting facility) connected
    # by the edges as a wire-frame model, shown from the top of the tower,
    # from the side, and from some general view. Export as 09_view1.pdf,
    # 09_view2.pdf, and 09_view3.pdf.

    X_3d = np.array(compute_Xs(u01, u23, P1, P2)).T

    fig = plt.figure()
    plt.title('Model reconstruction')
    ax = plt.axes(projection='3d')
    for i, j in zip(edges[0], edges[1]):
        ax.plot([X_3d[0, i], X_3d[0, j]],
                [X_3d[1, i], X_3d[1, j]],
                [X_3d[2, i], X_3d[2, j]],
                color="black")

    # borders = 3
    # ax.axes.set_xlim3d(left=-borders, right=borders)
    # ax.axes.set_ylim3d(bottom=-borders, top=borders)
    # ax.axes.set_zlim3d(bottom=-borders, top=borders)

    ax.plot3D(X_3d[0], X_3d[1], X_3d[2], 'b.')

    plt.show()

    # fig.savefig(".pdf")

    # Step 7
    # Save Fe, E, R, C, P1, P2, X, and u1, u2, point_sel_e as 09b_data.mat.
    # X = np.array(X).T
    # sio.savemat('09b_data.mat', {
    #     'u1': u01,
    #     'u2': u23,
    #     "Fe": sio.loadmat("09a_data.mat")["Fe"],
    #     "E": E,
    #     "R": R,
    #     "C": C.reshape(3, 1),
    #     "P1": P1,
    #     "P2": P2,
    #     "X": X,
    #     "point_sel_e": sio.loadmat("09a_data.mat")["point_sel_e"],
    #     })
