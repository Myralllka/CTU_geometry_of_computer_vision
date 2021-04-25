import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
# from mpl_toolkits import mplot3d
import scipy.linalg as slinalg
from hw04a import *
from hw03 import plot_csystem


def p3p_RC(N, u, X, K):
    """
    The function takes one configuration of η_i and returns a single R and C.
    Note that R must be orthonormal with determinant equal to +1.
    :param N: [η_1, η_2, η_3]
    :param u: coordinates of image points
    :param X: coordinates of world points
    :param K: Camera calibration mat
    :return: calibrated camera centre C and orientation R
    """

    # preparing data because of incorrect input (named as matlab form)...
    # u = u.T
    # X = X.T

    R, C = list(), list()
    new_u = list()
    inv_K = np.linalg.inv(K)
    for each in u:
            tmp = each
            if min(tmp.shape) == 2:
                tmp = list(tmp.reshape(2, ))
                tmp.append(1)
                tmp = inv_K @ tmp
            else:
                tmp = list(tmp.reshape(3, ))
                tmp = inv_K @ tmp
            new_u.append(tmp)

    u = new_u
    Y = [N[i] * (u[i] / np.linalg.norm(u[i])) for i in range(3)]

    Z2e = (Y[1] - Y[0]).reshape(3, )
    Z2d = (X[1] - X[0]).reshape(3, )
    Z3e = (Y[2] - Y[0]).reshape(3, )
    Z3d = (X[2] - X[0]).reshape(3, )

    Z1e = np.cross(Z2e, Z3e)
    Z1d = np.cross(Z2d, Z3d)

    R = np.c_[Z1e, Z2e, Z3e] @ np.linalg.inv(np.c_[Z1d, Z2d, Z3d])
    # print(R)
    # print(R.T @ R)
    # print(np.linalg.det(R))
    # print(round(np.linalg.det(R) - 1, 2))

    C = X[0].reshape(3, ) - (R.T @ Y[0].reshape(3, ))
    C = C.reshape(3, 1)

    return R, C


if __name__ == "__main__":
    # Step 1

    # C = np.array([1, 2, -3]).reshape(3, 1)
    # f = 1
    # K, R = np.eye(3), np.eye(3)
    # P_B = np.c_[1 / f * K @ R, -1 / f * K @ R @ C]
    # X = [np.array([[0], [0], [0], [1]]), np.array([[1], [0], [0], [1]]),
    #      np.array([[0], [1], [0], [1]])]
    # d12 = np.linalg.norm(X[1] - X[0])
    # d23 = np.linalg.norm(X[2] - X[1])
    # d31 = np.linalg.norm(X[0] - X[2])
    # x = list()
    # for each in X:
    #     tmp = P_B @ each
    #     tmp /= tmp[-1]
    #     x.append(tmp)
    # c12, c23, c31 = cos_s(x[0], x[1], x[2], K)
    # res = p3p_distances(d12, d23, d31, c12, c23, c31)
    # R, C = p3p_RC([res[0][0], res[1][0], res[2][0]],
    #               [i[:-1] for i in x],
    #               [i[:-1] for i in X], K)



    # Step 2


    K = sio.loadmat("K.mat")["K"]
    f = sio.loadmat("daliborka_01-ux.mat")

    ix = np.array([86, 77, 83, 7, 20, 45, 63, 74, 26, 38]) - 1

    x = f["x"]
    x = np.array([x.T[i] for i in ix])
    x_extended = np.c_[x, np.array([[1] * len(x)]).T]

    u = f["u"]
    u = np.array([u.T[i] for i in ix])
    u_extended = np.c_[u, np.array([[1] * len(u)]).T]

    R_list, C_list = list(), list()
    all_xs = np.array(f["x"]).T
    all_us = np.array(f["u"]).T
    all_errors_R_C_p = list()
    for inx in itertools.combinations(range(0, len(ix)), 3):
        X = x_extended[inx, :]
        d12 = np.linalg.norm(X[1] - X[0])
        d23 = np.linalg.norm(X[2] - X[1])
        d31 = np.linalg.norm(X[0] - X[2])
        X_s = u_extended[inx, :]
        c12, c23, c31 = cos_s(X_s[0], X_s[1], X_s[2], K)
        N1, N2, N3 = np.array(p3p_distances(d12, d23, d31, c12, c23, c31))
        X = x[inx, :]
        X_s = u[inx, :]
        for res in zip(N1, N2, N3):
            R, C = p3p_RC(res, X_s, X, K)
            Q = np.c_[K @ R, -K @ R @ C]

            errors = list()

            for point_i in range(len(all_xs)):
                point = list(all_xs[point_i])
                point.append(1)
                point = np.array(point)
                projected = Q @ point
                errors.append(np.linalg.norm((projected / projected[-1])
                                             [:-1] - all_us[point_i]))

            all_errors_R_C_p.append([max(errors), R, C, X_s, inx, errors])

    # fig = plt.figure()  # figure handle to be used later
    # fig.clf()
    # plt.title('Maximal reprojection error for each tasted P')
    # plt.xlabel('selection index')
    # plt.ylabel('log_10 of maximum reprojection error [px]')
    # plt.plot(np.log10(list(np.array(all_errors_R_C_p)[:, 0])), 'b.',
    #          fillstyle='none')
    # plt.show()
    # plt.legend(loc='best')
    # fig.savefig("04_RC_maxerr.pdf")

    all_errors_R_C_p.sort(key=lambda a: a[0])
    points_selected = all_errors_R_C_p[0][3].T
    R, C = all_errors_R_C_p[0][1], all_errors_R_C_p[0][2]
    # sio.savemat('04_p3p.mat', {
    #     'R': R,
    #     'C': C,
    #     'point_sel': ix[all_errors_R_C_p[0][-1]]
    #     })
    # print(all_errors_R_C_p[0][4])

    # print(Q)
    # print(K)
    # print(R)
    # print(C)
    # print(K@R)
    Q = np.c_[K @ R, (-K) @ R @ C]


    img = plt.imread('daliborka_01.jpg')
    img = img.copy()

    # fig = plt.figure()  # figure handle to be used later
    # fig.clf()
    # plt.title('Reprojection errors emphasized 100 times')
    # plt.imshow(img)
    # plt.xlabel('x [px]')
    # plt.ylabel('y [px]')
    u = f["u"]
    x = f["x"]
    # plt.plot(u[0], u[1], 'b.', fillstyle='none', label="Orig. pts")
    #
    # plt.plot(points_selected[0], points_selected[1],
    #          'yo',
    #          fillstyle='full',
    #          label="Used for P")

    pr_array = []
    errors = []
    for i in range(len(u[0])):
        point = list(x.T[i])
        point.append(1)
        point = np.array(point)
        # print("point")
        # print(point)
        projected = Q @ point
        # print("projected")
        # print(projected)
        projected /= projected[-1]
        projected = projected[:-1]
        pr_array.append(projected)

    pr_array = np.array(pr_array)
    pr_array = pr_array.T

    e = 100 * (pr_array - u)
    plt.plot((u[0][0], u[0][0] + e[0][0]), (u[1][0], u[1][0] + e[1][0]),
             'r-',
             fillstyle='none',
             label="Errors (100x)")
    plt.plot((u[0], u[0] + e[0]), (u[1], u[1] + e[1]),
             'r-',
             fillstyle='none')
    plt.legend(loc='best')
    plt.show()
    # fig.savefig("04_RC_projections_errors.pdf")


    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.title('All point reprojection errors for the best P')
    plt.xlabel('point')
    plt.ylabel('err [px]')
    plt.plot(all_errors_R_C_p[0][5])

    plt.show()
    plt.legend(loc='best')
    fig.savefig("04_RC_pointerr.pdf")

    Delta = np.eye(3)
    d = np.array([0, 0, 0])
    d = np.array([[d[0]], [d[1]], [d[2]]])

    Epsilon = Delta @ np.linalg.inv(R)
    e = C.copy()


    fig = plt.figure()
    plt.title('All tested RCs')
    ax = plt.axes(projection='3d')
    borders = 1
    ax.axes.set_xlim3d(left=-borders, right=borders)
    ax.axes.set_ylim3d(bottom=-borders, top=borders)
    ax.axes.set_zlim3d(bottom=-borders, top=borders)
    plot_csystem(ax, Epsilon, e, 'ε', 'magenta')
    plot_csystem(ax, Delta, d, 'δ', 'black')
    ax.plot3D(x[0], x[1], x[2], 'b.')
    all_C = np.array(all_errors_R_C_p)[:, 2]
    all_C = np.array([each.reshape(3,) for each in all_C])
    # print(all_C.shape)
    ax.plot3D(all_C[:, 0], all_C[:, 1], all_C[:, 2], 'r.')
    plt.show()
    fig.savefig("04_scene.pdf")
