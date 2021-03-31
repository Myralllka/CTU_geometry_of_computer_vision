import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as scpl


def estimate_Q(u, x, ix):
    """
    :return: Q, points_sel, err_max, err_points, Q_all
    where
    Q: best projection matrix
    points_sel: indices of the 6 points
    err_max: vector of all maximal errors for all tested matrices
    err_points: vector of point errors for the best camera
    Q_all: cell matrix containing all tested camera matrices
    """
    points_sel, Q, Q_all, m_all = list(), list(), list(), list()
    max_err = []
    xs = np.array([x.T[i] for i in ix])
    us = np.array([u.T[i] for i in ix])

    for i in range(len(ix)):
        m1 = list(xs[i])
        m2 = [0] * 4
        m1.append(1)
        m2.extend(m1)
        m1.extend([0] * 4)
        m1.extend((-us[i][0]) * np.array(m1)[:4])
        m2.extend((-us[i][1]) * np.array(m2)[4:])
        m_all.append([m1, m2])

    m_all = np.array(m_all)

    for inx in itertools.combinations(range(0, len(m_all)), 6):
        points_sel = np.array([ix[i] for i in range(len(ix)) if i in inx])
        M = m_all[inx, :]
        M = M.reshape(12, 12)
        for i in itertools.combinations(range(0, len(M)), 11):
            M11 = M[i, :]
            Q = scpl.null_space(M11).reshape(3, 4)
            errors = list()
            for point_i in range(len(x[0])):
                point = list(x.T[point_i])
                point.append(1)
                point = np.array(point)
                projected = Q @ point
                errors.append(np.linalg.norm((projected / projected[-1])[
                                             :-1] - u[:, point_i]))
            max_err.append(max(errors))
            Q_all.append((Q, max(errors), points_sel, errors))

    Q_all.sort(key=lambda x: x[1])

    return Q_all[0][0], Q_all[0][2], max_err, Q_all[0][3], Q_all


if __name__ == "__main__":
    img = plt.imread('daliborka_01.jpg')
    img = img.copy()
    f = sio.loadmat("daliborka_01-ux.mat")
    u = f["u"]
    x = f["x"]
    ix = np.array([86, 77, 83, 7, 20, 45, 63, 74, 26, 38])

    Q, points_sel, err_max, err_points, Q_all = estimate_Q(u, x, ix)

    ####### img 1 ########

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.title('Maximal reprojection error for each tasted Q')
    plt.xlabel('selection index')
    plt.ylabel('log_10 of maximum reprojection error [px]')
    plt.plot(np.log10(err_max))
    plt.show()
    plt.legend(loc='best')
    fig.savefig("02_Q_maxerr.pdf")

    ####### img 2 ########

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.title('original and reprojected points')
    plt.imshow(img)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.plot(u[0], u[1], 'b.', fillstyle='none', label="Orig. pts")
    plt.plot(u[0][points_sel], u[1][points_sel],
             'y.',
             fillstyle='full',
             label="Used for Q")
    pr_array = []
    for i in range(len(u[0])):
        point = list(x.T[i])
        point.append(1)
        point = np.array(point)
        projected = Q @ point
        projected /= projected[-1]
        projected = projected[:-1]
        pr_array.append(projected)

    pr_array = np.array(pr_array)
    pr_array = pr_array.T
    plt.plot(pr_array[0], pr_array[1],
             'ro',
             fillstyle='none',
             label="Reprojected")
    plt.legend(loc='best')
    plt.show()
    fig.savefig("02_Q_projections.pdf")

    ####### img 3 ########

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.title('Reprojected errors (100x enlarged)')
    plt.imshow(img)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.plot(u[0], u[1], 'b.', fillstyle='none', label="Orig. pts")
    plt.plot(u[0][points_sel], u[1][points_sel],
             'y.',
             fillstyle='full',
             label="Used for Q")

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
    fig.savefig("02_Q_projections_errors.pdf")

    ####### img 4 ########

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.title('All point reprojection errors for the best Q')
    plt.xlabel('point index')
    plt.ylabel('reprojection error [px]')
    plt.plot(Q_all[0][3])

    plt.show()
    plt.legend(loc='best')
    fig.savefig("02_Q_pointerr.pdf")