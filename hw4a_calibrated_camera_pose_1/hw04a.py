import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
# from mpl_toolkits import mplot3d
import scipy.linalg as slinalg


def p3p_polynom(d12, d23, d31, c12, c23, c31):
    """
    source: https://cw.felk.cvut.cz/brute/data/ae/release/2021l_gvg/ae/tools/p3p_polynom.m

    P3P_POLYNOM  Coefficients of polynom for calibrated camera pose estimation.
    [a0, a1, a2, a3, a4] = p3p_polynom( d12, d23, d31, c12, c23, c31 )
    """

    a4 = -4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c23 ** 2 + d23 ** 8 - 2 * d23 ** 6 * d12 ** 2 - 2 * d23 ** 6 * d31 ** 2 + d23 ** 4 * d12 ** 4 + 2 * d23 ** 4 * d12 ** 2 * d31 ** 2 + d23 ** 4 * d31 ** 4
    a3 = 8 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c12 * c23 ** 2 + 4 * d23 ** 6 * d12 ** 2 * c31 * c23 - 4 * d23 ** 4 * d12 ** 4 * c31 * c23 + 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c31 * c23 - 4 * d23 ** 8 * c12 + 4 * d23 ** 6 * d12 ** 2 * c12 + 8 * d23 ** 6 * d31 ** 2 * c12 - 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c12 - 4 * d23 ** 4 * d31 ** 4 * c12
    a2 = -8 * d23 ** 6 * d12 ** 2 * c31 * c12 * c23 - 8 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c31 * c12 * c23 + 4 * d23 ** 8 * c12 ** 2 - 4 * d23 ** 6 * d12 ** 2 * c31 ** 2 - 8 * d23 ** 6 * d31 ** 2 * c12 ** 2 + 4 * d23 ** 4 * d12 ** 4 * c31 ** 2 + 4 * d23 ** 4 * d12 ** 4 * c23 ** 2 - 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c23 ** 2 + 4 * d23 ** 4 * d31 ** 4 * c12 ** 2 + 2 * d23 ** 8 - 4 * d23 ** 6 * d31 ** 2 - 2 * d23 ** 4 * d12 ** 4 + 2 * d23 ** 4 * d31 ** 4
    a1 = 8 * d23 ** 6 * d12 ** 2 * c31 ** 2 * c12 + 4 * d23 ** 6 * d12 ** 2 * c31 * c23 - 4 * d23 ** 4 * d12 ** 4 * c31 * c23 + 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c31 * c23 - 4 * d23 ** 8 * c12 - 4 * d23 ** 6 * d12 ** 2 * c12 + 8 * d23 ** 6 * d31 ** 2 * c12 + 4 * d23 ** 4 * d12 ** 2 * d31 ** 2 * c12 - 4 * d23 ** 4 * d31 ** 4 * c12
    a0 = -4 * d23 ** 6 * d12 ** 2 * c31 ** 2 + d23 ** 8 - 2 * d23 ** 4 * d12 ** 2 * d31 ** 2 + 2 * d23 ** 6 * d12 ** 2 + d23 ** 4 * d31 ** 4 + d23 ** 4 * d12 ** 4 - 2 * d23 ** 6 * d31 ** 2

    return np.array([a0, a1, a2, a3, a4])


def p3p_dverify(N1, N2, N3, d12, d23, d31, c12, c23, c31):
    """
     function p3p_dverify for verification of computed camera-to-point distances using the cosine law. Use this
     function in p3p_distances. The function returns vector of three errors, one for each equation. Each
     computed error should be distance (not squared), relative to particular
    d_jk, i.e.
    :return: vector of three errors
    """
    e1 = (np.sqrt(N1 ** 2 + N2 ** 2 - 2 * N1 * N2 * c12) - d12) / d12
    e2 = (np.sqrt(N2 ** 2 + N3 ** 2 - 2 * N2 * N3 * c23) - d23) / d23
    e3 = (np.sqrt(N3 ** 2 + N1 ** 2 - 2 * N3 * N1 * c31) - d31) / d31
    return np.array([e1, e2, e3])


def p3p_distances(d12, d23, d31, c12, c23, c31):
    """
    function p3p_distances for computing distances of three spatial points from a center of calibrated camera.
    The function must return the distances η_i in N1, N2, N3 of the three points. Implement only the case 'A'
    of the computation. If there are more solutions, the returned variables are row vectors (matlab) or lists (python).
    If there is no solution by the case 'A', return empty vector/list ([]]). For constructing the fourth order
    polynomial, there is the p3p_polynom function in the tools repository, that can be used in your code.
    """
    n1, n2, n3 = list(), list(), list()
    epsilon = 1e-4
    a_s = p3p_polynom(d12, d23, d31, c12, c23, c31)
    C = np.array([[0, 0, 0, -a_s[0] / a_s[4]],
                  [1, 0, 0, -a_s[1] / a_s[4]],
                  [0, 1, 0, -a_s[2] / a_s[4]],
                  [0, 0, 1, -a_s[3] / a_s[4]]])
    n12 = slinalg.eigvals(C)
    # n12.filter(lambda x: np.iscomplex(x))
    for each in n12:
        flag = True
        if np.iscomplex(each):
            continue
        n = np.real(each)
        m1 = d12 ** 2
        m2 = (d31 ** 2) - d23 ** 2
        q1 = (d23 ** 2) * (1 + n ** 2 - 2 * n * c12) - (d12 ** 2) * (n ** 2)
        p1 = -2 * (d12 ** 2) * n * c23
        p2 = 2 * (d23 ** 2) * c31 - 2 * (d31 ** 2) * n * c23
        q2 = d23 ** 2 - (d31 ** 2) * (n ** 2)

        n13 = (m1 * q2 - m2 * q1) / (m1 * p2 - m2 * p1)
        N1 = d12 / np.sqrt(1 + n ** 2 - 2 * n * c12)
        N2 = N1 * n
        N3 = N1 * n13
        errors = p3p_dverify(N1, N2, N3, d12, d23, d31, c12, c23, c31)
        for each in errors:
            if each > epsilon:
                flag = False

        if flag:
            n1.append(N1)
            n2.append(N2)
            n3.append(N3)

    return n1, n2, n3


def cos_s(x1, x2, x3, K):
    if len(x1.shape) != 2:
        x1 = x1.reshape(3, 1)
        x2 = x2.reshape(3, 1)
        x3 = x3.reshape(3, 1)
    # tmp = x1.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x2
    c12 = ((x1.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x2) /
           (np.linalg.norm(np.linalg.inv(K) @ x1) * np.linalg.norm(np.linalg.inv(K) @ x2)))[0][0]
    c23 = ((x2.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x3) /
           (np.linalg.norm(np.linalg.inv(K) @ x2) * np.linalg.norm(np.linalg.inv(K) @ x3)))[0][0]
    c31 = ((x3.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x1) /
           (np.linalg.norm(np.linalg.inv(K) @ x3) * np.linalg.norm(np.linalg.inv(K) @ x1)))[0][0]

    return c12, c23, c31


if __name__ == "__main__":
    # Step 1
    # Construct a simple projection matrix where C = [1, 2, -3] T, f = 1, K = R = I(3×3 identity).
    # Project the 3D points X1 = [0, 0, 0] T, X2 = [1, 0, 0] T, X3 = [0, 1, 0] T
    # by the P and compute the cosines c12, c23 and c31 for the projected image points.
    # Using the 3D points and the cosines, compute the camera-points distances η and compare with correct
    # known values (C - Xi).

    # C = np.array([[1], [2], [-3]])
    # f = 1
    # K, R = [np.eye(3) for i in range(2)]
    # P_B = np.c_[1 / f * K @ R, -1 / f * K @ R @ C]
    # X = [np.array([[0], [0], [0], [1]]), np.array([[1], [0], [0], [1]]), np.array([[0], [1], [0], [1]])]
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
    # print(res)
    # for each in X:
    #     print(X)
    # print(np.linalg.norm(C - each[:-1]))

    # Step 2
    # X1 = np.array([[1], [0], [0], [1]])
    # X2 = np.array([[0], [2], [0], [1]])
    # X3 = np.array([[0], [0], [3], [1]])
    # c12 = 0.9037378393
    # c23 = 0.8269612542
    # c31 = 0.9090648231
    # d12 = np.linalg.norm(X2 - X1)
    # d23 = np.linalg.norm(X3 - X2)
    # d31 = np.linalg.norm(X1 - X3)
    # res = p3p_distances(d12, d23, d31, c12, c23, c31)
    # print(res)

    # Step 3
    K = sio.loadmat("K.mat")["K"]
    f = sio.loadmat("daliborka_01-ux.mat")

    ix = np.array([86, 77, 83, 7, 20, 45, 63, 74, 26, 38]) - 1

    x = f["x"]
    x = np.array([x.T[i] for i in ix])
    x = np.c_[x, np.array([[1] * len(x)]).T]

    u = f["u"]
    u = np.array([u.T[i] for i in ix])
    u = np.c_[u, np.array([[1] * len(u)]).T]
    N1, N2, N3 = list(), list(), list()

    for inx in itertools.combinations(range(0, len(ix)), 3):
        X = x[inx, :]
        d12 = np.linalg.norm(X[1] - X[0])
        d23 = np.linalg.norm(X[2] - X[1])
        d31 = np.linalg.norm(X[0] - X[2])
        X_s = u[inx, :]
        c12, c23, c31 = cos_s(X_s[0], X_s[1], X_s[2], K)
        res = np.array(p3p_distances(d12, d23, d31, c12, c23, c31))
        # print(d12, d23, d31, c12, c23, c31)
        N1.extend(res[0])
        N2.extend(res[1])
        N3.extend(res[2])

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.title('Distances')
    plt.xlabel('trial')
    plt.ylabel('distance[m]')
    plt.plot(N1, color='red', fillstyle='none', label="η_1")
    plt.plot(N2, color='green', fillstyle='none', label="η_2")
    plt.plot(N3, color='blue', fillstyle='none', label="η_3")
    plt.legend(loc='best')
    plt.show()
    # fig.savefig("04_distances.pdf")
