import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
# from hw02 import estimate_Q
from mpl_toolkits import mplot3d
import scipy.linalg as slinalg


def Q2KRC(Q: np.ndarray):
    """
    decomposing a camera matrix Q (3×4) into into the projection centre
    C (3×1), rotation matrix R (3×3)
    and upper triangular matrix K (3×3) such that
    Q = λ ( K R | - K R C )
    where K(3,3) = 1, K(1,1) > 0, and det(R) = 1.
    :return: K, R, C
    """
    m = Q.T[-1]
    M = np.delete(Q, 3, 1)
    K = np.zeros(9).reshape(3, 3)
    m3sq = pow(np.linalg.norm(M[2]), 2)
    K[1, 2] = M[1] @ M[2] / m3sq
    K[0, 2] = M[0] @ M[2] / m3sq
    K[1, 1] = np.sqrt(((M[1] @ M[1]) / m3sq) - K[1, 2] ** 2)
    K[0, 1] = (((M[0] @ M[1]) / m3sq) - (K[0, 2] * K[1, 2])) / K[1, 1]
    K[0, 0] = np.sqrt((M[0] @ M[0] / m3sq) - K[0, 1] ** 2 - K[0, 2] ** 2)
    K[2, 2] = 1
    s = 1 if np.linalg.det(M) > 0 else -1
    R = np.linalg.inv(K) @ (
            np.sign(np.linalg.det(M)) * M / np.linalg.norm(M[2]))

    # C = -np.linalg.inv(M) @ m
    # C = np.array([[C[0]], [C[1]], [C[2]]])
    # C.reshape(3, 1)
    c = slinalg.null_space(Q)
    C = c[:-1] / c[-1]
    return K, R, C


def plot_csystem(ax, base, origin, name, color):
    """
    drawing a coordinate system with base Base located in the origin b with a
    given name and color. The base and origin are expressed in the world
    coordinate system δ. The base consists of a two or three three-dimensional
    column vectors of coordinates. E.g.
    hw03.plot_csystem(ax,np.eye(3),np.zeros([3,1]),'k','d')
    δ_x, δ_y, δ_z
    :param base: 3x2 or 3x3 mat
    :param origin: 1x3 vec
    """
    ax.quiver3D(origin[0, 0], origin[1, 0], origin[2, 0],
                base[0][0],
                base[1][0],
                base[2][0],
                arrow_length_ratio=0.1,
                color=color)
    ax.text(base[0][0] + origin[0, 0],
            base[1][0] + origin[1, 0],
            base[2][0] + origin[2, 0],
            name + "_x")
    ax.quiver3D(origin[0], origin[1], origin[2],
                base[0][1] - origin[0, 0],
                base[1][1] - origin[1, 0],
                base[2][1] - origin[2, 0],
                arrow_length_ratio=0.1,
                color=color)
    ax.text(base[0][1], base[1][1], base[2][1], name + "_y")
    if base.shape[1] > 2:
        ax.quiver3D(origin[0, 0], origin[1, 0], origin[2, 0],
                    base[0][2] - origin[0, 0],
                    base[1][2] - origin[1, 0],
                    base[2][2] - origin[2, 0],
                    arrow_length_ratio=0.1,
                    color=color)
        ax.text(base[0][2], base[1][2], base[2][2], name + "_z")


if __name__ == "__main__":
    img = plt.imread('daliborka_01.jpg')
    img = img.copy()
    f = sio.loadmat("daliborka_01-ux.mat")
    u = f["u"]
    x = f["x"]
    ix = np.array([86, 77, 83, 7, 20, 45, 63, 74, 26, 38])

    Q, points_sel, err_max, err_points, Q_all = estimate_Q(u, x, ix)
    K, R, C = Q2KRC(Q)
    C_real = C.copy()

    b1 = 5 * 10**(-6)
    f = K[0, 0] * b1
    A = (1 / f) * (K @ R)
    Pb = np.c_[A, -A @ C]

    Delta = np.eye(3)
    d = np.array([0, 0, 0])
    d = np.array([[d[0]], [d[1]], [d[2]]])

    Epsilon = Delta @ np.linalg.inv(R)
    e = C.copy()

    Nu = Epsilon @ np.linalg.inv(K)
    n = C.copy()

    Kappa = Delta * f
    k = d.copy()

    Gamma = Kappa @ np.linalg.inv(R)
    g = C.copy()

    Beta = Nu * f
    b = C.copy()

    Alpha = Beta @ np.array([[1, 0], [0, 1], [0, 0]])
    C = np.array([C[0, 0], C[1, 0], C[2, 0]])
    a = C + Beta[2].T
    a = np.array([[a[0]], [a[1]], [a[2]]])

    # for i in (a, b, g, d, e, k, n):
    #     print(i.shape)
    sio.savemat('03_bases.mat', {
        'Pb': Pb, 'f': f,
        'Alpha': Alpha, 'a': a,
        'Beta': Beta, 'b': b,
        'Gamma': Gamma, 'g': g,
        'Delta': Delta, 'd': d,
        'Epsilon': Epsilon, 'e': e,
        'Kappa': Kappa, 'k': k,
        'Nu': Nu, 'n': n
        })
    #
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Task 1
    # borders = 10
    # ax.axes.set_xlim3d(left=-borders, right=borders)
    # ax.axes.set_ylim3d(bottom=-borders, top=borders)
    # ax.axes.set_zlim3d(bottom=-borders, top=borders)
    Beta = Beta @ np.array([[1100, 0, 0], [0, 850, 0], [0, 0, 1]])
    Alpha = Alpha @ np.array([[1100, 0], [0, 850]])
    plot_csystem(ax, Delta, d, 'δ', 'black')
    plot_csystem(ax, Beta * 50, b, 'β', 'red')
    plot_csystem(ax, Epsilon, e, 'ε', 'magenta')
    plot_csystem(ax, Kappa, k, 'κ', 'brown')
    plot_csystem(ax, Nu, n, 'υ', 'cyan')
    ax.plot3D(x[0], x[1], x[2], "b.")
    # fig.savefig("03_figure1.pdf")

    # Task 2
    # borders = 10
    # ax.axes.set_xlim3d(left=-borders, right=borders)
    # ax.axes.set_ylim3d(bottom=-borders, top=borders)
    # ax.axes.set_zlim3d(bottom=-borders, top=borders)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot_csystem(ax, Beta, b, 'β', 'red')
    plot_csystem(ax, Gamma, g, 'γ', 'blue')
    plot_csystem(ax, Alpha, a, 'α', 'green')
    ax.plot3D(x[0], x[1], x[2], 'b.')
    # fig.savefig("03_figure2.pdf")

    # # Task 3
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    borders = 1
    ax.axes.set_xlim3d(left=-borders, right=borders)
    ax.axes.set_ylim3d(bottom=-borders, top=borders)
    ax.axes.set_zlim3d(bottom=-borders, top=borders)
    plot_csystem(ax, Delta, d, 'δ', 'black')
    plot_csystem(ax, Epsilon, e, 'ε', 'magenta')
    ax.plot3D(x[0], x[1], x[2], 'b.')
    for q in Q_all:
        K, R, C = Q2KRC(q[0])
        ax.plot3D(C[0], C[1], C[2], 'r.')
    # plt.show()
    fig.savefig("04_scene.pdf")
