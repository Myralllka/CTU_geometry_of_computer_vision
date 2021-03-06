import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg
import tools
import cv2
import os

from hw04a import p3p_distances


def p3p_RC(N, X, u, K):
    """
    The function takes one configuration of η_i and returns a single R and C.
    Note that R must be orthonormal with determinant equal to +1.
    :param N: [η_1, η_2, η_3]
    :param u: coordinates of image points
    :param K: Camera calibration mat
    :return: calibrated camera centre C and orientation R
    """
    R, C = list(), list()
    new_u = list()
    inv_K = np.linalg.inv(K)
    for each in u:
        tmp = list(each.reshape(2, ))
        tmp.append(1)
        tmp = np.array(tmp)
        tmp = inv_K @ tmp
        new_u.append(tmp)
    u = new_u
    # X = np.array([[i[0], i[1], 1] for i in X])
    Y = [N[i] * (u[i] / np.linalg.norm(u[i])) for i in range(3)]

    Z2e = (Y[1] - Y[0]).reshape(3, )
    Z2d = (X[1] - X[0]).reshape(3, )
    Z3e = (Y[2] - Y[0]).reshape(3, )
    Z3d = (X[2] - X[0]).reshape(3, )
    Z1e = np.cross(Z2e, Z3e)
    Z1d = np.cross(Z2d, Z3d)
    R = np.c_[Z1e, Z2e, Z3e] @ np.linalg.inv(np.c_[Z1d, Z2d, Z3d])
    C = X[0].reshape(3, ) - (R.T @ Y[0].reshape(3, ))
    C = C.reshape(3, 1)
    return R, C


def cos_s(x1, x2, x3, K):
    x1 = np.array([x1[0], x1[1], 1]).reshape(3, 1)
    x2 = np.array([x2[0], x2[1], 1]).reshape(3, 1)
    x3 = np.array([x3[0], x3[1], 1]).reshape(3, 1)
    K = np.array(K)
    # tmp = x1.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x2
    c12 = ((x1.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x2) /
           (np.linalg.norm(np.linalg.inv(K) @ x1) * np.linalg.norm(
                   np.linalg.inv(K) @ x2)))[0][0]
    c23 = ((x2.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x3) /
           (np.linalg.norm(np.linalg.inv(K) @ x2) * np.linalg.norm(
                   np.linalg.inv(K) @ x3)))[0][0]
    c31 = ((x3.T @ (np.linalg.inv(K.T)) @ np.linalg.inv(K) @ x1) /
           (np.linalg.norm(np.linalg.inv(K) @ x3) * np.linalg.norm(
                   np.linalg.inv(K) @ x1)))[0][0]

    return c12, c23, c31


def plot_cube(tp, dn):
    for i in range(4):
        plt.plot([tp[i][0], dn[i][0]], [tp[i][1], dn[i][1]], 'b-')
        plt.plot([tp[i - 1][0], tp[i][0]], [tp[i - 1][1], tp[i][1]], 'b-')
        plt.plot([dn[i - 1][0], dn[i][0]], [dn[i - 1][1], dn[i][1]], 'b-')


def make_video():
    video_name = '07_seq_wire.avi'

    images = ["07_gif_{}.png".format(counter) for counter in range(0, 100)]
    frame = cv2.imread(images[0])

    height, width, layers = frame.shape
    # print(height, width)
    video = cv2.VideoWriter(video_name, 0, 60, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    for image in reversed(images):
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    # make_video()

    img1 = plt.imread("pokemon_10.jpg")
    img2 = plt.imread("pokemon_19.jpg")

    C = [[473.4, 507.2, 738.3, 736.2],
         [500.7, 346.0, 347.7, 505.7]]

    C2 = [[571.6, 384.9, 571.4, 755.5],
          [541.8, 420.6, 333.8, 434.6]]

    u = [[119.1, 463.7, 988.8, 798.0, 473.4, 507.2, 738.3, 736.2],
         [597.9, 218.7, 321.2, 813.5, 500.7, 346.0, 347.7, 505.7]]

    C_outer = [[119.1, 463.7, 988.8, 798.0],
               [597.9, 218.7, 321.2, 813.5]]

    C2_outer = [[397.4, 153.8, 701.0, 1104.9],
                [822.7, 345.3, 241.6, 582.8]]

    u2 = [[397.4, 153.8, 701.0, 1104.9, 571.6, 384.9, 571.4, 755.5],
          [822.7, 345.3, 241.6, 582.8, 541.8, 420.6, 333.8, 434.6]]

    Ck, Cb, Cko, Cbo = [], [], [], []
    Ck2, Cb2, Cko2, Cbo2 = [], [], [], []

    for i in range(4):
        # tmp = C[1][(i + 1) % 4] - C[1][i] / (C[0][(i + 1) % 4] - C[0][i])
        Ck.append((C[1][(i + 1) % 4] - C[1][i]) /
                  (C[0][(i + 1) % 4] - C[0][i]))
        Cb.append(C[1][i] - (Ck[i] * C[0][i]))

        Cko.append((C_outer[1][(i + 1) % 4] - C_outer[1][i]) /
                   (C_outer[0][(i + 1) % 4] - C_outer[0][i]))
        Cbo.append(C_outer[1][i] - (Cko[i] * C_outer[0][i]))
        # #######################
        Ck2.append((C2[1][(i + 1) % 4] - C2[1][i]) /
                   (C2[0][(i + 1) % 4] - C2[0][i]))
        Cb2.append(C2[1][i] - (Ck2[i] * C2[0][i]))

        Cko2.append((C2_outer[1][(i + 1) % 4] - C2_outer[1][i]) /
                    (C2_outer[0][(i + 1) % 4] - C2_outer[0][i]))
        Cbo2.append(C2_outer[1][i] - (Cko2[i] * C2_outer[0][i]))

    X1 = (Cb[2] - Cb[0]) / (Ck[0] - Ck[2])
    Y1 = Ck[0] * X1 + Cb[0]
    X2 = (Cb[3] - Cb[1]) / (Ck[1] - Ck[3])
    Y2 = Ck[3] * X2 + Cb[3]
    X3 = (Cbo[2] - Cbo[0]) / (Cko[0] - Cko[2])
    Y3 = Cko[0] * X3 + Cbo[0]
    X4 = (Cbo[3] - Cbo[1]) / (Cko[1] - Cko[3])
    Y4 = Cko[3] * X4 + Cbo[3]
    X5 = (Cb2[2] - Cb2[0]) / (Ck2[0] - Ck2[2])
    Y5 = Ck2[0] * X5 + Cb2[0]
    X6 = (Cb2[3] - Cb2[1]) / (Ck2[1] - Ck2[3])
    Y6 = Ck2[3] * X6 + Cb2[3]
    X7 = (Cbo2[2] - Cbo2[0]) / (Cko2[0] - Cko2[2])
    Y7 = Cko2[0] * X7 + Cbo2[0]
    X8 = (Cbo2[3] - Cbo2[1]) / (Cko2[1] - Cko2[3])
    Y8 = Cko2[3] * X8 + Cbo2[3]

    vp1 = [[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]]
    vp2 = [[X5, Y5], [X6, Y6], [X7, Y7], [X8, Y8]]

    # Step 1. Pictures

    # plt.plot([X1, X2], [Y1, Y2], 'bx')
    # plt.plot([X3, X4], [Y3, Y4], 'rx')
    #
    # for i in range(4):
    #     plt.plot([X1, C[0][i]], [Y1, C[1][i]], 'b-')
    #     plt.plot([X2, C[0][i]], [Y2, C[1][i]], 'b-')
    #     plt.plot([X3, C_outer[0][i]], [Y3, C_outer[1][i]], 'r-')
    #     plt.plot([X4, C_outer[0][i]], [Y4, C_outer[1][i]], 'r-')

    # plt.plot([vp1[1][0], vp1[2][0]], [vp1[1][1], vp1[2][1]], 'g-')
    # plt.imshow(img1)
    # plt.margins(0.01, 0.01)
    # plt.savefig("07_vp1_zoom.pdf",  bbox_inches="tight", dpi=1000)
    # plt.show()

    # fig = plt.figure()
    # fig.clf()

    # plt.plot([X5, X6], [Y5, Y6], 'bx')
    # plt.plot([X7, X8], [Y7, Y8], 'rx')
    #
    # for i in range(4):
    #     plt.plot([X5, C2[0][i]], [Y5, C2[1][i]], 'b-')
    #     plt.plot([X6, C2[0][i]], [Y6, C2[1][i]], 'b-')
    #     plt.plot([X7, C2_outer[0][i]], [Y7, C2_outer[1][i]], 'r-')
    #     plt.plot([X8, C2_outer[0][i]], [Y8, C2_outer[1][i]], 'r-')
    #
    # plt.plot([vp2[0][0], vp2[3][0]], [vp2[0][1], vp2[3][1]], 'g-')
    # plt.imshow(img2)
    # plt.margins(0.01, 0.01)
    # plt.savefig("07_vp2_zoom.pdf", bbox_inches="tight", dpi=1000)
    # plt.show()

    # fig = plt.figure()
    # fig.clf()

    # Step 2.1 Find K from vanishing points

    # Calculations...
    # [vp1[0][0], vp1[0][1], 1] * [[1, 0, O13],[0, 1, O23],[O13 O23 O33]] [vp1[1][0], vp1[1][1], 1]
    # [vp1[0][0] + vp1[1][0], vp1[0][1] + vp1[1][1], 1] * [O12, O23, O33] = -(vp1[1][0]*vp1[0][0] + vp1[1][1]*vp1[0][1])
    # [vp1[2][0] + vp1[3][0], vp1[2][1] + vp1[3][1], 1] * [O12, O23, O33] = -(vp1[3][0]*vp1[2][0] + vp1[3][1]*vp1[2][1])
    # [vp2[0][0] + vp2[1][0], vp2[0][1] + vp2[1][1], 1] * [O12, O23, O33] = -(vp2[1][0]*vp2[0][0] + vp2[1][1]*vp2[0][1])
    # Code

    A = np.array([[vp2[2][0] + vp2[3][0], vp2[2][1] + vp2[3][1], 1],
                  [vp1[2][0] + vp1[3][0], vp1[2][1] + vp1[3][1], 1],
                  [vp1[0][0] + vp1[1][0], vp1[0][1] + vp1[1][1], 1]])

    b = np.array([-(vp2[3][0] * vp2[2][0] + vp2[3][1] * vp2[2][1]),
                  -(vp1[3][0] * vp1[2][0] + vp1[3][1] * vp1[2][1]),
                  -(vp1[1][0] * vp1[0][0] + vp1[1][1] * vp1[0][1])])

    o = np.linalg.solve(A, b)
    k13 = -o[0]
    k23 = -o[1]
    k11 = pow((o[2] + k13 ** 2 + k23 ** 2), .5)
    K = np.array([[k11, 0, k13],
                  [0, k11, k23],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    # Step 2.2 Angle between sqr and rect
    ANGLE = 0

    ANGLE += ((np.array([X1, Y1, 1]) @ K_inv.T @ K_inv @ np.array(
            [X3, Y3, 1])) /
              (np.linalg.norm(K_inv @ np.array([X1, Y1, 1])) *
               np.linalg.norm(K_inv @ np.array([X3, Y3, 1]))))
    ANGLE += ((np.array([X2, Y2, 1]) @ K_inv.T @ K_inv @ np.array(
            [X4, Y4, 1])) /
              (np.linalg.norm(K_inv @ np.array([X2, Y2, 1])) *
               np.linalg.norm(K_inv @ np.array([X4, Y4, 1]))))
    ANGLE += ((np.array([X5, Y5, 1]) @ K_inv.T
               @ K_inv @ np.array([X7, Y7, 1])) /
              (np.linalg.norm(K_inv @ np.array([X5, Y5, 1])) *
               np.linalg.norm(K_inv @ np.array([X7, Y7, 1]))))
    ANGLE += ((np.array([X6, Y6, 1]) @ K_inv.T
               @ K_inv @ np.array([X8, Y8, 1])) /
              (np.linalg.norm(K_inv @ np.array([X6, Y6, 1])) *
               np.linalg.norm(K_inv @ np.array([X8, Y8, 1]))))
    ANGLE /= 4
    ANGLE = np.arccos(ANGLE)

    # Step 3.1 find R1, R2, C1, C2 using p3p

    points_u = np.array([[C[0][0], C[1][0]],
                         [C[0][1], C[1][1]],
                         [C[0][3], C[1][3]]])

    points_x = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [1, 0, 0]])

    c12, c23, c31 = cos_s(points_u[0], points_u[1], points_u[2], K)

    d12 = np.linalg.norm(points_x[1] - points_x[0])
    d23 = np.linalg.norm(points_x[2] - points_x[1])
    d31 = np.linalg.norm(points_x[0] - points_x[2])

    res = p3p_distances(d12, d23, d31, c12, c23, c31)

    R1, Camera1 = p3p_RC([res[0][1], res[1][1], res[2][1]], points_x, points_u,
                         K)

    points_u2 = np.array([[C2[0][1], C2[1][1]],
                          [C2[0][2], C2[1][2]],
                          [C2[0][3], C2[1][3]]])

    points_x2 = np.array([[0, 1, 0],
                          [1, 1, 0],
                          [1, 0, 0]])

    c12, c23, c31 = cos_s(points_u2[0], points_u2[1], points_u2[2], K)

    d12 = np.linalg.norm(points_x2[1] - points_x2[0])
    d23 = np.linalg.norm(points_x2[2] - points_x2[1])
    d31 = np.linalg.norm(points_x2[0] - points_x2[2])

    res = p3p_distances(d12, d23, d31, c12, c23, c31)

    R2, Camera2 = p3p_RC([res[0][1], res[1][1], res[2][1]], points_x2,
                         points_u2, K)

    # Step 3.2 create a virtual object. Place a cube on the image...
    P1 = np.c_[K @ R1, (-K) @ R1 @ Camera1]
    P2 = np.c_[K @ R2, (-K) @ R2 @ Camera2]
    cube = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                     [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])

    # top_arr = []
    # down_arr = []
    # for i in range(4):
    #     down = np.array([cube[i][0], cube[i][1], cube[i][2], 1])
    #     top = np.array([cube[i + 4][0], cube[i + 4][1], cube[i + 4][2], 1])
    #     down = P2 @ down
    #     top = P2 @ top
    #     down = (down / down[-1])[:-1]
    #     top = (top / top[-1])[:-1]
    #     down_arr.append(down)
    #     top_arr.append(top)
    # break

    # plot_cube(top_arr, down_arr)
    # plt.imshow(img2)
    # plt.savefig("07_box_wire2.pdf")
    # plt.show()

    # Step 3.3. Interpolation
    counter = 0
    for lmbd in [i / 20 for i in range(0, 21)]:
        fig = plt.figure()
        fig.clf()
        # for lmbd in [0.5]:
        c = Camera2 * lmbd + Camera1 * (1 - lmbd)
        r = scipy.linalg.fractional_matrix_power(R2 @ R1.T, lmbd).real @ R1
        p = np.c_[K @ r, (-K) @ r @ c]

        H = P1[:, [0, 1, 3]] @ np.linalg.inv(p[:, [0, 1, 3]])
        blank = np.full(img1.shape, 0, dtype="uint8")
        # print(blank)
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                point = H @ np.array([x, y, 1])
                point /= point[-1]
                point = np.array([round(point[0]), round(point[1])])
                if ((0 < point[1] < img1.shape[0])
                        and (0 < point[0] < img1.shape[1])):
                    blank[y][x] = img1[point[1]][point[0]]

        top_arr = []
        down_arr = []
        for i in range(4):
            down = np.array([cube[i][0], cube[i][1], cube[i][2], 1])
            top = np.array([cube[i + 4][0], cube[i + 4][1], cube[i + 4][2], 1])
            down = p @ down
            top = p @ top
            down = (down / down[-1])[:-1]
            top = (top / top[-1])[:-1]
            down_arr.append(down)
            top_arr.append(top)
        plot_cube(top_arr, down_arr)
        plt.axis('off')
        plt.imshow(blank)
        plt.savefig("07_gif_{}.png".format(counter),
                    bbox_inches='tight',
                    pad_inches=0.0,
                    dpi=400)
        counter += 1
    # Step 4.1. Save data in Mat

    # Camera1 = np.array(Camera1).reshape(3, 1)
    # Camera2 = np.array(Camera2).reshape(3, 1)
    # vp1 = np.array(vp1).T
    # vp2 = np.array(vp2).T
    # sio.savemat('07_data.mat', {
    #     'u1': u,
    #     'u2': u2,
    #     'R1': R1,
    #     'R2': R2,
    #     'C1': Camera1,
    #     'C2': Camera2,
    #     'vp1': vp1,
    #     'vp2': vp2,
    #     'K': K,
    #     'angle': ANGLE})
