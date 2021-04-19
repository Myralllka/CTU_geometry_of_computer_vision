import numpy as np  # for matrix computation and linear algebra
import math
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
from hw05 import u2h_optim
import matplotlib.colors as clrs

if __name__ == "__main__":
    f = sio.loadmat("bridge_corresp.mat")
    u = f["u"]

    # Step 1
    # fig = plt.figure()
    # plt.title('Transfer errors')
    # plt.xlabel('point index (sorted)')
    # plt.ylabel('err [px]')
    # for i in range(7):
    #     for j in range(7):
    #         if i == j or i > j or abs(i - j) > 1:
    #             continue
    #         _, H, inx = u2h_optim(u[i, j], u[j, i])
    #         errors = []
    #         for index in range(10):
    #             tmp = u[i, j].T[index]
    #             tmp = H @ np.array([tmp[0], tmp[1], 1])
    #             tmp = (tmp / tmp[-1])[:-1]
    #             errors.append(np.linalg.norm(u[j, i].T[index] - tmp))
    #
    #         plt.plot(range(10), sorted(errors),
    #                  label="{}-{}".format(i + 1, j + 1))
    #
    # plt.legend(loc='best')
    # plt.show()
    # fig.savefig("06_errors.pdf")

    #  Step 2
    #
    # fig = plt.figure()
    # fig.clf()
    # plt.title('Image borders in the coordinate system of image 4')
    # plt.xlabel('y_4 [px]')
    # plt.ylabel('x_4 [px]')
    #
    # def convert(u, H):
    #     return np.array(list(map(lambda x: (x / x[-1])[:-1], map(lambda x: H @ np.array([x[0], x[1], 1]), u.T)))).T
    #
    #
    # _, H34, _ = u2h_optim(u[2, 3], u[3, 2])
    # _, H24, _ = u2h_optim(u[1, 2], convert(u[2, 1], H34))
    # _, H14, _ = u2h_optim(u[0, 1], convert(u[1, 0], H24))
    # _, H54, _ = u2h_optim(u[4, 3], u[3, 4])
    # _, H64, _ = u2h_optim(u[5, 4], convert(u[4, 5], H54))
    # _, H74, _ = u2h_optim(u[6, 5], convert(u[5, 6], H64))
    #
    # _, H43, _ = u2h_optim(u[3, 2], u[2, 3])
    # _, H42, _ = u2h_optim(u[2, 1], convert(u[1, 2], H43))
    # _, H41, _ = u2h_optim(u[1, 0], convert(u[0, 1], H42))
    # _, H45, _ = u2h_optim(u[3, 4], u[3, 4])
    # _, H46, _ = u2h_optim(u[4, 5], convert(u[5, 4], H45))
    # _, H47, _ = u2h_optim(u[5, 6], convert(u[6, 5], H46))
    #
    # images = [plt.imread("bridge_0{}.jpg".format(i + 1)).copy() for i in range(7)]
    #
    # Hs = [H24, H34, np.eye(3), H54, H64]
    #
    # corners = [[0, 0],
    #            [len(images[0][0]), 0],
    #            [len(images[0][0]), len(images[0])],
    #            [0, len(images[0])]]
    # colors = ["g", "b", "y", "c", "m"]
    # for i in range(len(Hs)):
    #     tf = []
    #     for each in corners:
    #         point = Hs[i] @ np.array([each[0], each[1], 1])
    #         point = (point / point[-1])
    #         tf.append(point)
    #     plt.plot([tf[0][0], tf[1][0]], [tf[0][1], tf[1][1]], "{}-".format(colors[i]), label="{}".format(i+2))
    #     plt.plot([tf[0][0], tf[3][0]], [tf[0][1], tf[3][1]], "{}-".format(colors[i]))
    #     plt.plot([tf[3][0], tf[2][0]], [tf[3][1], tf[2][1]], "{}-".format(colors[i]))
    #     plt.plot([tf[1][0], tf[2][0]], [tf[1][1], tf[2][1]], "{}-".format(colors[i]))
    #
    # plt.legend(loc='best')
    # plt.savefig("06_borders.pdf")
    # plt.show()

    #  Step 3. Panorama

    # fig = plt.figure()
    # fig.clf()
    #
    # def convert(u, H):
    #     return np.array(list(map(lambda x: (x / x[-1])[:-1],
    #                              map(lambda x: H @ np.array([x[0], x[1], 1]),
    #                                  u.T)))).T
    #
    #
    # _, H34, _ = u2h_optim(u[2, 3], u[3, 2])
    # _, H54, _ = u2h_optim(u[4, 3], u[3, 4])
    #
    # images = [plt.imread("bridge_0{}.jpg".format(i + 1)).copy() for i in [2, 3, 4]]
    #
    # lu = H54 @ np.array([0, 0, 1])
    # lu /= lu[-1]
    # ld = H54 @ np.array([0, 900, 1])
    # ld /= ld[-1]
    # ru = H34 @ np.array([1200, 0, 1])
    # ru /= ru[-1]
    # rd = H34 @ np.array([1200, 900, 1])
    # rd /= rd[-1]
    # shift_right = abs(round(lu[0])) + 2
    # shift_down = abs(round(lu[1])) + 2
    # length = int(np.ceil(abs(min(lu[0], ld[0])) + abs(max(ru[0], rd[0]))) + 10)
    # height = int(np.ceil(abs(min(ld[1], rd[1])) + abs(max(ru[1], lu[1]))) + shift_down)
    #
    # result = np.ndarray((height, length, 3), dtype="uint8")
    #
    # for i in range(900):
    #     for j in range(1200):
    #         result[i+shift_down, j+shift_right] = images[1][i][j]
    #         point = H54 @ np.array([j, i, 1])
    #         point = np.int_(np.rint(point / point[-1])[:-1])
    #         result[point[1] + shift_down, point[0] + shift_right] = images[2][i][j]
    #         point = H34 @ np.array([j, i, 1])
    #         point = np.int_(np.rint(point / point[-1])[:-1])
    #         result[point[1] + shift_down, point[0] + shift_right] = images[0][i][j]
    #
    # # plt.imshow(result)
    # plt.imsave("06_panorama.png", result)
    # plt.show()

    #     Step 4. K
    #     Width:                    2400 inch
    #     Height:                   1800 inch
    #     FocalPlaneXResolution:    2160000/225 inch
    #     FocalPlaneYResolution:    1611200/168 inch
    #     FocalLength:              7400/1000 mm

    Fx = 7.4 * (2160000 / 225) / 2 / 25.4
    Cx = 2400 / 4
    Fy = 7.4 * (1611200 / 168) / 2 / 25.4
    Cy = 1800 / 4

    K = [[Fx, 0, Cx],
         [0, Fy, Cy],
         [0, 0, 1]]
    # sio.savemat('06_data.mat', {'K': K})
    # print(K)
    # Step 5. panorama cylinder

    fig = plt.figure()
    fig.clf()
    plt.title('Image borders in the coordinate system of image 4')
    plt.xlabel('y_4 [px]')
    plt.ylabel('x_4 [px]')


    def convert(u, H): return np.array(list(map(lambda x: (x / x[-1])[:-1], map(lambda x: H @ np.array([x[0], x[1], 1]), u.T)))).T

    _, H34, _ = u2h_optim(u[2, 3], u[3, 2])
    _, H24, _ = u2h_optim(u[1, 2], convert(u[2, 1], H34))
    _, H14, _ = u2h_optim(u[0, 1], convert(u[1, 0], H24))
    _, H54, _ = u2h_optim(u[4, 3], u[3, 4])
    _, H64, _ = u2h_optim(u[5, 4], convert(u[4, 5], H54))
    _, H74, _ = u2h_optim(u[6, 5], convert(u[5, 6], H64))

    images = [plt.imread("bridge_0{}.jpg".format(i + 1)).copy() for i in
              range(7)]

    # Hs = [H14, H24, H34, np.eye(3), H54, H64, H74]
    # Hs = [H34, np.eye(3), H54]

    colors = ["r", "g", "b", "y", "c", "m", "r"]
    Hs = [np.eye(3)]
    for hs_i in range(len(Hs)):
        rows = images[hs_i].shape[0]
        cols = images[hs_i].shape[1]
        center_x = cols / 2 + (hs_i - (len(Hs)//2)) * cols
        center_y = rows / 2 + (hs_i - (len(Hs)//2)) * rows
        print(center_y, center_x)
        for i in range(len(images[0][0])):
            point = Hs[hs_i] @ np.array([i, 0, 1])
            point = (point / point[-1])
            point_x = K[0][0] * math.tan((point[0] - center_x) / K[0][0]) + center_x
            point_y = (point[1] - center_y) / math.cos(math.atan((point[0] - center_x) / K[0][0])) + center_y
            # print(point_y, point_x)
            plt.plot([point_x], [point_y], "{}.".format(colors[hs_i]))

            point = Hs[hs_i] @ np.array([i, len(images[0]), 1])
            point = (point / point[-1])
            point_x = K[0][0] * math.tan((point[0] - center_x) / K[0][0]) + center_x
            point_y = (point[1] - center_y) / math.cos(math.atan((point[0] - center_x) / K[0][0])) + center_y
            plt.plot([point_x], [point_y], "{}.".format(colors[hs_i + 1]))

        # for i in range(len(images[0][0])):
        #     point = Hs[hs_i] @ np.array([i, 0, 1])
        #     point = (point / point[-1])
        #     point_x = K[0][0] * math.tan((point[0] - center_x) / K[0][0]) + center_x
        #     point_y = (point[1] - center_y) / math.cos(math.atan((point[0] - center_x) / K[0][0])) + center_y
        #     plt.plot([point_x], [point_y], "{}o".format(colors[hs_i]))
        #
        #     point = Hs[hs_i] @ np.array([len(images[0][0]), i, 1])
        #     point = (point / point[-1])
        #     point_x = K[0][0] * math.tan((point[0] - center_x) / K[0][0]) + center_x
        #     point_y = (point[1] - center_y) / math.cos(math.atan((point[0] - center_x) / K[0][0])) + center_y
        #     plt.plot([point_x], [point_y], "{}o".format(colors[hs_i + 1]))
        # plt.plot([tf[0][0], tf[1][0]], [tf[0][1], tf[1][1]], "{}-".format(colors[i]), label="{}".format(i + 2))
        # plt.plot([tf[0][0], tf[3][0]], [tf[0][1], tf[3][1]], "{}-".format(colors[i]))
        # plt.plot([tf[3][0], tf[2][0]], [tf[3][1], tf[2][1]], "{}-".format(colors[i]))
        # plt.plot([tf[1][0], tf[2][0]], [tf[1][1], tf[2][1]], "{}-".format(colors[i]))

