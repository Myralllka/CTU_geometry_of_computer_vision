import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg


def u2H(u, u0):
    """
    :param u: the image coordinates of points in the first image
    (2×4 matrix/np.array)
    :param u0: (2×4) be the image coordinates of the corresponding points in the second image.
    :return: H: a 3×3 homography matrix (np.array). Matrix H is regular.
    Return an empty array [] if there is no solution.
    """
    if (u.shape[0]) == 2:
        u = np.array(u).T
        u0 = np.array(u0).T

    M = list()
    for i in range(len(u)):
        m = list(u[i])
        m.extend([1, 0, 0, 0])
        m.extend(list(- u0[i][0] * u[i]))
        m.append(- u0[i][0])
        M.append(m)

    for i in range(len(u)):
        m = [0, 0, 0]
        m.extend(list(u[i]))
        m.append(1)
        m.extend(list(- u0[i][1] * u[i]))
        m.append(- u0[i][1])
        M.append(m)

    H = slinalg.null_space(M)

    if H.size == 0:
        return []
    return (H / H[-1]).reshape(3, 3)


def u2h_optim(u, u0):
    """
    Find the homography (3×3 matrix H) that maps your image to the reference
    image. Find it as the best homography by optimizing over all 210
    quadruplets among the ten matches. Minimize the maximal transfer error
    (in the domain of points u0) on all image matches. Create function u2h_optim
    (with arbitrary inputs and outputs) solving this step. The function
    will be used in the one of the future homeworks.
    :return : array of
    - minimized maximal error
    - corresponding H
    - corresponding indexes of points
    """
    u = np.array(u).T
    u0 = np.array(u0).T
    all_errors = list()

    for inx in itertools.combinations(range(0, len(u)), 4):
        current_u0 = u0[inx, :]
        current_u = u[inx, :]
        H = u2H(current_u, current_u0)

        errors = list()

        for i in range(len(u)):
            point = np.append(u[i], [1])
            projected = H @ point

            projected /= projected[-1]
            errors.append(np.linalg.norm(projected[:-1] - u0[i]))
        all_errors.append([max(errors), H, list(inx)])

    res = min(all_errors, key=lambda x: x[0])
    return res


def color_normalization(img0, img1, H):
    pass


if __name__ == "__main__":
    # Step 1

    # u = [[0, 0, 1, 1],
    #      [0, 1, 1, 0]]
    #
    # u0 = [[1, 2, 1.5, 1],
    #       [1, 2, 0.5, 0]]
    #
    # H = np.array([[1, 1, 1], [-1, 1, 1], [1, 0, 1]])
    #
    # res = u2H(u, u0)
    # print(res)
    # print(np.array([round(i, 1) for i in res.reshape(9,)]).reshape(3, 3))

    # Step 2. general

    U0 = [[142.4, 93.4, 139.1, 646.9, 1651.4, 1755.2, 1747.3, 1739.5, 1329.2,
           972.0],
          [1589.3, 866.7, 259.3, 305.6, 87.3, 624.8, 1093.5, 1593.8, 1610.2,
           1579.3]]
    U = [
        [783.8, 462.6, 243.7, 363.9, 465.2, 638.3, 784.7, 954.6, 927.8, 881.4],
        [747.5, 671.2, 586.8, 463.9, 248.5, 260.6, 291.5, 326.9, 412.8, 495.0]]
    C = [[474.4, 508.2, 739.3, 737.2],
         [501.7, 347.0, 348.7, 506.7]]

    _, H, inx = u2h_optim(U, U0)

    sio.savemat('05_homography.mat', {
        'u': U,
        'u0': U0,
        'point_sel': inx,
        'H': H
        })
    U = np.array(U)
    U0 = np.array(U0)
    img_00 = plt.imread('pokemon_00.jpg')
    img_10 = plt.imread('pokemon_10.jpg')
    img_00 = img_00.copy()
    img_10 = img_10.copy()

    fig = plt.figure()

    # Step 2

    # for i in range(len(img_10)):
    #     for j in range(len(img_10[0])):
    #         if sum(img_10[i][j]) / 3 < 35:
    #             tmp = H @ np.array([j, i, 1])
    #             tmp /= tmp[-1]
    #             tmp = tmp[:-1]
    #
    #             if ((round(tmp[1]) < img_00.shape[0])
    #                     and (round(tmp[0]) < img_00.shape[1])
    #                     and (round(tmp[0]) > 0)
    #                     and (round(tmp[1]) > 0)):
    #                 t = img_00[round(tmp[1])][round(tmp[0])]
    #                 img_10[i][j] = t
    #
    # black_points = np.array(black_points).T
    # plt.imshow(img_10)
    # plt.show()
    # fig.savefig("05_corrected.png")

    # Step 3

    fig.set_figheight(100)
    fig.set_figwidth(100)

    fig.suptitle('10 point correspondences')

    plt.subplot(121)
    plt.title('Label points in my image')
    plt.imshow(img_10)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    innx = [i for i in range(len(U[0])) if i not in inx]
    plt.plot(U[0][inx], U[1][inx],
             "r.",
             markersize=14,
             mfc="y",
             label="used for H")
    plt.plot(U[0][innx], U[1][innx],
             "rx",
             markersize=10,
             mew=2,
             label="point")
    for i in range(len(U[0])):
        if i in inx:
            plt.annotate(str(i),
                         (U[0][i], U[1][i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         backgroundcolor='red',
                         c='y', weight='bold')
        else:
            plt.annotate(str(i),
                         (U[0][i], U[1][i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         backgroundcolor='pink',
                         c='black', weight='bold')

    plt.subplot(122)
    plt.title('Label points in reference image')
    plt.imshow(img_00)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    innx = [i for i in range(len(U[0])) if i not in inx]
    plt.plot(U0[0][inx], U0[1][inx],
             "r.",
             markersize=14,
             mfc="y",
             label="used for H")

    plt.plot(U0[0][innx], U0[1][innx],
             "rx",
             markersize=10,
             mew=2,
             label="point")

    for i in range(len(U0[0])):
        if i in inx:
            plt.annotate(str(i),
                         (U0[0][i], U0[1][i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         backgroundcolor='red',
                         c='y', weight='bold')
        else:
            plt.annotate(str(i),
                         (U0[0][i], U0[1][i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         backgroundcolor='pink',
                         c='black', weight='bold')

    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
        break

    fig.legend(lines, labels, loc='lower center')

    plt.show()
    fig.savefig("05_homography.pdf")
