import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations


def estimate_A(u2: np.ndarray, u: np.ndarray):
    colors = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 0],
              [1, 1, 1]]
    # iterate all combinations
    errors = list()
    for inx in itertools.combinations(range(0, u.shape[1]), 3):
        # preprocessing
        u_i = u[:, inx]
        u_2i = u2[:, inx]
        u_i = np.append(u_i, np.array([[1, 1, 1]]), 0)
        u_2i = np.append(u_2i, np.array([[1, 1, 1]]), 0)
        u_2i_inv = np.linalg.inv(u_2i)
        ux = list()
        A = u_i @ u_2i_inv
        ###
        for i in range(len(u2[0])):
            ux.append(A @ [[u2[0][i]], [u2[1][i]], [1]])
        ux = np.array(ux)
        ux = np.array([[i[0], i[1]] for i in ux])
        ux = ux[:, :, 0]
        e = 100 * (ux - u.T)
        errors.append((sum([np.linalg.norm(i) for i in e]), e, A))

    errors.sort(key=lambda x: x[0])
    e = errors[0][1]

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.imshow(img)


    # draw all points (in proper color) and errors
    for i in range(len(u[0])):

        plt.plot(u.T[i, 0], u.T[i, 1], 'o', color=colors[i],
                 fillstyle='none')  # the 4-th point in magenta color
        plt.plot((u.T[i, 0], u.T[i, 0] + e[i, 0]), (u.T[i, 1], u.T[i,
                                                                   1] + e[i,
                                                                          1]),
                 'r-')  # the 4-th displacement

    plt.show()

    fig.savefig('01_daliborka_errs.pdf')
    return errors[0][2][:2]


if __name__ == "__main__":
    img = plt.imread('img/daliborka_01.jpg')
    img = img.copy()

    u = np.array([[147., 274., 298., 550., 641., 750., 958.],
                  [185., 211., 225., 316., 151., 283., 202.]])

    colors = [[255, 0, 0],
              [0, 255, 0],
              [0, 0, 255],
              [255, 0, 255],
              [0, 255, 255],
              [255, 255, 0],
              [255, 255, 255]]

    for i in range(7):
        img[int(u[1, i]), int(u[0, i])] = colors[i]

    # plt.imshow(img)
    # plt.show()

    # plt.imsave("01_daliborka_points.png", img)

    u2 = np.array([[-182.6, -170.5, -178.8, -202.6, 51.5, -78.0, 106.1],
                   [265.8, 447.0, 486.7, 851.9, 907.1, 1098.7, 1343.6]])

    A = estimate_A(u2, u)
    sio.savemat('01_points.mat', {'u': u, 'A': A})

    # print(A)
