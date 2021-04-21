import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg
import tools
import SeqWriter

if __name__ == "__main__":
    img1 = plt.imread("pokemon_10.jpg")
    img2 = plt.imread("pokemon_19.jpg")

    C = [[473.4, 507.2, 738.3, 736.2],
         [500.7, 346.0, 347.7, 505.7]]

    C2 = [[571.6, 384.9, 571.4, 755.5],
          [541.8, 420.6, 333.8, 434.6]]

    C_outer = [[119.1, 463.7, 988.8, 798.0],
               [597.9, 218.7, 321.2, 813.5]]

    C2_outer = [[397.4, 153.8, 701.0, 1104.9],
                [822.7, 345.3, 241.6, 582.8]]

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

    plt.plot([X1, X2], [Y1, Y2], 'bx')
    plt.plot([X3, X4], [Y3, Y4], 'rx')

    for i in range(4):
        plt.plot([X1, C[0][i]], [Y1, C[1][i]], 'b-')
        plt.plot([X2, C[0][i]], [Y2, C[1][i]], 'b-')
        plt.plot([X3, C_outer[0][i]], [Y3, C_outer[1][i]], 'r-')
        plt.plot([X4, C_outer[0][i]], [Y4, C_outer[1][i]], 'r-')

    plt.plot([vp1[1][0], vp1[2][0]], [vp1[1][1], vp1[2][1]], 'g-')
    plt.imshow(img1)
    # plt.margins(0.01, 0.01)
    # plt.savefig("07_vp1_zoom.pdf",  bbox_inches="tight", dpi=1000)
    # plt.show()

    fig = plt.figure()
    fig.clf()

    plt.plot([X5, X6], [Y5, Y6], 'bx')
    plt.plot([X7, X8], [Y7, Y8], 'rx')

    for i in range(4):
        plt.plot([X5, C2[0][i]], [Y5, C2[1][i]], 'b-')
        plt.plot([X6, C2[0][i]], [Y6, C2[1][i]], 'b-')
        plt.plot([X7, C2_outer[0][i]], [Y7, C2_outer[1][i]], 'r-')
        plt.plot([X8, C2_outer[0][i]], [Y8, C2_outer[1][i]], 'r-')

    plt.plot([vp2[0][0], vp2[3][0]], [vp2[0][1], vp2[3][1]], 'g-')
    plt.imshow(img2)
    # plt.margins(0.01, 0.01)
    # plt.savefig("07_vp2_zoom.pdf", bbox_inches="tight", dpi=1000)
    # plt.show()

    fig = plt.figure()
    fig.clf()

    
