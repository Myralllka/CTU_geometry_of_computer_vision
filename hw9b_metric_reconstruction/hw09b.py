import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg


def e2rc(E_e2rc):
    # G = tE
    # G_F = |t| sqrt(2) ||C_e1||
    frobenius_norm_e = np.sqrt(
            sum(list(
                map(lambda x: sum(list(map(lambda y: y ** 2, x))), E_e2rc))))
    U, S, V = np.linalg.svd(E_e2rc)
    print (U, S, V)


if __name__ == "__main__":
    # Data preprocessing
    img1 = plt.imread('daliborka_01.jpg')
    img2 = plt.imread('daliborka_23.jpg')

    f = sio.loadmat("daliborka_01_23-uu.mat")
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

    u1 = u01[:, ix]
    u2 = u23[:, ix]

    colors = ["dimgray", "rosybrown", "maroon", "peru",
              "moccasin", "yellow", "olivedrab", "lightgreen",
              "navy", "royalblue", "indigo", "hotpink"]

    # Step 1
    # Decompose the best E into relative rotation R and translation C
    # (four solutions). Choose such a solution that reconstructs (most of)
    # the points in front of both (computed) cameras.

    E = sio.loadmat("09a_data.mat")["E"]

    # R, C = e2rc(E)
    e2rc(E)

    # Step 2
    # Construct projective matrices P1, P2 (including K).

    # Step 3
    # Compute scene points X.

    # Step 4
    # Display the images, draw the input points as blue dots and the scene points X projected by appropriate P_i as red circles. Draw also the edges, connecting the original points as yellow lines. Export as 09_reprojection.pdf.

    # Step 5
    # Draw graph of reprojection errors and export as 09_errorsr.pdf.

    # Step 6
    # Draw the 3D point set (using 3D plotting facility) connected by the edges as a wire-frame model, shown from the top of the tower, from the side, and from some general view. Export as 09_view1.pdf, 09_view2.pdf, and 09_view3.pdf.

    # Step 7
    # Save Fe, E, R, C, P1, P2, X, and u1, u2, point_sel_e as 09b_data.mat.

    # sio.savemat('09a_data.mat', {
    #     'u1': u01,
    #     'u2': u23,
    #     "Fe": sio.loadmat("09a_data.mat")["Fe"],
    #     "E": sio.loadmat("09a_data.mat")["E"],
    #     "R": R,
    #     "C": C,
    #     "P1": P1,
    #     "P2": P2,
    #     "X": X,
    #     "point_sel_e": sio.loadmat("09a_data.mat")["point_sel_e"],
    #     })
