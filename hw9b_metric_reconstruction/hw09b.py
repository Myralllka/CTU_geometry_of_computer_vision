import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import scipy.io as sio  # for matlab file format output
import itertools  # for generating all combinations
import scipy.linalg as slinalg


def e2rc(E_e2rc):
    # E = R cros C
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    G_norm = E_e2rc / np.linalg.norm(E_e2rc, ord="fro")
    U, _, V = np.linalg.svd(E_e2rc)
    # W = U @ Rz
    R1 = U @ Rz @ V
    R2 = U @ Rx.T @ V
    # R_c = np.sign(np.linalg.det(W)) * W @ V * np.sign(np.linalg.det(V))
    C1 = V[2]
    C2 = -C1
    return R1, R2, C1, C2


def optimize_rc(R1_current, C1_current, R2_current, C2_current, K_current,
                u_current_1, u_current_2):
    result_index_R_C_Ps = []
    P1_c = np.append(K, np.array([0, 0, 0]).reshape(3, 1), axis=1)

    for R_loop, C_loop in [[R1_current, C1_current],
                           [R1_current, C2_current],
                           [R2_current, C1_current],
                           [R2_current, C2_current]]:
        counter = 0
        P2_c = np.append(K_current @ R_loop,
                         (K_current @ R_loop @ C_loop).reshape(
                                 3, 1), axis=1)
        for i in range(len(u_current_1[0])):
            u_tmp = np.array([u_current_1[0][i],
                              u_current_1[1][i], 1]).reshape(3, 1)
            v_tmp = np.array([u_current_2[0][i],
                              u_current_2[1][i], 1]).reshape(3, 1)
            M = np.vstack([np.hstack([u_tmp, np.zeros((3, 1)), -P1_c]),
                           np.hstack([np.zeros((3, 1)), v_tmp, -P2_c])])
            _, _, Vh = np.linalg.svd(M)
            X = Vh[-1, 2:5] / Vh[-1, -1]
            if np.dot(X, u_tmp) / (
                    np.linalg.norm(X) * np.linalg.norm(u_tmp)) > 0:
                counter += 1
        print(counter)
        result_index_R_C_Ps.append([counter, R_loop, C_loop, P1_c, P2_c])
    return max(result_index_R_C_Ps, key=lambda x: x[0])


if __name__ == "__main__":
    # Data preprocessing
    img1 = plt.imread('daliborka_01.jpg')
    img2 = plt.imread('daliborka_23.jpg')

    f = sio.loadmat("daliborka_01_23-uu.mat")

    K = sio.loadmat('K.mat')["K"]
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

    # u01_gamma = [np.linalg.inv(K) @ i for i in u01]
    # u23_gamma = [np.linalg.inv(K) @ i for i in u23]

    # u1 = u01_gamma[:, ix]
    # u2 = u23_gamma[:, ix]

    colors = ["dimgray", "rosybrown", "maroon", "peru",
              "moccasin", "yellow", "olivedrab", "lightgreen",
              "navy", "royalblue", "indigo", "hotpink"]

    # Step 1
    # Decompose the best E into relative rotation R and translation C
    # (four solutions). Choose such a solution that reconstructs (most of)
    # the points in front of both (computed) cameras.

    E = sio.loadmat("09a_data.mat")["E"]

    R1, C1, R2, C2 = e2rc(E)

    # Step 2
    # Construct projective matrices P1, P2 (including K).
    _, R, C, P1, P2 = optimize_rc(R1, R2, C1, C2, K, u01, u23)
    print(E)
    print(np.cross(R, C))
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
