
Construct simple projection matrix P where C = [1,2,-3]T, f=1, K = R = I. Project the 3D points X1 = [0, 0, 0]T, X2 = [1, 0, 0]T, X3 = [0, 1, 0]T by the P. Compute the distances η and camera pose using your p3p_RC for all solutions. Compare with correct known values of R, C.
Find optimal camera pose using point correspondences by a similar way as in HW02:
Compute camera poses for all 120 triplets of point correspondences chosen from your 10 points (as in HW-02). Use the matrix K from the A.E. Input data. (There can be more than one solutions for each triplet.)
For a particular camera pose R, C, compose camera matrix and compute the reprojection errors on all 109 points and find their maximum.
Select the best camera pose minimising the maximum reprojection error.
Export the optimal R, C, and point_sel (indices [i1, i2, i3] of the three points used for computing the optimal R, C) as 04_p3p.mat.
Display the image (daliborka_01) and draw u as blue dots, highlight the three points used for computing the best R, C by drawing them as yellow dots, and draw the displacements of reprojected points x multiplied 100 times as red lines. Export as 04_RC_projections_errors.pdf.
Plot the decadic logarithm (log10()) of the maximum reprojection error of all the computed poses as the function of their trial index and export as 04_RC_maxerr.pdf. Plot the errors as points, not lines, in this case.
Plot the reprojection error of the best R, C on all 109 points as the function of point index and export as 04_RC_pointerr.pdf.
Draw the coordinate systems δ (black), ε (magenta) of the optimal R, C, draw the 3D scene points (blue), and draw centers (red) of all cameras you have tested. Export as 04_scene.pdf.
Compare your graphs with the graphs in HW02.