Input Data
The input data can be downloaded from submission system: the image daliborka_01.jpg, coordinates of 3D points x and their projections u (in the file daliborka_01-ux.mat), and point index vector ix.

Task
Load the points u, x and the image into your matlab workspace.
Examine the image points u by displaying them over the image.
Examine the 3D points x by displaying them in a new figure. The 3D plot can be e.g. rotated.
Implement the estimation of camera projection matrix Q from given image points u and 3D points x given your selection index ix as a function
Q, points_sel, err_max, err_points, Q_all = estimate_Q( u, x, ix )
where Q is the best projection matrix, points_sel are indices of the 6 points (w.r.t to all 109 points). The other output arguments are optional (not tested by A.E.), for your convenience: err_max should be vector of all maximal errors for all tested matrices, err_points should be vector of point errors for the best camera and Q_all should be cell matrix containing all tested camera matrices (will be used in HW-03). Note: use native indexing, i.e. ix and point_sel must be 0-based in python implementation and 1-based in matlab implementation.
Perform all possible selections of 5 1/2 points from your 10 points (using ix)
For each selection compute the projection matrix Q projecting the selected 5 1/2 points exactly.
Compute the reprojection errors – Euclidean distances between measured image points u and the projections of 3D points x using the particular matrix Q (for all 109 points). Find the maximum error over all the correspondences.
From all computed projection matrices select the one that has the maximum reprojection error minimal.
Plot the decadic logarithm (log10()) of the maximum reprojection error of all the computed projection matrices as the function of their selection index and export the plot as a pdf file 02_Q_maxerr.pdf.
Display the image and plot u as blue dots (plot specifier 'b.'), highlight the points used for computing the best Q by plotting them as yellow dots ('y.'), and plot the projections of x using the best Q as red circles ('ro'). Export the plot as 02_Q_projections.pdf.
Display the image and plot u as blue dots, highlight the points used for computing the best Q by plotting them as yellow dots, and plot the displacements of projected points x multiplied 100 times as red lines. Export the plot as 02_Q_projections_errors.pdf.
Plot the reprojection error of the best Q on all 109 points as the function of point index and export as 02_Q_pointerr.pdf.
(Note: do not forget to create figure titles and describe axes where appropriate.)
