Input Data
The input data can be downloaded from submission system: the image daliborka_01.jpg and the set of seven 2D points u2.

Task overview
Load daliborka_01.jpg bitmap image into a 3D matrix/array (rows x cols x 3).
Display the array as image.
Manually acquire coordinates of the set of seven points in the image corresponding to the tops of the five chimneys and the two towers. The points must be ordered from left to right. Store the point coordinates in a 2×7 matrix, i.e. the point coordinates are column vectors.
In the bitmap image, colorize the pixels that are nearest to the acquired points (use rounding, not flooring/ceiling). Use the following colors in the following order: red = [255, 0, 0], green = [0, 255, 0], blue = [0, 0, 255], magenta = [255, 0, 255], cyan = [0, 255, 255], yellow = [255, 255, 0], white = [255, 255, 255] to colorize the respective seven points. The colors are defined by their R, G, B values: color = [R, G, B]. The order of colors must correspond to the order of points u. Store the modified bitmap image as a file 01_daliborka_points.png.
Create function A = estimate_A( u2, u ) for estimation of the affine transformation A (2×3 matrix) from n given points u2 to image points u. The inputs u2, and u are 2xn matrices, e.g., point coordinates are columns.
Perform all possible selections of 3 point correspondences from all n correspondences.
For each triple compute the affine transformation Ai (exactly).
Compute transfer errors of the particular transformation for every correspondence, i.e., the euclidean distances between points u and ux; the points ux are the points u2 transformed by A. Find the maximum error over the correspondences.
From the all computed transformations Ai select the one A that has the maximum transfer error minimal.
Assume general number of points, though we have only 7 points and 35 different triplets.
Display the image, the set of points u using the same colors as above, and 100× magnified transfer errors for the best A as red lines.
Export the graph as a pdf file 01_daliborka_errs.pdf.
Store the points u and the matrix A in the file 01_points.mat (Matlab format).

