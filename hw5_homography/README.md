Create function u2H computing homography from four image matches. Let u be the image coordinates of points in the first image (2×4 matrix/np.array) and u0 (2×4) be the image coordinates of the corresponding points in the second image. Then H is a 3×3 homography matrix (np.array), such that



Steps

Download the reference image and your specific image from the upload system (Input Data).
Download coordinates of 10 image matches (point correspondences) between the two images (Input Data). Store the matches as column vectors in 2×10 matrices u and u0 for your and the reference image, respectively.
Find the homography (3×3 matrix H) that maps your image to the reference image. Find it as the best homography by optimizing over all 210 quadruplets among the ten matches. Minimize the maximal transfer error (in the domain of points u0) on all image matches. Create function u2h_optim (with arbitrary inputs and outputs) solving this step. The function will be used in the one of the future homeworks.
Store the matches as u, u0, the array of indices of the four matches used to compute the homography as point_sel and the homography matrix H in 05_homography.mat file.
Fill the pixels of the black square in your image using the pixels from the reference image mapped by H. The pixels can be found e.g. by an image intensity thresholding, or coordinates C of the square corners from Input Data can be used. Store the corrected bitmap image as 05_corrected.png. Optionally, try some colour normalization of filled-in area (up to one bonus point).
Display both images side by side and draw the image matches to both as crosses with labels (1 to 10) and highlight the four points used for computing the best H. Export as 05_homography.pdf.
Filling-in the image
The part of reference image is transferred to your image, so we need the transformation of coordinates in the opposite direction.

Generate list of coordinates u of all pixels in your image that are to be filled. These are typically integers.
Transform these coordinates using the homography. Resulting coordinates u0 point to the reference image (typically not integers).
For each pixel i:
Look-up the color in the reference image at u0i. The easier way is to take the nearest pixel (rounding the coordinates). In general situations, the coordinates should be checked no to be outside the image.
Fill the color in your image (using corresponding coordinates ui).
