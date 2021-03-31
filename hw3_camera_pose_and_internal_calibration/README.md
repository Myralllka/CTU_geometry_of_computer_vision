Implementation
matlab	python
[ K, R, C ] = Q2KRC( Q )	K, R, C = hw03.Q2KRC( Q )
Create a function Q2KRC for decomposing a camera matrix Q (3×4) into into the projection centre C (3×1), rotation matrix R (3×3) and upper triangular matrix K (3×3) such that

Q = λ ( K R | - K R C )

where K(3,3) = 1, K(1,1) > 0, and det(R) = 1.

Create a function 'plot_csystem' for drawing a coordinate system with base Base located in the origin b with a given name and color. The base and origin are expressed in the world coordinate system δ. The base consists of a two or three three-dimensional column vectors of coordinates. E.g.

plot_csystem(eye(3),zeros(3,1),'k','\\delta');	hw03.plot_csystem(ax,np.eye(3),np.zeros([3,1]),'k','d')
should plot the δ system. The function should label each base vector (e.g. δ_x, δ_y, δ_z).

Steps
Decompose the optimal camera matrix Q you have recovered in HW-02. Let the horizontal pixel size be 5 μm. Compute f (in metres) and compose matrix Pb (Pβ) using K, R, C, and f.
For the camera, compute bases and centres of coordinate systems α, β, γ, δ, ε, κ, υ. Express all bases and centres in the world coordinate system δ. The bases should be stored in matrices Alpha, Beta, Gamma, Delta, Epsilon, Kappa, Nu, respectively, the coordinate system centres should be stored in matrices a, b, g, d, e, k, n, respectively.
Save Pb, f, all bases and coordinate system centres into 03_bases.mat.
For following plots, multiply the first vectors of bases α, β by image width (1100) and the second vectors of bases α, β by image height (850).
Draw the coordinate systems δ (black), ε (magenta), κ (brown), υ (cyan), draw the system β (red) with its base scaled-up 50 times additionally , and draw the 3D scene points (109 points, blue). Label each base vector (e.g. δ_x, δ_y, δ_z). Export as 03_figure1.pdf.
Draw the coordinate systems α (green), β (red), γ (blue), draw the image points (109 points, blue). Label each base vector. Export as 03_figure2.pdf.
Draw the coordinate systems δ (black), ε (magenta), plot the 3D scene points (blue), and plot centers (red) of all cameras you have tested in HW-02 (using the decomposition). Zoom-in such that the coordinate systems are clearly visible. Export as 03_figure3.pdf. Note that the coordinate system ε is for the optimal camera only.
