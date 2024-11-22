import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Function to calculate the minimum enclosing ellipse
def fit_minimum_enclosing_ellipse(points):
    def ellipse_area(params):
        xc, yc, a, b, theta = params
        return np.pi * a * b

    def constraints(params, points):
        xc, yc, a, b, theta = params
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_rotated = ((points[:, 0] - xc) * cos_theta + (points[:, 1] - yc) * sin_theta) ** 2 / a**2
        y_rotated = ((points[:, 1] - yc) * cos_theta - (points[:, 0] - xc) * sin_theta) ** 2 / b**2
        return x_rotated + y_rotated - 1  # Ellipse constraint: x^2/a^2 + y^2/b^2 <= 1

    hull = ConvexHull(points)  # Use the convex hull for efficiency
    hull_points = points[hull.vertices]

    # Initial parameters: center (mean of points), semi-major/minor axes (std of x/y), angle 0
    xc, yc = np.mean(hull_points, axis=0)
    a, b = np.std(hull_points, axis=0)
    theta = 0

    initial_params = [xc, yc, a, b, theta]
    bounds = [(None, None), (None, None), (1e-3, None), (1e-3, None), (-np.pi / 2, np.pi / 2)]

    result = minimize(ellipse_area, initial_params, bounds=bounds, constraints={'type': 'ineq', 'fun': constraints, 'args': (hull_points,)})
    if result.success:
        return result.x  # Returns xc, yc, a, b, theta
    else:
        return None

# Load an image and find contours
image = cv2.imread('./image/iris.png', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Choose the contour you want to fit
contour = contours[0]  # Just an example, select the contour of interest

# Flatten the contour to use with SciPy's optimization
points = contour.reshape(-1, 2)

# Fit the minimum enclosing ellipse
params = fit_minimum_enclosing_ellipse(points)

if params is not None:
    xc, yc, a, b, theta = params
    center = (int(xc), int(yc))
    axes = (int(a), int(b))
    angle = np.degrees(theta)

    # Draw the ellipse on the image
    result_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(result_image, (center, axes, angle), (0, 255, 0), 2)

    # Display the image with the minimum enclosing ellipse
    cv2.imshow("Minimum Enclosing Ellipse", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Ellipse fitting failed.")
