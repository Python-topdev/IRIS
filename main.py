import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from cvxopt import matrix, solvers
from matplotlib.patches import Ellipse
import cv2

data = {
    "Tireoide":(255,0,0),
    "Cordas Vocais Traqueia":(250,0,0),
    "Esofago":(245,0,0),
    "Bronquios":(240,0,0),
    "Tireoide":(235,0,0),
    "P":(230,0,0),
    "Amigdalas Laringe Faringe":(225,0,0),
    "Escapula":(220,0,0),
    "T":(215,0,0),
    "Coluna":(210,0,0),
    "Figado":(205,0,0)
}

result_ellipse_center=(0,0)
result_ellipse_axes=(0,0)
result_ellipse_angle=0

origin_center=(344,344)
origin_radius=66
origin_total_radius=344

origin_image=cv2.imread("./image/D.png")

image = cv2.imread('./image/iris.png', cv2.IMREAD_COLOR)
temp_image=image.copy()

pre_image=origin_image.copy()
gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
pre_image = cv2.inpaint(pre_image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
def ellipse_half_line_intersection(bx, by):
    # Convert angle to radians
    angle_rad = np.radians(result_ellipse_angle)
    
    # Translate points A and B to be relative to the ellipse center
    ta, tb = center_circle[0] - result_ellipse_center[0], center_circle[1] - result_ellipse_center[1]
    tbx, tby = bx - result_ellipse_center[0], by - result_ellipse_center[1]
    
    # Rotate points to align with the ellipse axes
    cos_angle, sin_angle = np.cos(-angle_rad), np.sin(-angle_rad)
    ra = (ta * cos_angle - tb * sin_angle, ta * sin_angle + tb * cos_angle)
    rb = (tbx * cos_angle - tby * sin_angle, tbx * sin_angle + tby * cos_angle)
    
    # Parametric form of the ray (A to B extended)
    dx, dy = rb[0] - ra[0], rb[1] - ra[1]

    # Substitute line equation into ellipse equation (x^2/a^2 + y^2/b^2 = 1)
    A = (dx**2) / (result_ellipse_axes[0]**2) + (dy**2) / (result_ellipse_axes[1]**2)
    B = 2 * ((ra[0] * dx) / (result_ellipse_axes[0]**2) + (ra[1] * dy) / (result_ellipse_axes[1]**2))
    C = (ra[0]**2) / (result_ellipse_axes[0]**2) + (ra[1]**2) / (result_ellipse_axes[1]**2) - 1
    
    # Solve quadratic equation: A * t^2 + B * t + C = 0
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return None  # No intersection

    # Calculate possible intersection points along the ray
    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    # Only consider the positive t (for the half-line)
    t = max(t1, t2) if t1 >= 0 or t2 >= 0 else None
    if t is None:
        return None  # No valid intersection on the half-line

    # Calculate intersection point in rotated space
    ix, iy = ra[0] + t * dx, ra[1] + t * dy
    
    # Rotate back to the original coordinate system
    original_ix = int(ix * cos_angle + iy * sin_angle + result_ellipse_center[0])
    original_iy = int(-ix * sin_angle + iy * cos_angle + result_ellipse_center[1])
    
    return original_ix, original_iy

def is_point_in_rotated_ellipse(px, py):
    # Convert angle to radians
    angle_rad = np.radians(result_ellipse_angle)
    
    # Translate point to the ellipse center
    tx, ty = px - result_ellipse_center[0], py - result_ellipse_center[1]
    
    # Rotate the point to align with the ellipse's axes
    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)
    rx = tx * cos_angle - ty * sin_angle
    ry = tx * sin_angle + ty * cos_angle
    
    # Check if the point lies within the ellipse's bounds
    distance = (rx ** 2) / (result_ellipse_axes[0] ** 2) + (ry ** 2) / (result_ellipse_axes[1] ** 2)
    
    return distance <= 1

def mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if is_point_in_rotated_ellipse(x,y):
            # print(np.arctan2((y-center_circle[1]), x-center_circle[0]))
            # cv2.circle(image,ellipse_half_line_intersection(x,y),5,(255,0,0))
            buf_image=pre_image.copy()
            interpoint=ellipse_half_line_intersection(x,y)
            angle=np.arctan2((y-center_circle[1]), x-center_circle[0])
            totalLength=np.sqrt((interpoint[0]-center_circle[0])**2+(interpoint[1]-center_circle[1])**2)-radius_circle
            length=np.sqrt((x-center_circle[0])**2+(y-center_circle[1])**2)-radius_circle
            # cv2.imshow('Detected Circles', image)
            
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # buf_image = cv2.dilate(buf_image, kernel, iterations=2)

            add_length=origin_radius+(origin_total_radius-origin_radius)*(length/totalLength)
            point_x=int(origin_center[0]+add_length*np.cos(angle))
            point_y=int(origin_center[1]+add_length*np.sin(angle))
            # cv2.circle(buf_image,(point_x,point_y),3,(255,0,0),5)
            mask = np.zeros((origin_total_radius*2 + 4, origin_total_radius*2 + 4), np.uint8)  # Mask must be 2 pixels larger than the image

            # Flood fill parameters
            lo_diff = (4, 4, 4)  # Lower brightness/color tolerance
            up_diff = (4, 4, 4)  # Upper brightness/color tolerance
            new_color = 255         # Mask fill color, does not affect the image

            for x in range(point_x-3,point_x+3):
                for y in range(point_y-3,point_y+3):
                    cv2.floodFill(
                        buf_image,
                        mask,
                        seedPoint=(x,y),
                        newVal=(new_color, new_color, new_color),
                        loDiff=(10, 10, 10),  # Lower color difference for filling
                        upDiff=(10, 10, 10),  # Upper color difference for filling
                        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8)  # Mask only + max brightness
                    )
            mask=mask[1:-1,1:-1]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.circle(buf_image,(point_x,point_y),3,(255,0,0),3)
            cv2.imshow("s",buf_image)
            # cv2.drawContours(origin_image,contours,-1,(255,0,0),10)
            xx, yy, w, h = cv2.boundingRect(contours[0])  # Get the bounding box of the largest contour
            masked_area = cv2.bitwise_and(origin_image, origin_image, mask=mask)
            cropped_masked_area = masked_area[yy:yy+h, xx:xx+w]
            # print(f"{length/totalLength},{angle}")
            roi = image[yy:yy+cropped_masked_area.shape[0], xx:xx+cropped_masked_area.shape[1]]
            # print(roi.shape,cropped_masked_area.shape)
            blended = cv2.addWeighted(roi, 0, cropped_masked_area, 1, 0)
            temp_image=image.copy()
            temp_image[y:y+cropped_masked_area.shape[0], x:x+cropped_masked_area.shape[1]]=blended
            print(image.shape,roi.shape)
            # cv2.rectangle(temp_image,(x,y),(x+100,y+100),(255,0,0),2)
            # buf[yy:yy+cropped_masked_area.shape[0], xx:xx+cropped_masked_area.shape[1]] = blended
            cv2.imshow("Detected Circles",temp_image)

def min_enclosing_ellipse(points):
    N = len(points)
    
    Q = np.column_stack((points, np.ones(N))) 

    P = matrix(np.dot(Q, Q.T).astype(np.double)) 
    q = matrix(-np.ones(N).astype(np.double))   
    G = matrix(-np.eye(N).astype(np.double))   
    h = matrix(np.zeros(N).astype(np.double))     
    A = matrix(np.ones((1, N)).astype(np.double)) 
    b = matrix(np.ones(1).astype(np.double))      

    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x']).flatten()
    alphas = np.clip(alphas, 0, None)
    alphas /= alphas.sum()
    center = np.dot(alphas, points)
    cov = np.cov(points.T, aweights=alphas)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis_length = 2 * np.sqrt(eigvals[1])
    minor_axis_length = 2 * np.sqrt(eigvals[0])

    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

    return center, major_axis_length, minor_axis_length, angle

HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
height, width = image.shape[:2]
image_center = (width // 2, height // 2)
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
gray = cv2.medianBlur(gray, 5)

# Detect circles in the image
# Apply GaussianBlur to reduce noise and improve circle detection
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using HoughCircles
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours=[]
for contour in contours:
  if cv2.contourArea(contour)>500:
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Avoid division by zero
        contour_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        contour_center = (0, 0)  # Default if contour is a line or point

    # Calculate the distance between the image center and the contour center
    distance = np.sqrt((image_center[0] - contour_center[0]) ** 2 + (image_center[1] - contour_center[1]) ** 2)

    # Define a threshold to determine if it's "almost" centered (e.g., 10% of image width)
    threshold_distance = 0.5 * min(width, height)  # 10% of the smaller image dimension

    # Check if the contour center is close to the image center
    if distance < threshold_distance:
      filtered_contours.append(contour)
# Check if any filtered contours were found
if filtered_contours:
    # Find the contour with the smallest area among the filtered contours
    smallest_contour = min(filtered_contours, key=cv2.contourArea)

(x, y), radius = cv2.minEnclosingCircle(smallest_contour)
center_circle = (int(x), int(y))  # Center of the circle
radius_circle = int(radius)       # Radius of the circle

# Draw the approximated circle on a copy of the original image
cv2.circle(image, center_circle, radius_circle, (0, 255, 0), 2)  # Draw the circle

lower_brown = np.array([70, 60, 60])  # Lower bound for brown color
upper_brown = np.array([140, 140, 160])  # Upper bound for brown color

kernel = np.ones((7, 7), np.uint8)  # Larger kernel for stronger effect
# binary_mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # Remove noise
# Create a mask for brown color
# image = cv2.bitwise_and(image, image, mask=binary_mask)
brown_mask = cv2.inRange(image, lower_brown, upper_brown)

kernel = np.ones((9, 9), np.uint8)  # 5x5 square kernel; you can change the size or shape

# Apply erosion
eroded_image = cv2.erode(brown_mask, kernel, iterations=1)

# Apply dilation
brown_mask = cv2.dilate(eroded_image, kernel, iterations=1)
# Use the mask to extract brown areas
brown_areas = cv2.bitwise_and(image, image, mask=brown_mask)
brown_gray=cv2.cvtColor(brown_areas,cv2.COLOR_BGRA2GRAY)
# _, brown_thresh = cv2.threshold(brown_gray, 50, 255, cv2.THRESH_BINARY_INV)
brown_contours,_=cv2.findContours(brown_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gray, brown_contours, -1, 255, 2)
ellipse_center = (200, 200)  # Center of the ellipse
major_axis_length = 100      # Major axis length
minor_axis_length = 50       # Minor axis length
angle = 30
for contour in brown_contours:
  if len(contour) >= 5:
    points = contour.reshape(-1,2)

    # Step 1: Find the two points with the longest distance
    max_distance = 0
    point1, point2 = None, None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_distance:
                max_distance = dist
                point1, point2 = points[i], points[j]

    # Step 2: Calculate the center (midpoint) of the ellipse
    ellipse_center = (point1 + point2) / 2

    # Step 3: Set the major axis length (longest distance)
    major_axis_length = max_distance

    # Step 4: Calculate the angle of the major axis
    dx, dy = point2 - point1
    angle = np.degrees(np.arctan2(dy, dx))

    # Step 5: Determine the minor axis length by gradually increasing it until all points are within the ellipse
    minor_axis_length = 0
    for point in points:
        # Calculate the perpendicular distance from the ellipse_center line (major axis) to the point
        diff = point - ellipse_center
        distance_to_major_axis = np.abs(-dy * diff[0] + dx * diff[1]) / np.linalg.norm([dx, dy])
        if distance_to_major_axis > minor_axis_length:
            minor_axis_length = distance_to_major_axis

    # Double the minor axis length to make it a full diameter
    minor_axis_length *= 2

    # Convert ellipse_center and axes lengths for cv2.ellipse
    ellipse_center = (int(ellipse_center[0]), int(ellipse_center[1]))
    axes = (int(major_axis_length / 2), int(minor_axis_length / 2))  # cv2.ellipse expects half-lengths of axes
    # Create a blank image to draw the ellipse
    # Draw the original points
    # Draw the enclosing ellipse
    if np.sqrt((ellipse_center[0] - center_circle[0]) ** 2 + (ellipse_center[1] - center_circle[1]) ** 2)<50 and radius>radius_circle:
        result_ellipse_center=ellipse_center
        result_ellipse_axes=axes
        result_ellipse_angle=angle
        cv2.ellipse(image, ellipse_center, axes, angle, 0, 360, (0, 255, 0), 2)

# cv2.imshow("ss",brown_areas)
target_radius=200
scaling_factor_x = target_radius / (major_axis_length / 2)
scaling_factor_y = target_radius / (minor_axis_length / 2)

# Step 3: Create a transformation matrix for scaling and translating
# Move ellipse center to origin, scale, then move to target center
transformation_matrix = cv2.getRotationMatrix2D(ellipse_center, 0, 1)
transformation_matrix[0, 0] *= scaling_factor_x
transformation_matrix[1, 1] *= scaling_factor_y

# Adjust translation to move scaled ellipse to target center
dx = center_circle[0] - ellipse_center[0] * scaling_factor_x
dy = center_circle[1] - ellipse_center[1] * scaling_factor_y
transformation_matrix[0, 2] += dx
transformation_matrix[1, 2] += dy

# Step 4: Apply the transformation to the entire image
transformed_image = cv2.warpAffine(gray, transformation_matrix, (gray.shape[1], gray.shape[0]))

# Draw the target circle on the transformed image for reference
cv2.circle(transformed_image, center_circle, target_radius, (0, 255, 0), 2)
# cv2.imshow("Transformed Image with Circle", transformed_image)
cv2.namedWindow('Detected Circles')
cv2.setMouseCallback("Detected Circles", mouse_position)
while True:
    # Show the image with the rectangle at the current mouse position
    # cv2.imshow("Detected Circles", temp_image)
    # temp_image=image.copy()
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break
# cv2.waitKey(0)
cv2.destroyAllWindows()
