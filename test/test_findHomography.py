import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_camera_pose(ax, homography, K=None):
    rot, trans, _, _ = cv2.decomposeHomographyMat(homography, K)
    camera_end = trans + 0.1 * rot

    ax.plot([trans[0][0], camera_end[0][0]], 
            [trans[1][0], camera_end[1][0]], 
            [trans[2][0], camera_end[2][0]], 
            color='green', label='Camera Direction')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Camera Position and Orientation')

def get_homography(image_points, object_points):
    return cv2.findHomography(image_points, object_points)[0]

# Sample data (replace with your actual detected ArUco markers and camera calibration)
# Sample data (replace with your own image and object points)
image_points = np.array([[
                          [299.,  95.], [365.,  96.], [359., 149.], [290., 149.],
                          [673.,  93.], [739.,  92.], [748., 146.], [680., 146.],
                          [720., 469.], [804., 468.], [819., 557.], [731., 557.],
                          [246., 475.], [332., 474.], [323., 562.], [233., 564.]
                         ]], dtype=np.float32)

object_points = 0.001 * np.array([[
                                  [12.5, 6.2], [99.5, 8], [98.6, 95], [11.5, 93.2], # top left tag
                                  [498.8, 8.2], [586, 8.2], [586.5, 95.4], [499.2, 95], # top right tag
                                  [499.5, 503], [586.7, 503.9], [585.4, 591.2], [498, 590.3], # bottom right tag
                                  [16.2, 503.1], [103.3, 503.2], [102.7, 590.5], [15.6, 590.5] # bottom left tag
                                  ]], dtype=np.float32)

# Define used camera
camera = "sony" # "sony", "gopro1", "gopro2"
# Import calibration parameters
# K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

# Dummy camera matrix for illustration
K = np.array([[1000, 0, 500],
              [0, 1000, 500],
              [0, 0, 1]])

# Get homography from point correspondences
homography = get_homography(image_points, object_points)

# Plot camera pose
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_camera_pose(ax, homography, K)
plt.show()
