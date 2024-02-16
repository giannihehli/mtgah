import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def estimate_camera_pose(image_points, marker_length, camera_matrix, dist_coeffs):
    _, rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(image_points, marker_length, camera_matrix, dist_coeffs)
    return rvecs, tvecs

def plot_aruco_markers(ax, image_points, marker_length):
    print(image_points)
    for i in range(4):
        print(image_points[:, i, 0], image_points[:, i, 1])
        ax.scatter(image_points[:, i, 0], image_points[:, i, 1], marker_length, color='blue', label='Detected ArUco Markers')
    
def plot_camera_pose(ax, rvecs, tvecs):
    for rvec, tvec in zip(rvecs, tvecs):
        ax.scatter(tvec[0][0], tvec[0][1], tvec[0][2], color='red', label='Camera Position')
        camera_end = tvec + np.dot(cv2.Rodrigues(rvec)[0], np.array([[0.1], [0], [0]]))
        ax.plot([tvec[0][0], camera_end[0][0]], [tvec[0][1], camera_end[0][1]], [tvec[0][2], camera_end[0][2]], color='green', label='Camera Direction')

# Sample data (replace with your actual detected ArUco markers and camera calibration)
# Sample data (replace with your own image and object points)
image_points = np.array([[
                          [299.,  95.], [365.,  96.], [359., 149.], [290., 149.],
#                          [673.,  93.], [739.,  92.], [748., 146.], [680., 146.],
#                          [720., 469.], [804., 468.], [819., 557.], [731., 557.],
#                          [246., 475.], [332., 474.], [323., 562.], [233., 564.]
                         ]], dtype=np.float32)
object_points = 0.001 * np.array([[[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0], # top left tag
#                                  [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0], # top right tag
#                                  [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0], # bottom right tag
#                                  [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0] # bottom left tag
                                  ]], dtype=np.float32)

# Define used camera
camera = "sony" # "sony", "gopro1", "gopro2
# Import calibration parameters
K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

marker_length = 0.0875  # Length of the ArUco marker (in meters)

# Sample camera parameters (replace with your actual camera calibration)
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)  # Camera matrix
dist_coeffs = np.zeros((4, 1))  # Distortion coefficients
#camera_matrix = K
#dist_coeffs = d

# Estimate camera pose
rvecs, tvecs = estimate_camera_pose(image_points, marker_length, camera_matrix, dist_coeffs)

# Plot camera pose and detected ArUco markers
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_aruco_markers(ax, image_points, marker_length)
plot_camera_pose(ax, rvecs, tvecs)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Detected ArUco Markers and Camera Pose')

plt.show()
