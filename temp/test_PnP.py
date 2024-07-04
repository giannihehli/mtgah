import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_camera_pose(image_points, object_points, K, d):
    # Solve PnP problem to get rotation and translation vectors
    _, rvec, tvec = cv2.solvePnP(object_points, image_points, K, d)
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec

def plot_camera_pose(ax, R, tvec):
    # Plot camera position
    ax.scatter(tvec[0], tvec[1], tvec[2], color='red', label='Camera Position')
    
    # Plot camera direction
    camera_end = tvec + np.dot(R, np.array([[0.1], [0], [0]]))
    ax.plot([tvec[0], camera_end[0]], [tvec[1], camera_end[1]], [tvec[2], camera_end[2]], color='green', label='Camera Direction')

def plot_object_points(ax, object_points):
    # Plot object points
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], color='blue', label='Object Points')


# Sample data (replace with your own image and object points)
image_points = np.array([[299.,  95.], [365.,  96.], [359., 149.], [290., 149.],
                          [673.,  93.], [739.,  92.], [748., 146.], [680., 146.],
                          [720., 469.], [804., 468.], [819., 557.], [731., 557.],
                          [246., 475.], [332., 474.], [323., 562.], [233., 564.]], dtype=np.float32)
object_points = 0.001 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0], # top left tag
                                  [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0], # top right tag
                                  [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0], # bottom right tag
                                  [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0] # bottom left tag
                                  ], dtype=np.float32)

# Define used camera
camera = "sony" # "sony", "gopro1", "gopro2
# Import calibration parameters
K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]


# Get camera pose
R, tvec = get_camera_pose(image_points, object_points, K, d)

print("Rotation matrix:")
print(R)
print("Translation vector:")
print(tvec)

# Plot camera pose and object points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_camera_pose(ax, R, tvec)
plot_object_points(ax, object_points)
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Position, Orientation, and Object Points')
plt.show()
