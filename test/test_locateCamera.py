import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycalib.plot import plotCamera
from termcolor import colored

# Importing user-defined modules
from detectMarkers import detect
from undistortImage import undistort

def find_camera_pose(image_points_cc, object_points_wc, camera_matrix, dist_coeffs):
#    _, rvec, tvec = cv2.solvePnP(object_points_wc, image_points_cc, camera_matrix, dist_coeffs)
#    rvec, tvec = cv2.solvePnPRefineLM(object_points_wc, image_points_cc, camera_matrix, dist_coeffs, rvec, tvec) 
    print("image_points_cc = ", image_points_cc)
    print("object_points_wc = ", object_points_wc)
    _, K, d, rvec, tvec = cv2.calibrateCamera(
        objectPoints=object_points_wc,
        imagePoints=image_points_cc,
        imageSize=(5472, 3648),
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs)
    
    print("K = ", K)
    print("d = ", d)
    print("rvec = ", rvec[0])
    print("tvec = ", tvec)
    
    return rvec, tvec

def plot_location(rvec, tvec, object_points_wc, corners_sort_cc, pattern_wc):
    # plotCamera() config
    plot_range  = 4 # target volume [-plot_range:plot_range]
    camera_size = 0.3  # size of the camera in plot

    # 3D PLOT
    fig_in = plt.figure()
    fig_in.show()
    ax_in = Axes3D(fig_in, auto_add_to_figure=False)
    fig_in.add_axes(ax_in)
    
    ax_in.set_xlim(-plot_range, plot_range)
    ax_in.set_ylim(-plot_range, plot_range)
    ax_in.set_zlim(-plot_range, plot_range)

    ax_in.set_xlabel('X')
    ax_in.set_ylabel('Y')
    ax_in.set_zlabel('Z')
    ax_in.legend()
    ax_in.set_title('Camera Position and chessboards in 3D space')
    

    R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[0])[0]) # Camera orientation in world coordinate system
    t_c2w = -R_c2w.dot(tvec).reshape((1,3)) # Camera position in world coordinate system
#    R_c2w = cv2.Rodrigues(rvec[0]) # Camera orientation in world coordinate system
#    t_c2w = tvec[0].reshape((1,3)) # Camera position in world coordinate system
        
    plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=camera_size)
    print("Plot camera at", t_c2w)

    ax_in.plot(pattern_wc[:,0], pattern_wc[:,1], pattern_wc[:,2], ".")
    
    plt.show()
    cv2.waitKey(0)
#    plt.savefig("calibration/" + camera + "/result.pdf")

def plot_camera_pose(ax, rvec, tvec, object_points_wc, camera_length=0.1):
    # Plot object points
    ax.scatter(object_points_wc[:, 0], object_points_wc[:, 1], object_points_wc[:, 2], color='blue', label='Object Points')
    
    # Plot camera position
    ax.scatter(tvec[0][0], tvec[1][0], tvec[2][0], color='red', label='Camera Position')
    
    # Plot camera direction
    camera_end = tvec + camera_length * rvec.squeeze()
    ax.plot([tvec[0][0], camera_end[0][0]], [tvec[1][0], camera_end[1][0]], [tvec[2][0], camera_end[2][0]], color='green', label='Camera Direction')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Camera Position and Orientation')


if __name__ == "__main__":
    # Define used camera
    camera = "sony" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Get corners and ids of detected markers
    marker = "DICT_4X4_1000"
#    image = cv2.imread("H:/data/calibration/sony/moving2/DSC00290.JPG")
    image = cv2.imread("H:/data/aruco/DSC00233.JPG")
    img_undst = undistort(K, d, image)
#    img_undst = image
    cv2.imshow("undistorted", cv2.resize(img_undst, (1800, 1200)))
    cv2.waitKey(0)
    img_det, corners, ids = detect(img_undst, marker)
    cv2.imshow("detected", cv2.resize(img_det, (1800, 1200)))
    cv2.waitKey(0)
    # Arange corners and ids in clockwise order
    # Initialize corner_sort array with needed array dimension
    corners_sort_cc = np.ones((len(ids)*4, 2), dtype=np.float32)

    # Sort corners and ids in clockwise order
    for i in range(len(ids)):
        corners_sort_cc[ids[i]*4:ids[i]*4+4] = corners[i]

    # Define reference pattern in clockwise order in world frame (3D) in [m]
    pattern_wc = 0.001 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0], # top left tag
                        [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0], # top right tag
                        [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0], # bottom right tag
                        [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]
                        ], dtype=np.float32)


    # Sample data (replace these with your own image and object points)
#    image_points_cc = corners_sort_cc
#    object_points_wc = [pattern_wc]
    image_points_cc = [corners_sort_cc]
    object_points_wc = [pattern_wc]

    # Sample camera parameters (replace these with your own camera matrix and distortion coefficients)
    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    camera_matrix = K
    dist_coeffs = d

    # Find camera pose
    rvec, tvec = find_camera_pose(image_points_cc, object_points_wc, camera_matrix, dist_coeffs)

    # Plot camera pose
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    plot_camera_pose(ax, rvec, tvec, object_points_wc)
    plt.show()
    cv2.waitKey(0)

    # Print camera pose
    plot_location(rvec, tvec, object_points_wc, corners_sort_cc, pattern_wc)

    print("Rotation vector:")
    print(rvec)
    print("Translation vector:")
    print(tvec)