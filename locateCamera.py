# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycalib.plot import plotCamera

# Importing user-defined modules
from detectMarkers import detect
from undistortImage import undistort

def locate(corners, pattern, K, d):

    # Get pose estimation
    _, rvec, tvec = cv2.solvePnP(pattern, corners, K, d)

    print("rvec ", rvec)
    print("tvec: ", tvec)

    return rvec, tvec

def get_rpe(imgpoints, objpoints, K, rvec, tvec):
    # Function to get reprojection error of pose estimation
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvec, tvec, K, np.zeros((5,1)))
        error = cv2.norm(imgpoints[i]- imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    rpe = mean_error/len(objpoints)
    print("rpe: ", rpe)

    return  rpe

def plot_result(rvec, tvec, objpoints):
    # plotCamera() config
    plot_range  = 4000 # target volume [-plot_range:plot_range]
    camera_size = 100  # size of the camera in plot

    # 3D PLOT
    fig_in = plt.figure()
    fig_in.show()
    ax_in = Axes3D(fig_in, auto_add_to_figure=False)
    fig_in.add_axes(ax_in)

    ax_in.set_xlim(-plot_range, plot_range)
    ax_in.set_ylim(-plot_range, plot_range)
    ax_in.set_zlim(-plot_range, plot_range)

    R_c2w = np.linalg.inv(cv2.Rodrigues(rvec)[0]) # Camera orientation in world coordinate system
    t_c2w = -tvec # Camera position in world coordinate system
    print("Camera position in world coordinate system: ", t_c2w)
    print("Camera orientation in world coordinate system: ", R_c2w)

    plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=camera_size)

    # Plot pattern and camera position
    ax_in.plot(objpoints[:,0], objpoints[:,1], objpoints[:,2], ".")
    ax_in.quiver(-tvec[0], -tvec[1], -tvec[2], tvec[0], tvec[1], tvec[2], color="r") # tvec points from the camera cooridnate origin to the world coordinate origin

    plt.show()
    cv2.waitKey(0)

if __name__ == "__main__":

    # Define used camera
    camera = "sony" # "sony", "gopro1", "gopro2

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    
    # Load and undistort image
    image = cv2.imread("DSC00233.JPG")
    img_undst = undistort(K, d, image)

    # Show undistorted image
    cv2.imshow("undistorted", cv2.resize(img_undst, (1800, 1200)))
    cv2.waitKey(0)

    # Define marker type and detect markers in image
    marker = "DICT_4X4_1000"
    img_det, corners, ids = detect(img_undst, marker)

    # Print detected corners and corresponding ids
    print("corners = ", corners)
    print("ids = ", ids)

    cv2.imshow("detected", cv2.resize(img_det, (1800, 1200)))
    cv2.waitKey(0)

    ## Arange corners and ids in clockwise order
    # Initialize corner_sort array with needed array dimension
    corners_sort = np.ones((len(ids)*4, 2))

    # Sort corners and ids in clockwise order
    for i in range(len(ids)):
        corners_sort[ids[i]*4:ids[i]*4+4] = corners[i]

    # Define reference pattern in clockwise order in world frame (3D) in [mm]
    pattern = np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                        [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                        [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                        [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]])

    print("corners_sort = ", corners_sort)
    print("pattern = ", pattern)

    # Calculate camera position and rotation vetor
    rvec, tvec = locate(corners_sort, pattern, K, d)

    # Get reprojectioin error of pose estimation
    rpe = get_rpe(corners_sort, pattern, K, rvec, tvec)

    # Plot result
    plot_result(rvec, tvec, pattern)
