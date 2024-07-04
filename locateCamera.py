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
#    _, rvec, tvec, rpe = cv2.solvePnPGeneric(pattern, corners, K, d)
    rpe, rvec, tvec, inliers = cv2.solvePnPRansac(pattern, corners, K, d, 
                                                  flags=cv2.SOLVEPNP_EPNP)
    # Ensure rvec and tvec are in the correct format
    rvec = np.array(rvec).reshape(-1, 3)
    tvec = np.array(tvec).reshape(-1, 3)

    print("rvec: ", rvec)
    print("tvec: ", tvec)
    print("rpe PnP: ", rpe)

    return rvec[0], tvec[0]

def get_rpe(img_undst, imgpoints, objpoints, K, rvec, tvec):
    # Function to get reprojection error of pose estimation
    mean_error = 0
    for i in range(len(objpoints)):
        reprojected, _ = cv2.projectPoints(objpoints[i], rvec, tvec, K, np.zeros((5,1)))
        img_rep = cv2.circle(img_undst, (int(reprojected[0][0][0]), 
                                         int(reprojected[0][0][1])), 5, (0,0,255), -1)
        img_corners = cv2.circle(img_rep, (int(imgpoints[i][0]), int(imgpoints[i][1])), 
                                 5, (0,255,0), -1)
        error = cv2.norm(imgpoints[i]- reprojected, cv2.NORM_L2)/len(reprojected)
        print("error ", i, ": ", error)
        mean_error += error

    rpe = mean_error/len(objpoints)
    print("rpe: ", rpe)

    # Show reprojected image
#    cv2.imshow("reprojected", cv2.resize(img_corners, (1920, 1080)))
#    cv2.imwrite("data/reprojected_scaled.jpg", img_rep)
#    cv2.waitKey(0)

    return  rpe

def plot_result(rvec, tvec, K, objpoints):
    # plotCamera() config
    plot_range  = 1 # target volume [-plot_range:plot_range] [m]
    camera_width = 0.0132  # width of the camera in plot [m]
    camera_height = 0.0088  # height of the camera in plot [m]

    # 3D PLOT
    fig_in = plt.figure()
    fig_in.show()
    ax_in = Axes3D(fig_in, auto_add_to_figure=False)
    fig_in.add_axes(ax_in)

    ax_in.set_xlim(-plot_range, plot_range)
    ax_in.set_ylim(-plot_range, plot_range)
    ax_in.set_zlim(-plot_range, plot_range)

    R_c2w = cv2.Rodrigues(rvec)[0] # Camera orientation in world coordinate system
    t_c2w = -R_c2w.T @ tvec # Camera position in world coordinate system
    print("Camera position in world coordinate system: ", t_c2w)
    print("Camera orientation in world coordinate system: ", R_c2w)

    plotCamera(ax_in, R_c2w.T, t_c2w, color="b", scale=0.1) 
    # width=camera_width, height=camera_height, focal_length=0.01, label="Camera")

    # Plot pattern and camera position
    ax_in.plot(objpoints[:,0], objpoints[:,1], objpoints[:,2], ".")
    ax_in.quiver(t_c2w[0], t_c2w[1], t_c2w[2], -t_c2w[0], -t_c2w[1], -t_c2w[2], color="r") 
    # tvec points from the camera cooridnate origin to the world coordinate origin

#    plt.show()
#    cv2.waitKey(0)

if __name__ == "__main__":
    ####################################################################################
    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Define used basis
    basis =  "rough" # "rough", "smooth"

    # Load image and define reference pattern in clockwise order in world frame (3D) in [mm]
    match basis:
        case "smooth":
            image = cv2.imread("data/orig.png")
            pattern = 0.001 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], 
                                        [98.6, 95, 0], [11.5, 93.2, 0],
                                        [498.8, 8.2, 0], [586, 8.2, 0], 
                                        [586.5, 95.4, 0], [499.2, 95, 0],
                                        [499.5, 503, 0], [586.7, 503.9, 0], 
                                        [585.4, 591.2, 0], [498, 590.3, 0],
                                        [16.2, 503.1, 0], [103.3, 503.2, 0], 
                                        [102.7, 590.5, 0], [15.6, 590.5, 0]]
                                        )
        case "rough":
            image = cv2.imread("data/orig.png")
            pattern = 0.001 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], 
                                        [98.7, 98.5, 0], [13.1, 98.8, 0],
                                        [499.2, 12.7, 0], [585.3, 12.8, 0], 
                                        [585.1, 98.7, 0], [499.1, 98.6, 0],
                                        [499.5, 501.2, 0], [585.5, 501.1, 0], 
                                        [585.6, 587.1, 0], [499.6, 587.1, 0],
                                        [12.7, 501.2, 0], [98.3, 501.2, 0], 
                                        [98.4, 587.5, 0], [12.6, 587.3, 0]]
                                        )

    # Undistort image
    img_undst = undistort(K, d, image)

    # Show undistorted image
#    cv2.imshow("undistorted", cv2.resize(img_undst, (1920, 1080)))
#    cv2.waitKey(0)

    # Define marker type and detect markers in image
    marker = "DICT_4X4_50"
    img_det, corners, ids = detect(img_undst, marker)

    print("image shape: ", image.shape)
    print("image undistorted shape: ", img_undst.shape)

    # Print detected corners and corresponding ids
#    print("corners = ", corners)
#    print("ids = ", ids)

    cv2.imshow("detected", cv2.resize(img_det, (1920, 1080)))
    cv2.waitKey(0)

#    print("pattern = ", pattern)

    # Calculate camera position and rotation vetor
    rvec, tvec = locate(corners, pattern, K, d)

    # Get reprojectioin error of pose estimation
    rpe = get_rpe(img_undst, corners, pattern, K, rvec, tvec)

    # Plot result
    plot_result(rvec, tvec, K, pattern)