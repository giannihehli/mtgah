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
    retval, rvec, tvec, rpe = cv2.solvePnPGeneric(pattern, corners, K, d, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)
    print(cv2.solvePnP(pattern, corners, K, d))
    print("retval: ", retval)
    print("rvec: ", rvec[0])
    print("tvec: ", tvec[0])
    print("rpe: ", rpe)

    if retval:
        # Refine pose estimation with Levenber-Marquardt iterative minimization
        # rvec, tvec = cv2.solvePnPRefineLM(pattern, corners, K, d, rvec, tvec)
        print("Reprojection error: ", rpe)

        # Get reprojectioin error of pose estimation
        get_rpe(corners, pattern, K, d, rvec[0], tvec[0])

        return rvec[0], tvec[0]

    elif not retval:
        print("Pose estimation failed. Try different pattern or camera parameters.")
        exit()

def plot_result(rvec, tvec, X_W, output_path):
    # plotCamera() config
    plot_range  = 2000 # target volume [-plot_range:plot_range]
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
    t_c2w = -R_c2w.dot(tvec).reshape((1,3)) # Camera position in world coordinate system
    print("Camera position in world coordinate system: ", t_c2w)
    print("Camera orientation in world coordinate system: ", R_c2w)

    plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=camera_size)

    # Plot pattern and camera position
    ax_in.plot(X_W[:,0], X_W[:,1], X_W[:,2], ".")
    ax_in.quiver(-tvec[0], -tvec[1], -tvec[2], tvec[0], tvec[1], tvec[2], color="r")

    plt.savefig(output_path + "result.pdf")
    plt.show()
    cv2.waitKey(0)

def get_rpe(imgpoints, objpoints, K, d, rvec, tvec):
    # Function to get reprojectioin error of pose estimation
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvec, tvec, K, d)
        error = cv2.norm(imgpoints[i]- imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)))    

if __name__ == "__main__":
    camera = "sony" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Get corners and ids of detected markers
    marker = "DICT_4X4_50"
    image = cv2.imread("H:/data/aruco/DSC00233.JPG")
    img_undst = undistort(K, d, image)
    #img_undst = image
    cv2.imshow("undistorted", cv2.resize(img_undst, (1800, 1200)))
    cv2.waitKey(0)
    img_det, corners, ids = detect(img_undst, marker)
    cv2.imshow("detected", cv2.resize(img_det, (1800, 1200)))
    cv2.waitKey(0)
    # Arange corners and ids in clockwise order
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
    
    pattern_2d = np.float32([[12.5, 6.2], [99.5, 8], [98.6, 95], [11.5, 93.2],
                        [498.8, 8.2], [586, 8.2], [586.5, 95.4], [499.2, 95],
                        [499.5, 503], [586.7, 503.9], [585.4, 591.2], [498, 590.3],
                        [16.2, 503.1], [103.3, 503.2], [102.7, 590.5], [15.6, 590.5]])

    rvec, tvec = locate(corners_sort, pattern, K, d)

    corners_float = corners_sort.astype(np.float32)
    print("corners_sort = ", corners_sort)
    print("pattern = ", pattern_2d)

    # Get homography matrix
    h = cv2.findHomography(pattern_2d, corners_sort, cv2.RANSAC)

    print(h)

    retval, rotM, tvec, test = cv2.decomposeHomographyMat(h[0], K)
    print("retval: ", retval)
    print("rotM: ", cv2.Rodrigues(rotM[i]))
    print("rvec: ", rvec)
    print("tvec: ", tvec)
    print("test: ", test)

    for i in range(len(rotM)):
        plot_result(cv2.Rodrigues(rotM[i])[0], tvec[i], pattern, "H:/data/aruco/")