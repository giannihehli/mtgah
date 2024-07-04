# This script was written to calibrate the camera. However, a different version was found and used instead.

# IMPORT LIBRARIES
import sys, os, cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from termcolor import colored

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from pycalib.plot import plotCamera

def calibrate(camera, view, check):
        
    # CALIBRATION PARAMETERS
    # Chessboard configuration
    rows = 9   # Number of corners (not cells) in row
    cols = 6  # Number of corners (not cells) in column
    size = 0.0243 # [m] Physical size of a cell (the distance between neighrboring corners). Any positive number works.

    # Theroetical object points for the chessboard we're calibrating against
    X_W = np.empty([rows * cols, 3], dtype=np.float32)
    for i_row in range(0, rows):
        for i_col in range(0, cols):
            X_W[i_row*cols+i_col] = np.array([size*i_col, size*i_row, 0], dtype=np.float32)
    
    # Input images capturing the chessboard above
    match camera:
        case "sony":
            input_files = "H:/data/calibration/sony/moving2/*.jpg"
            output_path = "H:/data/calibration/sony/moving2/"
        case "gopro1":
            input_files = "H:/data/calibration/gopro1/*.jpg"
            output_path = "H:/data/calibration/gopro1/"
        case "gopro2":
            input_files = "H:/data/calibration/gopro2/*.jpg"
            output_path = "H:/data/calibration/gopro2/"

    # 2D POSITIONS OF THE CHESS CORNERS
    objpoints = []
    imgpoints = []

    # Count variables
    count_found = 0
    count_failed = 0

    for image_path in glob(input_files): # for each chessboard image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)      # load the image
        ret, corners = cv2.findChessboardCorners(image, (cols, rows)) # detech the chess corners

        if ret: # if found
            count_found += 1
            print(colored("Detection successful : ", "green"), os.path.basename(image_path))
            term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            corners_sub = cv2.cornerSubPix(image, corners, (5,5), (-1,-1), term) # refine the corner positions
            objpoints.append(X_W)     # the chess corner in 3D
            imgpoints.append(corners_sub) # is projected to this 2D position

            if check: # if check show detected Corners
                fnl = cv2.drawChessboardCorners(image, (cols, rows), corners, ret)
                cv2.namedWindow('resizedImg', cv2.WINDOW_NORMAL)
                cv2.imshow("resizedImg", fnl)
                cv2.imwrite(output_path + "detected corners/detected_" + os.path.basename(image_path), fnl)
                #cv2.waitKey(0)

        else:     # if not found
            count_failed += 1
            print(colored("Detection failed : ", "red"), image_path)
            continue 

    # CALIBRATION
    rpe, K, d, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (image.shape[1], image.shape[0]), None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)
    print("Calibration successful / RPE: ", rpe, " / found: ", count_found, " / failed: ", count_failed)
    np.savetxt(output_path + "K.txt", K)
    print("Save intrinsic parameter K = ", K)
    np.savetxt(output_path + "d.txt", d)
    print("Save Distortion parameters d = (k1, k2, p1, p2, k3) = ", d)

    print("RMS re-projection error :", rpe)

    if view:
        plot_calibration(rvec, tvec, X_W, output_path)

    return K, d, rvec, tvec, X_W

def plot_calibration(rvec, tvec, X_W, output_path):
    # plotCamera() config
    plot_mode   = 0    # 0: fixed camera / moving chessboard,  1: fixed chessboard, moving camera
    plot_range  = 1 # target volume [-plot_range:plot_range]
    camera_size = 0.03  # size of the camera in plot

    # 3D PLOT
    fig_in = plt.figure()
    fig_in.show()
    ax_in = Axes3D(fig_in, auto_add_to_figure=False)
    fig_in.add_axes(ax_in)

    ax_in.set_xlim(-plot_range, plot_range)
    ax_in.set_ylim(-plot_range, plot_range)
    ax_in.set_zlim(-plot_range, plot_range)

    if plot_mode == 0: # fixed camera = plot in CCS
        
        plotCamera(ax_in, np.eye(3), np.zeros((1,3)), color="b", scale=camera_size) # camera is at (0,0,0)

        for i_ex in range(len(rvec)):
            X_C = np.zeros((X_W.shape))
            for i_x in range(X_W.shape[0]):
                R_w2c = cv2.Rodrigues(rvec[i_ex])[0] # convert to the rotation matrix
                t_w2c = tvec[i_ex].reshape(3)
                X_C[i_x,:] = R_w2c.dot(X_W[i_x,:]) + t_w2c # Transform chess corners in WCS to CCS
                    
            ax_in.plot(X_C[:,0], X_C[:,1], X_C[:,2], ".") # plot chess corners in CCS

    elif plot_mode == 1: # fixed chessboard = plot in WCS
        
        for i_ex in range(len(rvec)):
            R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[i_ex])[0]) # Camera orientation in world coordinate system
            t_c2w = -R_c2w.dot(tvec[i_ex]).reshape((1,3)) # Camera position in world coordinate system
            
            plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=camera_size)
            print("Plot camera", i_ex, "at", t_c2w)

        ax_in.plot(X_W[:,0], X_W[:,1], X_W[:,2], ".")
    plt.savefig(output_path + "result.pdf")
    plt.show()
    cv2.waitKey(0)

if __name__=="__main__":
    K, d, rvec, tvec, X_W = calibrate("sony", view=True, check=False)