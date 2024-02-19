# Resources: 
# - OpenCV-Python tutorial for calibration: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#   - Variable names were changed for clarity

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycalib.plot import plotCamera
from termcolor import colored

def calibrate(camera, data_path):

    # Chessboard variables
    rows = 9
    cols = 6
    size = 0.0243 # [m] Physical size of a cell (the distance between neighrboring corners). Any positive number works.

    # Theoretical object points for the chessboard we"re calibrating against,
    # These will come out like: 
    #     size * (0, 0, 0), (1, 0, 0), ..., 
    #     size * (rows-1, cols-1, 0)
    # Note that the Z value for all stays at 0, as this is a printed out 2D image
    # And also that the max point is -1 of the max because we"re zero-indexing
    # The following line generates all the tuples needed at (0, 0, 0)
    objp = np.zeros((rows*cols,3), np.float32)
    
    # The following line fills the tuples just generated with their values size * (0, 0, 0), size * (1, 0, 0), ...
    objp[:,:2] = size*np.mgrid[0:rows,0:cols].T.reshape(-1, 2)

    # Need a set of images or a video taken with the camera you want to calibrate
    # Input images capturing the chessboard above
    input_files = data_path + "/*.jpg"
    output_path = "H:/data/calibration/" + camera + "/moving2"
    #input_files = "H:/data/calibration/" + camera + "/*.jpg"
    #output_path = "H:/data/calibration/" + camera

    # All images used should be the same size, which if taken with the same camera shouldn"t be a problem
    # I"m using a set of images taken with the camera with the naming convention:
    # "camera-pic-of-chessboard-<NUMBER>.jpg"
    images = glob.glob(input_files)
    imageSize = None # Determined at runtime

    # Create arrays you"ll use to store object points and image points from all images processed
    objpoints = [] # 3D point in real world space where chess squares are
    imgpoints = [] # 2D point in image plane, determined by CV2

    # Count variables
    count_found = 0
    count_failed = 0

    # Loop through images glob"ed
    for image_path in images:
        # Open the image
        img = cv2.imread(image_path)
        # Grayscale of the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard in the image, setting PatternSize(2nd arg) to a tuple of (#rows, #columns)
        ret, corners = cv2.findChessboardCorners(gray, (rows,cols), None)

        # If a chessboard was found, let"s collect image/corner points
        if ret == True:
            count_found += 1
            print(colored("Detection successful : ", "green"), os.path.basename(image_path))
            # Add the points in 3D that we just discovered
            objpoints.append(objp)
            
            # Enhance corner accuracy with cornerSubPix
            corners_acc = cv2.cornerSubPix(
                    image=gray, 
                    corners=corners, 
                    winSize=(11, 11), 
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)) # Last parameter is about termination critera
            imgpoints.append(corners_acc)

            # If our image size is unknown, set it now
            if not imageSize:
                imageSize = (gray.shape[1], gray.shape[0])
                print("Image size: ", imageSize)
                print("objpoints: ", objpoints)
                print("imgpoints: ", imgpoints)
        
            # Draw the corners to a new image to show whoever is performing the calibration
            # that the board was properly detected
            img = cv2.drawChessboardCorners(img, (rows, cols), corners_acc, ret)
            # Pause to display each image, waiting for key press
            #cv2.imshow("Chessboard", img)
            #cv2.waitKey(0)
            cv2.imwrite(output_path + "/detected/det_" + os.path.basename(image_path), img)
        else:     # if not found
            count_failed += 1
            print(colored("Detection failed : ", "red"), os.path.basename(image_path))
            continue 

    # Destroy any open CV windows
    cv2.destroyAllWindows()

    # Make sure at least one image was found
    if len(images) < 1:
        # Calibration failed because there were no images, warn the user
        print("Calibration was unsuccessful. No images of chessboards were found. Add images of chessboards and use or alter the naming conventions used in this file.")
        # Exit for failure
        exit()

    # Make sure we were able to calibrate on at least one chessboard by checking
    # if we ever determined the image size
    if not imageSize:
        # Calibration failed because we didn"t see any chessboards of the PatternSize used
        print("Calibration was unsuccessful. We couldn't detect chessboards in any of the images supplied. Try changing the patternSize passed into findChessboardCorners(), or try different pictures of chessboards.")
        # Exit for failure
        exit()
    
    # Now that we"ve seen all of our images, perform the camera calibration
    # based on the set of points we"ve discovered
    rpe, K, d, rvec, tvec = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=imageSize,
            cameraMatrix=None,
            distCoeffs=None)
        
    # Save values to be used where matrix+dist is required, for instance for posture estimation
    print("Calibration successful / RPE: ", rpe, " / found: ", count_found, " / failed: ", count_failed)
    np.savetxt("calibration/" + camera + "/K.txt", K)
    print("Saved intrinsic parameter K = ", K)
    np.savetxt("calibration/" + camera + "/d.txt", d)
    print("Saved Distortion parameters d = (k1, k2, p1, p2, k3) = ", d)

    print("RMS re-projection error :", rpe)
    print("translation vectors : ", tvec)
    
    plot_calibration(rvec, tvec, objp)

    # Print to console our success

def plot_calibration(rvec, tvec, X_W):
    # plotCamera() config
    plot_mode   = 0    # 0: fixed camera / moving chessboard,  1: fixed chessboard, moving camera
    plot_range  = 0.5 # target volume [-plot_range:plot_range]
    camera_size = 0.03  # size of the camera in plot

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

    if plot_mode == 0: # fixed camera = plot in CCS
        
        plotCamera(ax_in, np.eye(3), np.zeros((1,3)), color="b", scale=camera_size) # camera is at (0,0,0)

        for i_ex in range(len(rvec)):
            X_C = np.zeros((X_W.shape))
            for i_x in range(X_W.shape[0]):
                R_w2c = cv2.Rodrigues(rvec[i_ex])[0] # convert to the rotation matrix
                t_w2c = tvec[i_ex].reshape(3)
                X_C[i_x,:] = R_w2c.dot(X_W[i_x,:]) + t_w2c # Transform chess corners in WCS to CCS
                    
            ax_in.plot(X_C[:,0], X_C[:,1], X_C[:,2], ".") # plot chess corners in CCS
            print("Rotation: ", R_w2c)
            print("Translation: ", t_w2c)

    elif plot_mode == 1: # fixed chessboard = plot in WCS
        
        for i_ex in range(len(rvec)):
            R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[i_ex])[0]) # Camera orientation in world coordinate system
            t_c2w = -R_c2w.dot(tvec[i_ex]).reshape((1,3)) # Camera position in world coordinate system
            
            plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=camera_size)
            print("Plot camera", i_ex, "at", t_c2w)
            print("Rotation: ", R_c2w)
            print("Translation: ", t_c2w)

        ax_in.plot(X_W[:,0], X_W[:,1], X_W[:,2], ".")
    
    plt.show()
    cv2.waitKey(0)
    plt.savefig("calibration/" + camera + "/result.pdf")

if __name__ == "__main__":
    # Camera selection
    camera = "sony" # "sony", "gopro1", "gopro2

    # Define data path
    data_path = "H:/data/calibration/" + camera + "/moving2"

    calibrate(camera, data_path)
