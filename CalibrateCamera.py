# Resources: 
# - OpenCV-Python tutorial for calibration: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#   - Variable names were changed for clarity

import numpy as np
import cv2
import pickle
import glob
import os
from termcolor import colored

def calibrate(camera):

    # Create arrays you"ll use to store object points and image points from all images processed
    objpoints = [] # 3D point in real world space where chess squares are
    imgpoints = [] # 2D point in image plane, determined by CV2

    # Chessboard variables
    CHESSBOARD_CORNERS_ROWCOUNT = 9
    CHESSBOARD_CORNERS_COLCOUNT = 6
    #CHESSBOARD_CORNERS_SIZE = 243 # Physical size of a cell (the distance between neighrboring corners). Any positive number works.

    # Theoretical object points for the chessboard we"re calibrating against,
    # These will come out like: 
    #     (0, 0, 0), (1, 0, 0), ..., 
    #     (CHESSBOARD_CORNERS_ROWCOUNT-1, CHESSBOARD_CORNERS_COLCOUNT-1, 0)
    # Note that the Z value for all stays at 0, as this is a printed out 2D image
    # And also that the max point is -1 of the max because we"re zero-indexing
    # The following line generates all the tuples needed at (0, 0, 0)
    objp = np.zeros((CHESSBOARD_CORNERS_ROWCOUNT*CHESSBOARD_CORNERS_COLCOUNT,3), np.float32)
    # The following line fills the tuples just generated with their values (0, 0, 0), (1, 0, 0), ...
    objp[:,:2] = np.mgrid[0:CHESSBOARD_CORNERS_ROWCOUNT,0:CHESSBOARD_CORNERS_COLCOUNT].T.reshape(-1, 2)

    # Need a set of images or a video taken with the camera you want to calibrate
    # Input images capturing the chessboard above
    input_files = "H:/data/calibration/" + camera + "/*.jpg"
    output_path = "H:/data/calibration/" + camera


    # All images used should be the same size, which if taken with the same camera shouldn"t be a problem
    # I"m using a set of images taken with the camera with the naming convention:
    # "camera-pic-of-chessboard-<NUMBER>.jpg"
    images = glob.glob(input_files)
    imageSize = None # Determined at runtime

    # Count variables
    count_found = 0
    count_failed = 0

    # Loop through images glob"ed
    for image_path in images:
        # Open the image
        img = cv2.imread(image_path)
        # Grayscale the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard in the image, setting PatternSize(2nd arg) to a tuple of (#rows, #columns)
        board, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_ROWCOUNT,CHESSBOARD_CORNERS_COLCOUNT), None)

        # If a chessboard was found, let"s collect image/corner points
        if board == True:
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
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)) # Last parameter is about termination critera
            imgpoints.append(corners_acc)

            # If our image size is unknown, set it now
            if not imageSize:
                imageSize = (gray.shape[1], gray.shape[0])
        
            # Draw the corners to a new image to show whoever is performing the calibration
            # that the board was properly detected
            img = cv2.drawChessboardCorners(img, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), corners_acc, board)
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
    print(imageSize)
    # Now that we"ve seen all of our images, perform the camera calibration
    # based on the set of points we"ve discovered
    rep, K, d, rvec, tvec = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=imageSize,
            cameraMatrix=None,
            distCoeffs=None)
        
    # Save values to be used where matrix+dist is required, for instance for posture estimation
    np.savetxt("calibration/" + camera + "/K.txt", K)
    print("Save intrinsic parameter K = ", K)
    np.savetxt("calibration/" + camera + "/d.txt", d)
    print("Save Distortion parameters d = (k1, k2, p1, p2, k3) = ", d)
        
    # Print to console our success
    print("Calibration successful on " + str(count_found) + " images.")

if __name__ == "__main__":
    # Camera selection
    camera = "sony" # "sony", "gopro1", "gopro2
    calibrate(camera)