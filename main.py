# IMPORT PACKAGES AND MODULES
import sys, os, cv2
import numpy as np
from undistortImage import undistortImage

# IMPORT USER-DEFINED MODULES
import videostoframes
import calibration

# Define used parameters
camera = "gopro1" # "sony", "gopro1", "gopro2
calib_path = "H:/data/calibration/" + camera + "/"
data_path = "H:/data/calibration/" + camera + "/"

# Calibrate camera

if os.path.isfile(calib_path + "K.txt"):
    print("Camera already calibrated. Loading calibration parameters...")
    K = np.loadtxt(calib_path + "K.txt")  # calibration matrix[3x3]
    d = np.loadtxt(calib_path + "d.txt")  # distortion coefficients[5x1]
else:
    print("Calibrating camera...")
    rep, K, d, rvec, tvec, X_W = calibration.calibrate(camera, True)

# Undistort images
undistortImage(camera, K, d, calib_path, "414.JPG")